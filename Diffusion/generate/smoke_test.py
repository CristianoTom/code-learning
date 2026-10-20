"""CPU 数值与端到端检查。临时文件自动清理，不会覆盖正式训练结果。"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

from dataset import ImageDataset
from diffusion import Diffusion
from legacy_model import UNet as LegacyUNet
from model import UNet, build_model
from train import learning_rate
from utils import ROOT, load_checkpoint, prune_checkpoints, to_pil


def run(*arguments):
    subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True)


def check_math(images):
    time = torch.tensor([0, 7])
    noise = torch.randn_like(images)
    for prediction_type in ('epsilon', 'v'):
        diffusion = Diffusion(8, prediction_type)
        noisy = diffusion.add_noise(images, time, noise)
        target = diffusion.training_target(images, time, noise)
        recovered = diffusion.predict_clean(noisy, time, target)
        assert torch.allclose(recovered, images, atol=1e-4), prediction_type

        class Oracle(torch.nn.Module):
            # 一个知道正确 x0 的理想预测器，用来检查采样公式而不是训练质量。
            def forward(self, x, t):
                alpha = diffusion.extract(diffusion.alpha_bars, t)
                correct_noise = (x - alpha.sqrt() * images) / (1 - alpha).sqrt()
                return diffusion.training_target(images, t, correct_noise)

        for method, steps in [('ddpm', 8), ('ddim', 4)]:
            sample = diffusion.sample(Oracle(), 2, 16, torch.Generator().manual_seed(2),
                                      progress=False, method=method, steps=steps)
            assert torch.allclose(sample, images, atol=2e-4), (prediction_type, method)

    assert learning_rate(0, 1e-4, 10, 100, 0.1) == 1e-5
    assert learning_rate(10, 1e-4, 10, 100, 0.1) == 1e-4
    assert math.isclose(learning_rate(100, 1e-4, 10, 100, 0.1), 1e-5)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    dataset = ImageDataset(ROOT / 'data', image_size=16, limit_images=2, cache=True)
    images = torch.stack([dataset.clean_image(0), dataset.clean_image(1)])
    assert images.shape == (2, 3, 16, 16)
    assert images.min() >= -1 and images.max() <= 1
    check_math(images)
    model = UNet(8, 16, dropout=0)
    diffusion = Diffusion(8, 'v')
    loss = diffusion.loss(model, images)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    a = diffusion.sample(model, 2, 16, torch.Generator().manual_seed(1), progress=False,
                         method='ddim', steps=4)
    b = diffusion.sample(model, 2, 16, torch.Generator().manual_seed(1), progress=False,
                         method='ddim', steps=4)
    assert torch.equal(a, b) and torch.isfinite(a).all()
    assert model.training
    # 256 输入也必须下采样到 8；注意力分辨率不超过 16。
    for size in (32, 64, 128, 256):
        network = UNet(8, size, dropout=0).eval()
        assert len(network.downsamples) == int(math.log2(size)) - 3
        with torch.no_grad():
            result = network(torch.zeros(1, 3, size, size), torch.zeros(1, dtype=torch.long))
        assert result.shape == (1, 3, size, size) and torch.isfinite(result).all()

    (ROOT / 'outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ddpm_smoke_', dir=ROOT / 'outputs') as temporary:
        directory = Path(temporary)
        snapshots = directory / 'retention'
        snapshots.mkdir()
        for name in ('epoch_0001.pt', 'epoch_0002.pt', 'epoch_0003.pt', 'last.pt', 'best.pt'):
            (snapshots / name).touch()
        prune_checkpoints(snapshots, 2)
        assert {p.name for p in snapshots.iterdir()} == {'epoch_0002.pt', 'epoch_0003.pt', 'last.pt', 'best.pt'}
        data = directory / 'data'
        data.mkdir()
        for index, image in enumerate(images):
            to_pil(image).save(data / f'{index}.png')
        output = directory / 'output'
        run('train.py', '--data-dir', data, '--output-dir', output,
            '--image-size', 16, '--base-channels', 8, '--timesteps', 8,
            '--epochs', 1, '--batch-size', 2, '--sample-every', 1, '--preview-count', 2, '--save-every', 1,
            '--warmup-steps', 0, '--decay-steps', 100, '--lr', 0.001, '--device', 'cpu')
        checkpoint = output / 'checkpoints/last.pt'
        first = load_checkpoint(checkpoint)
        assert first['global_step'] == 1 and first['epoch'] == 1
        assert first['config']['prediction_type'] == 'v'
        assert (output / 'checkpoints/epoch_0001.pt').is_file()
        run('train.py', '--data-dir', data, '--output-dir', output,
            '--resume', checkpoint, '--epochs', 2, '--batch-size', 2, '--lr', 0.0002,
            '--sample-every', 0, '--device', 'cpu')
        second = load_checkpoint(checkpoint)
        assert second['global_step'] == 2 and second['epoch'] == 2
        assert second['config']['lr'] == 0.0002
        expected_lr = learning_rate(1, 0.0002, 0, 100, 0.1)
        assert math.isclose(second['optimizer']['param_groups'][0]['lr'], expected_lr)
        assert any(not torch.equal(first['model'][k], second['model'][k]) for k in first['model'])
        for method in ('ddim', 'ddpm'):
            destination = directory / method
            run('sample.py', '--checkpoint', checkpoint, '--output-dir', destination,
                '--num-images', 3, '--batch-size', 2, '--sampler', method, '--device', 'cpu')
            for index in range(3):
                with Image.open(destination / f'image_{index:04d}.png') as image:
                    assert image.size == (16, 16) and image.mode == 'RGB'
            assert (destination / 'grid.png').is_file()
            assert json.loads((destination / 'generation.json').read_text(encoding='utf-8'))['actual_steps'] == 8
        assert (output / 'previews/ema_0001.png').is_file()
        assert (output / 'previews/raw_0001.png').is_file()
        with Image.open(output / 'diagnostics/denoise_0001.png') as image:
            assert image.size == (32, 112)

        # 老权重的网络和 epsilon 日程不能被误当成 v2/v 预测。
        legacy = LegacyUNet(8)
        legacy_config = dict(image_size=16, base_channels=8, timesteps=8)
        assert isinstance(build_model(legacy_config), LegacyUNet)
        legacy_path = directory / 'legacy.pt'
        torch.save(dict(config=legacy_config, model=legacy.state_dict(),
                        ema_model=legacy.state_dict(), global_step=0), legacy_path)
        run('sample.py', '--checkpoint', legacy_path, '--output-dir', directory / 'legacy',
            '--num-images', 1, '--device', 'cpu', '--sampler', 'ddpm')
    print('全部通过：两种预测公式、理想预测器采样、16～256 网络、反向梯度、训练、诊断、续训学习率覆盖、新旧权重生成。')


if __name__ == '__main__':
    main()

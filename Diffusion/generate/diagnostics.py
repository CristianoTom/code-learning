"""固定训练样本、固定噪声的去噪检查；不是独立验证集的泛化指标。"""

import csv

import torch
from PIL import Image

from utils import to_pil


@torch.no_grad()
def save_diagnostics(model, diffusion, clean_images, directory, epoch):
    directory.mkdir(parents=True, exist_ok=True)
    device = clean_images.device
    generator = torch.Generator(device=device).manual_seed(2026)
    noise = torch.randn(clean_images.shape, device=device, generator=generator)
    rows = [clean_images]
    records = []
    was_training = model.training
    model.eval()
    try:
        # 每个噪声强度两行：带噪输入、模型一次预测的 x0。
        for fraction in (0.1, 0.5, 0.9):
            step = min(diffusion.timesteps - 1, int(fraction * diffusion.timesteps))
            time = torch.full((len(clean_images),), step, dtype=torch.long, device=device)
            noisy = diffusion.add_noise(clean_images, time, noise)
            prediction = model(noisy, time).float()
            target = diffusion.training_target(clean_images, time, noise)
            recovered = diffusion.predict_clean(noisy, time, prediction)
            records.append([epoch, fraction, (prediction - target).square().mean().item(),
                            (recovered - clean_images).square().mean().item()])
            rows.extend([noisy.clamp(-1, 1), recovered.clamp(-1, 1)])
        size = clean_images.shape[-1]
        canvas = Image.new('RGB', (len(clean_images) * size, len(rows) * size), 'white')
        for row, images in enumerate(rows):
            for column, image in enumerate(images):
                canvas.paste(to_pil(image), (column * size, row * size))
        canvas.save(directory / f'denoise_{epoch:04d}.png')
        with (directory / 'metrics.csv').open('a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['epoch', 'noise_fraction', 'target_mse', 'x0_mse_unclipped'])
            writer.writerows(records)
    finally:
        model.train(was_training)

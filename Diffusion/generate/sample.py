"""从随机高斯噪声出发，使用已训练的模型生成新图片。"""

import argparse
import json
from pathlib import Path

import torch

from diffusion import Diffusion
from model import build_model
from utils import ROOT, choose_device, load_checkpoint, save_grid, to_pil


def main():
    parser = argparse.ArgumentParser(description='使用 DDPM 权重生成新图片')
    parser.add_argument('--checkpoint', default=str(ROOT / 'outputs_v2/checkpoints/last.pt'))
    parser.add_argument('--output-dir', default=str(ROOT / 'outputs_v2/generated'))
    parser.add_argument('--num-images', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=4, help='显存不足时减小此值')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--raw-weights', action='store_true', help='使用原始权重；默认使用 EMA 权重')
    parser.add_argument('--sampler', choices=['ddim', 'ddpm'], default='ddim')
    parser.add_argument('--steps', type=int, help='默认 DDIM 100 步；DDPM 必须使用全部训练时间步')
    args = parser.parse_args()
    if min(args.num_images, args.batch_size) < 1:
        parser.error('num-images 和 batch-size 必须为正数')
    device = choose_device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint['config']
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint['model' if args.raw_weights else 'ema_model'])
    diffusion = Diffusion(config['timesteps'], config.get('prediction_type', 'epsilon')).to(device)
    steps = args.steps
    if steps is None:
        steps = min(100, config['timesteps']) if args.sampler == 'ddim' else config['timesteps']
    if not 2 <= steps <= config['timesteps']:
        parser.error('--steps 必须在 2 和训练扩散步数之间')
    if args.sampler == 'ddpm' and steps != config['timesteps']:
        parser.error('DDPM 不能跳过时间步；请使用 --sampler ddim 或移除 --steps')
    generator = torch.Generator(device=device).manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_images = []
    for start in range(0, args.num_images, args.batch_size):
        count = min(args.batch_size, args.num_images - start)
        images = diffusion.sample(model, count, config['image_size'], generator,
                                  method=args.sampler, steps=steps).cpu()
        for offset, image in enumerate(images):
            to_pil(image).save(output / f'image_{start + offset:04d}.png')
        all_images.append(images)
    save_grid(torch.cat(all_images), output / 'grid.png')
    metadata = dict(vars(args), actual_steps=steps, model_config=config,
                    checkpoint_global_step=checkpoint['global_step'])
    (output / 'generation.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'已生成 {args.num_images} 张图片：{output.resolve()}')


if __name__ == '__main__':
    main()

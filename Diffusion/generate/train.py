"""小数据集扩散训练：多尺度 U-Net、v 预测、EMA、预热与余弦学习率。"""

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import ImageDataset
from diagnostics import save_diagnostics
from diffusion import Diffusion
from model import build_model
from utils import ROOT, choose_device, load_checkpoint, prune_checkpoints, save_checkpoint, save_grid, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description='从零训练适合小图片集的扩散模型')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--output-dir', default=str(ROOT / 'outputs_v2'))
    parser.add_argument('--image-size', type=int, choices=[16, 32, 64, 128, 256], default=64)
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--prediction-type', choices=['v', 'epsilon'], default='v')
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=5000, help='总轮数上限，不是续训增加的轮数')
    parser.add_argument('--max-steps', type=int, default=40000, help='总优化步数上限；0 表示只受 epochs 限制')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None, help='初始默认 1e-4；续训显式指定时会覆盖旧值')
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--decay-steps', type=int, default=40000, help='余弦衰减结束的总优化步数')
    parser.add_argument('--min-lr-ratio', type=float, default=0.1)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--cache', action=argparse.BooleanOptionalAction, default=True,
                        help='在内存缓存缩小后的图片；可用 --no-cache 关闭')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--amp', action='store_true', help='启用 CUDA 混合精度')
    parser.add_argument('--horizontal-flip', action='store_true', help='可选镜像增强，含文字时不要启用')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=50, help='每隔多少轮保存最新权重及轮次权重')
    parser.add_argument('--keep-checkpoints', type=int, default=3, help='保留最近 N 个轮次权重；0 保留全部')
    parser.add_argument('--sample-every', type=int, default=100, help='对照预览与诊断的轮次间隔；0 关闭')
    parser.add_argument('--preview-count', type=int, default=4)
    parser.add_argument('--preview-steps', type=int, default=100, help='DDIM 预览步数，自动限制在 timesteps 内')
    parser.add_argument('--limit-images', type=int, default=0, help='只取前 N 张做拟合检查；0 使用全部数据')
    parser.add_argument('--resume', help='恢复新版 checkpoint；旧版网络需要重新训练')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.save_every, args.preview_count) < 1:
        parser.error('epochs、batch-size、save-every、preview-count 必须为正数')
    if min(args.num_workers, args.sample_every, args.max_steps, args.limit_images, args.keep_checkpoints) < 0:
        parser.error('workers、采样间隔、最大步数和图片限制不能为负数')
    if args.base_channels < 8 or args.base_channels % 2 or args.timesteps < 2 or args.preview_steps < 2:
        parser.error('基础通道必须是至少为 8 的偶数；扩散及预览步数至少为 2')
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        parser.error('学习率必须为有限正数')
    if not 0 <= args.dropout < 1 or not 0 <= args.ema_decay < 1:
        parser.error('dropout 和 ema-decay 必须在 [0, 1) 内')
    if args.warmup_steps < 0 or args.decay_steps <= args.warmup_steps or not 0 < args.min_lr_ratio <= 1:
        parser.error('要求 0 <= warmup-steps < decay-steps，且 0 < min-lr-ratio <= 1')
    return args


def learning_rate(step, base_lr, warmup_steps, decay_steps, min_ratio):
    """按已经完成的优化步数计算下一步学习率；恢复训练不会重新预热。"""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / (decay_steps - warmup_steps))
    factor = min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * progress)) / 2
    return base_lr * factor


@torch.no_grad()
def update_ema(ema_model, model, decay, global_step):
    # 前期及时跟随网络，避免 EMA 长时间停留在初始化附近。
    effective_decay = min(decay, (1 + global_step) / (10 + global_step))
    for average, current in zip(ema_model.parameters(), model.parameters()):
        average.lerp_(current, 1 - effective_decay)
    for average, current in zip(ema_model.buffers(), model.buffers()):
        average.copy_(current)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))

    checkpoint = load_checkpoint(args.resume) if args.resume else None
    requested_lr = args.lr
    if checkpoint:
        if checkpoint['config'].get('architecture_version', 1) != 2:
            raise ValueError('旧版小网络与新版结构不兼容。请使用新的输出目录从头训练；旧权重仍可用 sample.py 生成。')
        # 模型与日程必须匹配，不能在续训时悄悄更换预测目标或分辨率。
        restored_keys = ('image_size', 'base_channels', 'timesteps', 'dropout', 'prediction_type',
                         'ema_decay', 'horizontal_flip', 'warmup_steps', 'decay_steps',
                         'min_lr_ratio', 'limit_images', 'seed')
        for key in restored_keys:
            setattr(args, key, checkpoint['config'][key])
        args.lr = checkpoint['config']['lr'] if requested_lr is None else requested_lr
    elif args.lr is None:
        args.lr = 1e-4

    output = Path(args.output_dir)
    if not checkpoint and ((output / 'checkpoints/last.pt').exists() or (output / 'loss.csv').exists()):
        raise ValueError('输出目录已有训练结果。请用 --resume 续训，或者指定新的 --output-dir。')
    output.mkdir(parents=True, exist_ok=True)
    dataset = ImageDataset(args.data_dir, args.image_size, args.horizontal_flip,
                           cache=args.cache, limit_images=args.limit_images)
    data_files = [p.relative_to(Path(args.data_dir)).as_posix() for p in dataset.paths]
    if checkpoint and checkpoint['data_files'] != data_files:
        raise ValueError('数据文件列表与保存时不同。请检查 data-dir；更换数据集请在新目录从头训练。')
    (output / 'data_files.json').write_text(json.dumps(data_files, indent=2, ensure_ascii=False), encoding='utf-8')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                        persistent_workers=args.num_workers > 0)
    save_grid(torch.stack([dataset.clean_image(i) for i in range(min(16, len(dataset)))]),
              output / 'training_data.png')
    # 均匀取固定的训练样本做诊断，不把它误称为独立验证集。
    indices = torch.linspace(0, len(dataset) - 1, min(args.preview_count, len(dataset))).long().tolist()
    diagnostic_images = torch.stack([dataset.clean_image(i) for i in indices]).to(device)

    config = dict(vars(args), architecture_version=2)
    model = build_model(config).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = Diffusion(args.timesteps, args.prediction_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp_enabled = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    completed_epochs, global_step = 0, 0
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if amp_enabled and checkpoint['scaler']:
            scaler.load_state_dict(checkpoint['scaler'])
        completed_epochs = checkpoint['epoch']
        global_step = checkpoint['global_step']
        torch.set_rng_state(checkpoint['rng_state'])
        if device.type == 'cuda' and checkpoint.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'], device=device)
        print(f'恢复：已完成 {completed_epochs} 轮、{global_step} 步；基础学习率 {args.lr:g}')
        # optimizer.load_state_dict 会带回旧学习率。循环内每一步显式重新设置。
    del checkpoint
    (output / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'设备：{device} | 图片：{len(dataset)} | 分辨率：{args.image_size} | '
          f'参数量：{sum(p.numel() for p in model.parameters()):,} | 预测目标：{args.prediction_type}')
    print(f'每轮约 {len(loader)} 步；上限 {args.epochs} 轮 / {args.max_steps or "不限"} 步，以先到者为准。')
    if args.image_size > 128:
        print('建议先在 64/128 分辨率确认结构已学会，再训练 256；小数据量不能保证生成质量。')
    if completed_epochs >= args.epochs or (args.max_steps and global_step >= args.max_steps):
        print('已经达到目标轮数或步数，无需继续。')
        return

    def save_state(path):
        save_checkpoint(dict(config=config, model=model.state_dict(), ema_model=ema_model.state_dict(),
                             optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                             epoch=completed_epochs, global_step=global_step, data_files=data_files,
                             rng_state=torch.get_rng_state(),
                             cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None), path)

    try:
        with (output / 'loss.csv').open('a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['epoch', 'global_step', 'loss', 'lr'])
            for epoch in range(completed_epochs, args.epochs):
                model.train()
                loss_sum, seen = 0.0, 0
                progress = tqdm(loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
                for batch_index, images in enumerate(progress):
                    current_lr = learning_rate(global_step, args.lr, args.warmup_steps,
                                               args.decay_steps, args.min_lr_ratio)
                    for group in optimizer.param_groups:
                        group['lr'] = current_lr
                    images = images.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=amp_enabled):
                        loss = diffusion.loss(model, images)
                    if not torch.isfinite(loss):
                        raise RuntimeError('损失出现 NaN/Inf；请降低学习率或关闭 AMP')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(gradient_norm):
                        if not amp_enabled:
                            raise RuntimeError('梯度出现 NaN/Inf，请降低学习率')
                        scaler.step(optimizer)
                        scaler.update()
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                    global_step += 1
                    update_ema(ema_model, model, args.ema_decay, global_step)
                    loss_sum += loss.item() * len(images)
                    seen += len(images)
                    progress.set_postfix(loss=f'{loss.item():.4f}', lr=f'{current_lr:.2g}', step=global_step)
                    if args.max_steps and global_step >= args.max_steps:
                        break
                if not seen:
                    raise RuntimeError('本轮没有成功的优化步骤，请关闭 AMP 后重试')
                # 若在轮中间停止，恢复时重跑这一轮，不声称精确恢复到某张图片。
                completed_epochs = epoch + 1 if batch_index == len(loader) - 1 else epoch
                writer.writerow([epoch + 1, global_step, loss_sum / seen, current_lr])
                file.flush()
                stopping = bool(args.max_steps and global_step >= args.max_steps) or epoch + 1 == args.epochs
                if (epoch + 1) % args.save_every == 0 or stopping:
                    save_state(output / 'checkpoints/last.pt')
                    if (epoch + 1) % args.save_every == 0:
                        save_state(output / 'checkpoints' / f'epoch_{epoch + 1:04d}.pt')
                        prune_checkpoints(output / 'checkpoints', args.keep_checkpoints)
                if args.sample_every and ((epoch + 1) % args.sample_every == 0 or stopping):
                    for name, preview_model in [('ema', ema_model), ('raw', model)]:
                        generator = torch.Generator(device=device).manual_seed(args.seed + 10000)
                        preview = diffusion.sample(preview_model, args.preview_count, args.image_size,
                                                   generator, method='ddim',
                                                   steps=min(args.preview_steps, args.timesteps))
                        save_grid(preview, output / 'previews' / f'{name}_{epoch + 1:04d}.png')
                    save_diagnostics(ema_model, diffusion, diagnostic_images, output / 'diagnostics', epoch + 1)
                if stopping:
                    break
    except KeyboardInterrupt:
        save_state(output / 'checkpoints/last.pt')
        print('已保存中断时权重，恢复时会从当前未完成轮次的开头继续。')
        return
    print(f'训练结束。最新权重：{output / "checkpoints/last.pt"}')


if __name__ == '__main__':
    main()

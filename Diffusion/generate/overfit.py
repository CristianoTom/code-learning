"""用 8 张图片检查拟合能力。它是排查工具，不是正式训练，也不是画质保证。"""

import argparse
import subprocess
import sys

from utils import ROOT


def main():
    parser = argparse.ArgumentParser(description='小数据拟合检查：先确认模型能学会少量同风格图片')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--output-dir', default=str(ROOT / 'outputs_overfit'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    command = [sys.executable, str(ROOT / 'train.py'), '--data-dir', args.data_dir,
               '--output-dir', args.output_dir, '--device', args.device,
               '--image-size', '64', '--base-channels', '32', '--dropout', '0',
               '--limit-images', '8', '--batch-size', '8', '--epochs', '3000',
               '--max-steps', '3000', '--warmup-steps', '50', '--decay-steps', '3000',
               '--save-every', '500', '--sample-every', '250', '--preview-steps', '100']
    if args.amp:
        command.append('--amp')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()

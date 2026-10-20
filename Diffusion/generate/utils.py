"""训练与生成共用的小工具。"""

import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('当前 PyTorch 无法使用 CUDA，请安装与设备匹配的 CUDA 版 PyTorch')
    return torch.device(name)


def to_pil(tensor):
    pixels = ((tensor.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return Image.fromarray(pixels.permute(1, 2, 0).numpy())


def save_grid(images, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = math.ceil(math.sqrt(len(images)))
    size = images.shape[-1]
    grid = Image.new('RGB', (columns * size, math.ceil(len(images) / columns) * size), 'white')
    for index, image in enumerate(images):
        grid.paste(to_pil(image), ((index % columns) * size, (index // columns) * size))
    grid.save(path)


def load_checkpoint(path):
    # 本项目保存的内容由张量和基础 Python 类型组成，无需加载任意对象。
    return torch.load(path, map_location='cpu', weights_only=True)


def save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def prune_checkpoints(directory, keep):
    """只清理本项目按轮次生成的快照，last.pt 和手工命名的权重不受影响。"""
    if keep == 0:
        return
    snapshots = [p for p in Path(directory).iterdir()
                 if p.is_file() and not p.is_symlink() and re.fullmatch(r'epoch_\d+\.pt', p.name)]
    snapshots.sort(key=lambda p: int(p.stem.split('_')[1]))
    for path in snapshots[:-keep]:
        path.unlink()

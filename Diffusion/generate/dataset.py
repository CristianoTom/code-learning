"""读取普通图片文件夹，不需要标签或额外的数据整理。"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class ImageDataset(Dataset):
    def __init__(self, root, image_size=64, horizontal_flip=False, cache=False, limit_images=0):
        self.paths = sorted(
            p for p in Path(root).rglob('*')
            if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        )
        if not self.paths:
            raise ValueError(f'没有找到图片，请检查数据目录：{root}')
        if limit_images:
            # 只用于小数据拟合检查，正常训练保持为 0，使用所有图片。
            self.paths = self.paths[:limit_images]
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        self.cached = None
        if cache:
            # 缓存缩小后的 uint8 图片，避免每一轮重复解码几千像素的原图。
            self.cached = [self.read_image(p) for p in tqdm(self.paths, desc='缓存预处理图片')]

    def __len__(self):
        return len(self.paths)

    def read_image(self, path):
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert('RGBA')
                # 透明区域使用白底，避免直接转 RGB 后变成黑底。
                background = Image.new('RGBA', image.size, 'white')
                image = Image.alpha_composite(background, image).convert('RGB')
                # 等比缩放后补白边，不裁剪图案，也不改变长宽比例。
                image = ImageOps.pad(
                    image, (self.image_size, self.image_size),
                    method=Image.Resampling.LANCZOS, color='white',
                )
                array = np.array(image, dtype=np.uint8, copy=True)
        except Exception as error:
            raise RuntimeError(f'无法读取图片：{path}') from error
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def clean_image(self, index):
        pixels = self.cached[index] if self.cached is not None else self.read_image(self.paths[index])
        return pixels.float() / 127.5 - 1.0

    def __getitem__(self, index):
        image = self.clean_image(index)
        if self.horizontal_flip and torch.rand(()).item() < 0.5:
            image = image.flip(-1)
        return image

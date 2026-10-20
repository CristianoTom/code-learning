"""多尺度 U-Net：先理解整体结构，再逐级恢复细节。只用 PyTorch 基础层。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def normalization(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


class TimeEmbedding(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        half = dimension // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
        self.register_buffer('frequencies', frequencies)

    def forward(self, time):
        angles = time.float()[:, None] * self.frequencies[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=1)


class ResidualBlock(nn.Module):
    """用时间向量调节特征的缩放和偏移，告知网络当前噪声强度。"""

    def __init__(self, in_channels, out_channels, time_dimension, dropout):
        super().__init__()
        self.first = nn.Sequential(
            normalization(in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(time_dimension, out_channels * 2))
        self.norm = normalization(out_channels)
        self.second = nn.Sequential(
            nn.SiLU(), nn.Dropout(dropout), nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_embedding):
        hidden = self.first(x)
        scale, shift = self.time_projection(time_embedding).chunk(2, dim=1)
        hidden = self.norm(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.shortcut(x) + self.second(hidden)


class Attention(nn.Module):
    """仅在 <=16×16 的特征上使用注意力，避免大图占用巨量显存。"""

    def __init__(self, channels):
        super().__init__()
        self.heads = math.gcd(4, channels)
        self.norm = normalization(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x)).reshape(batch, 3, self.heads, channels // self.heads, height * width)
        q, k, v = qkv.unbind(dim=1)
        hidden = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2),
        )
        hidden = hidden.transpose(-1, -2).reshape(batch, channels, height, width)
        return x + self.output(hidden)


class Stage(nn.Module):
    """每个分辨率使用两个残差块，而不是只做一次卷积。"""

    def __init__(self, in_channels, out_channels, time_dimension, dropout, use_attention):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(in_channels, out_channels, time_dimension, dropout),
            ResidualBlock(out_channels, out_channels, time_dimension, dropout),
        ])
        self.attention = Attention(out_channels) if use_attention else nn.Identity()

    def forward(self, x, time_embedding):
        for block in self.blocks:
            x = block(x, time_embedding)
        return self.attention(x)


class UNet(nn.Module):
    def __init__(self, base_channels=64, image_size=64, dropout=0.1):
        super().__init__()
        if base_channels < 8 or base_channels % 2:
            raise ValueError('base_channels 必须是至少为 8 的偶数')
        if image_size < 16 or image_size > 256 or image_size & (image_size - 1):
            raise ValueError('image_size 请选择 16、32、64、128 或 256')
        self.image_size = image_size
        # 64 -> 32 -> 16 -> 8；128、256 自动增加层数，最深处保持 8×8。
        levels = int(math.log2(image_size)) - 2
        multipliers = (1, 2, 3, 4, 4, 4)[:levels]
        channels = [base_channels * multiplier for multiplier in multipliers]
        time_dimension = base_channels * 4
        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels), nn.Linear(base_channels, time_dimension),
            nn.SiLU(), nn.Linear(time_dimension, time_dimension),
        )
        self.input = nn.Conv2d(3, channels[0], 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        previous = channels[0]
        for level, current in enumerate(channels):
            resolution = image_size // (2 ** level)
            self.encoders.append(Stage(previous, current, time_dimension, dropout, resolution <= 16))
            if level < levels - 1:
                self.downsamples.append(nn.Conv2d(current, current, 3, stride=2, padding=1))
            previous = current
        self.middle = Stage(channels[-1], channels[-1], time_dimension, dropout, True)
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(levels)):
            current = channels[level]
            self.decoders.append(Stage(previous + current, current, time_dimension, dropout,
                                       image_size // (2 ** level) <= 16))
            if level > 0:
                self.upsamples.append(nn.Conv2d(current, current, 3, padding=1))
            previous = current
        self.output = nn.Sequential(normalization(channels[0]), nn.SiLU(),
                                    nn.Conv2d(channels[0], 3, 3, padding=1))

    def forward(self, x, time):
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f'模型需要 {self.image_size}×{self.image_size} 的输入')
        time_embedding = self.time_embedding(time)
        hidden = self.input(x)
        skips = []
        for level, encoder in enumerate(self.encoders):
            hidden = encoder(hidden, time_embedding)
            skips.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)
        hidden = self.middle(hidden, time_embedding)
        for level, decoder in enumerate(self.decoders):
            hidden = decoder(torch.cat([hidden, skips.pop()], dim=1), time_embedding)
            if level < len(self.upsamples):
                hidden = F.interpolate(hidden, scale_factor=2, mode='nearest')
                hidden = self.upsamples[level](hidden)
        return self.output(hidden)


def build_model(config):
    version = config.get('architecture_version', 1)
    if version == 1:
        # 原项目权重仍可生成，但不能用于新网络续训。
        from legacy_model import UNet as LegacyUNet
        return LegacyUNet(config['base_channels'])
    if version != 2:
        raise ValueError(f'不支持的网络版本：{version}')
    return UNet(config['base_channels'], config['image_size'], config['dropout'])

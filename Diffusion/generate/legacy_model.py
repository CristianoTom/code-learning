"""从基础 PyTorch 层构建小型 U-Net：输入带噪图片和时间，输出预测噪声。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def normalization(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


class TimeEmbedding(nn.Module):
    """把整数时间步编码成不同频率的正弦、余弦向量。"""

    def __init__(self, dimension):
        super().__init__()
        half = dimension // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
        self.register_buffer('frequencies', frequencies)

    def forward(self, time):
        angles = time.float()[:, None] * self.frequencies[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dimension):
        super().__init__()
        self.first = nn.Sequential(
            normalization(in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(time_dimension, out_channels))
        self.second = nn.Sequential(
            normalization(out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.shortcut = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x, time_embedding):
        hidden = self.first(x)
        hidden = hidden + self.time_projection(time_embedding)[:, :, None, None]
        return self.second(hidden) + self.shortcut(x)


class Attention(nn.Module):
    """只在最低分辨率使用自注意力，让远处的像素交换信息。"""

    def __init__(self, channels):
        super().__init__()
        self.norm = normalization(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).flatten(2).chunk(3, dim=1)
        weights = torch.bmm(q.transpose(1, 2), k) * (channels ** -0.5)
        weights = weights.softmax(dim=-1)
        hidden = torch.bmm(v, weights.transpose(1, 2)).reshape(batch, channels, height, width)
        return x + self.output(hidden)


class UNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        if base_channels < 8 or base_channels % 2:
            raise ValueError('base_channels 必须是大于等于 8 的偶数')
        c = base_channels
        time_dimension = c * 4
        self.time_embedding = nn.Sequential(
            TimeEmbedding(c), nn.Linear(c, time_dimension), nn.SiLU(),
            nn.Linear(time_dimension, time_dimension),
        )
        self.input = nn.Conv2d(3, c, 3, padding=1)
        self.encoder1 = ResidualBlock(c, c, time_dimension)
        self.down1 = nn.Conv2d(c, c, 4, stride=2, padding=1)
        self.encoder2 = ResidualBlock(c, c * 2, time_dimension)
        self.down2 = nn.Conv2d(c * 2, c * 2, 4, stride=2, padding=1)
        self.middle1 = ResidualBlock(c * 2, c * 4, time_dimension)
        self.attention = Attention(c * 4)
        self.middle2 = ResidualBlock(c * 4, c * 2, time_dimension)
        self.decoder2 = ResidualBlock(c * 4, c, time_dimension)
        self.decoder1 = ResidualBlock(c * 2, c, time_dimension)
        self.output = nn.Sequential(normalization(c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))

    def forward(self, x, time):
        time_embedding = self.time_embedding(time)
        skip1 = self.encoder1(self.input(x), time_embedding)
        skip2 = self.encoder2(self.down1(skip1), time_embedding)
        hidden = self.middle1(self.down2(skip2), time_embedding)
        hidden = self.middle2(self.attention(hidden), time_embedding)
        hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode='nearest')
        hidden = self.decoder2(torch.cat([hidden, skip2], dim=1), time_embedding)
        hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode='nearest')
        hidden = self.decoder1(torch.cat([hidden, skip1], dim=1), time_embedding)
        return self.output(hidden)

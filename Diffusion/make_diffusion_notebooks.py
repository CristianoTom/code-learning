import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_IMPORTS = r"""
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
"""


SCHEDULE_CODE = r"""
class DiffusionSchedule:
    def __init__(self, timesteps=300, beta_start=1e-4, beta_end=0.02, device=device):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - torch.cat([torch.ones(1, device=device), self.alpha_bars[:-1]])) / (1.0 - self.alpha_bars)


def extract(values, t, x_shape):
    batch_size = t.shape[0]
    out = values.gather(0, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def q_sample(x0, t, schedule, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    return (
        extract(schedule.sqrt_alpha_bars, t, x0.shape) * x0
        + extract(schedule.sqrt_one_minus_alpha_bars, t, x0.shape) * noise
    )
"""


TIME_EMBED_CODE = r"""
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
"""


MLP_DENOISER_CODE = r"""
class MLPDenoiser(nn.Module):
    def __init__(self, data_dim=2, time_dim=64, hidden_dim=256, num_classes=None):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
        )
        self.label_emb = nn.Embedding(num_classes, hidden_dim) if num_classes is not None else None
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x, t, y=None):
        h = self.time_mlp(t)
        if self.label_emb is not None and y is not None:
            h = h + self.label_emb(y)
        return self.net(torch.cat([x, h], dim=1))
"""


UNET_CODE = r"""
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, t):
        h = self.bnorm1(F.silu(self.conv1(x)))
        time_emb = F.silu(self.time_mlp(t))
        h = h + time_emb[..., None, None]
        h = self.bnorm2(F.silu(self.conv2(h)))
        return self.transform(h)


class SimpleUnet(nn.Module):
    def __init__(self, image_channels=1, base_channels=32, time_dim=128, num_classes=None):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.label_emb = nn.Embedding(num_classes, time_dim) if num_classes is not None else None
        self.conv0 = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        self.downs = nn.ModuleList([
            Block(base_channels, base_channels * 2, time_dim),
            Block(base_channels * 2, base_channels * 4, time_dim),
        ])
        self.ups = nn.ModuleList([
            Block(base_channels * 4, base_channels * 2, time_dim, up=True),
            Block(base_channels * 2, base_channels, time_dim, up=True),
        ])
        self.output = nn.Conv2d(base_channels, image_channels, 1)

    def forward(self, x, timestep, y=None):
        t = self.time_mlp(timestep)
        if self.label_emb is not None and y is not None:
            t = t + self.label_emb(y)

        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)
"""


SAMPLING_CODE = r"""
@torch.no_grad()
def p_sample(model, x, t, t_index, schedule, y=None):
    betas_t = extract(schedule.betas, t, x.shape)
    sqrt_one_minus_alpha_bars_t = extract(schedule.sqrt_one_minus_alpha_bars, t, x.shape)
    sqrt_recip_alphas_t = extract(schedule.sqrt_recip_alphas, t, x.shape)

    predicted_noise = model(x, t, y) if y is not None else model(x, t)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alpha_bars_t)

    if t_index == 0:
        return model_mean
    posterior_variance_t = extract(schedule.posterior_variance, t, x.shape)
    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample(model, shape, schedule, y=None):
    model.eval()
    x = torch.randn(shape, device=device)
    for i in tqdm(reversed(range(schedule.timesteps)), total=schedule.timesteps):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, i, schedule, y=y)
    return x
"""


IMAGE_HELPERS = r"""
def show_images(x, nrow=8, title=None):
    from torchvision.utils import make_grid

    x = x.detach().cpu()
    x = (x.clamp(-1, 1) + 1) / 2
    grid = make_grid(x, nrow=nrow)
    plt.figure(figsize=(nrow, max(2, len(x) // nrow)))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()
"""


def write_notebook(name, cells):
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    path = NOTEBOOK_DIR / name
    path.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=2), encoding="utf-8")


def build():
    write_notebook(
        "00_diffusion_math_and_forward_process.ipynb",
        [
            md("""# 00 Diffusion 数学与前向加噪

目标：先不训练神经网络，只理解 DDPM 的核心变量。

你会看到：
- 为什么扩散模型要逐步加噪；
- `beta`, `alpha`, `alpha_bar` 分别是什么；
- 如何从原图 `x0` 一步采样到任意时刻 `xt`；
- 为什么训练目标通常是预测噪声 `epsilon`。
"""),
            code(COMMON_IMPORTS),
            code(SCHEDULE_CODE),
            md("""## 观察噪声调度

`alpha_bar_t` 越小，说明原始信号保留越少，噪声占比越高。"""),
            code(r"""
schedule = DiffusionSchedule(timesteps=300)

plt.figure(figsize=(8, 4))
plt.plot(schedule.betas.cpu(), label="beta")
plt.plot(schedule.alpha_bars.cpu(), label="alpha_bar")
plt.legend()
plt.title("Noise schedule")
plt.show()
"""),
            md("""## 用一张手写数字演示前向扩散

如果第一次运行缺少 `torchvision`，请先在终端或 notebook 中安装依赖：

```bash
pip install torch torchvision matplotlib tqdm
```
"""),
            code(r"""
from torchvision import datasets, transforms
from torchvision.utils import make_grid

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1),
])
dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
x0, label = dataset[0]
x0 = x0.unsqueeze(0).to(device)

timesteps_to_show = torch.linspace(0, schedule.timesteps - 1, 10).long().to(device)
images = []
for t in timesteps_to_show:
    xt = q_sample(x0, t.view(1), schedule)
    images.append(xt)

grid = make_grid(torch.cat(images).cpu().clamp(-1, 1), nrow=10, normalize=True, value_range=(-1, 1))
plt.figure(figsize=(14, 2))
plt.imshow(grid.permute(1, 2, 0), cmap="gray")
plt.axis("off")
plt.title(f"Forward diffusion on MNIST label={label}")
plt.show()
"""),
            md("""## 训练目标

前向过程：

```text
xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
```

训练一个网络 `epsilon_theta(xt, t)` 去预测真实噪声 `epsilon`，损失就是：

```text
MSE(epsilon, epsilon_theta(xt, t))
```

后续 notebook 会把这个训练目标接到 MLP、U-Net 和条件生成上。"""),
        ],
    )

    write_notebook(
        "01_case_toy_2d_from_noise_to_points.ipynb",
        [
            md("""# 01 案例一：2D 玩具数据扩散生成

这个案例用二维点代替图片。它训练很快，适合先把完整 DDPM 流程跑通：

1. 构造真实数据分布；
2. 随机选择时间步 `t`；
3. 前向加噪得到 `xt`；
4. 用 MLP 预测噪声；
5. 从纯噪声反向采样生成新点。
"""),
            code(COMMON_IMPORTS),
            code(SCHEDULE_CODE),
            code(TIME_EMBED_CODE),
            code(MLP_DENOISER_CODE),
            code(SAMPLING_CODE),
            code(r"""
def make_two_moons(n=20000, noise=0.06):
    angles = torch.rand(n) * math.pi
    moon1 = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    moon2 = torch.stack([1 - torch.cos(angles), 0.5 - torch.sin(angles)], dim=1)
    x = torch.cat([moon1[: n // 2], moon2[n // 2 :]], dim=0)
    x = x + noise * torch.randn_like(x)
    x = (x - x.mean(0)) / x.std(0)
    return x.float()

data = make_two_moons()
loader = DataLoader(TensorDataset(data), batch_size=512, shuffle=True)

plt.figure(figsize=(4, 4))
plt.scatter(data[:3000, 0], data[:3000, 1], s=3, alpha=0.5)
plt.axis("equal")
plt.title("Training data")
plt.show()
"""),
            code(r"""
schedule = DiffusionSchedule(timesteps=200)
model = MLPDenoiser(data_dim=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

epochs = 20
for epoch in range(epochs):
    model.train()
    total = 0.0
    for (x0,) in loader:
        x0 = x0.to(device)
        t = torch.randint(0, schedule.timesteps, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, schedule, noise)
        pred_noise = model(xt, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"epoch {epoch + 1:02d} loss={total / len(loader):.4f}")
"""),
            code(r"""
generated = sample(model, (5000, 2), schedule).cpu()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(data[:5000, 0], data[:5000, 1], s=2, alpha=0.5)
plt.axis("equal")
plt.title("Real")
plt.subplot(1, 2, 2)
plt.scatter(generated[:, 0], generated[:, 1], s=2, alpha=0.5)
plt.axis("equal")
plt.title("Generated")
plt.show()
"""),
            md("""可以尝试：
- 把 `timesteps` 从 200 改成 50 或 500；
- 把 MLP 的 `hidden_dim` 改小；
- 把数据改成圆环、高斯混合或螺旋线。"""),
        ],
    )

    write_notebook(
        "02_case_mnist_unconditional_generation.ipynb",
        [
            md("""# 02 案例二：MNIST 无条件图片生成

这是第一个真正的图片生成案例。网络不知道类别，只学习“手写数字长什么样”。

为了能在普通电脑上跑通，默认设置偏轻量：训练 1 个 epoch 能看到雏形，训练更久效果会明显提升。"""),
            code(COMMON_IMPORTS),
            code(r"""
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1),
])
dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
"""),
            code(SCHEDULE_CODE),
            code(TIME_EMBED_CODE),
            code(UNET_CODE),
            code(SAMPLING_CODE),
            code(IMAGE_HELPERS),
            code(r"""
schedule = DiffusionSchedule(timesteps=300)
model = SimpleUnet(image_channels=1, base_channels=32).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

epochs = 1
for epoch in range(epochs):
    model.train()
    pbar = tqdm(loader)
    for x0, _ in pbar:
        x0 = x0.to(device)
        t = torch.randint(0, schedule.timesteps, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, schedule, noise)
        pred_noise = model(xt, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"epoch {epoch + 1} loss={loss.item():.4f}")
"""),
            code(r"""
samples = sample(model, (16, 1, 28, 28), schedule)
show_images(samples, nrow=8, title="MNIST samples")
"""),
            code(r"""
Path("../checkpoints").mkdir(exist_ok=True)
torch.save(model.state_dict(), "../checkpoints/mnist_unconditional_unet.pt")
"""),
            md("""可以尝试：
- 把 `epochs` 改成 5 或 10；
- 把 `base_channels=32` 改成 64；
- 把 `timesteps=300` 改成 1000，采样更慢但理论上更细。"""),
        ],
    )

    write_notebook(
        "03_case_mnist_class_conditional_generation.ipynb",
        [
            md("""# 03 案例三：MNIST 类别条件生成

这个案例在 U-Net 里加入类别 embedding，让模型按指定数字生成图片。

核心变化只有两处：
- 模型初始化加 `num_classes=10`；
- 训练和采样时都把标签 `y` 传给模型。"""),
            code(COMMON_IMPORTS),
            code(r"""
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1),
])
dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
"""),
            code(SCHEDULE_CODE),
            code(TIME_EMBED_CODE),
            code(UNET_CODE),
            code(SAMPLING_CODE),
            code(IMAGE_HELPERS),
            code(r"""
schedule = DiffusionSchedule(timesteps=300)
model = SimpleUnet(image_channels=1, base_channels=32, num_classes=10).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

epochs = 1
for epoch in range(epochs):
    model.train()
    pbar = tqdm(loader)
    for x0, y in pbar:
        x0 = x0.to(device)
        y = y.to(device)
        t = torch.randint(0, schedule.timesteps, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, schedule, noise)
        pred_noise = model(xt, t, y)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"epoch {epoch + 1} loss={loss.item():.4f}")
"""),
            code(r"""
labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2, device=device)
samples = sample(model, (len(labels), 1, 28, 28), schedule, y=labels)
show_images(samples, nrow=10, title="Class-conditional MNIST: 0-9")
"""),
            code(r"""
Path("../checkpoints").mkdir(exist_ok=True)
torch.save(model.state_dict(), "../checkpoints/mnist_class_conditional_unet.pt")
"""),
            md("""加深理解：
- label embedding 和 timestep embedding 在这里直接相加；
- 这是一种最简条件控制方式；
- 更强的文本到图像模型会把条件换成文本 encoder 输出，并用 cross-attention 注入 U-Net。"""),
        ],
    )

    write_notebook(
        "04_case_fashion_mnist_generation.ipynb",
        [
            md("""# 04 案例四：Fashion-MNIST 服饰图片生成

这个案例和 MNIST 使用几乎相同的代码，但数据分布更复杂：衣服、鞋、包的轮廓比数字更丰富。

学习重点：同一套 DDPM 代码可以迁移到另一个数据集。"""),
            code(COMMON_IMPORTS),
            code(r"""
from torchvision import datasets, transforms

class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1),
])
dataset = datasets.FashionMNIST(root="../data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
"""),
            code(SCHEDULE_CODE),
            code(TIME_EMBED_CODE),
            code(UNET_CODE),
            code(SAMPLING_CODE),
            code(IMAGE_HELPERS),
            code(r"""
schedule = DiffusionSchedule(timesteps=300)
model = SimpleUnet(image_channels=1, base_channels=32, num_classes=10).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

epochs = 1
for epoch in range(epochs):
    model.train()
    pbar = tqdm(loader)
    for x0, y in pbar:
        x0 = x0.to(device)
        y = y.to(device)
        t = torch.randint(0, schedule.timesteps, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, schedule, noise)
        pred_noise = model(xt, t, y)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"epoch {epoch + 1} loss={loss.item():.4f}")
"""),
            code(r"""
labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2, device=device)
samples = sample(model, (len(labels), 1, 28, 28), schedule, y=labels)
show_images(samples, nrow=10, title="Fashion-MNIST conditional samples")
print(dict(enumerate(class_names)))
"""),
            md("""可以比较：
- MNIST 与 Fashion-MNIST 哪个 loss 降得更慢；
- 同样训练 1 个 epoch，哪个生成结果更清楚；
- 条件标签是否真的能控制类别。"""),
        ],
    )

    write_notebook(
        "05_case_cifar10_rgb_generation.ipynb",
        [
            md("""# 05 案例五：CIFAR-10 RGB 图片生成

这个案例把输入从单通道 `1x28x28` 扩展到三通道 `3x32x32`。

它比 MNIST 慢很多，也更难训练。建议先在 GPU 上跑；CPU 也能跑通流程，但训练效果会比较粗糙。"""),
            code(COMMON_IMPORTS),
            code(r"""
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1),
])
dataset = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
"""),
            code(SCHEDULE_CODE),
            code(TIME_EMBED_CODE),
            code(UNET_CODE),
            code(SAMPLING_CODE),
            code(IMAGE_HELPERS),
            code(r"""
schedule = DiffusionSchedule(timesteps=300)
model = SimpleUnet(image_channels=3, base_channels=48, num_classes=10).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

epochs = 1
for epoch in range(epochs):
    model.train()
    pbar = tqdm(loader)
    for x0, y in pbar:
        x0 = x0.to(device)
        y = y.to(device)
        t = torch.randint(0, schedule.timesteps, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, schedule, noise)
        pred_noise = model(xt, t, y)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"epoch {epoch + 1} loss={loss.item():.4f}")
"""),
            code(r"""
labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7] * 2, device=device)
samples = sample(model, (len(labels), 3, 32, 32), schedule, y=labels)
show_images(samples, nrow=8, title="CIFAR-10 RGB conditional samples")
print(dataset.classes)
"""),
            md("""进一步探索：
- CIFAR-10 上的小 U-Net 容量明显不足，这是正常现象；
- 可以增加 `base_channels`、训练 epoch、或者加入 attention；
- 真正高质量图像生成通常还会用 EMA、cosine schedule、更深 U-Net、classifier-free guidance 等技巧。"""),
        ],
    )

    write_notebook(
        "06_bonus_classifier_free_guidance_mnist.ipynb",
        [
            md("""# 06 Bonus：Classifier-Free Guidance 思想最小实现

这是进阶补充，不计入五个基础案例。它演示如何在训练时随机丢弃条件，让同一个模型同时学会：

- 有条件预测噪声；
- 无条件预测噪声。

采样时把二者按 guidance scale 组合，可以增强条件控制。"""),
            code(COMMON_IMPORTS),
            code(r"""
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1),
])
dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
"""),
            code(SCHEDULE_CODE),
            code(TIME_EMBED_CODE),
            code(UNET_CODE),
            code(r"""
class CFGUnet(SimpleUnet):
    def __init__(self, image_channels=1, base_channels=32, num_classes=10):
        super().__init__(image_channels=image_channels, base_channels=base_channels, num_classes=num_classes + 1)
        self.null_label = num_classes


@torch.no_grad()
def cfg_sample(model, shape, schedule, labels, guidance_scale=3.0):
    model.eval()
    x = torch.randn(shape, device=device)
    null_labels = torch.full_like(labels, model.null_label)
    for i in tqdm(reversed(range(schedule.timesteps)), total=schedule.timesteps):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)

        eps_cond = model(x, t, labels)
        eps_uncond = model(x, t, null_labels)
        guided_eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        betas_t = extract(schedule.betas, t, x.shape)
        sqrt_one_minus_alpha_bars_t = extract(schedule.sqrt_one_minus_alpha_bars, t, x.shape)
        sqrt_recip_alphas_t = extract(schedule.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * guided_eps / sqrt_one_minus_alpha_bars_t)
        if i == 0:
            x = model_mean
        else:
            posterior_variance_t = extract(schedule.posterior_variance, t, x.shape)
            x = model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)
    return x
"""),
            code(IMAGE_HELPERS),
            code(r"""
schedule = DiffusionSchedule(timesteps=300)
model = CFGUnet(image_channels=1, base_channels=32, num_classes=10).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

epochs = 1
drop_prob = 0.1
for epoch in range(epochs):
    model.train()
    pbar = tqdm(loader)
    for x0, y in pbar:
        x0 = x0.to(device)
        y = y.to(device)
        drop_mask = torch.rand_like(y.float()) < drop_prob
        y_train = y.clone()
        y_train[drop_mask] = model.null_label

        t = torch.randint(0, schedule.timesteps, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, schedule, noise)
        pred_noise = model(xt, t, y_train)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"epoch {epoch + 1} loss={loss.item():.4f}")
"""),
            code(r"""
labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2, device=device)
samples = cfg_sample(model, (len(labels), 1, 28, 28), schedule, labels, guidance_scale=3.0)
show_images(samples, nrow=10, title="Classifier-free guidance MNIST")
"""),
            md("""调参建议：
- `drop_prob` 常见取值 0.1 到 0.2；
- `guidance_scale` 越大越贴近条件，但过大会导致失真；
- 可以对比 `guidance_scale=0, 1, 3, 7`。"""),
        ],
    )

    readme = ROOT / "README.md"
    readme.write_text(
        """# Diffusion 图片生成教程

这个文件夹包含一组从零搭建 DDPM 扩散模型的 Jupyter notebook。

## 建议学习顺序

1. `notebooks/00_diffusion_math_and_forward_process.ipynb`：扩散数学和前向加噪。
2. `notebooks/01_case_toy_2d_from_noise_to_points.ipynb`：2D 玩具数据完整训练与采样。
3. `notebooks/02_case_mnist_unconditional_generation.ipynb`：MNIST 无条件图片生成。
4. `notebooks/03_case_mnist_class_conditional_generation.ipynb`：MNIST 指定数字生成。
5. `notebooks/04_case_fashion_mnist_generation.ipynb`：Fashion-MNIST 服饰生成。
6. `notebooks/05_case_cifar10_rgb_generation.ipynb`：CIFAR-10 RGB 图片生成。
7. `notebooks/06_bonus_classifier_free_guidance_mnist.ipynb`：进阶的 classifier-free guidance。

## 安装依赖

```bash
pip install torch torchvision matplotlib tqdm notebook
```

如果你有 NVIDIA GPU，请按 PyTorch 官网说明安装匹配 CUDA 的版本。
""",
        encoding="utf-8",
    )

    requirements = ROOT / "requirements.txt"
    requirements.write_text(
        "torch\n"
        "torchvision\n"
        "matplotlib\n"
        "tqdm\n"
        "notebook\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()
    print(f"Wrote notebooks to {NOTEBOOK_DIR}")

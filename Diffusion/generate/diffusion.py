"""扩散过程：余弦加噪日程，v/epsilon 预测，DDPM 和 DDIM 采样。"""

import math

import torch
from torch import nn
from tqdm.auto import tqdm


class Diffusion(nn.Module):
    def __init__(self, timesteps=1000, prediction_type='epsilon'):
        super().__init__()
        if timesteps < 2 or prediction_type not in ('epsilon', 'v'):
            raise ValueError('timesteps 至少为 2；prediction_type 只能是 epsilon 或 v')
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        positions = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        cumulative = torch.cos(((positions / timesteps + 0.008) / 1.008) * math.pi / 2) ** 2
        cumulative = cumulative / cumulative[0]
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999).float()
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        previous = torch.cat([torch.ones(1), alpha_bars[:-1]])
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('posterior_variance', betas * (1 - previous) / (1 - alpha_bars))
        self.register_buffer('mean_x0', betas * previous.sqrt() / (1 - alpha_bars))
        self.register_buffer('mean_xt', (1 - previous) * alphas.sqrt() / (1 - alpha_bars))

    @staticmethod
    def extract(values, time):
        return values[time][:, None, None, None]

    def add_noise(self, clean_images, time, noise):
        alpha_bar = self.extract(self.alpha_bars, time)
        return alpha_bar.sqrt() * clean_images + (1 - alpha_bar).sqrt() * noise

    def training_target(self, clean_images, time, noise):
        if self.prediction_type == 'epsilon':
            return noise
        alpha_bar = self.extract(self.alpha_bars, time)
        # v 是 x0 和 noise 的旋转组合，还原 x0 时无需除以很小的 alpha。
        return alpha_bar.sqrt() * noise - (1 - alpha_bar).sqrt() * clean_images

    def predict_clean(self, noisy_images, time, prediction):
        alpha_bar = self.extract(self.alpha_bars, time)
        prediction = prediction.float()
        if self.prediction_type == 'v':
            return alpha_bar.sqrt() * noisy_images - (1 - alpha_bar).sqrt() * prediction
        return (noisy_images - (1 - alpha_bar).sqrt() * prediction) / alpha_bar.sqrt()

    def loss(self, model, clean_images):
        batch = len(clean_images)
        # 每批覆盖不同噪声强度，再随机打乱，减少小数据训练的采样波动。
        uniform = (torch.arange(batch, device=clean_images.device) + torch.rand(batch, device=clean_images.device)) / batch
        time = (uniform * self.timesteps).long()[torch.randperm(batch, device=clean_images.device)]
        noise = torch.randn_like(clean_images)
        noisy = self.add_noise(clean_images, time, noise)
        prediction = model(noisy, time)
        target = self.training_target(clean_images, time, noise)
        return (prediction.float() - target).square().mean()

    @torch.no_grad()
    def sample(self, model, count, image_size, generator=None, progress=True,
               method='ddpm', steps=None):
        if count < 1 or method not in ('ddpm', 'ddim'):
            raise ValueError('count 必须为正数，method 必须是 ddpm 或 ddim')
        if method == 'ddpm' and steps not in (None, self.timesteps):
            raise ValueError('DDPM 必须使用全部训练时间步；少步采样请选择 DDIM')
        steps = self.timesteps if steps is None else steps
        if not 2 <= steps <= self.timesteps:
            raise ValueError('采样步数必须在 2 和训练时间步数之间')
        device = self.alpha_bars.device
        was_training = model.training
        model.eval()
        try:
            images = torch.randn(count, 3, image_size, image_size, device=device, generator=generator)
            times = torch.linspace(self.timesteps - 1, 0, steps).long().tolist()
            for index, step in enumerate(tqdm(times, desc=f'生成图片 ({method})', disable=not progress)):
                time = torch.full((count,), step, device=device, dtype=torch.long)
                clean = self.predict_clean(images, time, model(images, time)).clamp(-1, 1)
                if method == 'ddpm':
                    mean = self.mean_x0[step] * clean + self.mean_xt[step] * images
                    images = mean
                    if step > 0:
                        noise = torch.randn(images.shape, device=device, generator=generator)
                        images = mean + self.posterior_variance[step].sqrt() * noise
                elif index == len(times) - 1:
                    images = clean
                else:
                    # DDIM eta=0：沿同一预测方向跳到下一个选定时间步。
                    alpha = self.alpha_bars[step]
                    next_alpha = self.alpha_bars[times[index + 1]]
                    noise = (images - alpha.sqrt() * clean) / (1 - alpha).sqrt()
                    images = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
            return images.clamp(-1, 1)
        finally:
            model.train(was_training)

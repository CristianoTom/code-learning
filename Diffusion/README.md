# Diffusion 图片生成教程

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

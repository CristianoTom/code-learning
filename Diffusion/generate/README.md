# 小数据集 Diffusion 图片生成 · v2

这个项目面向约 208 张同风格角色图片，从零训练无条件扩散模型。手写 U-Net、扩散过程、DDPM/DDIM 采样，不依赖 diffusers，不下载预训练权重；保留中文注释和清楚的文件划分。

**这次修改的目的：先学出完整的人物结构，再提高细节。208 张图片不能保证生成质量或泛化能力。** 推荐先运行 64×64 实验，再尝试 128×128，不要直接沿用旧的 256×256、通道 32 配置。清楚的生成图也可能是对训练图的记忆，不能仅凭清晰度判断模型学得好。

## 这次具体改变了什么

- 多尺度 U-Net：64 输入逐级变成 32、16、8；128/256 自动增加层数，最深处保持 8×8。每层两个残差块，使用时间调制和多尺度跳跃连接。
- 注意力只放在 16×16 及以下特征上，使用 PyTorch 的多头 scaled dot product attention，避免旧网络在 256 输入时建立 4096×4096 注意力矩阵。
- 默认 v 预测：强噪声处恢复 x0 不需要除以极小数。仍可用 `--prediction-type epsilon` 做对照，两种目标的 loss 数值不能直接比较。
- 按总优化步数进行学习率预热、余弦衰减；续训可以真正覆盖基础学习率。
- EMA 前期快速跟随，后期平滑；预览同时输出 EMA 和原始权重结果。
- DDIM 支持真正的少步采样，DDPM 仍保留完整的 1000 步采样。
- 缓存缩小后的图片，避免每轮重复解码高分辨率原图；新增固定噪声的去噪检查。
- 新旧权重按版本区分。新版训练默认写入 `outputs_v2`，不会接着旧网络训练。

## 1. 安装

把整个 `generate` 文件夹复制到 GPU 设备，在该目录下运行。建议 Python 3.10～3.12。

```powershell
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux 改为 source .venv/bin/activate
python -m pip install --upgrade pip
```

先按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)为你的 GPU 和驱动安装 CUDA 版 PyTorch，再执行：

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

正式训练前确认 CUDA 为 True。项目使用 PyTorch 2.5 及以上接口；本地只有 CPU，GPU 显存消耗和 CUDA 混合精度没有在本地实测。

## 2. 准备数据与检查代码

把实际使用的图片放在 `data/`，可使用子目录。代码自动使用所有找到的图片，**不会强制截取 208 张，也不会删除原图**。图片数量会打印到终端，实际文件清单保存在输出目录的 `data_files.json`。

图片修正 EXIF 朝向、透明部分合成白底、等比缩放并补白边，最后归一化到 [-1, 1]。不裁剪角色。默认不水平翻转，避免反转文字；没有文字且允许镜像时可添加 `--horizontal-flip`。

```powershell
python smoke_test.py
```

此命令在 CPU 上检查 v/epsilon 公式、理想预测器的 DDPM/DDIM 采样、16～256 各分辨率网络、反向传播、训练、诊断、保存、续训学习率覆盖及旧权重兼容。使用真实数据的前两张图做微型训练，临时输出自动清理。它不验证正式训练的生成质量。

## 3. 推荐的第一次正式训练：64×64

```powershell
python train.py --device cuda --amp
```

默认配置：分辨率 64、基础通道 64、dropout 0.1、v 预测、1000 个扩散时间步、batch size 16、基础学习率 0.0001。前 500 次更新预热，随后余弦衰减，到 40000 步降至基础学习率的 10%。默认轮数上限 5000、总步数上限 40000，以先到者为准。

208 张图片、batch size 16 时，每轮约 13 次更新，40000 步约 3077 轮。281 张则每轮约 18 次更新，步数上限自动在更少轮数内达到。**这里用总更新步数控制训练量，不能把轮数直接当成画质指标。40000 步是实验预算，不是清晰度保证。**

每 50 轮保存最新权重和轮次权重，每 100 轮保存固定种子的 EMA/原始权重预览及去噪诊断。**默认只保留最近 3 个 `epoch_XXXX.pt` 快照和 `last.pt`**，避免优化器权重占满磁盘；需要保留所有轮次请加 `--keep-checkpoints 0`，重要权重也可另存为手工命名的文件，例如 `best.pt`。手工命名权重不会被自动清理。训练结束时也保存最新权重和预览；`--sample-every 0` 关闭全部预览和诊断。按 Ctrl+C 会尝试保存当前进度；进程被强制终止或断电时只能恢复上一次已经写入的权重。

显存不足先减小 `--batch-size`，例如 8。输入分辨率越高，训练越贵；`--amp` 减少显存使用，出现数值溢出时先不加它重试。无需 torchvision。

缩小后的图片默认缓存为 uint8，在内存中保存；208 张 64×64 RGB 图片的像素缓存约 2.4 MiB，不包括网络、原图解码时的临时内存及 Python 开销。内存紧张可用 `--no-cache`。Windows 默认 `--num-workers 0`，有需要再尝试 2 或 4。

## 4. 如何判断是否在变好

### `training_data.png`

输出目录中前 16 张图片的预处理预览，模型真正看到的输入。它不是生成图。先确认人物没有变形、黑底或过小。

### `previews/ema_XXXX.png` 和 `raw_XXXX.png`

同一个种子的纯噪声生成结果，用于比较不同轮次与不同权重。默认用 100 步 DDIM。观察脸型、眼睛、头发的位置是否稳定，不能只看 loss。EMA 与 raw 都很差时，不能把问题全部归因于 EMA。

### `diagnostics/denoise_XXXX.png`

使用固定的训练图片和固定的噪声，纵向 7 行：

1. 干净训练图片。
2. 约 10% 时间步的带噪图片。
3. 从第 2 行一次预测的干净图片。
4. 约 50% 时间步的带噪图片。
5. 从第 4 行一次预测的干净图片。
6. 约 90% 时间步的带噪图片。
7. 从第 6 行一次预测的干净图片。

这里的百分比表示时间步在日程中的位置，**不是像素噪声比例**。高噪声下原始信息很少，一次预测模糊是可能的，不能要求第 7 行精确还原。完整生成还要执行多步去噪。

`diagnostics/metrics.csv` 按三种噪声强度记录预测目标 MSE 和未裁剪的 x0 MSE，便于定位问题。它们是**固定训练样本上的拟合诊断，不是独立验证集分数**，也不是清晰度或泛化能力的证明。

### `loss.csv`

记录每轮平均目标 MSE、总更新步数和实际学习率。默认改为 v 预测后，不要拿它和旧版 epsilon loss≈0.024 比大小，也不要设一个“低于多少就一定清楚”的门槛。

## 5. 先用 8 张图做拟合检查（可选）

如果几千步后始终没有人物结构，可以先运行：

```powershell
python overfit.py --device cuda --amp
```

它单独写入 `outputs_overfit`：使用前 8 张图、64×64、通道 32、dropout 0，训练最多 3000 步。这个实验刻意允许记忆少量样本，用来检查基本拟合能力，不用于正式生成。查看固定预览和去噪诊断；如果仍然完全无法成形，应检查数据、学习率与实现，而不是直接再训数万步。

它没有保证 3000 步内一定成功。需要延长或恢复时，直接用 train.py，例如：

```powershell
python train.py --device cuda --amp --output-dir outputs_overfit --resume outputs_overfit/checkpoints/last.pt --epochs 6000 --max-steps 6000
```

正式训练不要添加 `--limit-images`；它仅用于这个诊断实验。

## 6. 续训与调整学习率

```powershell
python train.py --device cuda --amp --resume outputs_v2/checkpoints/last.pt
```

显式降低**基础学习率**：

```powershell
python train.py --device cuda --amp --resume outputs_v2/checkpoints/last.pt --lr 0.00005
```

新学习率会在加载优化器后按全局步数重新应用，不会被旧值覆盖。实际步进学习率还要乘预热/余弦系数，查看 `loss.csv` 的 lr 列即可确认。

若已达到 40000 步，继续训练需同时提高轮数和步数上限：

```powershell
python train.py --device cuda --amp --resume outputs_v2/checkpoints/last.pt --epochs 7000 --max-steps 60000
```

续训会恢复模型、优化器、EMA、AMP 缩放器、随机状态，以及结构、分辨率、预测目标、增强和学习率日程参数。超过原始衰减步数后保持最低学习率，**不会因为延长训练而重新开始余弦日程**。`--lr` 省略时沿用旧的基础值，显式给出才覆盖。batch size、workers、设备、目标轮数/步数可变。

数据文件列表必须一致，避免把不同数据集的 loss 混在一起。图片内容改变但文件名没变不会被这个清单检查发现。跨硬件或多进程不保证逐位复现。轮中间停止时从该轮开头继续，不能精确恢复到某个 batch。输出到自定义目录时，续训也请传入同一个 `--output-dir`。

## 7. 生成图片

```powershell
python sample.py --device cuda --num-images 16 --sampler ddim --steps 100
```

默认加载 `outputs_v2/checkpoints/last.pt` 的 EMA 权重，输出到 `outputs_v2/generated`。分辨率、结构与预测目标自动从权重恢复。结果包含独立 PNG、拼图 `grid.png` 和生成配置 `generation.json`。

对比更密集的 DDIM，或完整 DDPM：

```powershell
python sample.py --device cuda --sampler ddim --steps 250 --output-dir outputs_v2/generated_ddim250
python sample.py --device cuda --sampler ddpm --output-dir outputs_v2/generated_ddpm
```

增加采样步数不能弥补模型没有学会结构。DDPM 必须使用全部训练步数；DDIM 才能选择较少的时间步。不同采样器的 seed 相同也不意味着生成图必须相同。

换种子或比较 raw 权重：

```powershell
python sample.py --device cuda --seed 456 --output-dir outputs_v2/generated_456
python sample.py --device cuda --raw-weights --output-dir outputs_v2/generated_raw
```

重复用同一个输出目录会覆盖同名图片；新数量更少时旧的多余图片仍会存在，建议每批使用新目录。相同设备、权重、seed、采样器和 batch size 通常可复现，改变 batch size 会改变随机数顺序。默认输出白底 RGB 图片，不保留透明通道。

## 8. 学会结构后尝试 128×128

```powershell
python train.py --device cuda --amp --image-size 128 --base-channels 64 --batch-size 8 --output-dir outputs_128
python sample.py --device cuda --checkpoint outputs_128/checkpoints/last.pt --output-dir outputs_128/generated
```

这是一组**独立的从头训练实验**，不要加载 64 或旧 256 的权重。128 会增加计算量；208 张图的数据分布、姿态一致性、裁剪和主体占比仍会限制效果。若64尚且没有完整结构，直接升到128/256通常不是优先排查方法。需要稳定高分辨率质量时，预训练模型微调是另一条路线，本项目依然遵循从零实现的要求。

## 旧项目兼容

旧的 256×256 小网络保留在 `legacy_model.py`，sample.py 会自动识别没有版本字段的旧权重，使用旧网络和 epsilon 预测：

```powershell
python sample.py --device cuda --checkpoint outputs/checkpoints/last.pt --sampler ddpm --output-dir outputs/legacy_generated
```

旧权重不能恢复到新版多尺度网络训练，train.py 会明确拒绝，而不是部分加载。原始数据和旧 outputs 不会被修改。

## 文件和阅读顺序

```text
dataset.py       图片预处理、缓存、可选翻转
model.py         新版多尺度 U-Net
diffusion.py     v/epsilon 目标、DDPM/DDIM
train.py         优化循环、学习率、EMA、版本化权重
diagnostics.py   固定样本、固定噪声的去噪检查
sample.py        新旧权重采样和生成记录
overfit.py       8 张图片拟合检查入口
smoke_test.py    CPU 数值与端到端验证
utils.py         图片、设备和权重工具
legacy_model.py  旧网络，仅用于兼容旧权重生成
```

建议按列表前四个文件依次阅读。`python train.py --help` 和 `python sample.py --help` 可查看全部参数。默认路径相对于脚本目录，显式传入的相对路径相对于终端当前目录。

算法参考：[DDPM](https://arxiv.org/abs/2006.11239)、[Improved DDPM 与多尺度架构](https://github.com/openai/improved-diffusion)、[v 参数化](https://arxiv.org/abs/2202.00512)、[DDIM](https://arxiv.org/abs/2010.02502)。代码是便于阅读的独立实现，没有实现蒸馏，也不是上述论文全部架构和实验的复刻。

# 神经算子解 PDE：从入门到 PINO

本教程面向会基础 Python、希望亲手实现神经算子的学习者。所有讲解为中文，图中使用英文以避免系统字体问题。每章都有 PDE 定义、边界条件、数据生成、核心代码、误差评估和练习。无需下载数据集、预训练权重或登录服务。

## 学习顺序

| 顺序 | 笔记本 | 难度 | 核心内容 |
|---|---|---|---|
| 00 | [算子与 PDE 基础](00_operator_and_pde_basics.ipynb) | 入门 | Poisson 解析解、有限差分、数据划分 |
| 01 | [MLP 基线](01_mlp_poisson_baseline.ipynb) | 初级 | 固定网格、硬边界约束、完整训练流程 |
| 02 | [DeepONet](02_deeponet_poisson.ipynb) | 中级 | branch/trunk、任意输出坐标、传感器限制 |
| 03 | [1D FNO 热方程](03_fno1d_heat.ipynb) | 中级 | 复数频谱层、周期边界、跨分辨率 |
| 04 | [Burgers 方程](04_fno_burgers.ipynb) | 进阶 | 非线性、伪谱 RK4、数值参考解收敛 |
| 05 | [2D FNO](05_fno2d_screened_poisson.ipynb) | 高级 | 二维反应 Poisson、正负频率块、场图 |
| 06 | [PINO](06_pino_physics_informed.ipynb) | 高级 | PDE 残差、频谱微分、少标签对照实验 |
| 07 | [三维 FNO 椭圆方程](07_fno3d_screened_poisson.ipynb) | 进阶案例 | 三个空间维度、四个频率角块、体积误差 |
| 08 | [参数化三维热方程](08_parametric_fno3d_heat.ipynb) | 进阶案例 | 初值 + 扩散率 + 时间，范围内与范围外测试 |
| 09 | [变系数三维扩散](09_fno3d_variable_diffusion.ipynb) | 高级案例 | 系数场 + forcing，守恒离散、PCG 与网格误差 |
| 10 | [三维连续时空 DeepONet](10_deeponet3d_continuous_time.ipynb) | 高级案例 | 固定初值传感器、四维查询、自动微分残差 |

每章可以独立从头运行，无需先执行或保存上一章的模型。建议先完成 00–03，再做 04–06。每章阅读、运行和练习可预留 30–60 分钟；训练耗时随 CPU/GPU 而变，二维和 PINO 较慢。

进阶路线：先学 07 的三维频谱结构，再学 08 的参数输入；09 进一步处理变系数数值标签；10 转向连续空间和时间查询。07–09 的默认训练网格为 12³，快速模式为 8³；10 使用 6³ 个初值传感器。三维教学配置可在 CPU 上运行，无需增加依赖。网格每轴加倍会使体素数增为八倍，请逐步增加网格、batch 和通道宽度。新章节的模型与 PCG 求解器直接位于笔记本，`tutorial_3d_utils.py` 仅提供网格、解析函数族和切片图。

## 原理部分的阅读方式

11 个笔记本现已在章节简介后加入共 35 个“原理详解”单元。每章先解释数学问题与结构，再对应数据生成、网络代码和测试指标。建议按“推导公式 → 核对张量形状 → 运行代码 → 用物理性质解释结果”的顺序学习。

| 章节 | 新增原理重点 |
|---|---|
| 00 | 函数空间、边界与唯一性、Green 函数、正弦本征展开、差分收敛阶 |
| 01 | 固定网格线性映射、MLP 参数维度、硬边界与软约束、损失尺度 |
| 02 | 基函数分离、latent 容量、branch/trunk 矩阵乘法、传感器信息限制 |
| 03 | 热方程模式 ODE、积分核到 Fourier 层、复数权重、FFT 归一化 |
| 04 | 非线性频率耦合、均值与能量、混叠和 2/3 截断、RK4 稳定性 |
| 05 | 二维频谱解、反应项与零模、正负频率块、物理域尺度与复杂度 |
| 06 | PINN/PINO 的对象、残差与误差、预条件等价关系、微分与损失单位 |
| 07 | 三维本征值、四角块、通道运算、复数参数计数、体积与导数误差 |
| 08 | 参数化解算子、nu×T 等价性、无量纲组合、参数归一化与外推 |
| 09 | 变系数弱形式、共享通量、调和平均、SPD、PCG、误差分解 |
| 10 | 时空分离结构、有限频带与零模限制、paired 求导、链式法则、初值权重 |

本次只扩充 Markdown 讲解；代码、执行计数和原有训练图表经逐章哈希核对保持一致，全部笔记本格式与代码语法检查通过。修改记录见 `theory_update_report.json`。学习过程中先分清“方程原理”“网络表示”“数值求解器”和“测试协议”，避免把模型可以执行、误差较小与严格满足物理混为一谈。

## 开始使用

在终端进入本文件夹，用自己的 Python 环境运行：

```powershell
cd D:\study\code-learning\PINO\neural_operator_tutorials
python -m pip install -r requirements.txt
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
python -m jupyterlab
```

打开 00，选择 Python 3 内核，并使用 **Restart Kernel and Run All**。从 PINO 根目录启动 Jupyter 也可。若已有 PyTorch，先确认 Jupyter 内核使用同一环境：笔记本中查看 `sys.executable`。CPU 可运行全套；CUDA 可用时自动选择 GPU。依赖给出最低版本，不是锁定环境，CPU/CUDA 的安装按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择。

本机 Windows Anaconda 存在 NumPy/PyTorch 的 Intel OpenMP 冲突。上面的 MKL 设置让 NumPy 的 MKL 采用顺序执行，避免重复运行库；PyTorch 仍可使用自己的线程。这只影响当前终端及其子进程，不修改系统安装。验证器在 Windows 自动使用此设置。已有 Jupyter 服务需要退出并从设置过变量的终端重启。

`tutorial_utils.py` 只集中通用训练循环、随机种子、绘图和简单解析数据函数；各章的模型、频谱层和 Burgers 求解器均直接写在笔记本里，方便学习和修改。损失用缩放后的解时，章节明确说明缩放，图表恢复物理量。

## 快速模式与验证

先在终端设置环境变量，再启动 Jupyter，可用每章 2 个 epoch 和少量样本检查流程：

```powershell
$env:NO_TUTORIAL_SMOKE = "1"
python -m jupyterlab
```

快速模式误差大是正常的，它仅验证流程。恢复完整训练前：

```powershell
Remove-Item Env:NO_TUTORIAL_SMOKE -ErrorAction SilentlyContinue
Remove-Item Env:NO_TUTORIAL_EPOCHS -ErrorAction SilentlyContinue
```

已有 Jupyter 服务/内核不会自动感知后来修改的终端环境，请重新启动服务，或在各章导入公共工具前设置环境变量并重启内核。可用 `NO_TUTORIAL_EPOCHS=20` 自定义轮数。默认训练轮数为 180/250/120/150/100/120；00 无训练。

07–10 默认训练轮数为 80/100/100/200；其快速模式同样只训练 2 轮。

命令行验证器检查 notebook 格式、Python 语法，并按独立内核执行所有单元格：

```powershell
python verify_tutorials.py --mode smoke
python verify_tutorials.py --mode full --save
```

只执行新增进阶案例：

```powershell
python verify_tutorials.py --start 7 --end 10 --mode smoke
python verify_tutorials.py --start 7 --end 10 --mode full --save
```

`--start`/`--end` 是包含端点的章节编号。局部验证报告使用 `verification_07_10_smoke.json` / `verification_07_10_full.json`，不会覆盖原有全套报告。不指定章节范围时会执行全部现有笔记本。

`--save` 将实际输出写回笔记本；否则只保存 `verification_smoke.json` 或 `verification_full.json` 报告。失败时退出码非零并记录错误，不将失败 notebook 标为成功。完整模式会清除快速模式和 epoch 覆盖，以验证默认配置。

## 正确理解结果

- 按完整输入函数划分训练、验证和测试，避免坐标级数据泄漏。
- 00–02 为零 Dirichlet Poisson；03–06 为周期方程，FFT 不是通用边界处理方法。
- 03–04 只学固定参数、固定终止时间，不学任意时间或任意黏性。
- 05 用常系数反应 Poisson 入门二维场，不等同于可变系数 Darcy。
- 06 是 PINO 核心思想的简化教学实验；线性可解析例子的预条件残差等价于解误差，不是复杂 PDE 上优势的证明。
- 快速执行通过不意味着收敛；完整训练的指标以实际输出为准。CUDA 与 CPU 的结果可能不同。
- 这些低频、光滑、规则区域案例不是工业求解器，也不代表对任意函数分布或复杂几何有泛化保证。
- 07–09 的三轴都是空间；10 的查询为三个空间坐标加时间。新增案例仍用周期立方体边界，不包含壁面、复杂几何或三维 Navier–Stokes。
- 09 的 PCG 代数误差和连续 PDE 的网格误差分开检查；神经模型不能靠错误标签学到正确物理。
- 10 的空间特征带有限频带先验，PDE 自动微分残差为训练后诊断，未用于监督训练。

## 本次执行记录

2026-10-09，Windows / Python 3.12.7 / PyTorch 2.5.1 CPU：7/7 快速验证通过，7/7 默认完整训练通过。笔记本已保存完整训练的输出和图表，可直接打开阅读。详细记录见 `verification_smoke.json` 与 `verification_full.json`；抽取的结果图位于 `verification_figures/`。

| 模型 | 独立测试集平均相对 L2 |
|---|---:|
| 固定网格 MLP | 5.69% |
| DeepONet，129 个查询点 | 6.91% |
| 热方程 FNO | 0.267% |
| Burgers FNO | 约 0.30% |
| 二维反应 Poisson FNO | 0.604% |
| 少标签 FNO 对照 | 0.812% |
| PINO 示例 | 0.254% |

这些数值来自各章不同 PDE 和数据分布，不能横向当作模型排名；只作为可复现的教学运行记录。各章训练数据量和训练轮数不同。

## 新增三维案例执行记录

2026-10-09，同一 Windows / Python 3.12.7 / PyTorch 2.5.1 CPU 环境：新增 07–10 共 4/4 快速验证通过、4/4 默认完整训练通过，输出与切片图已保存回笔记本。当前共 11 个笔记本，格式与 Python 语法检查全部通过。新案例运行详情见 `verification_07_10_smoke.json` 和 `verification_07_10_full.json`。

| 新案例 | 独立测试集平均相对 L2 | 补充检查 |
|---|---:|---|
| 07：三维周期椭圆 FNO | 1.12% | 18³ 网格约 1.12%，奇数网格重建通过 |
| 08：参数化三维热方程 | 4.11% | 范围外参数约 57.7%，参考热半群、均值、能量检查通过 |
| 09：变系数三维扩散 | 3.54% | 细网格代理误差约 4.11%，参考解粗细网格差异约 1.70% |
| 10：连续时空 DeepONet | 4.59%（随机时空点） | t=0.19 体积误差约 4.39%，解析自动微分残差检查通过 |

09 的 PCG 最大相对代数残差小于 1e-9，但这不等于连续 PDE 的网格误差。10 的模型 PDE 残差明显不为零，即使解误差已较小，也说明数据拟合与导数正确性需要分别评估。详细原始数值位于各笔记本，不同案例的误差不作模型排名。

## 原始论文与接口参考

- [DeepONet](https://arxiv.org/abs/1910.03193)
- [Fourier Neural Operator](https://arxiv.org/abs/2010.08895)
- [Physics-Informed Neural Operator](https://arxiv.org/abs/2111.03794)
- [PyTorch torch.fft](https://docs.pytorch.org/docs/stable/fft.html)

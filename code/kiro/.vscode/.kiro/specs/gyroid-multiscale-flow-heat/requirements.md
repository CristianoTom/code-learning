# 需求文档

## 简介

本功能实现一种面向大规模周期 Gyroid 阵列快速流热预测的体素化等效多尺度框架。该框架通过将细尺度 TPMS（三周期极小曲面）几何聚合为粗体素单元，并利用随机场神经闭合模型学习等效输运参数，在无需解析真实边界层与复杂三维 CFD 网格的情况下，实现对大规模周期 Gyroid 阵列流热性能的快速预测。

整体流程分为三个阶段：
1. **几何体素化**：基于 Gyroid 隐式函数对 TPMS 几何进行高分辨率采样，并将细尺度几何聚合为粗体素 RVE 单元，赋予体素标签与等效输运属性。
2. **多尺度求解**：在规则体素网格上，根据体素标签分区求解 Brinkman–Forchheimer 流动方程、傅里叶导热方程，并通过 LTNE 双温模型实现流固耦合。
3. **神经闭合建模**：基于少量小尺度 CFD 数据，训练随机场神经网络，学习体素级等效参数（K、CF、hv、keff），并通过宏观响应（Re、Nu、f、j、Δp、h、出口温度）进行校验。

---

## 术语表

- **Gyroid**：一种三周期极小曲面（TPMS），由隐式函数 `sin(wx)cos(wy) + sin(wy)cos(wz) + sin(wz)cos(wx) = C` 定义，其中 `w` 为空间频率，`C` 为水平集参数（控制固体体积分数）。
- **TPMS**：三周期极小曲面（Triply Periodic Minimal Surface），具有周期性、高比表面积和良好流动特性的多孔结构。
- **RVE**：代表性体积单元（Representative Volume Element），用于均匀化细尺度几何特征的最小重复单元。
- **Voxel**：三维规则网格中的体积像素单元，本框架中为粗尺度计算的基本单元。
- **体素标签**：每个体素被赋予的区域类型标识：0=固体、1=流体、2=界面、3=加热板。
- **LTNE**：局部热非平衡模型（Local Thermal Non-Equilibrium），用双温场（流体温度 Tf、固体温度 Ts）描述固液非热平衡换热。
- **Brinkman–Forchheimer 方程**：描述多孔介质中流动的动量方程，包含黏性阻力项（Darcy 项）和惯性阻力项（Forchheimer 项）。
- **K**：渗透率张量（Permeability），表征多孔介质对流体流动的阻力，单位 m²。
- **CF**：Forchheimer 惯性阻力系数，表征高雷诺数下的惯性效应，无量纲。
- **hv**：体积换热系数（Volumetric Heat Transfer Coefficient），描述单位体积内固液界面的换热强度，单位 W/(m³·K)。
- **keff**：等效导热率（Effective Thermal Conductivity），均匀化后的体素级导热参数，单位 W/(m·K)。
- **Gyroid_Sampler**：负责 Gyroid 隐式函数采样与体素化的模块。
- **Voxel_Labeler**：负责根据局部几何特征为体素赋予标签与等效属性的模块。
- **BF_LTNE_Solver**：负责在体素网格上求解 Brinkman–Forchheimer + LTNE 方程组的求解器模块。
- **Neural_Closure_Model**：负责从 CFD 数据学习体素级等效参数的神经网络模型。
- **CFD_Dataset**：用于训练神经闭合模型的小尺度周期阵列 CFD 仿真数据集。
- **Re**：雷诺数（Reynolds Number），表征流动惯性与黏性之比。
- **Nu**：努塞尔数（Nusselt Number），表征对流换热与导热之比。
- **f**：摩擦因子（Friction Factor），表征流动阻力。
- **j**：科尔本 j 因子（Colburn j-factor），综合表征传热性能。

---

## 需求

### 需求 1：Gyroid 几何采样与体素化

**用户故事：** 作为研究人员，我希望通过 Gyroid 隐式函数对 TPMS 几何进行高分辨率空间采样，并将其体素化为规则网格，以便后续多尺度求解使用。

#### 验收标准

1. THE Gyroid_Sampler SHALL 接受空间频率参数 `w`、水平集参数 `C`、计算域尺寸及采样分辨率作为输入，生成三维标量场。
2. WHEN 三维标量场生成后，THE Gyroid_Sampler SHALL 根据隐式函数值将每个采样点分类为固体域（函数值 ≤ 0）或流体域（函数值 > 0）。
3. THE Gyroid_Sampler SHALL 支持 2D 截面可视化，输出固体/流体分布的二值图、热力图和等值线图三种模式。
4. WHEN 采样分辨率变化时，THE Gyroid_Sampler SHALL 保持 Gyroid 几何的拓扑连通性，即固体域和流体域均保持连通。
5. THE Gyroid_Sampler SHALL 支持三维周期性边界条件，使采样结果在 x、y、z 三个方向上均满足周期性。
6. WHEN 参数 `C` 超出有效范围（导致固体或流体体积分数为 0），THE Gyroid_Sampler SHALL 返回包含有效范围提示的错误信息。

---

### 需求 2：RVE 聚合与体素标签赋值

**用户故事：** 作为研究人员，我希望将细尺度 Gyroid 几何聚合为粗体素 RVE 单元，并根据局部几何特征为每个体素赋予标签与等效输运属性，以便在粗尺度上进行高效求解。

#### 验收标准

1. THE Voxel_Labeler SHALL 将细尺度采样网格聚合为粗体素网格，聚合比例（细尺度分辨率与粗体素尺寸之比）可由用户指定。
2. WHEN 聚合完成后，THE Voxel_Labeler SHALL 为每个粗体素计算局部孔隙率（流体体积分数）、比表面积、界面分布特征及壁面邻近特征。
3. THE Voxel_Labeler SHALL 根据局部孔隙率阈值将每个粗体素赋予以下四种标签之一：0（固体，孔隙率 < 下限阈值）、1（流体，孔隙率 > 上限阈值）、2（界面，介于两阈值之间）、3（加热板，由用户指定的底部边界层）。
4. WHEN 体素标签赋值完成后，THE Voxel_Labeler SHALL 输出每个体素的标签场、孔隙率场、比表面积场，供 BF_LTNE_Solver 和 Neural_Closure_Model 使用。
5. THE Voxel_Labeler SHALL 保证标签场中加热板（标签 3）仅出现在用户指定的底部边界层，且厚度不超过 1 个体素层。
6. IF 聚合比例导致粗体素网格中任意方向的体素数量少于 3，THEN THE Voxel_Labeler SHALL 返回错误信息，提示用户增大计算域或减小聚合比例。

---

### 需求 3：Brinkman–Forchheimer 流动求解

**用户故事：** 作为研究人员，我希望在体素网格上求解 Brinkman–Forchheimer 流动方程，以获得体素级速度场和压力场，用于后续传热计算。

#### 验收标准

1. THE BF_LTNE_Solver SHALL 在标签为 1（流体）和 2（界面）的体素上求解 Brinkman–Forchheimer 动量方程，方程形式为：`μ/K·u + ρ·CF·|u|·u = -∇p`。
2. THE BF_LTNE_Solver SHALL 在标签为 0（固体）和 3（加热板）的体素上将速度场强制置零。
3. WHEN 压力-速度迭代收敛时，THE BF_LTNE_Solver SHALL 满足质量守恒条件，即全域速度散度的 L2 范数小于用户指定的收敛阈值（默认 1×10⁻⁴）。
4. THE BF_LTNE_Solver SHALL 支持以下边界条件：入口（x=0 面）指定均匀速度 `u_in`，出口（x=L 面）指定零压力，y 和 z 方向采用周期性边界条件。
5. WHEN 入口雷诺数 Re 在 1 到 1000 范围内变化时，THE BF_LTNE_Solver SHALL 在不超过 10000 次迭代内收敛。
6. IF 渗透率 K 在任意体素处为零或负值，THEN THE BF_LTNE_Solver SHALL 将该体素的速度置零并继续求解，不中断计算。

---

### 需求 4：LTNE 双温传热求解

**用户故事：** 作为研究人员，我希望通过 LTNE 双温模型求解流体温度场和固体温度场，以准确描述固液非热平衡换热过程。

#### 验收标准

1. THE BF_LTNE_Solver SHALL 在标签为 1（流体）和 2（界面）的体素上求解流体能量方程：`ρ·cp·(u·∇Tf) = ∇·(kf·∇Tf) + hv·(Ts - Tf)`。
2. THE BF_LTNE_Solver SHALL 在标签为 0（固体）、2（界面）和 3（加热板）的体素上求解固体能量方程：`∇·(ks·∇Ts) - hv·(Ts - Tf) + Q_wall = 0`，其中 `Q_wall` 仅在加热板体素（标签 3）处非零。
3. WHEN 底部加热板存在时，THE BF_LTNE_Solver SHALL 通过热通量源项 `q_wall`（单位 W/m²）向固体温度方程引入热量，热通量值由用户指定。
4. THE BF_LTNE_Solver SHALL 支持以下温度边界条件：入口（x=0 面）指定均匀流体入口温度 `T_in`，出口（x=L 面）采用零梯度边界条件，y 和 z 方向采用周期性边界条件。
5. WHEN 流固温度迭代收敛时，THE BF_LTNE_Solver SHALL 满足能量守恒条件，即加热板输入总热量与出口流体带走热量之差的相对误差小于 5%。
6. THE BF_LTNE_Solver SHALL 在每次迭代后输出出口平均流体温度 `T_out`、压降 `Δp`、速度散度范数，供用户监控收敛过程。

---

### 需求 5：宏观性能指标计算

**用户故事：** 作为研究人员，我希望从求解结果中提取标准化的宏观流热性能指标，以便与实验数据和 CFD 结果进行对比验证。

#### 验收标准

1. THE BF_LTNE_Solver SHALL 基于求解结果计算雷诺数 `Re = ρ·u_mean·Dh / μ`，其中 `Dh` 为水力直径，`u_mean` 为入口截面平均速度。
2. THE BF_LTNE_Solver SHALL 基于求解结果计算努塞尔数 `Nu = h·Dh / kf`，其中 `h` 为基于加热板面积和流固温差的平均换热系数。
3. THE BF_LTNE_Solver SHALL 基于求解结果计算摩擦因子 `f = Δp·Dh / (0.5·ρ·u_mean²·L)`，其中 `L` 为流动方向计算域长度。
4. THE BF_LTNE_Solver SHALL 基于求解结果计算科尔本 j 因子 `j = Nu / (Re·Pr^(1/3))`，其中 `Pr` 为普朗特数。
5. THE BF_LTNE_Solver SHALL 将所有宏观性能指标（Re、Nu、f、j、Δp、h、T_out）以结构化字典形式返回，键名与上述定义一致。
6. WHEN 入口平均速度 `u_mean` 小于 1×10⁻⁶ m/s 时，THE BF_LTNE_Solver SHALL 在性能指标字典中将 Re、f、j 标记为 NaN，并输出警告信息，而非产生除零错误。

---

### 需求 6：CFD 数据集构建与管理

**用户故事：** 作为研究人员，我希望构建和管理用于训练神经闭合模型的小尺度 CFD 数据集，以确保训练数据的质量和可复现性。

#### 验收标准

1. THE CFD_Dataset SHALL 支持存储来自不同阵列规模（1×1×1、1×2×1、2×1×1、2×2×1）和不同入口流速的 CFD 仿真结果，每条记录包含阵列规模、入口流速、体素级特征场和宏观响应指标。
2. THE CFD_Dataset SHALL 支持以 HDF5 或 NumPy npz 格式持久化存储，并提供加载接口，加载后数据结构与存储前一致（往返一致性）。
3. WHEN 新的 CFD 数据记录被添加时，THE CFD_Dataset SHALL 验证数据完整性，即检查所有必需字段（阵列规模、入口流速、体素标签场、K、CF、hv、keff、Re、Nu、f、j、Δp、h、T_out）均存在且数值有效（无 NaN 或 Inf）。
4. THE CFD_Dataset SHALL 提供数据集划分功能，按用户指定的比例将数据集划分为训练集、验证集和测试集，划分结果可通过随机种子复现。
5. IF 数据记录中存在 NaN 或 Inf 值，THEN THE CFD_Dataset SHALL 拒绝该记录并返回包含字段名称的错误信息，不将其写入数据集。

---

### 需求 7：神经闭合模型训练

**用户故事：** 作为研究人员，我希望基于 CFD 数据集训练随机场神经闭合模型，学习体素级等效参数（K、CF、hv、keff），以替代昂贵的细尺度 CFD 计算。

#### 验收标准

1. THE Neural_Closure_Model SHALL 以体素级几何特征（孔隙率、比表面积、界面分布、壁面邻近特征）和入口流速作为输入，输出体素级等效参数（K、CF、hv、keff）。
2. THE Neural_Closure_Model SHALL 支持通过有限差分/有限体积迭代，使模型预测的宏观响应（Re、Nu、f、j、Δp、h、T_out）与 CFD 数据集中的参考值匹配，损失函数包含所有七个宏观响应指标的加权均方误差。
3. WHEN 训练完成后，THE Neural_Closure_Model SHALL 在测试集上对所有七个宏观响应指标的相对误差均不超过 10%。
4. THE Neural_Closure_Model SHALL 保证输出的等效参数物理合理性：K > 0、CF ≥ 0、hv ≥ 0、keff > 0，通过激活函数约束实现。
5. THE Neural_Closure_Model SHALL 支持模型权重的保存与加载，保存格式为标准的 PyTorch `.pt` 文件或等效格式，加载后推理结果与保存前一致（往返一致性）。
6. WHEN 训练数据集样本数量少于 10 条时，THE Neural_Closure_Model SHALL 输出警告信息，提示数据量不足可能导致过拟合，但不中断训练流程。

---

### 需求 8：大规模阵列快速预测

**用户故事：** 作为研究人员，我希望利用训练好的神经闭合模型和多尺度求解器，对任意规模的大规模周期 Gyroid 阵列进行快速流热性能预测，无需解析细尺度几何。

#### 验收标准

1. THE BF_LTNE_Solver SHALL 接受由 Neural_Closure_Model 输出的体素级等效参数场（K、CF、hv、keff）作为输入，替代固定参数进行多尺度求解。
2. WHEN 计算域规模从 1×1×1 扩展到 N×M×P（N、M、P 为正整数）时，THE BF_LTNE_Solver SHALL 通过周期性复制体素标签场和等效参数场构建大规模计算域，无需重新运行细尺度 CFD。
3. THE BF_LTNE_Solver SHALL 在单台工作站（16 核 CPU，64 GB 内存）上，对不超过 100×100×100 体素的计算域完成一次完整流热求解（收敛至默认阈值），耗时不超过 30 分钟。
4. WHEN 大规模预测完成后，THE BF_LTNE_Solver SHALL 输出全域压力场、速度场、流体温度场、固体温度场及所有宏观性能指标，并支持以 VTK 或 NumPy npz 格式导出，供后处理可视化使用。
5. THE BF_LTNE_Solver SHALL 支持用户通过配置文件（YAML 或 JSON 格式）指定所有求解参数（计算域尺寸、体素分辨率、流体物性、边界条件、收敛阈值），无需修改源代码。

---

### 需求 9：解析器与序列化（往返一致性）

**用户故事：** 作为研究人员，我希望框架中所有涉及数据读写的模块均满足往返一致性，以确保数据在存储、传输和加载过程中不发生信息损失。

#### 验收标准

1. THE CFD_Dataset SHALL 实现配置文件的解析功能，支持读取 YAML 和 JSON 格式的求解参数配置文件，并将其解析为内部参数对象。
2. THE CFD_Dataset SHALL 实现配置文件的序列化功能，支持将内部参数对象格式化输出为合法的 YAML 或 JSON 文件。
3. FOR ALL 合法的内部参数对象，解析后序列化再解析 SHALL 产生与原始对象等价的参数对象（往返性质）。
4. WHEN 配置文件中存在非法字段或缺失必需字段时，THE CFD_Dataset SHALL 返回包含字段名称和期望类型的描述性错误信息，不产生未捕获的异常。
5. IF 配置文件中的数值参数超出物理合理范围（如负渗透率、负导热率），THEN THE CFD_Dataset SHALL 返回包含参数名称、实际值和合理范围的错误信息。

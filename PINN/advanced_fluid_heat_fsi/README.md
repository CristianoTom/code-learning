# PINN 进阶案例：流体传热与流固耦合

这个文件夹放置更贴近流体传热、流固耦合方向的 PINN 教程案例。相比上一级目录中的基础例子，这里的 notebook 更强调：

- 二维不可压 Navier-Stokes 方程残差
- 速度、压力、温度多物理场联合输出
- 对流扩散传热方程
- 共轭传热界面条件：温度连续、热流连续
- 热浮力 Boussinesq 耦合
- 流固耦合中的接口位移、载荷和结构方程约束

## 建议顺序

1. `A00_common_pinn_blocks.ipynb`  
   进阶案例通用代码块：MLP、自动微分、采样、绘图。

2. `A01_steady_ns_lid_driven_cavity.ipynb`  
   二维稳态不可压 N-S：方腔顶盖驱动流。重点是连续方程、动量方程、无滑移边界和压力参考点。

3. `A02_ns_taylor_green_manufactured.ipynb`  
   二维 N-S 制造解算例。用已知解析场验证 PINN 残差、压力和速度学习是否正确。

4. `A03_channel_convection_heat_transfer.ipynb`  
   二维通道内给定速度场的对流扩散传热。重点是入口、壁面和出口边界处理。

5. `A04_conjugate_heat_transfer_interface.ipynb`  
   流体-固体共轭传热 CHT。重点是界面温度连续和热流连续。

6. `A05_boussinesq_natural_convection_cavity.ipynb`  
   热浮力自然对流方腔。重点是速度-压力-温度三场耦合。

7. `A06_toy_fsi_channel_beam_coupling.ipynb`  
   简化流固耦合：流体压力载荷驱动弹性壁面位移。重点是接口条件和结构方程残差。

## 运行说明

推荐环境：

```powershell
pip install torch numpy matplotlib jupyter
jupyter notebook
```

这些案例为了教学可运行，训练轮数默认设置得偏保守。实际研究中通常需要：

- 增加网络宽度和深度
- 使用更多 collocation points
- 引入学习率调度或 Adam + L-BFGS
- 做变量无量纲化
- 对边界、界面和高梯度区域进行加密采样
- 与 CFD/FEM 数据进行混合训练或误差验证

建议先完整跑通每个 notebook，再逐步把几何、边界条件和物性参数替换成自己的课题设置。

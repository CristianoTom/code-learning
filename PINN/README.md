# PINN 从零入门教程

这个目录包含一组可直接运行的 Jupyter Notebook，用 PyTorch 从零开始学习 Physics-Informed Neural Networks, PINN。

如果你关注流体传热、流固耦合方向，进阶案例已单独放在：

- `advanced_fluid_heat_fsi/`

其中包含二维不可压 Navier-Stokes、二维通道对流传热、共轭传热界面条件、Boussinesq 自然对流和简化 FSI 耦合案例。

## Notebook 路线

1. `00_pinn_autograd_warmup.ipynb`  
   学习 PyTorch 张量、神经网络、自动微分，并用一个小函数拟合例子理解 PINN 的基础工具。

2. `01_pinn_ode_exponential_decay.ipynb`  
   用 PINN 求解一阶常微分方程：`du/dt = -u`，`u(0)=1`。

3. `02_pinn_1d_poisson.ipynb`  
   用 PINN 求解一维 Poisson 方程：`u''(x) = -pi^2 sin(pi x)`，`u(0)=u(1)=0`。

4. `03_pinn_heat_inverse_parameter.ipynb`  
   用 PINN 同时求解热方程并反演未知扩散系数。

5. `04_pinn_hard_boundary_conditions.ipynb`  
   学习用输出变换强制满足边界条件，减少边界损失调参。

6. `05_pinn_advection_equation.ipynb`  
   用 PINN 求解一维线性输运方程，并处理周期边界条件。

7. `06_pinn_heat_forward.ipynb`  
   用 PINN 求解热方程正问题，理解初值、边界和 PDE 残差的配合。

8. `07_pinn_burgers_viscous.ipynb`  
   用 PINN 求解粘性 Burgers 方程，接触非线性 PDE。

9. `08_pinn_wave_equation.ipynb`  
   用 PINN 求解一维波动方程，练习二阶时间导数和速度初值。

10. `09_pinn_2d_poisson.ipynb`  
    用 PINN 求解二维 Poisson 方程，学习多输入网络和二维可视化。

11. `10_pinn_training_tips.ipynb`  
    比较采样点数量、损失权重和网络规模对训练效果的影响。

12. `11_pinn_reusable_template.ipynb`  
    一个可复用的 PyTorch PINN 模板，用于快速改写自己的方程。

## 推荐环境

```powershell
pip install torch numpy matplotlib jupyter
jupyter notebook
```

如果你使用 VS Code，也可以直接打开这些 `.ipynb` 文件逐个运行。

## 学习建议

先按编号顺序运行。每个 notebook 都尽量自包含，建议你运行完之后尝试修改：

- 神经网络层数和宽度
- 采样点数量
- PDE/ODE 残差损失权重
- 训练轮数
- 激活函数

PINN 的学习重点不是只看最终曲线，而是观察损失项、采样点、边界条件和网络表达能力之间如何互相影响。

## 建议运行顺序

如果你是零基础，先运行 `00` 到 `03`。之后按兴趣进入：

- 想掌握约束技巧：运行 `04`
- 想看时间相关 PDE：运行 `05`、`06`、`08`
- 想挑战非线性 PDE：运行 `07`
- 想进入二维问题：运行 `09`
- 想系统调参：运行 `10`
- 想迁移到自己的课题：从 `11` 开始改模板

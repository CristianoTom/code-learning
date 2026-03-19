import matplotlib.pyplot as plt
import numpy as np

# --- 数据提取与模拟 (根据原图趋势选取关键点) ---
epochs = np.linspace(1, 10, 20) # 选取20个散点以保证曲线平滑度

# 1. Train Loss: 初始下降极快，随后呈阶梯状缓慢下降
# 观察点：(1, 0.8), (1.2, 0.45), (2, 0.4), (3, 0.35), (6, 0.25), (10, 0.18)
train_loss = [
    0.80, 0.45, 0.41, 0.39, 0.38, 0.31, 0.31, 0.30, 0.25, 0.24, 
    0.23, 0.22, 0.21, 0.20, 0.16, 0.15, 0.15, 0.11, 0.10, 0.09
]

# 2. Train Accuracy: 从~0.7快速升至0.85，最终接近0.95
train_acc = [
    0.68, 0.83, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.91,
    0.92, 0.92, 0.93, 0.93, 0.94, 0.94, 0.95, 0.95, 0.96, 0.96
]

# 3. Test Accuracy: 初始略高于训练集，后续与训练集高度同步
test_acc = [
    0.86, 0.86, 0.87, 0.87, 0.88, 0.89, 0.90, 0.90, 0.91, 0.91,
    0.92, 0.92, 0.92, 0.93, 0.93, 0.92, 0.93, 0.93, 0.93, 0.93
]

# 2. 设置图表风格
plt.figure(figsize=(5, 4), dpi=120)
plt.rcParams['axes.linewidth'] = 1.5  # 边框加粗

# 3. 绘制曲线 (线宽和颜色严格对应)
plt.plot(epochs, train_loss, label='train loss', color='#1f77b4', linewidth=2)
plt.plot(epochs, train_acc, label='train acc', color='m', linestyle='--', linewidth=2)
plt.plot(epochs, test_acc, label='test acc', color='g', linestyle='-.', linewidth=2)

# 4. 坐标轴范围与刻度 (完全匹配原图)
plt.xlim(1, 10)
plt.ylim(-0.04, 1.05)
plt.xticks([2, 4, 6, 8, 10], fontsize=13)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=13)

# 5. 细节处理：网格与标签
plt.xlabel('epoch', fontsize=14)
plt.grid(True, linestyle='-', linewidth=1.0, color='#C0C0C0') # 深色实线网格

# 6. 图例 (位置与透明度)
plt.legend(loc='center right', fontsize=13, framealpha=0.8, edgecolor='#D3D3D3')

# 7. 自动调整布局，去除白边
plt.tight_layout()

# 显示结果
plt.show()

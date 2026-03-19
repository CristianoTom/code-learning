import matplotlib.pyplot as plt
import numpy as np

# 1. 精准构造数据点 (对应图1的阶梯感和数值)
epochs = np.array([1, 1.2, 1.3, 2, 2.2, 3, 3.2, 4, 4.2, 5, 5.2, 6, 6.3, 7, 7.2, 8, 8.5, 9, 9.2, 10])

# Train Loss: 明显的台阶下降
train_loss = [0.5, 0.35, 0.25, 0.25, 0.18, 0.18, 0.13, 0.14, 0.09, 0.10, 0.06, 0.07, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01]

# Train Acc: 虚线，在 0.85 到 1.0 之间
train_acc = [0.82, 0.90, 0.90, 0.91, 0.93, 0.93, 0.95, 0.95, 0.96, 0.96, 0.98, 0.97, 0.98, 0.98, 0.99, 0.99, 0.995, 0.995, 0.998, 1.0]

# Test Acc: 点划线，在 0.8 到 0.9 之间有波动
test_acc = [0.86, 0.87, 0.87, 0.90, 0.88, 0.81, 0.82, 0.85, 0.86, 0.90, 0.87, 0.83, 0.84, 0.86, 0.86, 0.85, 0.86, 0.91, 0.92, 0.92]

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
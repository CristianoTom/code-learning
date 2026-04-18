import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

x = np.arange(1, 11)
y = np.array([12.3, 13.1, 12.7, 14.2, 15.0, 14.6, 16.2, 16.0, 17.1, 18.0])

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
ax.plot(x, y, linewidth=2.2, marker="o", markersize=5, label="实验数据")

ax.set_title("中文折线图示例", fontsize=16)
ax.set_xlabel("迭代次数", fontsize=13)
ax.set_ylabel("指标值", fontsize=13)
ax.grid(True, alpha=0.25)
ax.legend()

fig.tight_layout()
plt.show()

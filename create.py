import numpy as np
import matplotlib.pyplot as plt

# 参数
c = 1.0          # 变换参数
R = 1.3 * c      # 圆半径（必须 > c）
e = 0.15 * c      # 圆心偏移（控制弯度）

# θ
theta = np.linspace(0, 2*np.pi, 2000)

# ζ 平面中的偏心圆
zeta = (e + 0j) + R * np.exp(1j * theta)

# Joukowski 变换
z = zeta + c**2 / zeta

# 提取实部虚部
x = np.real(z)
y = np.imag(z)

# 绘图
plt.figure(figsize=(8, 4))
plt.plot(x, y, 'b', linewidth=2)

# 坐标轴
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)

plt.gca().set_aspect('equal', adjustable='box')
plt.title("Joukowski Airfoil")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

plt.show()
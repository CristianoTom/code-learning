import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 5))

# 第一个3D子图（1行2列第1个）
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot([0,1,2], [0,1,2], [0,1,4], color='red', linewidth=2)
ax1.set_title('3D Line Plot')

# 第二个3D子图（1行2列第2个）
ax2 = fig.add_subplot(122, projection='3d')
ax2.bar3d([0,1,2], [0,1,2], [0,0,0], 0.5, 0.5, [1,2,3], color='blue', alpha=0.6)
ax2.set_title('3D Bar Plot')

plt.tight_layout()
plt.show()
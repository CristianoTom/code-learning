import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from pathlib import Path

# =============================
# 1) 从 CSV 自动读取目标点
# =============================
CSV_PATH = None
OBJ_COLS = None  # 例如 [6, 7, 8]；None 表示自动取最后 3 列
MINIMIZE = None  # 例如 [True, False, True]；None 表示全部最小化


def resolve_csv_path(csv_path=None):
    if csv_path is not None:
        p = Path(csv_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"CSV file not found: {p}")

    here = Path(__file__).resolve().parent
    candidates = [
        here / ".." / "vae" / "sample_design_data.csv",
        here / ".." / "new" / "performance.csv",
    ]
    for p in candidates:
        p = p.resolve()
        if p.exists():
            return p
    raise FileNotFoundError("No CSV file found. Please set CSV_PATH manually.")


def load_csv_points(csv_path, obj_cols=None):
    # 用浮点读取，兼容首行表头（首行会变成 NaN）。
    arr = np.genfromtxt(csv_path, delimiter=",", dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[0] == 0:
        raise ValueError("CSV has no data rows.")

    # 若首行全 NaN，视为表头并丢弃。
    if np.isnan(arr[0]).all():
        arr = arr[1:]

    # 丢弃含 NaN 的脏数据行。
    arr = arr[~np.isnan(arr).any(axis=1)]

    if arr.shape[0] == 0:
        raise ValueError("CSV has no valid numeric rows.")

    if obj_cols is None:
        if arr.shape[1] < 3:
            raise ValueError("Need at least 3 columns for 3D Pareto plotting.")
        obj_cols = list(range(arr.shape[1] - 3, arr.shape[1]))

    points = arr[:, obj_cols]
    if points.shape[1] != 3:
        raise ValueError("3D plot requires exactly 3 objective columns.")
    return points, obj_cols


csv_file = resolve_csv_path(CSV_PATH)
X_raw, used_cols = load_csv_points(csv_file, OBJ_COLS)

num_obj = X_raw.shape[1]
if MINIMIZE is None:
    minimize_mask = np.ones(num_obj, dtype=bool)
else:
    minimize_mask = np.asarray(MINIMIZE, dtype=bool)
    if minimize_mask.size != num_obj:
        raise ValueError("MINIMIZE length must equal number of objectives.")

# 非支配排序统一按最小化处理。
X_sort = X_raw.copy()
X_sort[:, ~minimize_mask] *= -1.0

print(f"CSV: {csv_file}")
print(f"Objective columns: {used_cols}")
print(f"Samples: {len(X_raw)}, Objectives: {num_obj}")

# =============================
# 2) 非支配排序：得到全部 Pareto 层
# =============================
def dominates(a, b):
    """最小化问题下 a 是否支配 b"""
    return np.all(a <= b) and np.any(a < b)

def non_dominated_sort(points):
    N = len(points)
    S = [[] for _ in range(N)]      # S[p]: p 支配的点索引
    n_dom = np.zeros(N, dtype=int)  # n_dom[p]: 支配 p 的点数量
    fronts = [[]]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(points[p], points[q]):
                S[p].append(q)
            elif dominates(points[q], points[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()  # 去掉最后空层
    return fronts

fronts = non_dominated_sort(X_sort)
print(f"Total Pareto fronts: {len(fronts)}")

# =============================
# 3) 曲面近似函数（每层都尝试画）
# =============================
def smooth_surface_from_points(points_3d, grid_n=32):
    """
    用 PCA(3D->2D 参数域) + griddata 插值回 3D 曲面
    """
    m = len(points_3d)
    if m < 8:
        return None  # 点太少，不足以稳定拟合曲面

    pca = PCA(n_components=2)
    uv = pca.fit_transform(points_3d)
    u, v = uv[:, 0], uv[:, 1]

    uu = np.linspace(u.min(), u.max(), grid_n)
    vv = np.linspace(v.min(), v.max(), grid_n)
    U, V = np.meshgrid(uu, vv)

    # 线性插值
    Xg = griddata((u, v), points_3d[:, 0], (U, V), method="linear")
    Yg = griddata((u, v), points_3d[:, 1], (U, V), method="linear")
    Zg = griddata((u, v), points_3d[:, 2], (U, V), method="linear")

    # 最近邻补 NaN
    Xn = griddata((u, v), points_3d[:, 0], (U, V), method="nearest")
    Yn = griddata((u, v), points_3d[:, 1], (U, V), method="nearest")
    Zn = griddata((u, v), points_3d[:, 2], (U, V), method="nearest")

    Xg = np.where(np.isnan(Xg), Xn, Xg)
    Yg = np.where(np.isnan(Yg), Yn, Yg)
    Zg = np.where(np.isnan(Zg), Zn, Zg)

    return Xg, Yg, Zg

# =============================
# 4) 颜色：按层数自动生成浅色系
# =============================
num_fronts = len(fronts)
cmap = plt.get_cmap("Pastel1", max(num_fronts, 2))
edge_cmap = plt.get_cmap("tab20", max(num_fronts, 2))

# =============================
# 5) 绘制：每个 Pareto 面都画
# =============================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

# 背景候选点（弱化）
ax.scatter(
    X_raw[:, 0], X_raw[:, 1], X_raw[:, 2],
    s=10, c="gray", alpha=0.12, depthshade=False, label="All candidates"
)

for i, idx in enumerate(fronts):
    pts = X_raw[idx]
    face_c = cmap(i)      # 浅色面
    edge_c = edge_cmap(i) # 深色边界

    # --- 曲面：每层都尝试画 ---
    surf = smooth_surface_from_points(pts, grid_n=34)
    if surf is not None:
        Xg, Yg, Zg = surf
        ax.plot_surface(
            Xg, Yg, Zg,
            color=face_c,
            alpha=0.36,          # 浅色透明
            linewidth=0,
            antialiased=True,
            shade=True,
            zorder=1
        )

    # --- 散点：边界清晰 ---
    # 外圈白底+深色边（提高清晰度）
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=52, c="white",
        edgecolors=edge_c, linewidths=1.3,
        alpha=1.0, depthshade=False, zorder=5
    )
    # 内圈浅色填充（表示所属层）
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=24, c=[face_c],
        edgecolors="none",
        alpha=1.0, depthshade=False, zorder=6,
        label=f"Front {i+1} (n={len(idx)})"
    )

# 轴标签
ax.set_xlabel("Objective f1 (min)")
ax.set_ylabel("Objective f2 (min)")
ax.set_zlabel("Objective f3 (min)")
ax.set_title(f"All Pareto Front Surfaces in 3D (Total fronts = {num_fronts})", pad=14)

# 视觉优化
ax.xaxis.pane.set_alpha(0.08)
ax.yaxis.pane.set_alpha(0.08)
ax.zaxis.pane.set_alpha(0.08)
ax.grid(alpha=0.23)
ax.view_init(elev=22, azim=-55)

# 图例太多时可注释下一行，或者改成只显示前10层
ax.legend(loc="upper left", fontsize=8, ncol=1)

plt.tight_layout()
plt.show()
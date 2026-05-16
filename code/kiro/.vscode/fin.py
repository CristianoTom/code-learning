"""
fin.py  翅片流道稳态传热  FVM + SIMPLE
流道: 20x20x240 mm, 空气沿 z 方向流动
翅片区: z=80~160 mm 中间 80 mm, 底部恒温热源
入口: 3 m/s 均匀来流, 取 x-z 截面 2D 求解
"""
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# ── 几何 (m) ──────────────────────────────────
LX, LZ       = 0.020, 0.240
FIN_Z0, FIN_Z1 = 0.080, 0.160
N_FINS       = 4
FIN_T        = 0.001   # 翅片厚度
FIN_H        = 0.010   # 翅片高度 (从底壁 x=0 向上)

# ── 网格 ──────────────────────────────────────
NX, NZ = 30, 90        # 适当降低网格数加快收敛
dx, dz = LX/NX, LZ/NZ

# ── 空气物性 (~300 K) ─────────────────────────
RHO   = 1.177
MU    = 1.846e-5
CP    = 1005.0
K_F   = 0.02624
K_S   = 205.0          # 铝
GAMMA_F = K_F / (RHO * CP)

# ── 边界条件 ──────────────────────────────────
W_IN  = 3.0
T_IN  = 300.0
T_HOT = 360.0

# ── SIMPLE 松弛 ───────────────────────────────
MAX_ITER = 500
AU = 0.5
AP_R = 0.3
AT = 0.9
TOL = 1e-5

# ═══════════════════════════════════════════════
# 翅片固体掩码  solid[i,k], i=0..NX-1, k=0..NZ-1
# ═══════════════════════════════════════════════
kz0 = int(round(FIN_Z0 / dz))
kz1 = int(round(FIN_Z1 / dz)) - 1
ix_h = int(round(FIN_H / dx))          # 翅片高度格数
fin_half = max(1, int(round(FIN_T / dx / 2)))

# 翅片均匀分布在 x 方向
fin_xs = np.linspace(LX / (N_FINS + 1), LX * N_FINS / (N_FINS + 1), N_FINS)

solid = np.zeros((NX, NZ), dtype=bool)
for xc in fin_xs:
    ic = int(xc / dx)
    for di in range(-fin_half, fin_half + 1):
        xi = ic + di
        if 0 <= xi < ix_h:           # 只在底部 ix_h 行
            if kz0 <= kz1:
                solid[xi, kz0:kz1+1] = True

# ── 场变量 (含 ghost, 1-based 内部) ──────────
u = np.zeros((NX+2, NZ+2))
w = np.zeros((NX+2, NZ+2))
p = np.zeros((NX+2, NZ+2))
T = np.full((NX+2, NZ+2), T_IN)

# 初始化主流速度
w[1:NX+1, 1:NZ+1] = W_IN

# 有效扩散系数 (NX x NZ)
gamma = np.where(solid, K_S / (RHO * CP), GAMMA_F)
mu_f  = np.where(solid, MU * 1e8, MU)   # 固体给极大粘度


def idx(i, k):
    return i * NZ + k


def apply_bc():
    # 入口 (k=0): 均匀来流
    w[:, 0] = 2 * W_IN - w[:, 1]
    u[:, 0] = -u[:, 1]
    T[:, 0] = 2 * T_IN - T[:, 1]
    # 出口 (k=NZ+1): 零梯度
    w[:, NZ+1] = w[:, NZ]
    u[:, NZ+1] = u[:, NZ]
    T[:, NZ+1] = T[:, NZ]
    # 底壁 (i=0): 无滑移; 翅片区恒温, 其余绝热
    u[0, :] = -u[1, :]
    w[0, :] = -w[1, :]
    for k in range(1, NZ+1):
        zc = (k - 0.5) * dz
        T[0, k] = (2*T_HOT - T[1, k]) if FIN_Z0 <= zc <= FIN_Z1 else T[1, k]
    # 顶壁 (i=NX+1): 无滑移绝热
    u[NX+1, :] = -u[NX, :]
    w[NX+1, :] = -w[NX, :]
    T[NX+1, :] = T[NX, :]
    # 固体单元速度为零
    u[1:NX+1, 1:NZ+1][solid] = 0.0
    w[1:NX+1, 1:NZ+1][solid] = 0.0


def pl(Pe):
    Pe = np.clip(Pe, -1e6, 1e6)
    return np.maximum(0.0, (1.0 - 0.1 * np.abs(Pe))**5)


def build_and_solve(phi, diff, Fe, Fw, Fn, Fs, src, is_solid, bc_fixed=None):
    """
    用 sparse 直接求解器求解对流扩散方程.
    bc_fixed: dict {(i,k): value} 用于 Dirichlet 边界
    """
    N = NX * NZ
    A = lil_matrix((N, N))
    b = np.zeros(N)

    De = diff * dz / dx;  Dw = diff * dz / dx
    Dn = diff * dx / dz;  Ds = diff * dx / dz

    aE_a = De * pl(Fe / (De + 1e-30)) + np.maximum(-Fe, 0)
    aW_a = Dw * pl(Fw / (Dw + 1e-30)) + np.maximum( Fw, 0)
    aN_a = Dn * pl(Fn / (Dn + 1e-30)) + np.maximum(-Fn, 0)
    aS_a = Ds * pl(Fs / (Ds + 1e-30)) + np.maximum( Fs, 0)
    aP_a = aE_a + aW_a + aN_a + aS_a + (Fe - Fw + Fn - Fs)
    aP_a = np.maximum(aP_a, 1e-30)

    for i in range(NX):
        for k in range(NZ):
            n = idx(i, k)
            ii, kk = i+1, k+1

            if is_solid[i, k]:
                # 固体: phi = 当前值 (保持不变)
                A[n, n] = 1.0
                b[n] = phi[ii, kk]
                continue

            aE = aE_a[i, k]; aW = aW_a[i, k]
            aN = aN_a[i, k]; aS = aS_a[i, k]
            aP = aP_a[i, k]

            A[n, n] = aP
            if i < NX-1 and not is_solid[i+1, k]: A[n, idx(i+1,k)] = -aE
            else: b[n] += aE * phi[ii+1, kk]

            if i > 0 and not is_solid[i-1, k]: A[n, idx(i-1,k)] = -aW
            else: b[n] += aW * phi[ii-1, kk]

            if k < NZ-1 and not is_solid[i, k+1]: A[n, idx(i,k+1)] = -aN
            else: b[n] += aN * phi[ii, kk+1]

            if k > 0 and not is_solid[i, k-1]: A[n, idx(i,k-1)] = -aS
            else: b[n] += aS * phi[ii, kk-1]

            b[n] += src[i, k]

    sol = spsolve(A.tocsr(), b)
    phi_new = phi.copy()
    phi_new[1:NX+1, 1:NZ+1] = sol.reshape(NX, NZ)
    return phi_new, aP_a


def face_fluxes():
    ui = u[1:NX+1, 1:NZ+1]
    wi = w[1:NX+1, 1:NZ+1]
    Fe = RHO * 0.5 * (ui + u[2:NX+2, 1:NZ+1]) * dz
    Fw = RHO * 0.5 * (ui + u[0:NX,   1:NZ+1]) * dz
    Fn = RHO * 0.5 * (wi + w[1:NX+1, 2:NZ+2]) * dx
    Fs = RHO * 0.5 * (wi + w[1:NX+1, 0:NZ  ]) * dx
    # 固体单元通量清零
    Fe[solid] = Fw[solid] = Fn[solid] = Fs[solid] = 0.0
    return Fe, Fw, Fn, Fs


def pressure_correction_step(aP_u, aP_w):
    global u, w, p
    ui = u[1:NX+1, 1:NZ+1]; wi = w[1:NX+1, 1:NZ+1]
    Fe = RHO*0.5*(ui+u[2:NX+2,1:NZ+1])*dz - RHO*0.5*(ui+u[0:NX,1:NZ+1])*dz
    Fn = RHO*0.5*(wi+w[1:NX+1,2:NZ+2])*dx - RHO*0.5*(wi+w[1:NX+1,0:NZ])*dx
    b_pc = -(Fe + Fn)
    b_pc[solid] = 0.0

    apu = np.maximum(aP_u, 1e-30)
    apw = np.maximum(aP_w, 1e-30)
    ae_p = RHO * dz**2 / apu
    aw_p = RHO * dz**2 / apu
    an_p = RHO * dx**2 / apw
    as_p = RHO * dx**2 / apw
    ap_p = ae_p + aw_p + an_p + as_p

    N = NX * NZ
    A = lil_matrix((N, N))
    b_vec = np.zeros(N)

    for i in range(NX):
        for k in range(NZ):
            n = idx(i, k)
            if solid[i, k]:
                A[n, n] = 1.0; continue
            ap = ap_p[i, k]
            A[n, n] = ap
            if i < NX-1: A[n, idx(i+1,k)] -= ae_p[i,k]
            if i > 0:    A[n, idx(i-1,k)] -= aw_p[i,k]
            if k < NZ-1: A[n, idx(i,k+1)] -= an_p[i,k]
            if k > 0:    A[n, idx(i,k-1)] -= as_p[i,k]
            b_vec[n] = b_pc[i, k]

    # 固定参考压力
    A[0, 0] *= 2
    pc_flat = spsolve(A.tocsr(), b_vec)
    pc = pc_flat.reshape(NX, NZ)

    p[1:NX+1, 1:NZ+1] += AP_R * pc
    # 速度修正 (仅流体单元)
    fluid = ~solid
    corr_u = np.zeros((NX, NZ))
    corr_w = np.zeros((NX, NZ))
    pc_pad = np.zeros((NX+2, NZ+2))
    pc_pad[1:NX+1, 1:NZ+1] = pc
    corr_u[fluid] = (dz * (pc_pad[0:NX,1:NZ+1] - pc_pad[2:NX+2,1:NZ+1])[fluid]
                     / (2 * apu[fluid]))
    corr_w[fluid] = (dx * (pc_pad[1:NX+1,0:NZ] - pc_pad[1:NX+1,2:NZ+2])[fluid]
                     / (2 * apw[fluid]))
    u[1:NX+1, 1:NZ+1] += corr_u
    w[1:NX+1, 1:NZ+1] += corr_w


def run():
    global u, w, p, T
    print("开始 SIMPLE 迭代...")
    apply_bc()

    for it in range(MAX_ITER):
        apply_bc()
        Fe, Fw, Fn, Fs = face_fluxes()

        # ── u 动量 ──
        src_u = -(p[2:NX+2,1:NZ+1] - p[0:NX,1:NZ+1]) * dz / 2
        src_u[solid] = 0.0
        u_new, aP_u = build_and_solve(u, mu_f, Fe, Fw, Fn, Fs, src_u, solid)
        u = AU * u_new + (1 - AU) * u

        # ── w 动量 ──
        src_w = -(p[1:NX+1,2:NZ+2] - p[1:NX+1,0:NZ]) * dx / 2
        src_w[solid] = 0.0
        w_new, aP_w = build_and_solve(w, mu_f, Fe, Fw, Fn, Fs, src_w, solid)
        w = AU * w_new + (1 - AU) * w

        # ── 压力修正 ──
        pressure_correction_step(aP_u, aP_w)
        apply_bc()

        # ── 温度 ──
        Fe2, Fw2, Fn2, Fs2 = face_fluxes()
        T_new, _ = build_and_solve(T, gamma, Fe2, Fw2, Fn2, Fs2,
                                   np.zeros((NX, NZ)), solid)
        T = AT * T_new + (1 - AT) * T
        apply_bc()

        # ── 收敛 ──
        Fe_, Fw_, Fn_, Fs_ = face_fluxes()
        res = np.max(np.abs((Fe_ - Fw_) + (Fn_ - Fs_)))
        if it % 50 == 0:
            Tmax = T[1:NX+1, 1:NZ+1].max()
            print(f"  iter={it:4d}  res={res:.2e}  T_max={Tmax:.1f} K")
        if res < TOL:
            print(f"收敛! iter={it}, res={res:.2e}")
            break
    else:
        print(f"达到最大迭代次数, 最终 res={res:.2e}")


def plot_results():
    x = (np.arange(NX) + 0.5) * dx * 1000   # mm
    z = (np.arange(NZ) + 0.5) * dz * 1000   # mm

    Temp  = T[1:NX+1, 1:NZ+1]   # (NX, NZ)
    W_arr = w[1:NX+1, 1:NZ+1]
    U_arr = u[1:NX+1, 1:NZ+1]

    # contourf 期望 (ny, nx) -> 转置
    Z2d, X2d = np.meshgrid(z, x)   # 均为 (NX, NZ)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('翅片流道稳态传热 (x-z 截面)', fontsize=13)

    # 温度云图
    ax = axes[0]
    cf = ax.contourf(Z2d, X2d, Temp, levels=50, cmap='hot')
    plt.colorbar(cf, ax=ax, label='温度 [K]')
    ax.axvline(FIN_Z0*1000, color='cyan', lw=1.5, ls='--')
    ax.axvline(FIN_Z1*1000, color='cyan', lw=1.5, ls='--', label='翅片区')
    for xc in fin_xs:
        ax.add_patch(plt.Rectangle(
            (FIN_Z0*1000, (xc - FIN_T/2)*1000),
            (FIN_Z1 - FIN_Z0)*1000, FIN_T*1000,
            lw=1.2, ec='white', fc='none'))
    ax.set_xlabel('z [mm]'); ax.set_ylabel('x [mm]')
    ax.set_title('温度场 [K]'); ax.legend(fontsize=9)

    # 速度云图 + 流线
    ax = axes[1]
    speed = np.sqrt(U_arr**2 + W_arr**2)
    cf2 = ax.contourf(Z2d, X2d, speed, levels=30, cmap='viridis')
    plt.colorbar(cf2, ax=ax, label='速度 [m/s]')
    # streamplot: x轴=z(NZ,), y轴=x(NX,), u分量=W_arr(NX,NZ), v分量=U_arr(NX,NZ)
    ax.streamplot(z, x, W_arr, U_arr, color='white', lw=0.6, density=1.2)
    ax.axvline(FIN_Z0*1000, color='cyan', lw=1.5, ls='--', label='翅片区')
    ax.axvline(FIN_Z1*1000, color='cyan', lw=1.5, ls='--')
    ax.set_xlabel('z [mm]'); ax.set_ylabel('x [mm]')
    ax.set_title('速度场 [m/s]'); ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('fin_results.png', dpi=150)
    plt.show()
    print("已保存 fin_results.png")


if __name__ == '__main__':
    run()
    plot_results()

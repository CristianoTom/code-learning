"""
2D Finite Volume Method (FVM) for coupled flow and heat transfer
- Incompressible steady-state flow (lid-driven cavity)
- Convection-diffusion energy equation
- SIMPLE algorithm for pressure-velocity coupling
- Power-law scheme for convection-diffusion
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# ─────────────────────────────────────────────
# Grid parameters
# ─────────────────────────────────────────────
NX, NY = 40, 40          # number of control volumes
LX, LY = 1.0, 1.0        # domain size [m]
dx = LX / NX
dy = LY / NY

# Physical properties
RHO   = 1.0              # density [kg/m³]
MU    = 0.01             # dynamic viscosity [Pa·s]  → Re = 100
GAMMA = 0.01             # thermal diffusivity [W/(m·K)]
U_LID = 1.0              # lid velocity [m/s]
T_HOT = 1.0              # hot wall temperature (bottom)
T_COLD = 0.0             # cold wall temperature (top)

# Solver settings
MAX_ITER = 2000
ALPHA_U  = 0.7           # under-relaxation for velocity
ALPHA_P  = 0.3           # under-relaxation for pressure
ALPHA_T  = 0.9           # under-relaxation for temperature
TOL      = 1e-5

# ─────────────────────────────────────────────
# Field arrays  (cell-centred, 1-based ghost cells via padding)
# ─────────────────────────────────────────────
u  = np.zeros((NX+2, NY+2))   # x-velocity
v  = np.zeros((NX+2, NY+2))   # y-velocity
p  = np.zeros((NX+2, NY+2))   # pressure
T  = np.zeros((NX+2, NY+2))   # temperature
pc = np.zeros((NX+2, NY+2))   # pressure correction

# Face velocities (staggered-like, stored at cell faces)
uf = np.zeros((NX+1, NY))     # u at i+1/2 faces
vf = np.zeros((NX, NY+1))     # v at j+1/2 faces


def power_law(Pe):
    """Power-law differencing scheme coefficient."""
    return np.maximum(0.0, (1.0 - 0.1 * np.abs(Pe))**5)


def build_momentum_matrix(phi, rho, mu, dx, dy, F_e, F_w, F_n, F_s, source):
    """
    Assemble FVM coefficients for one momentum component.
    Returns coefficient arrays aP, aE, aW, aN, aS, b  (interior cells only).
    """
    ni, nj = NX, NY
    D_e = mu * dy / dx
    D_w = mu * dy / dx
    D_n = mu * dx / dy
    D_s = mu * dx / dy

    Pe_e = F_e / D_e
    Pe_w = F_w / D_w
    Pe_n = F_n / D_n
    Pe_s = F_s / D_s

    aE = D_e * power_law(Pe_e) + np.maximum(-F_e, 0)
    aW = D_w * power_law(Pe_w) + np.maximum( F_w, 0)
    aN = D_n * power_law(Pe_n) + np.maximum(-F_n, 0)
    aS = D_s * power_law(Pe_s) + np.maximum( F_s, 0)
    aP = aE + aW + aN + aS + (F_e - F_w + F_n - F_s)

    b = source.copy()
    return aP, aE, aW, aN, aS, b


def solve_scalar_field(phi, aP, aE, aW, aN, aS, b):
    """Gauss-Seidel sweep for a scalar field (in-place)."""
    for i in range(1, NX+1):
        for j in range(1, NY+1):
            phi[i, j] = (aE[i-1, j-1]*phi[i+1, j] +
                         aW[i-1, j-1]*phi[i-1, j] +
                         aN[i-1, j-1]*phi[i, j+1] +
                         aS[i-1, j-1]*phi[i, j-1] +
                         b [i-1, j-1]) / aP[i-1, j-1]
    return phi


# ─────────────────────────────────────────────
# Face mass-flux arrays (interior faces)
# ─────────────────────────────────────────────
def compute_face_fluxes():
    """Interpolate cell-centre velocities to faces."""
    Fe = np.zeros((NX, NY))
    Fw = np.zeros((NX, NY))
    Fn = np.zeros((NX, NY))
    Fs = np.zeros((NX, NY))
    for i in range(NX):
        for j in range(NY):
            ii, jj = i+1, j+1
            Fe[i,j] = RHO * 0.5*(u[ii,jj] + u[ii+1,jj]) * dy
            Fw[i,j] = RHO * 0.5*(u[ii,jj] + u[ii-1,jj]) * dy
            Fn[i,j] = RHO * 0.5*(v[ii,jj] + v[ii,jj+1]) * dx
            Fs[i,j] = RHO * 0.5*(v[ii,jj] + v[ii,jj-1]) * dx
    return Fe, Fw, Fn, Fs


# ─────────────────────────────────────────────
# Boundary conditions
# ─────────────────────────────────────────────
def apply_velocity_bc():
    # Lid (top, j=NY+1): u = U_LID, v = 0
    u[:, NY+1] = 2*U_LID - u[:, NY]
    v[:, NY+1] = 0.0
    # Bottom wall (j=0): no-slip
    u[:, 0] = -u[:, 1]
    v[:, 0] = 0.0
    # Left wall (i=0): no-slip
    u[0, :] = 0.0
    v[0, :] = -v[1, :]
    # Right wall (i=NX+1): no-slip
    u[NX+1, :] = 0.0
    v[NX+1, :] = -v[NX, :]


def apply_temperature_bc():
    # Bottom hot wall
    T[:, 0] = 2*T_HOT - T[:, 1]
    # Top cold wall
    T[:, NY+1] = 2*T_COLD - T[:, NY]
    # Left/Right: zero-flux (Neumann)
    T[0, :]    = T[1, :]
    T[NX+1, :] = T[NX, :]


# ─────────────────────────────────────────────
# Pressure-correction (SIMPLE)
# ─────────────────────────────────────────────
def pressure_correction_step(aP_u, aP_v):
    """Solve pressure-correction equation and correct fields."""
    global u, v, p

    # Build pressure-correction source (continuity residual)
    b_pc = np.zeros((NX, NY))
    for i in range(NX):
        for j in range(NY):
            ii, jj = i+1, j+1
            Fe = RHO * 0.5*(u[ii,jj] + u[ii+1,jj]) * dy
            Fw = RHO * 0.5*(u[ii,jj] + u[ii-1,jj]) * dy
            Fn = RHO * 0.5*(v[ii,jj] + v[ii,jj+1]) * dx
            Fs = RHO * 0.5*(v[ii,jj] + v[ii,jj-1]) * dx
            b_pc[i,j] = -(Fe - Fw + Fn - Fs)

    # Assemble sparse system for pc
    N = NX * NY
    A = lil_matrix((N, N))
    b_vec = np.zeros(N)

    def idx(i, j): return i * NY + j

    for i in range(NX):
        for j in range(NY):
            ii, jj = i+1, j+1
            k = idx(i, j)
            apu = aP_u[i, j] if aP_u[i, j] > 1e-30 else 1e-30
            apv = aP_v[i, j] if aP_v[i, j] > 1e-30 else 1e-30

            ae = RHO * dy**2 / apu
            aw = RHO * dy**2 / apu
            an = RHO * dx**2 / apv
            as_ = RHO * dx**2 / apv

            ap = ae + aw + an + as_

            A[k, k] += ap
            if i < NX-1: A[k, idx(i+1,j)] -= ae
            if i > 0:    A[k, idx(i-1,j)] -= aw
            if j < NY-1: A[k, idx(i,j+1)] -= an
            if j > 0:    A[k, idx(i,j-1)] -= as_
            b_vec[k] = b_pc[i, j]

    # Fix reference pressure
    A[0, 0] *= 2
    pc_flat = spsolve(A.tocsr(), b_vec)
    pc[1:NX+1, 1:NY+1] = pc_flat.reshape(NX, NY)

    # Correct pressure and velocity
    p[1:NX+1, 1:NY+1] += ALPHA_P * pc[1:NX+1, 1:NY+1]

    for i in range(NX):
        for j in range(NY):
            ii, jj = i+1, j+1
            apu = aP_u[i, j] if aP_u[i, j] > 1e-30 else 1e-30
            apv = aP_v[i, j] if aP_v[i, j] > 1e-30 else 1e-30
            u[ii, jj] += dy * (pc[ii-1,jj] - pc[ii+1,jj]) / (2*apu)
            v[ii, jj] += dx * (pc[ii,jj-1] - pc[ii,jj+1]) / (2*apv)


# ─────────────────────────────────────────────
# Main SIMPLE loop
# ─────────────────────────────────────────────
def run():
    global u, v, p, T

    print("Starting SIMPLE iterations...")
    for it in range(MAX_ITER):
        apply_velocity_bc()
        apply_temperature_bc()

        Fe, Fw, Fn, Fs = compute_face_fluxes()

        # ── u-momentum ──
        src_u = np.zeros((NX, NY))
        for i in range(NX):
            for j in range(NY):
                ii, jj = i+1, j+1
                src_u[i,j] = -(p[ii+1,jj] - p[ii-1,jj]) * dy / 2
        aP_u, aE_u, aW_u, aN_u, aS_u, b_u = build_momentum_matrix(
            u, RHO, MU, dx, dy, Fe, Fw, Fn, Fs, src_u)
        u_star = u.copy()
        u_star = solve_scalar_field(u_star, aP_u, aE_u, aW_u, aN_u, aS_u, b_u)
        u = ALPHA_U * u_star + (1 - ALPHA_U) * u

        # ── v-momentum ──
        src_v = np.zeros((NX, NY))
        for i in range(NX):
            for j in range(NY):
                ii, jj = i+1, j+1
                src_v[i,j] = -(p[ii,jj+1] - p[ii,jj-1]) * dx / 2
        aP_v, aE_v, aW_v, aN_v, aS_v, b_v = build_momentum_matrix(
            v, RHO, MU, dx, dy, Fe, Fw, Fn, Fs, src_v)
        v_star = v.copy()
        v_star = solve_scalar_field(v_star, aP_v, aE_v, aW_v, aN_v, aS_v, b_v)
        v = ALPHA_U * v_star + (1 - ALPHA_U) * v

        # ── pressure correction ──
        pressure_correction_step(aP_u, aP_v)
        apply_velocity_bc()

        # ── energy equation ──
        Fe_t, Fw_t, Fn_t, Fs_t = compute_face_fluxes()
        src_T = np.zeros((NX, NY))
        aP_T, aE_T, aW_T, aN_T, aS_T, b_T = build_momentum_matrix(
            T, RHO, GAMMA, dx, dy, Fe_t, Fw_t, Fn_t, Fs_t, src_T)
        T_new = T.copy()
        T_new = solve_scalar_field(T_new, aP_T, aE_T, aW_T, aN_T, aS_T, b_T)
        T = ALPHA_T * T_new + (1 - ALPHA_T) * T
        apply_temperature_bc()

        # ── convergence check ──
        res_u = np.max(np.abs(u[1:NX+1, 1:NY+1]))
        res_cont = 0.0
        for i in range(NX):
            for j in range(NY):
                ii, jj = i+1, j+1
                Fe_ = RHO * 0.5*(u[ii,jj]+u[ii+1,jj])*dy
                Fw_ = RHO * 0.5*(u[ii,jj]+u[ii-1,jj])*dy
                Fn_ = RHO * 0.5*(v[ii,jj]+v[ii,jj+1])*dx
                Fs_ = RHO * 0.5*(v[ii,jj]+v[ii,jj-1])*dx
                res_cont = max(res_cont, abs(Fe_-Fw_+Fn_-Fs_))

        if it % 100 == 0:
            print(f"  iter={it:4d}  continuity_res={res_cont:.2e}")

        if res_cont < TOL:
            print(f"Converged at iteration {it}, residual={res_cont:.2e}")
            break

    return u, v, p, T


# ─────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────
def plot_results():
    x = np.linspace(dx/2, LX-dx/2, NX)
    y = np.linspace(dy/2, LY-dy/2, NY)
    X, Y = np.meshgrid(x, y, indexing='ij')

    U = u[1:NX+1, 1:NY+1]
    V = v[1:NX+1, 1:NY+1]
    Temp = T[1:NX+1, 1:NY+1]
    speed = np.sqrt(U**2 + V**2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Velocity magnitude + streamlines
    ax = axes[0]
    cf = ax.contourf(X, Y, speed, levels=20, cmap='viridis')
    ax.streamplot(x, y, U.T, V.T, color='white', linewidth=0.5, density=1.2)
    plt.colorbar(cf, ax=ax, label='Speed [m/s]')
    ax.set_title('Velocity Field')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    # Pressure
    ax = axes[1]
    cf = ax.contourf(X, Y, p[1:NX+1, 1:NY+1], levels=20, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax, label='Pressure [Pa]')
    ax.set_title('Pressure Field')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    # Temperature
    ax = axes[2]
    cf = ax.contourf(X, Y, Temp, levels=20, cmap='hot')
    plt.colorbar(cf, ax=ax, label='Temperature')
    ax.set_title('Temperature Field')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig('.vscode/fvm_results.png', dpi=150)
    plt.show()
    print("Results saved to .vscode/fvm_results.png")


if __name__ == '__main__':
    run()
    plot_results()

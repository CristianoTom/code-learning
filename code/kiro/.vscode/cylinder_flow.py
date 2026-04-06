"""
2D Flow past a Cylinder — FVM + Immersed Boundary Method (IBM)
- Unsteady incompressible Navier-Stokes (fractional-step / projection method)
- Cartesian grid with cylinder enforced via direct forcing IBM
- Visualises: vorticity, pressure, velocity magnitude, streamlines
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
# Domain
LX, LY = 8.0, 4.0          # [m]  channel length × height
NX, NY  = 320, 160          # grid cells
dx, dy  = LX/NX, LY/NY

# Cylinder
CX, CY = 2.0, 2.0           # centre
R      = 0.25               # radius  → D = 0.5

# Flow
U_INF = 1.0                 # inlet velocity
RE    = 100.0               # Reynolds number  (laminar vortex shedding)
NU    = U_INF * (2*R) / RE  # kinematic viscosity

# Time
DT        = 0.005
N_STEPS   = 4000            # total steps  (20 s physical time)
PLOT_EVERY = 500            # save snapshot every N steps

print(f"Re={RE:.0f},  nu={NU:.4f},  CFL≈{U_INF*DT/dx:.3f}")

# ─────────────────────────────────────────────
# Grid (cell-centred, ghost layers at boundaries)
# ─────────────────────────────────────────────
x = (np.arange(NX) + 0.5) * dx   # cell centres
y = (np.arange(NY) + 0.5) * dy

X, Y = np.meshgrid(x, y, indexing='ij')   # shape (NX, NY)

# IBM mask: True inside cylinder
MASK = (X - CX)**2 + (Y - CY)**2 <= R**2

# ─────────────────────────────────────────────
# Field arrays
# ─────────────────────────────────────────────
u  = np.ones((NX, NY)) * U_INF   # x-velocity
v  = np.zeros((NX, NY))          # y-velocity
p  = np.zeros((NX, NY))          # pressure

# Enforce cylinder at t=0
u[MASK] = 0.0
v[MASK] = 0.0


# ─────────────────────────────────────────────
# Helper: periodic-safe index clamp
# ─────────────────────────────────────────────
def clamp(i, n):
    return np.clip(i, 0, n-1)


# ─────────────────────────────────────────────
# Boundary conditions
# ─────────────────────────────────────────────
def apply_bc(u, v):
    # Inlet (left): uniform flow
    u[0, :]  = U_INF
    v[0, :]  = 0.0
    # Outlet (right): zero-gradient (convective)
    u[-1, :] = u[-2, :]
    v[-1, :] = v[-2, :]
    # Top / bottom: slip (symmetry)
    u[:, 0]  = u[:, 1]
    v[:, 0]  = 0.0
    u[:, -1] = u[:, -2]
    v[:, -1] = 0.0
    return u, v


# ─────────────────────────────────────────────
# Laplacian (2nd-order central, Neumann at walls)
# ─────────────────────────────────────────────
def laplacian(f, dx, dy):
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        (f[2:,  1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2 +
        (f[1:-1, 2:]  - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dy**2
    )
    return lap


# ─────────────────────────────────────────────
# Divergence of (u,v)
# ─────────────────────────────────────────────
def divergence(u, v, dx, dy):
    div = np.zeros_like(u)
    div[1:-1, 1:-1] = (
        (u[2:,  1:-1] - u[:-2, 1:-1]) / (2*dx) +
        (v[1:-1, 2:]  - v[1:-1, :-2]) / (2*dy)
    )
    return div


# ─────────────────────────────────────────────
# Pressure Poisson solver (Gauss-Seidel)
# ─────────────────────────────────────────────
def solve_pressure(p, rhs, dx, dy, n_iter=50):
    dx2, dy2 = dx**2, dy**2
    denom = 2*(1/dx2 + 1/dy2)
    for _ in range(n_iter):
        pn = p.copy()
        p[1:-1, 1:-1] = (
            (pn[2:,  1:-1] + pn[:-2, 1:-1]) / dx2 +
            (pn[1:-1, 2:]  + pn[1:-1, :-2]) / dy2 -
            rhs[1:-1, 1:-1]
        ) / denom
        # Neumann BCs for pressure
        p[0,  :] = p[1,  :]
        p[-1, :] = p[-2, :]
        p[:,  0] = p[:,  1]
        p[:, -1] = p[:, -2]
    return p


# ─────────────────────────────────────────────
# Advection: 2nd-order upwind (van Leer-like)
# ─────────────────────────────────────────────
def advect(f, u, v, dx, dy):
    """Upwind advection of scalar f by velocity (u,v)."""
    adv = np.zeros_like(f)
    # x-direction
    ux_p = np.maximum(u, 0)
    ux_m = np.minimum(u, 0)
    adv[1:-1, 1:-1] += (
        ux_p[1:-1, 1:-1] * (f[1:-1, 1:-1] - f[:-2,  1:-1]) / dx +
        ux_m[1:-1, 1:-1] * (f[2:,   1:-1] - f[1:-1, 1:-1]) / dx
    )
    # y-direction
    vy_p = np.maximum(v, 0)
    vy_m = np.minimum(v, 0)
    adv[1:-1, 1:-1] += (
        vy_p[1:-1, 1:-1] * (f[1:-1, 1:-1] - f[1:-1, :-2 ]) / dy +
        vy_m[1:-1, 1:-1] * (f[1:-1, 2:  ] - f[1:-1, 1:-1]) / dy
    )
    return adv


# ─────────────────────────────────────────────
# Vorticity
# ─────────────────────────────────────────────
def vorticity(u, v, dx, dy):
    w = np.zeros_like(u)
    w[1:-1, 1:-1] = (
        (v[2:,  1:-1] - v[:-2, 1:-1]) / (2*dx) -
        (u[1:-1, 2:]  - u[1:-1, :-2]) / (2*dy)
    )
    return w


# ─────────────────────────────────────────────
# Main time-stepping loop (fractional-step)
# ─────────────────────────────────────────────
snapshots = []

for step in range(1, N_STEPS + 1):

    # 1. Intermediate velocity (no pressure gradient)
    adv_u = advect(u, u, v, dx, dy)
    adv_v = advect(v, u, v, dx, dy)
    lap_u = laplacian(u, dx, dy)
    lap_v = laplacian(v, dx, dy)

    u_star = u + DT * (-adv_u + NU * lap_u)
    v_star = v + DT * (-adv_v + NU * lap_v)

    # 2. IBM: zero velocity inside cylinder
    u_star[MASK] = 0.0
    v_star[MASK] = 0.0

    # 3. Apply BCs to intermediate field
    u_star, v_star = apply_bc(u_star, v_star)

    # 4. Pressure Poisson  ∇²p = (1/dt) ∇·u*
    div = divergence(u_star, v_star, dx, dy)
    rhs = div / DT
    p = solve_pressure(p, rhs, dx, dy, n_iter=30)

    # 5. Velocity correction
    u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - DT * (p[2:,  1:-1] - p[:-2, 1:-1]) / (2*dx)
    v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - DT * (p[1:-1, 2:]  - p[1:-1, :-2]) / (2*dy)

    # 6. IBM + BCs on corrected velocity
    u[MASK] = 0.0
    v[MASK] = 0.0
    u, v = apply_bc(u, v)

    # 7. Save snapshot
    if step % PLOT_EVERY == 0:
        w = vorticity(u, v, dx, dy)
        snapshots.append({
            'step': step,
            'u': u.copy(), 'v': v.copy(),
            'p': p.copy(), 'w': w.copy()
        })
        print(f"  step {step}/{N_STEPS}  |w|_max={np.max(np.abs(w)):.3f}")

print("Simulation complete.")

# ─────────────────────────────────────────────
# Plot last snapshot
# ─────────────────────────────────────────────
snap = snapshots[-1]
U, V, P, W = snap['u'], snap['v'], snap['p'], snap['w']
speed = np.sqrt(U**2 + V**2)

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
fig.suptitle(f"Flow past a Cylinder  Re={RE:.0f}  t={snap['step']*DT:.1f}s", fontsize=13)

cyl_patch = lambda ax: ax.add_patch(
    plt.Circle((CX, CY), R, color='gray', zorder=5))

# (a) Vorticity
ax = axes[0, 0]
lim = max(abs(W.min()), abs(W.max())) * 0.8
norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
cf = ax.contourf(X.T, Y.T, W.T, levels=64, cmap='RdBu_r', norm=norm)
plt.colorbar(cf, ax=ax, label='ω [1/s]')
cyl_patch(ax)
ax.set_title('Vorticity')
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')

# (b) Pressure
ax = axes[0, 1]
cf = ax.contourf(X.T, Y.T, P.T, levels=64, cmap='coolwarm')
plt.colorbar(cf, ax=ax, label='p [Pa]')
cyl_patch(ax)
ax.set_title('Pressure')
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')

# (c) Velocity magnitude + streamlines
ax = axes[1, 0]
cf = ax.contourf(X.T, Y.T, speed.T, levels=64, cmap='viridis')
plt.colorbar(cf, ax=ax, label='|u| [m/s]')
# Streamlines on coarser grid
skip = 4
ax.streamplot(x[::skip], y[::skip],
              U[::skip, ::skip].T, V[::skip, ::skip].T,
              color='white', linewidth=0.5, density=1.5, arrowsize=0.8)
cyl_patch(ax)
ax.set_title('Velocity Magnitude + Streamlines')
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')

# (d) u-velocity profile at x = 4D downstream
ax = axes[1, 1]
x_probe = CX + 4 * (2*R)
ix = int(x_probe / dx)
ax.plot(U[ix, :], y, 'b-', linewidth=1.5, label=f'u at x={x_probe:.1f}')
ax.axvline(U_INF, color='gray', linestyle='--', label='U_inf')
ax.set_xlabel('u [m/s]'); ax.set_ylabel('y [m]')
ax.set_title(f'u-profile at x = {x_probe:.1f} m')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('.vscode/cylinder_flow.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → .vscode/cylinder_flow.png")

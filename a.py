import argparse

import numpy as np


def sample_centers(*, cell_length: float, step: float) -> tuple[np.ndarray, np.ndarray]:
    half = float(cell_length) * 0.5
    start = -half + float(step) * 0.5
    x = np.arange(start, half, float(step), dtype=np.float32)
    y = np.arange(start, half, float(step), dtype=np.float32)
    return x, y


def field_value(x: np.ndarray, y: np.ndarray, *, w: float, C: float) -> np.ndarray:
    w_f = float(w)
    return -np.abs(np.sin(w_f * x) * np.cos(w_f * y) + np.sin(w_f * y) + np.cos(w_f * x)) + float(C)


def draw_pattern(
    *,
    cell_length: float,
    step: float,
    w: float,
    C: float,
    mode: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    levels: int,
    colorbar: bool,
    save_path: str | None,
) -> None:
    import matplotlib

    try:
        import tkinter  # noqa: F401

        matplotlib.use("TkAgg", force=True)
    except Exception:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x, y = sample_centers(cell_length=cell_length, step=step)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    f_raw = field_value(xx, yy, w=w, C=C)
    f_plot = f_raw
    if mode in ("heat", "contour"):
        absmax_raw = float(np.nanmax(np.abs(f_raw)))
        eps = max(1e-12, absmax_raw * 1e-6)
        inside = f_raw > 0.0
        boundary = np.abs(f_raw) <= eps
        boundary[1:, :] |= inside[1:, :] != inside[:-1, :]
        boundary[:-1, :] |= inside[:-1, :] != inside[1:, :]
        boundary[:, 1:] |= inside[:, 1:] != inside[:, :-1]
        boundary[:, :-1] |= inside[:, :-1] != inside[:, 1:]
        fill_mask = inside & ~boundary
        if np.any(fill_mask):
            f_plot = f_raw.copy()
            f_plot[fill_mask] = float(np.nanmax(f_raw))

    half = float(cell_length) * 0.5
    plt.figure(figsize=(6, 6), dpi=150)
    if mode == "binary":
        img = (~(f_raw > 0.0)).astype(np.float32)
        plt.imshow(
            img,
            cmap="gray",
            origin="lower",
            extent=(-half, half, -half, half),
            interpolation="nearest",
        )
    elif mode == "heat":
        if vmin is None and vmax is None:
            absmax = float(np.nanmax(np.abs(f_raw)))
            if absmax > 0:
                vmin, vmax = -absmax, absmax
        im = plt.imshow(
            f_plot,
            cmap=cmap,
            origin="lower",
            extent=(-half, half, -half, half),
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        if colorbar:
            plt.colorbar(im, fraction=0.046, pad=0.04)
    elif mode == "contour":
        if vmin is None and vmax is None:
            absmax = float(np.nanmax(np.abs(f_raw)))
            if absmax > 0:
                vmin, vmax = -absmax, absmax
        cs = plt.contourf(
            xx,
            yy,
            f_plot,
            levels=int(levels),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if colorbar:
            plt.colorbar(cs, fraction=0.046, pad=0.04)
    else:
        raise ValueError(f"未知 mode: {mode!r}，可选: binary/heat/contour")

    fmin = float(np.nanmin(f_raw))
    fmax = float(np.nanmax(f_raw))
    if fmin <= 0.0 <= fmax:
        plt.contour(xx, yy, f_raw, levels=[0.0], colors="k", linewidths=1.6)

    plt.axis("equal")
    plt.axis("off")
    plt.title(f"sin(wx)cos(wy)+sin(wy)+cos(wx)-C, mode={mode}, w={w:g}, C={C:g}, step={step:g}")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=float, default=40.0)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--w", type=float, default=float(2.0 * np.pi / 20.0))
    parser.add_argument("--C", type=float, default=0.4)
    parser.add_argument("--mode", type=str, default="contour", choices=("binary", "heat", "contour"))
    parser.add_argument("--cmap", type=str, default="RdBu_r")
    parser.add_argument("--vmin", type=float, default=-1)
    parser.add_argument("--vmax", type=float, default=1)
    parser.add_argument("--levels", type=int, default=30)
    parser.add_argument("--colorbar", action="store_true")
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    vmin = None if np.isnan(float(args.vmin)) else float(args.vmin)
    vmax = None if np.isnan(float(args.vmax)) else float(args.vmax)
    save_path = args.save.strip() or None
    draw_pattern(
        cell_length=args.cell,
        step=args.step,
        w=args.w,
        C=args.C,
        mode=args.mode,
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
        levels=args.levels,
        colorbar=bool(args.colorbar),
        save_path=save_path,
    )


if __name__ == "__main__":
    main()

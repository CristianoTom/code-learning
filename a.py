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
    return np.sin(w_f * x) * np.cos(w_f * y) + np.sin(w_f * y) + np.cos(w_f * x) - float(C)


def draw_binary_pattern(*, cell_length: float, step: float, w: float, C: float, save_path: str | None) -> None:
    import matplotlib

    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt

    x, y = sample_centers(cell_length=cell_length, step=step)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    f = field_value(xx, yy, w=w, C=C)
    mask_black = f > 0.0
    img = (~mask_black).astype(np.float32)

    half = float(cell_length) * 0.5
    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(
        img,
        cmap="gray",
        origin="lower",
        extent=(-half, half, -half, half),
        interpolation="nearest",
    )
    plt.axis("equal")
    plt.axis("off")
    plt.title(f"sin(wx)cos(wy)+sin(wy)+cos(wx)-C, w={w:g}, C={C:g}, step={step:g}")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=float, default=40.0)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--w", type=float, default=float(2.0 * np.pi / 20.0))
    parser.add_argument("--C", type=float, default=0.2)
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    save_path = args.save.strip() or None
    draw_binary_pattern(cell_length=args.cell, step=args.step, w=args.w, C=args.C, save_path=save_path)


if __name__ == "__main__":
    main()

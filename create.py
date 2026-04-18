from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np


def perturb_series_keep_trend(
    values: List[float],
    *,
    rel_noise: float = 0.05,
    max_rel_perturb: float | None = 0.02,
    seed: int | None = 42,
    keep: str = "auto",
    smooth_strength: float = 0.25,
    decimals: int = 2,
) -> List[float]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("输入必须是一维列表")
    if arr.size < 2:
        return np.round(arr, decimals).tolist()

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=rel_noise, size=arr.shape)
    if max_rel_perturb is not None:
        noise = np.clip(noise, -float(max_rel_perturb), float(max_rel_perturb))

    noisy = arr * (1.0 + noise)
    noisy = (1.0 - smooth_strength) * noisy + smooth_strength * arr

    if keep not in {"auto", "nonincreasing", "nondecreasing"}:
        raise ValueError("keep 只能是 auto/nonincreasing/nondecreasing")

    if keep == "auto":
        d = np.diff(arr)
        if np.all(d <= 0):
            keep_mode = "nonincreasing"
        elif np.all(d >= 0):
            keep_mode = "nondecreasing"
        else:
            keep_mode = "nonincreasing" if float(arr[-1] - arr[0]) <= 0 else "nondecreasing"
    else:
        keep_mode = keep

    out = noisy.copy()
    if keep_mode == "nonincreasing":
        for i in range(1, out.size):
            out[i] = min(out[i], out[i - 1])
    else:
        for i in range(1, out.size):
            out[i] = max(out[i], out[i - 1])

    return np.round(out, decimals).tolist()


if __name__ == "__main__":
    demo = [66.1, 60.15, 55.74, 52.45, 50.44, 48.16, 46.53, 45.27, 44.0, 43.07]
    new_demo = perturb_series_keep_trend(demo, rel_noise=0.05, max_rel_perturb=0.01, seed=123, decimals=2)
    print(new_demo)

    x = np.arange(len(demo))
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=True)
    ax0, ax1 = axes

    ax0.plot(x, demo, linewidth=2.0)
    ax0.set_title("原始数据", fontsize=14)
    ax0.set_xlabel("index")
    ax0.set_ylabel("value")
    ax0.grid(True, alpha=0.25)

    ax1.plot(x, new_demo, linewidth=2.0, color="tab:red")
    ax1.set_title("扰动后数据", fontsize=14)
    ax1.set_xlabel("index")
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    plt.show()

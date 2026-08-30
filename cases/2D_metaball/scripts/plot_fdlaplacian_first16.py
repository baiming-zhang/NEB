"""Plot the first sixteen saved FD Laplacian streamfunction modes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.basis, allow_pickle=False) as archive:
        eigenvalues = archive["eigenvalues"].astype(np.float64)
        psi = archive["psi_valid"].astype(np.float32)
        valid = archive["valid_mask"].astype(bool)
        token = int(archive["geometry_token"])
    if psi.shape[1] < 16:
        raise ValueError(f"basis contains only {psi.shape[1]} modes")

    fields = np.full((*valid.shape, 16), np.nan, dtype=np.float32)
    fields[valid] = psi[:, :16]
    limit = float(np.nanmax(np.abs(fields)))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 8, figsize=(19.2, 5.8), constrained_layout=True)
    image = None
    for index, axis in enumerate(axes.ravel()):
        image = axis.imshow(
            fields[:, :, index],
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.contour(
            valid.astype(np.float32),
            levels=[0.5],
            colors=["#252525"],
            linewidths=0.45,
            origin="lower",
        )
        axis.set_title(
            rf"Mode {index + 1:02d}" + "\n" + rf"$\lambda={eigenvalues[index]:.2f}$",
            fontsize=12,
            pad=4,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
    if image is None:
        raise RuntimeError("no modes were rendered")
    colorbar = fig.colorbar(
        image,
        ax=axes,
        orientation="vertical",
        fraction=0.012,
        pad=0.012,
        aspect=35,
    )
    colorbar.set_label(r"Streamfunction $\psi_k$ (discrete $L^2$ normalized)", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
    fig.suptitle(
        rf"Geometry-aware FD Dirichlet Laplacian · first 16 of 1024 modes · unseen token {token}",
        fontsize=16,
        fontweight="bold",
    )
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

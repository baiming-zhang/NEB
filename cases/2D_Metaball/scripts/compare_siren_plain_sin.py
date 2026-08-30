"""Matched plain-sin versus raw spatial-SIREN comparison at epoch 500."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metaball_runtime import backend, install_backend


BLUE = "#2166AC"
RED = "#B2182B"
TEXT = "#222222"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_history(run: Path) -> list[dict]:
    path = run / "data" / "hardcoded_training_history.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows or int(rows[-1]["epoch"]) < 500:
        raise RuntimeError(f"epoch-500 history is not ready: {path}")
    return [row for row in rows if int(row["epoch"]) <= 500]


def preflight_metrics(root: Path) -> dict:
    text = (root / "preflight_k1024" / "hardcoded_train.log").read_text(
        encoding="utf-8"
    )
    matches = re.findall(
        r"self-test epoch=\d+ vorticity_R=([0-9.eE+-]+) "
        r"orth=([0-9.eE+-]+) rank=(\d+)",
        text,
    )
    if not matches:
        raise RuntimeError(f"preflight metrics missing under {root}")
    loss, orth, rank = matches[-1]
    return {"loss": float(loss), "orth": float(orth), "rank": int(rank)}


def epoch_row(rows: list[dict], epoch: int) -> dict:
    return next(row for row in rows if int(row["epoch"]) == epoch)


def raw_gram_spectrum(checkpoint: Path, validation_pack: Path) -> np.ndarray:
    metadata = json.loads(
        checkpoint.with_suffix(".json").read_text(encoding="utf-8")
    )
    config = dict(metadata["config"])
    install_backend(config)
    cfg = backend.cfg_namespace(config)
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with np.load(validation_pack, allow_pickle=False) as archive:
        points = archive["validation_points"][0]
        token = float(archive["validation_beta"][0])
    model = backend.HardcodedModel(cfg, arrays)
    raw = backend.cp.empty((len(points), cfg.modes), dtype=backend.F32)
    for mode_index, (_, field, _) in enumerate(
        model.forward_stream(cfg, points, token, 0.0, False)
    ):
        raw[:, mode_index] = -(
            field.hess[:, 0, 0] + field.hess[:, 0, 2]
        )
    gram = raw.T @ raw
    gram = np.float32(0.5) * (gram + gram.T)
    values = backend.cp.asnumpy(backend.cp.linalg.eigvalsh(gram))[::-1]
    values = np.maximum(values, 0.0)
    return values / max(float(values[0]), np.finfo(np.float32).tiny)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain-root", type=Path, required=True)
    parser.add_argument("--siren-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plain_run = args.plain_root / "run_001"
    siren_run = args.siren_root / "run_001"
    plain_history = load_history(plain_run)
    siren_history = load_history(siren_run)
    plain_pre = preflight_metrics(args.plain_root)
    siren_pre = preflight_metrics(args.siren_root)
    plain_500 = epoch_row(plain_history, 500)
    siren_500 = epoch_row(siren_history, 500)

    plain_checkpoint = plain_run / "checkpoints" / "hardcoded_epoch0500.npz"
    siren_checkpoint = siren_run / "checkpoints" / "hardcoded_epoch0500.npz"
    validation_pack = args.siren_root / "inputs" / "fixed_validation_scipy_sobol_v1.npz"
    plain_spectrum = raw_gram_spectrum(plain_checkpoint, validation_pack)
    siren_spectrum = raw_gram_spectrum(siren_checkpoint, validation_pack)

    rows = []
    for label, omega0, pre, final, checkpoint in (
        ("plain sin", 1.0, plain_pre, plain_500, plain_checkpoint),
        ("spatial-SIREN", 32.0, siren_pre, siren_500, siren_checkpoint),
    ):
        rows.append(
            {
                "method": label,
                "spatial_omega0": omega0,
                "initial_rank": pre["rank"],
                "initial_loss": pre["loss"],
                "epoch500_train_rank": int(final["train_rank"]),
                "epoch500_train_loss": float(final["train_loss"]),
                "epoch500_validation_rank": int(final["validation_min_rank"]),
                "epoch500_validation_loss": float(final["validation_loss"]),
                "epoch500_validation_orth_error": float(final["validation_orth_error"]),
                "epoch500_train_projection_ratio": float(
                    final["train_mean_projection_ratio"]
                ),
                "checkpoint_sha256": sha256(checkpoint),
            }
        )

    with (args.output_dir / "plain_sin_vs_siren_epoch500.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "matched_controls": {
            "modes": 1024,
            "epochs_compared": 500,
            "input": "raw xy plus the same nine circle parameters",
            "initial_npz_sha256_plain": sha256(
                args.plain_root / "inputs" / "initial_epoch0000.npz"
            ),
            "initial_npz_sha256_siren": sha256(
                args.siren_root / "inputs" / "initial_epoch0000.npz"
            ),
            "sole_variable": "first-layer raw spatial omega0 (1 versus 32)",
            "precision": "all FP32",
        },
        "results": rows,
        "conclusion": (
            "Plain sin is bandwidth-starved and collapses under the common "
            "metaball envelope; canonical raw SIREN preserves a broad, "
            "well-conditioned vorticity subspace without positional encoding."
        ),
    }
    (args.output_dir / "plain_sin_vs_siren_epoch500.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
        }
    )
    fig = plt.figure(figsize=(11.7, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 0.55))
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(3)]
    histories = [("plain sin", plain_history, RED), ("spatial-SIREN", siren_history, BLUE)]
    plots = (
        ("train_loss", "Rayleigh loss", True),
        ("train_rank", "effective raw rank", False),
        ("train_orth_error", "orthogonality error", True),
        ("train_mean_projection_ratio", "projection ratio", True),
    )
    for axis, (key, title, log_scale) in zip(axes[:4], plots):
        for label, history, color in histories:
            x = np.asarray([row["epoch"] for row in history])
            y = np.asarray([row[key] for row in history], dtype=float)
            axis.plot(x, y, color=color, linewidth=1.0, label=label)
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.18, linewidth=0.45)
    axes[0].legend(frameon=False, fontsize=7)
    for label, spectrum, color in (
        ("plain sin", plain_spectrum, RED),
        ("spatial-SIREN", siren_spectrum, BLUE),
    ):
        axes[4].plot(
            np.arange(1, len(spectrum) + 1), spectrum,
            color=color, linewidth=0.9, label=label,
        )
    axes[4].axhline(
        np.finfo(np.float32).eps * 4096,
        color="#777777", linestyle="--", linewidth=0.7, label="rank threshold",
    )
    axes[4].set_yscale("log")
    axes[4].set_xlabel("Gram eigenvalue index")
    axes[4].set_title("normalized raw-vorticity Gram spectrum")
    axes[4].grid(alpha=0.18, linewidth=0.45)
    axes[4].legend(frameon=False, fontsize=6)

    categories = ["initial rank", "epoch-500 val rank"]
    xx = np.arange(2)
    axes[5].bar(
        xx - 0.18,
        [rows[0]["initial_rank"], rows[0]["epoch500_validation_rank"]],
        width=0.36, color=RED, label="plain sin",
    )
    axes[5].bar(
        xx + 0.18,
        [rows[1]["initial_rank"], rows[1]["epoch500_validation_rank"]],
        width=0.36, color=BLUE, label="spatial-SIREN",
    )
    axes[5].set_xticks(xx, categories, rotation=12)
    axes[5].set_ylim(0, 1024)
    axes[5].set_title("rank retention")
    axes[5].grid(axis="y", alpha=0.18, linewidth=0.45)

    table_axis = fig.add_subplot(grid[2, :])
    table_axis.axis("off")
    table_data = [
        [
            row["method"], f'{row["spatial_omega0"]:.0f}',
            str(row["initial_rank"]), str(row["epoch500_validation_rank"]),
            f'{row["epoch500_validation_loss"]:.3e}',
            f'{row["epoch500_validation_orth_error"]:.2e}',
        ]
        for row in rows
    ]
    table = table_axis.table(
        cellText=table_data,
        colLabels=["method", "omega0", "rank@0", "val rank@500", "val loss@500", "val orth@500"],
        loc="upper center", cellLoc="center", colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.25)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0.0)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_edgecolor("#555555")
            cell.set_linewidth(0.6)
    table_axis.text(
        0.5, 0.02,
        "Conclusion: the failure is missing raw-SIREN spatial bandwidth, not a need for positional encoding.",
        ha="center", va="bottom", fontsize=8.5, weight="bold",
    )
    fig.suptitle(
        "2D metaball · native CUDA Jet3 · matched plain sin vs raw spatial-SIREN",
        fontsize=12, weight="bold",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            args.output_dir / f"plain_sin_vs_siren_epoch500.{suffix}",
            dpi=260 if suffix == "png" else None,
            bbox_inches="tight", pad_inches=0.03,
        )
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

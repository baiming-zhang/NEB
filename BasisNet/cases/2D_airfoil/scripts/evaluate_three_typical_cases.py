from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


CASE_ROOT = Path(__file__).resolve().parents[1]


def read_gt_metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if archive["u"].shape != (41, 801, 1001):
            raise RuntimeError(f"unexpected GT shape in {path}: {archive['u'].shape}")
    solver = metadata["solver_config"]
    expected = {
        "nx": 1001,
        "ny": 801,
        "reynolds": 1500.0,
        "steps": 60000,
        "warmup_steps": 40000,
        "snapshot_every": 500,
        "collision_model": "trt",
        "require_separated_gt": True,
    }
    for key, value in expected.items():
        if solver.get(key) != value:
            raise RuntimeError(
                f"GT metadata mismatch for {key} in {path}: "
                f"{solver.get(key)!r} != {value!r}"
            )
    return metadata


def validate_gt(path: Path, beta: float, alpha_deg: float) -> dict:
    metadata = read_gt_metadata(path)
    if not math.isclose(float(metadata["beta"]), beta, abs_tol=1.0e-8):
        raise RuntimeError(f"beta mismatch in {path}")
    if not math.isclose(
        float(metadata["alpha_deg"]), alpha_deg, abs_tol=1.0e-8
    ):
        raise RuntimeError(f"alpha mismatch in {path}")
    return metadata


def evaluation_complete(output_dir: Path, beta: float, alpha_deg: float) -> bool:
    required = [
        output_dir / "evaluation_summary.json",
        output_dir / "reconstruction_sample.npz",
        output_dir / "velocity_optimal_comparison.png",
        output_dir / "basis_lowest6" / "contact_sheet.png",
        output_dir / "basis_lowest6" / "summary.json",
        *[
            output_dir / "basis_lowest6" / f"basis_{index:02d}.png"
            for index in range(1, 7)
        ],
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        summary = json.loads(required[0].read_text(encoding="utf-8"))
        with np.load(required[1], allow_pickle=False) as sample:
            if str(sample["coefficient_fit"]) != "velocity_l2":
                return False
            if int(sample["ritz_rank"]) != 1024:
                return False
            if "vorticity_opt_u" in sample.files:
                return False
            scalars = [
                float(sample[name])
                for name in sample.files
                if name.endswith("_error") and sample[name].ndim == 0
            ]
        return (
            summary.get("status") == "complete"
            and summary.get("coefficient_fit") == "velocity_l2"
            and math.isclose(float(summary["beta"]), beta, abs_tol=1.0e-6)
            and math.isclose(
                float(summary["alpha_deg"]), alpha_deg, abs_tol=1.0e-6
            )
            and all(math.isfinite(value) for value in scalars)
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def write_state(path: Path, **values: object) -> None:
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def run(command: list[str]) -> None:
    print("[command] " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def stack_images(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    try:
        width = max(image.width for image in images)
        resized = []
        for image in images:
            if image.width == width:
                resized.append(image.copy())
            else:
                height = round(image.height * width / image.width)
                resized.append(image.resize((width, height), Image.Resampling.LANCZOS))
        canvas = Image.new("RGB", (width, sum(image.height for image in resized)), "white")
        y = 0
        for image in resized:
            canvas.paste(image, (0, y))
            y += image.height
        canvas.save(output, quality=95)
    finally:
        for image in images:
            image.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate three typical alpha/beta airfoil cases with velocity-only fitting"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-gt", type=Path, required=True)
    parser.add_argument("--middle-gt", type=Path, required=True)
    parser.add_argument("--phase", type=float, default=0.5)
    parser.add_argument("--ritz-quadrature", type=int, default=4096)
    parser.add_argument("--field-batch", type=int, default=32768)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    gt_dir = output_root / "gt"
    gt_dir.mkdir(exist_ok=True)
    state_path = output_root / "pipeline_state.json"
    bottom_gt = gt_dir / "gt_alpha_m10_beta050.npz"
    cases = [
        {
            "label": "alpha_top_p10_beta050",
            "beta": 0.5,
            "alpha_deg": 10.0,
            "gt": args.top_gt.resolve(),
        },
        {
            "label": "alpha_middle_0_beta085",
            "beta": 0.85,
            "alpha_deg": 0.0,
            "gt": args.middle_gt.resolve(),
        },
        {
            "label": "alpha_bottom_m10_beta050",
            "beta": 0.5,
            "alpha_deg": -10.0,
            "gt": bottom_gt,
        },
    ]

    validate_gt(cases[0]["gt"], 0.5, 10.0)
    validate_gt(cases[1]["gt"], 0.85, 0.0)
    if bottom_gt.is_file():
        validate_gt(bottom_gt, 0.5, -10.0)
    else:
        write_state(
            state_path,
            stage="generating_bottom_gt",
            completed_cases=0,
            total_cases=3,
            beta=0.5,
            alpha_deg=-10.0,
        )
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts" / "solve_ns.py"),
                "--config",
                str(
                    CASE_ROOT
                    / "configs"
                    / "k1024_c2_sobol_10000_fp16_hybrid50.json"
                ),
                "--solver-config",
                str(CASE_ROOT / "configs" / "gt_re1500_separated.json"),
                "--beta=0.5",
                "--alpha-deg=-10",
                "--output",
                str(bottom_gt),
            ]
        )
        validate_gt(bottom_gt, 0.5, -10.0)

    summaries = []
    for index, case in enumerate(cases, start=1):
        output_dir = output_root / case["label"]
        write_state(
            state_path,
            stage="velocity_evaluation",
            completed_cases=index - 1,
            total_cases=3,
            current_case=case["label"],
            beta=case["beta"],
            alpha_deg=case["alpha_deg"],
        )
        if not evaluation_complete(
            output_dir, case["beta"], case["alpha_deg"]
        ):
            run(
                [
                    sys.executable,
                    "-u",
                    str(CASE_ROOT / "scripts" / "evaluate_single_gt.py"),
                    "--checkpoint",
                    str(args.checkpoint.resolve()),
                    "--gt",
                    str(case["gt"]),
                    "--output-dir",
                    str(output_dir),
                    "--phase",
                    str(args.phase),
                    "--ritz-quadrature",
                    str(args.ritz_quadrature),
                    "--field-batch",
                    str(args.field_batch),
                    "--velocity-only",
                    "--render-lowest-six",
                    "--basis-beta",
                    str(case["beta"]),
                    "--basis-alpha-deg",
                    str(case["alpha_deg"]),
                    "--basis-grid-nx",
                    "501",
                    "--basis-grid-ny",
                    "401",
                ]
            )
        if not evaluation_complete(
            output_dir, case["beta"], case["alpha_deg"]
        ):
            raise RuntimeError(f"incomplete evaluation: {case['label']}")
        summary = json.loads(
            (output_dir / "evaluation_summary.json").read_text(
                encoding="utf-8"
            )
        )
        summaries.append(summary)
        write_state(
            state_path,
            stage="velocity_evaluation",
            completed_cases=index,
            total_cases=3,
            last_case=case["label"],
            beta=case["beta"],
            alpha_deg=case["alpha_deg"],
        )

    comparison_sheet = output_root / "three_velocity_comparisons.png"
    basis_sheet = output_root / "three_basis_lowest6.png"
    stack_images(
        [Path(summary["velocity_optimal_figure"]) for summary in summaries],
        comparison_sheet,
    )
    stack_images(
        [Path(summary["basis_lowest6"]["contact_sheet"]) for summary in summaries],
        basis_sheet,
    )
    aggregate = {
        "status": "complete",
        "coefficient_fit": "velocity_l2_only",
        "checkpoint": str(args.checkpoint.resolve()),
        "phase": args.phase,
        "cases": summaries,
        "three_velocity_comparisons": str(comparison_sheet),
        "three_basis_lowest6": str(basis_sheet),
    }
    summary_path = output_root / "three_case_summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    write_state(
        state_path,
        stage="complete",
        completed_cases=3,
        total_cases=3,
        summary=str(summary_path),
    )
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()

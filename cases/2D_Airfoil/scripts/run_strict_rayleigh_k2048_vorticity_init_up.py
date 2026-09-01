#!/usr/bin/env python3
"""Run the physical upper-airfoil K2048 rollout from vorticity projection."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CASE_ROOT.parents[1]
STRICT_ROOT = (
    CASE_ROOT / "results"
    / "airfoil_strict_velocity_vorticity_k2048_e2000_seed20260823_stable_mgs_v2"
)
EVALUATION_ROOT = STRICT_ROOT / "strict_prefix_evaluation_three_geometries"
OUTPUT_ROOT = STRICT_ROOT / "strict_temporal_rollout_k2048_vorticity_init_up"
ROLLOUT = Path(__file__).with_name("rollout_gt30_fd_vs_potential_k2048.py")
LAPLACIAN_ROOT = (
    CASE_ROOT / "results" / "siren_original_k64_to_k1024_e2000_sequence"
    / "shared_geometry_laplacian_cache_k2048"
)
CHECKPOINTS = {
    "potential": (
        CASE_ROOT / "results" / "siren_original_jet1_gpu_k128_to_k2048_e2000_sequence"
        / "arms" / "k2048" / "run_001" / "checkpoints" / "hardcoded_sss_frozen.npz"
    ),
    "velocity": (
        STRICT_ROOT / "arms" / "seed_20260823" / "velocity" / "run_001"
        / "checkpoints" / "hardcoded_sss_frozen.npz"
    ),
    "vorticity": (
        STRICT_ROOT / "arms" / "seed_20260823" / "vorticity" / "run_001"
        / "checkpoints" / "hardcoded_sss_frozen.npz"
    ),
}


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = OUTPUT_ROOT / "runner_state.json"
    state: dict[str, object] = {
        "status": "running",
        "started_at": now(),
        "completed_tasks": [],
        "total_tasks": 3,
        "protocol": {
            "orientation": "up",
            "modes": 2048,
            "frames": 30,
            "initial_projection_field": "vorticity",
            "viscosity_scale": 1.0,
            "physical_viscosity_retained": True,
            "all_other_rollout_settings_identical_to_physical_baseline": True,
        },
    }
    write_json(state_path, state)

    for index, objective in enumerate(("potential", "velocity", "vorticity"), 1):
        methods = ["potential", "fd_mimetic"] if objective == "potential" else [objective]
        method_root = OUTPUT_ROOT / ("potential_fd" if objective == "potential" else objective)
        peeled = (
            EVALUATION_ROOT / "up" / "basis_state"
            / f"{objective}_k2048_prefix_state.npz"
        )
        state.update(
            {
                "task_index": index,
                "objective": objective,
                "methods": methods,
                "updated_at": now(),
            }
        )
        write_json(state_path, state)
        command = [
            sys.executable, "-u", str(ROLLOUT),
            "--evaluation-root", str(EVALUATION_ROOT),
            "--laplacian-cache-root", str(LAPLACIAN_ROOT),
            "--checkpoint", str(CHECKPOINTS[objective]),
            "--peeled-state", str(peeled),
            "--output-root", str(method_root),
            "--orientations", "up",
            "--methods", *methods,
            "--modes", "2048",
            "--frames", "30",
            "--steps-per-frame", "500",
            "--grid-stride", "2",
            "--substeps-per-frame", "10",
            "--base-flow-closure", "frozen-base-subtracted",
            "--initial-projection-field", "vorticity",
            "--viscosity-scale", "1",
        ]
        print(f"[{index}/3] physical-mu vorticity-init up {'/'.join(methods)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        completed = list(state["completed_tasks"])
        completed.append(
            {
                "objective": objective,
                "completed_at": now(),
                "summary": str(method_root / "up" / "comparison_summary.json"),
            }
        )
        state["completed_tasks"] = completed
        write_json(state_path, state)

    state.update({"status": "complete", "completed_at": now()})
    write_json(state_path, state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()

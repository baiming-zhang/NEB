#!/usr/bin/env python3
"""Run restartable strict K2048 temporal rollouts for all three Rayleigh spaces.

The cached geometry-aware FD-Laplacian is evaluated once per geometry alongside
Potential.  Velocity and Vorticity use their own frozen checkpoints and the
geometry-specific K2048 transforms produced by the strict prefix evaluation.
Every child invocation uses the same mimetic spatial complex, GT frames,
temporal mean, time step, viscosity, outlet closure, and implicit integrator.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CASE_ROOT.parents[1]
STRICT_ROOT = (
    CASE_ROOT
    / "results"
    / "airfoil_strict_velocity_vorticity_k2048_e2000_seed20260823_stable_mgs_v2"
)
EVALUATION_ROOT = STRICT_ROOT / "strict_prefix_evaluation_three_geometries"
OUTPUT_ROOT = STRICT_ROOT / "strict_temporal_rollout_k2048"
ROLLOUT = Path(__file__).with_name("rollout_gt30_fd_vs_potential_k2048.py")
LAPLACIAN_ROOT = (
    CASE_ROOT
    / "results"
    / "siren_original_k64_to_k1024_e2000_sequence"
    / "shared_geometry_laplacian_cache_k2048"
)
POTENTIAL = (
    CASE_ROOT
    / "results"
    / "siren_original_jet1_gpu_k128_to_k2048_e2000_sequence"
    / "arms"
    / "k2048"
    / "run_001"
    / "checkpoints"
    / "hardcoded_sss_frozen.npz"
)
VELOCITY = (
    STRICT_ROOT
    / "arms"
    / "seed_20260823"
    / "velocity"
    / "run_001"
    / "checkpoints"
    / "hardcoded_sss_frozen.npz"
)
VORTICITY = (
    STRICT_ROOT
    / "arms"
    / "seed_20260823"
    / "vorticity"
    / "run_001"
    / "checkpoints"
    / "hardcoded_sss_frozen.npz"
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, object]] = []
    checkpoints = {
        "potential": POTENTIAL,
        "velocity": VELOCITY,
        "vorticity": VORTICITY,
    }
    for orientation in ("up", "middle", "down"):
        for objective in ("potential", "velocity", "vorticity"):
            tasks.append(
                {
                    "orientation": orientation,
                    "objective": objective,
                    "methods": (
                        ["potential", "fd_mimetic"]
                        if objective == "potential"
                        else [objective]
                    ),
                    "checkpoint": checkpoints[objective],
                    "peeled_state": (
                        EVALUATION_ROOT
                        / orientation
                        / "basis_state"
                        / f"{objective}_k2048_prefix_state.npz"
                    ),
                    "output_root": OUTPUT_ROOT / (
                        "potential_fd" if objective == "potential" else objective
                    ),
                }
            )

    state_path = OUTPUT_ROOT / "runner_state.json"
    state: dict[str, object] = {
        "status": "running",
        "started_at": now(),
        "completed_tasks": [],
        "total_tasks": len(tasks),
        "protocol": {
            "modes": 2048,
            "frames": 30,
            "steps_per_frame": 500,
            "grid_stride": 2,
            "substeps_per_frame": 10,
            "base_flow_closure": "frozen-base-subtracted",
            "shared_mimetic_chain": True,
            "fd_basis_reused_without_eigensolve": True,
        },
    }
    write_json(state_path, state)

    for task_index, task in enumerate(tasks, start=1):
        orientation = str(task["orientation"])
        objective = str(task["objective"])
        state.update(
            {
                "task_index": task_index,
                "orientation": orientation,
                "objective": objective,
                "methods": task["methods"],
                "updated_at": now(),
            }
        )
        write_json(state_path, state)
        command = [
            sys.executable,
            "-u",
            str(ROLLOUT),
            "--evaluation-root",
            str(EVALUATION_ROOT),
            "--laplacian-cache-root",
            str(LAPLACIAN_ROOT),
            "--checkpoint",
            str(task["checkpoint"]),
            "--peeled-state",
            str(task["peeled_state"]),
            "--output-root",
            str(task["output_root"]),
            "--orientations",
            orientation,
            "--methods",
            *[str(value) for value in task["methods"]],
            "--modes",
            "2048",
            "--frames",
            "30",
            "--steps-per-frame",
            "500",
            "--grid-stride",
            "2",
            "--substeps-per-frame",
            "10",
            "--base-flow-closure",
            "frozen-base-subtracted",
        ]
        print(
            f"[{task_index}/{len(tasks)}] {orientation} {objective}: "
            f"{'/'.join(str(value) for value in task['methods'])}",
            flush=True,
        )
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        completed = list(state["completed_tasks"])
        completed.append(
            {
                "orientation": orientation,
                "objective": objective,
                "completed_at": now(),
                "summary": str(
                    Path(task["output_root"])
                    / orientation
                    / "comparison_summary.json"
                ),
            }
        )
        state["completed_tasks"] = completed
        write_json(state_path, state)

    state.update({"status": "complete", "completed_at": now()})
    write_json(state_path, state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()

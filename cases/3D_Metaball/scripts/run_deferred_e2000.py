"""Wait for other GEN training jobs, then run the original K64 setup for 2000 epochs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


CASE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CASE_ROOT / "configs" / "k64_three_sphere_connected_siren_dirichlet_2000.json"
RESULT_ROOT = CASE_ROOT / "results" / "k64_siren_dirichlet_e2000_original"
RUN_DIR = RESULT_ROOT / "run_001"
CHECKPOINT = RUN_DIR / "checkpoints" / "latest.pt"
COMPLETION = RUN_DIR / "completion.json"
STATUS = RESULT_ROOT / "launcher_status.json"
LOG = RESULT_ROOT / "launcher.log"
TRAIN_STDOUT = RESULT_ROOT / "training.stdout.log"
TRAIN_STDERR = RESULT_ROOT / "training.stderr.log"
POLL_SECONDS = 60
RETRY_SECONDS = 300
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
TRAINING_MARKERS = (
    "train_metaball.py",
    "train_validate.py",
    "run_hyperparameter_sweep.py",
    "evaluate_sweep_winner_128.py",
    "evaluate_unseen_vs_laplacian.py",
)


def configure_paths(config: Path, result_root: Path) -> None:
    global CONFIG, RESULT_ROOT, RUN_DIR, CHECKPOINT, COMPLETION, STATUS, LOG
    global TRAIN_STDOUT, TRAIN_STDERR
    CONFIG = config.resolve()
    RESULT_ROOT = result_root.resolve()
    RUN_DIR = RESULT_ROOT / "run_001"
    CHECKPOINT = RUN_DIR / "checkpoints" / "latest.pt"
    COMPLETION = RUN_DIR / "completion.json"
    STATUS = RESULT_ROOT / "launcher_status.json"
    LOG = RESULT_ROOT / "launcher.log"
    TRAIN_STDOUT = RESULT_ROOT / "training.stdout.log"
    TRAIN_STDERR = RESULT_ROOT / "training.stderr.log"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(100):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.02)


def append_log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp()}] {message}\n")


def completed() -> bool:
    if not COMPLETION.exists():
        return False
    try:
        return bool(json.loads(COMPLETION.read_text(encoding="utf-8")).get("complete"))
    except (OSError, json.JSONDecodeError):
        return False


def training_blockers() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    query = (
        "Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.ProcessId -ne {os.getpid()} -and $_.Name -match '^python(w)?\\.exe$' }} | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", query],
        check=False,
        capture_output=True,
        text=True,
        creationflags=CREATE_NO_WINDOW,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        append_log(f"process query parse failure: {result.stdout[:300]!r}")
        return []
    processes = payload if isinstance(payload, list) else [payload]
    blockers = []
    for process in processes:
        command = str(process.get("CommandLine") or "").lower()
        if any(marker in command for marker in TRAINING_MARKERS):
            blockers.append(process)
    return blockers


def wait_until_free() -> None:
    previous: tuple[int, ...] | None = None
    while True:
        blockers = training_blockers()
        blocker_ids = tuple(sorted(int(item["ProcessId"]) for item in blockers))
        if not blockers:
            atomic_json(
                STATUS,
                {"state": "resource_free", "checked_at": timestamp(), "blockers": []},
            )
            append_log("no other GEN training/evaluation process found; starting")
            return
        atomic_json(
            STATUS,
            {"state": "waiting_for_resource", "checked_at": timestamp(), "blockers": blockers},
        )
        if blocker_ids != previous:
            summary = ", ".join(
                f"pid={item['ProcessId']} {Path(str(item.get('CommandLine') or '')).name}"
                for item in blockers
            )
            append_log(f"waiting for: {summary}")
            previous = blocker_ids
        time.sleep(POLL_SECONDS)


def run_training() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(CASE_ROOT / "scripts" / "train_validate.py"),
        "--config",
        str(CONFIG),
        "--run-dir",
        str(RUN_DIR),
    ]
    if CHECKPOINT.exists():
        command.extend(("--resume", str(CHECKPOINT)))
        launch_kind = "resume"
    else:
        launch_kind = "cold_start"
    with TRAIN_STDOUT.open("a", encoding="utf-8") as stdout, TRAIN_STDERR.open(
        "a", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=CASE_ROOT.parents[1],
            stdout=stdout,
            stderr=stderr,
            creationflags=CREATE_NO_WINDOW,
        )
        atomic_json(
            STATUS,
            {
                "state": "training",
                "launch_kind": launch_kind,
                "started_at": timestamp(),
                "pid": process.pid,
                "run_dir": str(RUN_DIR.resolve()),
            },
        )
        append_log(f"{launch_kind} training pid={process.pid}")
        return process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    configure_paths(args.config, args.result_root)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    append_log(f"deferred launcher started pid={os.getpid()}")
    while not completed():
        wait_until_free()
        exit_code = run_training()
        if completed():
            atomic_json(
                STATUS,
                {"state": "complete", "completed_at": timestamp(), "exit_code": exit_code},
            )
            append_log("training complete")
            return
        atomic_json(
            STATUS,
            {
                "state": "retry_wait",
                "failed_at": timestamp(),
                "exit_code": exit_code,
                "resume_checkpoint_exists": CHECKPOINT.exists(),
                "retry_seconds": RETRY_SECONDS,
            },
        )
        append_log(
            f"training exited code={exit_code}; retrying from latest checkpoint in {RETRY_SECONDS}s"
        )
        time.sleep(RETRY_SECONDS)
    atomic_json(STATUS, {"state": "complete", "completed_at": timestamp()})


if __name__ == "__main__":
    main()

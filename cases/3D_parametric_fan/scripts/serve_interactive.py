"""Serve the trained four-blade fan as a pure-SVG interactive inference app."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import warnings
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cupy as cp
import numpy as np
from scipy.stats import qmc

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid_sss.basisnet.parametric_fan3d.hardcoded_backend import (  # noqa: E402
    HardcodedFanModel,
    cfg_namespace,
    fluid_points,
    peel_and_adjoint,
    raw_fields,
    raw_velocity,
    rotor_distance,
)

if "torch" in sys.modules:
    raise RuntimeError(
        "PyTorch was imported transitively; refusing framework-free inference"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            CASE_ROOT
            / "results"
            / "k1024_manual_100_run002"
            / "checkpoints"
            / "latest_inference.npz"
        ),
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=CASE_ROOT / "interactive" / "index.html",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def visual_points(cfg: Any, parameters: np.ndarray, count: int = 240) -> np.ndarray:
    """Deterministic non-lattice points; no mesh is exposed or rendered."""
    sampler = qmc.Sobol(3, scramble=True, seed=cfg.seed + 42_001)
    pieces: list[np.ndarray] = []
    remaining = count
    while remaining:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The balance properties of Sobol",
                category=UserWarning,
            )
            sample = sampler.random(max(3 * remaining, 256))
        x = cfg.xmin + (cfg.xmax - cfg.xmin) * sample[:, 0]
        radius = cfg.farfield_radius * np.sqrt(sample[:, 1])
        angle = 2.0 * math.pi * sample[:, 2]
        candidate = np.column_stack(
            (x, radius * np.cos(angle), radius * np.sin(angle))
        ).astype(np.float32)
        accepted = candidate[
            np.asarray(rotor_distance(cfg, candidate, parameters)) > 0.04
        ][:remaining]
        pieces.append(accepted)
        remaining -= len(accepted)
    return np.concatenate(pieces)


class InferenceEngine:
    def __init__(self, checkpoint: Path) -> None:
        metadata_path = checkpoint.with_suffix(".json")
        if not checkpoint.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"trained checkpoint is not ready: {checkpoint}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = metadata["config"]
        if config["blade_count"] != 4 or config["parameter_dim"] != 3:
            raise ValueError("checkpoint violates the four-blade/three-parameter contract")
        with np.load(checkpoint, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        # Inference checkpoints intentionally omit Adam; create zero placeholders
        # because the same framework-free model class owns both execution paths.
        template = {
            name: value
            for name, value in arrays.items()
        }
        layer_count = config["hidden_layers"] + 1
        parameter_shapes: list[tuple[int, ...]] = []
        for layer in range(config["hidden_layers"]):
            parameter_shapes.extend(
                (
                    template[f"l{layer}.weight"].shape,
                    template[f"l{layer}.bias"].shape,
                )
            )
        parameter_shapes.extend(
            (
                template["output.weight"].shape,
                template["output.bias"].shape,
            )
        )
        for index, shape in enumerate(parameter_shapes):
            template[f"adam.p{index:02d}.exp_avg"] = np.zeros(shape, np.float32)
            template[f"adam.p{index:02d}.exp_avg_sq"] = np.zeros(shape, np.float32)
        template["adam.steps"] = np.zeros(
            (config["modes"], 2 * layer_count), dtype=np.int64
        )
        self.cfg = cfg_namespace(config)
        self.model = HardcodedFanModel(self.cfg, template)
        self.lock = threading.Lock()
        self.cache_key: tuple[float, float, float] | None = None
        self.cache: dict[str, Any] = {}

    def normalize(self, beta: float, chord: float, thickness: float) -> np.ndarray:
        cfg = self.cfg
        values = np.asarray(
            (
                (beta - cfg.beta_tip_min_deg)
                / (cfg.beta_tip_max_deg - cfg.beta_tip_min_deg),
                (chord - cfg.chord_tip_min)
                / (cfg.chord_tip_max - cfg.chord_tip_min),
                (thickness - cfg.thickness_min)
                / (cfg.thickness_max - cfg.thickness_min),
            ),
            dtype=np.float32,
        )
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("one or more geometry parameters are outside training bounds")
        return values

    def query(
        self,
        beta: float,
        chord: float,
        thickness: float,
        rank: int,
    ) -> dict[str, Any]:
        if not 1 <= rank <= self.cfg.modes:
            raise ValueError(f"mode rank must be in [1,{self.cfg.modes}]")
        parameters = self.normalize(beta, chord, thickness)
        key = tuple(float(round(value, 7)) for value in parameters)
        with self.lock:
            cache_hit = key == self.cache_key
            if not cache_hit:
                quadrature = fluid_points(
                    self.cfg,
                    self.cfg.validation_points,
                    parameters,
                    self.cfg.seed + 9_000_000,
                )
                raw = raw_fields(
                    self.cfg, self.model, quadrature, parameters
                )
                peeled = peel_and_adjoint(
                    self.cfg,
                    raw,
                    len(quadrature),
                    need_adjoint=False,
                    need_coefficients=True,
                )
                points = visual_points(self.cfg, parameters)
                visual_raw = raw_velocity(
                    self.cfg, self.model, points, parameters
                )
                self.cache = {
                    "points": points,
                    "raw_velocity": visual_raw,
                    "coefficients": peeled["coefficient_matrix"],
                    "mode_order": peeled["mode_order"],
                    "mode_energies": peeled["mode_energies"],
                    "energy_order": np.argsort(
                        peeled["mode_energies"], kind="stable"
                    ),
                    "orth_error": peeled["orth_error"],
                    "effective_raw_rank": peeled["effective_raw_rank"],
                }
                self.cache_key = key
            rank_index = rank - 1
            basis_column = int(self.cache["energy_order"][rank_index])
            velocity = (
                self.cache["raw_velocity"]
                @ self.cache["coefficients"][:, basis_column]
            )
            velocity = cp.asnumpy(
                velocity.reshape(3, len(self.cache["points"])).T
            )
            speed = np.linalg.norm(velocity, axis=1)
            scale = float(np.quantile(speed, 0.92))
            if not math.isfinite(scale) or scale <= 1.0e-12:
                scale = 1.0
            normalized = velocity / scale
            return {
                "parameters": {
                    "beta_tip_deg": beta,
                    "chord_tip": chord,
                    "thickness_ratio": thickness,
                },
                "rank": rank,
                "causal_column": basis_column + 1,
                "raw_network": int(self.cache["mode_order"][basis_column]) + 1,
                "rayleigh": float(self.cache["mode_energies"][basis_column]),
                "orth_error": float(self.cache["orth_error"]),
                "effective_raw_rank": int(self.cache["effective_raw_rank"]),
                "points": np.round(self.cache["points"], 5).tolist(),
                "vectors": np.round(normalized, 5).tolist(),
                "vector_scale_p92": scale,
                "parameter_cache_hit": cache_hit,
                "theta_policy": "client exact SO(2); theta was not sent to the network",
            }


class Handler(SimpleHTTPRequestHandler):
    engine: InferenceEngine
    html_path: Path

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/mode":
            try:
                query = parse_qs(parsed.query)
                payload = self.engine.query(
                    beta=float(query["beta"][0]),
                    chord=float(query["chord"][0]),
                    thickness=float(query["thickness"][0]),
                    rank=int(query["rank"][0]),
                )
                body = json.dumps(payload, separators=(",", ":")).encode()
                self.send_response(200)
            except Exception as error:  # browser receives an actionable message
                body = json.dumps({"error": str(error)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path in {"/", "/index.html"}:
            body = self.html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[interactive] {fmt % args}", flush=True)


def main() -> None:
    args = parse_args()
    Handler.engine = InferenceEngine(args.checkpoint.resolve())
    Handler.html_path = args.html.resolve()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"fan inference ready at http://{args.host}:{args.port} "
        f"(checkpoint={args.checkpoint.resolve()})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

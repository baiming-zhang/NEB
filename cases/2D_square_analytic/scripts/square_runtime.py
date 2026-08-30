"""Square-domain hooks for the unmodified GitHub 2D-airfoil Jet3 backend.

The production backend is imported directly from ``cases/2D_airfoil/src``.
Only domain sampling, the hard boundary envelope, input encoding, and hidden
activation are replaced. The WIRE hook implements the official reference
``RealGaborLayer`` as two affine projections followed by
``cos(omega*u) * exp(-(sigma*v)**2)``. The Jet3 layout, linear kernels,
residual peeling, Rayleigh quotient, handwritten parameter reverse,
optimizer, and precision paths remain those of the production module.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cupy as cp
import numpy as np
from scipy.stats import qmc


CASE_ROOT = Path(__file__).resolve().parents[1]
AIRFOIL_ROOT = CASE_ROOT.parent / "2D_airfoil"
BACKEND_PATH = (
    AIRFOIL_ROOT
    / "src/eigenfluid_sss/basisnet/airfoil2d/hardcoded_backend.py"
)
sys.path.insert(0, str(AIRFOIL_ROOT / "src"))

from eigenfluid_sss.basisnet.airfoil2d import hardcoded_backend as backend  # noqa: E402


_PRODUCTION_SINE_FORWARD = backend.sine_forward
_PRODUCTION_SINE_BACKWARD = backend.sine_backward
_PRODUCTION_PEEL_STEP = backend.peel_step
_PRODUCTION_ADAM_STEP = backend.HardcodedMode.adam_step


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def square_envelope_jet(
    cfg: SimpleNamespace,
    x: backend.Jet3,
    y: backend.Jet3,
    beta: float,
    alpha_deg: float,
) -> backend.Jet3:
    """Single-zero Dirichlet streamfunction envelope on a rectangle.

    In normalized coordinates this is ``16*x*(1-x)*y*(1-y)``.  A
    single-zero streamfunction is the boundary condition associated with the
    analytic square Laplacian modes sin(m*pi*x) sin(n*pi*y).
    """
    del beta, alpha_deg
    xn = (x - cfg.xmin) / (cfg.xmax - cfg.xmin)
    yn = (y - cfg.ymin) / (cfg.ymax - cfg.ymin)
    return 16.0 * xn * (1.0 - xn) * yn * (1.0 - yn)


def _concatenate(features: list[backend.Jet3]) -> backend.Jet3:
    return backend.Jet3(
        cp.concatenate([item.value for item in features], axis=-1),
        cp.concatenate([item.grad for item in features], axis=-2),
        cp.concatenate([item.hess for item in features], axis=-2),
        cp.concatenate([item.third for item in features], axis=-2),
    )


def _feature_slice(source: backend.Jet3, start: int, stop: int) -> backend.Jet3:
    """Slice the feature axis for either scalar or mode-batched Jet3 arrays."""
    return backend.Jet3(
        source.value[..., start:stop],
        source.grad[..., start:stop, :],
        source.hess[..., start:stop, :],
        source.third[..., start:stop, :],
    )


def square_encoding_jet(
    cfg: SimpleNamespace,
    x: backend.Jet3,
    y: backend.Jet3,
    beta: float,
    alpha_deg: float,
) -> backend.Jet3:
    """Activation-appropriate square input encoding.

    SIREN and ReLU retain the historical ``4*frequencies+4`` storage width for
    the earlier matched-input study. WIRE uses the official RealGaborLayer's
    true two-coordinate input so its first ``nn.Linear``-equivalent fan-in and
    initialization are faithful to the reference implementation.
    """
    del beta, alpha_deg
    xu = (x - cfg.xmin) / (cfg.xmax - cfg.xmin)
    yu = (y - cfg.ymin) / (cfg.ymax - cfg.ymin)
    normalization = str(
        getattr(cfg, "coordinate_normalization", "centered_unit")
    )
    if normalization == "unit_interval":
        xn, yn = xu, yu
    elif normalization == "centered_unit":
        xn, yn = 2.0 * xu - 1.0, 2.0 * yu - 1.0
    elif normalization == "standardized_uniform":
        # Uniform[0,1] has mean 1/2 and standard deviation 1/sqrt(12).
        scale = float(np.sqrt(12.0))
        xn, yn = scale * (xu - 0.5), scale * (yu - 0.5)
    else:
        raise ValueError(
            f"unsupported square coordinate_normalization={normalization!r}"
        )
    width = 4 * int(cfg.frequencies) + 4
    encoding = str(getattr(cfg, "input_encoding", "positional_encoding"))
    shape = x.value.shape
    if encoding == "raw_xy":
        if str(getattr(cfg, "activation", "siren")) != "siren":
            raise ValueError("true fan-in=2 raw_xy is reserved for SIREN")
        if not bool(getattr(cfg, "canonical_siren", False)):
            omega0 = float(
                getattr(cfg, "siren_first_layer_spatial_omega0", 1.0)
            )
            xn = omega0 * xn
            yn = omega0 * yn
        return _concatenate([xn, yn])
    if encoding == "raw_xy_padded":
        if str(getattr(cfg, "activation", "siren")) == "wire":
            return _concatenate([xn, yn])
        # A raw-coordinate SIREN needs an explicit first-layer spatial scale.
        # With omega0=1 the initialized first layer sees less than one cycle
        # across the square and empirically caps the attainable Ritz spectrum.
        # Scaling only x/y preserves raw inputs (there are no Fourier features)
        # while exposing the requested spatial bandwidth to the first layer.
        if str(getattr(cfg, "activation", "siren")) == "siren":
            omega0 = float(
                getattr(cfg, "siren_first_layer_spatial_omega0", 1.0)
            )
            xn = omega0 * xn
            yn = omega0 * yn
        zero = backend.Jet3.constant(0.0, shape, x.value.dtype.type)
        return _concatenate([xn, yn] + [zero] * (width - 2))
    if encoding != "positional_encoding":
        raise ValueError(f"unsupported square input_encoding={encoding!r}")

    features: list[backend.Jet3] = []
    frequencies = 2.0 ** np.arange(cfg.frequencies)
    for spatial in (xn, yn):
        for frequency in frequencies:
            phase = spatial * float(frequency)
            sine = cp.sin(phase.value)
            cosine = cp.cos(phase.value)
            features.append(backend.unary(phase, sine, cosine, -sine, -cosine))
    for spatial in (xn, yn):
        for frequency in frequencies:
            phase = spatial * float(frequency)
            sine = cp.sin(phase.value)
            cosine = cp.cos(phase.value)
            features.append(backend.unary(phase, cosine, -sine, -cosine, sine))
    zero = backend.Jet3.constant(0.0, shape, x.value.dtype.type)
    features.extend([zero, zero, zero, zero])
    if len(features) != width:
        raise RuntimeError("square encoding width mismatch")
    return _concatenate(features)


def relu_forward(source: backend.Jet3) -> backend.Jet3:
    """Compact Jet3 ReLU, using the zero subgradient at the kink."""
    mask = (source.value > 0.0).astype(source.value.dtype, copy=False)
    zero = cp.zeros_like(source.value)
    return backend.unary(
        source, cp.maximum(source.value, 0.0), mask, zero, zero
    )


def relu_backward(source: backend.Jet3, adjoint: backend.Jet3) -> backend.Jet3:
    """Exact transpose of ``relu_forward`` away from its measure-zero kink."""
    mask = (source.value > 0.0).astype(source.value.dtype, copy=False)
    zero = cp.zeros_like(source.value)
    return backend.unary_backward(source, adjoint, mask, zero, zero, zero)


def _cosine_terms(value: cp.ndarray, omega: float) -> tuple[cp.ndarray, ...]:
    """Value through fourth derivative of ``cos(omega*x)``."""
    phase = omega * value
    cosine = cp.cos(phase)
    sine = cp.sin(phase)
    return (
        cosine,
        -omega * sine,
        -(omega**2) * cosine,
        (omega**3) * sine,
        (omega**4) * cosine,
    )


def _gaussian_terms(value: cp.ndarray, sigma: float) -> tuple[cp.ndarray, ...]:
    """Value through fourth derivative of ``exp(-(sigma*x)**2)``."""
    a = sigma * sigma
    x2 = value * value
    gaussian = cp.exp(-a * x2)
    return (
        gaussian,
        (-2.0 * a * value) * gaussian,
        (4.0 * a * a * x2 - 2.0 * a) * gaussian,
        (12.0 * a * a * value - 8.0 * a**3 * value**3) * gaussian,
        (12.0 * a * a - 48.0 * a**3 * x2 + 16.0 * a**4 * x2**2)
        * gaussian,
    )


def make_wire_activation(config: dict[str, Any]):
    """Build the official real-Gabor WIRE forward and exact Jet3 transpose.

    The incoming affine output has ``2*hidden`` channels. Its first half is
    the reference layer's frequency projection ``u`` and its second half is
    the independent Gaussian-scale projection ``v``. The returned feature
    width is therefore ``hidden``.
    """
    omega = float(config["wire_omega0"])
    sigma = float(config["wire_sigma0"])

    def split(source: backend.Jet3) -> tuple[backend.Jet3, backend.Jet3]:
        width = int(source.value.shape[-1])
        if width % 2:
            raise ValueError("WIRE affine output width must be even")
        half = width // 2
        return _feature_slice(source, 0, half), _feature_slice(source, half, width)

    def wire_forward(source: backend.Jet3) -> backend.Jet3:
        frequency_source, scale_source = split(source)
        frequency_terms = _cosine_terms(frequency_source.value, omega)
        scale_terms = _gaussian_terms(scale_source.value, sigma)
        frequency = backend.unary(frequency_source, *frequency_terms[:4])
        gaussian = backend.unary(scale_source, *scale_terms[:4])
        return frequency * gaussian

    def wire_backward(source: backend.Jet3, adjoint: backend.Jet3) -> backend.Jet3:
        frequency_source, scale_source = split(source)
        frequency_terms = _cosine_terms(frequency_source.value, omega)
        scale_terms = _gaussian_terms(scale_source.value, sigma)
        frequency = backend.unary(frequency_source, *frequency_terms[:4])
        gaussian = backend.unary(scale_source, *scale_terms[:4])
        frequency_adjoint = backend.envelope_product_backward(gaussian, adjoint)
        scale_adjoint = backend.envelope_product_backward(frequency, adjoint)
        return _concatenate(
            [
                backend.unary_backward(
                    frequency_source, frequency_adjoint, *frequency_terms[1:]
                ),
                backend.unary_backward(
                    scale_source, scale_adjoint, *scale_terms[1:]
                ),
            ]
        )

    return wire_forward, wire_backward


def square_fluid_points(
    cfg: SimpleNamespace,
    count: int,
    beta: float,
    alpha: float,
    seed: int,
) -> np.ndarray:
    """Uniform quadrature points over the square area.

    ``uniform_random`` is an IID PCG64 draw with no reflected or rotated
    companions.  The legacy Sobol/D4 paths remain readable for reproducing
    earlier runs, but the generic square configuration uses only the IID path.
    """
    del beta, alpha
    sampling_mode = str(getattr(cfg, "sampling_mode", "uniform_random"))
    symmetry = bool(getattr(cfg, "square_symmetry_quadrature", False))
    if sampling_mode == "uniform_random":
        if symmetry:
            raise ValueError("uniform_random forbids D4 symmetry augmentation")
        sample = np.random.default_rng(seed).random((count, 2))
    elif sampling_mode in {
        "uniform_area",
        "uniform_physical_area_sobol",
        "uniform_fluid_area_sobol",
    }:
        if symmetry and count % 8:
            raise ValueError(
                "D4-symmetric square quadrature count must be divisible by 8"
            )
        draw_count = count // 8 if symmetry else count
        sampler = qmc.Sobol(2, scramble=True, seed=seed)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The balance properties of Sobol",
                category=UserWarning,
            )
            sample = sampler.random(draw_count)
    else:
        raise ValueError(f"unsupported square sampling_mode={sampling_mode!r}")
    if symmetry:
        x, y = sample[:, 0], sample[:, 1]
        sample = np.concatenate(
            [
                np.column_stack((x, y)),
                np.column_stack((1.0 - x, y)),
                np.column_stack((x, 1.0 - y)),
                np.column_stack((1.0 - x, 1.0 - y)),
                np.column_stack((y, x)),
                np.column_stack((1.0 - y, x)),
                np.column_stack((y, 1.0 - x)),
                np.column_stack((1.0 - y, 1.0 - x)),
            ],
            axis=0,
        )
        sample = sample[np.random.default_rng(seed + 17).permutation(count)]
    points = np.column_stack(
        (
            cfg.xmin + sample[:, 0] * (cfg.xmax - cfg.xmin),
            cfg.ymin + sample[:, 1] * (cfg.ymax - cfg.ymin),
        )
    )
    return points.astype(np.float32, copy=False)


def square_continuous_parameters(
    cfg: SimpleNamespace, epoch: int
) -> tuple[float, float]:
    del cfg, epoch
    return 0.0, 0.0


def square_peel_step(
    cfg: SimpleNamespace,
    model: Any,
    points: np.ndarray,
    beta: float,
    alpha: float,
    training: bool,
    reference_model: Any | None = None,
    full_diagnostics: bool = True,
) -> dict[str, Any]:
    """Enter the production peel with its legacy airfoil guard satisfied.

    The guard predates case hooks and checks only a geometry label.  The
    production numerical function is called unchanged after temporarily
    presenting that label; square sampling, area, envelope, and encoding are
    already installed as process-local globals.
    """
    geometry_mode = cfg.geometry_mode
    try:
        cfg.geometry_mode = "cinf_ellipse"
        return _PRODUCTION_PEEL_STEP(
            cfg,
            model,
            points,
            beta,
            alpha,
            training,
            reference_model=reference_model,
            full_diagnostics=full_diagnostics,
        )
    finally:
        cfg.geometry_mode = geometry_mode


def all_fp32_adam_step(self: Any, gradients: list[cp.ndarray], cfg: SimpleNamespace) -> None:
    """Preserve the production master-to-forward synchronization in FP32.

    ``HardcodedModel`` builds batched forward stacks with ``cp.stack``, so the
    stacks never alias the per-mode master arrays. The GitHub airfoil path
    refreshes those stacks only inside its FP16 branch. The all-FP32 square
    path requires the identical refresh after every Adam update.
    """
    _PRODUCTION_ADAM_STEP(self, gradients, cfg)
    if self.compute_dtype is backend.F32:
        for shadow, master in zip(self.shadow_weights, self.weights):
            shadow[...] = master
        for shadow, master in zip(self.shadow_biases, self.biases):
            shadow[...] = master


def validate_square_config(
    config: dict[str, Any], production_validate: Any
) -> None:
    """Apply all production checks, substituting only the case labels."""
    if config.get("case_domain") != "unit_square_analytic_dirichlet":
        raise ValueError("case_domain must be unit_square_analytic_dirichlet")
    if config.get("geometry_mode") != "square_unit":
        raise ValueError("geometry_mode must be square_unit")
    if config.get("envelope_mode") != "cinf_square_dirichlet_single_zero":
        raise ValueError("unexpected square envelope_mode")
    if config.get("activation") not in {"relu", "siren", "wire"}:
        raise ValueError("activation must be relu, siren, or wire")
    if config.get("input_encoding") not in {
        "raw_xy", "raw_xy_padded", "positional_encoding"
    }:
        raise ValueError("unsupported square input encoding")
    if config.get("coordinate_normalization", "centered_unit") not in {
        "unit_interval", "centered_unit", "standardized_uniform"
    }:
        raise ValueError("unsupported square coordinate normalization")
    if config.get("basis_objective") not in {
        "pure_vorticity_rayleigh", "streamfunction_dirichlet_rayleigh"
    }:
        raise ValueError("unsupported square basis_objective")
    sampling_mode = str(config.get("sampling_mode", "uniform_random"))
    if sampling_mode not in {
        "uniform_random",
        "uniform_area",
        "uniform_physical_area_sobol",
        "uniform_fluid_area_sobol",
    }:
        raise ValueError("unsupported square sampling_mode")
    if sampling_mode == "uniform_random" and bool(
        config.get("square_symmetry_quadrature", False)
    ):
        raise ValueError("generic IID sampling forbids D4 symmetry augmentation")
    if (
        config.get("activation") == "siren"
        and float(config.get("siren_first_layer_spatial_omega0", 0.0)) <= 0.0
    ):
        raise ValueError("SIREN first-layer spatial omega0 must be positive")
    if bool(config.get("canonical_siren", False)):
        if config.get("activation") != "siren":
            raise ValueError("canonical_siren requires activation=siren")
        if config.get("input_encoding") != "raw_xy":
            raise ValueError("canonical square SIREN requires true fan-in=2 raw_xy")
        if float(config.get("siren_first_omega0", 0.0)) <= 0.0:
            raise ValueError("canonical SIREN first omega0 must be positive")
        if float(config.get("siren_hidden_omega0", 0.0)) <= 0.0:
            raise ValueError("canonical SIREN hidden omega0 must be positive")
    if config.get("activation") == "wire":
        if config.get("input_encoding") != "raw_xy_padded":
            raise ValueError("the WIRE square study requires raw xy inputs")
        if config.get("wire_variant") != "official_real_gabor":
            raise ValueError("wire_variant must be official_real_gabor")
        if float(config.get("wire_omega0", 0.0)) <= 0.0:
            raise ValueError("WIRE omega0 must be positive")
        if float(config.get("wire_sigma0", 0.0)) <= 0.0:
            raise ValueError("WIRE sigma0 must be positive")
    if [config.get(key) for key in ("xmin", "xmax", "ymin", "ymax")] != [
        0.0, 1.0, 0.0, 1.0
    ]:
        raise ValueError("analytic comparison requires the unit square")
    precision_keys = (
        "mlp_compute_dtype",
        "mlp_accumulation_dtype",
        "jet_dtype",
        "parameter_backward_dtype",
        "quadrature_dtype",
        "ritz_dtype",
        "master_weight_dtype",
        "optimizer_dtype",
    )
    wrong_precision = {
        key: config.get(key) for key in precision_keys
        if str(config.get(key, "")).lower() != "float32"
    }
    if wrong_precision:
        raise ValueError(f"the square ablation is all-FP32: {wrong_precision}")
    production_shadow = dict(config)
    production_shadow["geometry_mode"] = "cinf_ellipse"
    production_shadow["envelope_mode"] = "cinf_farfield_freestream"
    production_shadow["sampling_mode"] = "uniform_fluid_area_sobol"
    production_shadow["square_symmetry_quadrature"] = False
    # The GitHub validator hard-codes its historical FP16 airfoil protocol.
    # Runtime precision is independently and strictly checked above.
    production_shadow["mlp_compute_dtype"] = "float16"
    production_validate(production_shadow)


def install_backend(config: dict[str, Any]) -> ModuleType:
    """Install the case hooks into one process-local backend module."""
    backend.envelope_jet = square_envelope_jet
    backend.encoding_jet = square_encoding_jet
    backend.fluid_points = square_fluid_points
    backend.continuous_parameters = square_continuous_parameters
    backend.airfoil_area = lambda cfg, beta: 0.0
    backend.peel_step = square_peel_step
    backend.HardcodedMode.adam_step = all_fp32_adam_step
    if config.get("activation") == "relu":
        backend.sine_forward = relu_forward
        backend.sine_backward = relu_backward
    elif config.get("activation") == "siren":
        backend.sine_forward = _PRODUCTION_SINE_FORWARD
        backend.sine_backward = _PRODUCTION_SINE_BACKWARD
    elif config.get("activation") == "wire":
        backend.sine_forward, backend.sine_backward = make_wire_activation(config)
    else:
        raise ValueError(f"unsupported activation={config.get('activation')!r}")
    return backend


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_domain": config["case_domain"],
        "activation": config["activation"],
        "input_encoding": config["input_encoding"],
        "coordinate_normalization": config.get(
            "coordinate_normalization", "centered_unit"
        ),
        "canonical_siren": bool(config.get("canonical_siren", False)),
        "siren_first_omega0": config.get("siren_first_omega0"),
        "siren_hidden_omega0": config.get("siren_hidden_omega0"),
        "siren_first_layer_spatial_omega0": config.get(
            "siren_first_layer_spatial_omega0"
        ),
        "wire_variant": config.get("wire_variant"),
        "wire_omega0": config.get("wire_omega0"),
        "wire_sigma0": config.get("wire_sigma0"),
        "wire_reference": (
            "vishwa91/wire modules/wire.py RealGaborLayer at "
            "7e05f699488dff25fc335ee2985052fdacefc8bb"
            if config.get("activation") == "wire" else None
        ),
        "production_backend": str(BACKEND_PATH.resolve()),
        "production_backend_sha256": sha256(BACKEND_PATH),
        "case_hooks": [
            (
                "IID PCG64 uniform unit-square sampler"
                if config.get("sampling_mode") == "uniform_random"
                else "uniform unit-square Sobol sampler"
            ),
            (
                "exact D4-orbit quadrature"
                if bool(config.get("square_symmetry_quadrature", False))
                else "no symmetry augmentation"
            ),
            "single-zero square Dirichlet streamfunction envelope",
            "activation-appropriate square input encoding",
            "hidden activation selector",
            "official two-projection real-Gabor WIRE layer",
            "FP32 master-to-batched-forward synchronization",
        ],
        "unchanged_production_paths": [
            "compact symmetric native CUDA Jet3 layout",
            "configured all-FP32 MLP and Jet3 numerical path",
            (
                "dynamic Rayleigh-sorted residual peeling"
                if config.get("causal_sorting") == "geometry_rayleigh"
                else "fixed-order residual peeling"
            ),
            config["basis_objective"],
            "handwritten Jet3 parameter reverse",
            "Adam, scheduler, and checkpointing",
        ],
    }


def write_provenance(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance(config), indent=2), encoding="utf-8")

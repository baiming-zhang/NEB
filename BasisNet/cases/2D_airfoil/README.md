# 2D parametric airfoil (CUDA)

This is the canonical EigenFluid-SSS airfoil implementation. The production
trainer is an explicit CuPy fixed graph with a handwritten reverse pass; it
does not use PyTorch autograd or finite-difference spatial derivatives.

## Canonical configuration

[`k1024_ncl_cinf_sobol_10000_fp16_pure_vorticity_rayleigh.json`](configs/k1024_ncl_cinf_sobol_10000_fp16_pure_vorticity_rayleigh.json)
defines the current run:

- 1,024 learned modes in every optimizer step;
- 10,000 configured steps, with the published checkpoint evaluated at step 5,000;
- continuously varying C-infinity ellipse thickness and angle of attack;
- scrambled Sobol samples over the physical fluid area;
- antisymmetric NCL tensor representation with `u = div(A)`;
- analytic compact symmetric third-order spatial jets;
- one unscaled pure-vorticity Rayleigh objective;
- FP16 neural-network forward evaluation and FP32 for all other numerical
  paths; and
- deterministic initialization and fixed validation inputs.

Step-5000 figures, metrics, modes, and the network-only Jet3 validation are in
[`publication/step5000_cuda`](publication/step5000_cuda).

## Layout

```text
2D_airfoil/
|-- configs/           JSON run and reference-flow configurations
|-- reference_inputs/  compact deterministic protocol metadata
|-- scripts/           preparation, training, evaluation, and ROM rollout
|-- src/               installable airfoil implementation
`-- tests/             smoke, analytic-Jet3, and ROM tests
```

Large generated `.npz` inputs are reproducible from the checked-in metadata
and are deliberately not versioned.

## Install and verify

From the repository root, install a CuPy build matching the local CUDA
toolkit. For CUDA 12:

```bash
python -m pip install -e ".[dev,cuda12]"
python -m pytest -q BasisNet/cases/2D_airfoil/tests
python -m ruff check BasisNet/cases/2D_airfoil
```

## Prepare and train

```powershell
$case = "BasisNet/cases/2D_airfoil"
$config = "$case/configs/k1024_ncl_cinf_sobol_10000_fp16_pure_vorticity_rayleigh.json"
$run = "$case/results/k1024_ncl_cinf_sobol_10000_fp16_pure_vorticity_rayleigh/run_001"

python "$case/scripts/prepare_hardcoded_run.py" `
  --config $config `
  --output-dir "$run/reference_inputs"

python -u "$case/scripts/train_hardcoded.py" `
  --resume "$run/reference_inputs/initial_epoch0000.npz" `
  --validation-pack "$run/reference_inputs/fixed_validation_scipy_sobol_v1.npz" `
  --run-dir $run `
  --epochs 5000
```

For a one-epoch smoke run using the canonical wrapper:

```powershell
python "$case/scripts/train.py" --config $config --run-dir "$case/results/smoke" --self-test
```

Training output, checkpoints, generated reference fields, plots, and videos
remain local under ignored output directories.

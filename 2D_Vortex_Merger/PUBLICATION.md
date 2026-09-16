# Publication snapshot (2026-09-16)

This snapshot publishes the current local vortex-merger implementation, including
periodic gauge-canonical and no-gauge potential-Rayleigh training, analytic
spatial derivatives, GT generation, full-mass autonomous Galerkin evaluation,
Fourier-Laplacian controls, and comparison rendering.

The completed K64/K1024 report, per-mode and per-frame CSV metrics, compact
verification/provenance JSON, PNG/PDF plots, and both comparison MP4 videos are
included under `GAUGE_CANONICAL_EIGENFLUID/outputs`. These are previously computed
results, not new training performed for publication. Original provenance records
retain historical absolute paths and hashes.

## Environment and validation

Use Python 3.11+ with NumPy, SciPy, h5py, Matplotlib, Pillow, pytest, and
CuPy matching the installed CUDA runtime for neural training/evaluation. Video
rendering also requires FFmpeg. Run commands from the repository root.

```sh
python -m pytest 2D_Vortex_Merger/tests/test_generate_gt.py -q
python 2D_Vortex_Merger/scripts/generate_gt.py --separation 0.9 --output 2D_Vortex_Merger/outputs/gt_128_re350_l0p90_t6pi --validate-time
python 2D_Vortex_Merger/GAUGE_CANONICAL_EIGENFLUID/scripts/prepare_gauge_canonical.py --config 2D_Vortex_Merger/GAUGE_CANONICAL_EIGENFLUID/configs/k64_periodic_gauge_canonical_e2000.json --output-dir 2D_Vortex_Merger/GAUGE_CANONICAL_EIGENFLUID/outputs/neural_k64_e2000
```

Training and evaluation entry points expose their arguments with `--help`.
K1024 canonical and no-gauge configurations are also included. The no-gauge arm
uses `train_gauge_ablation.py`; paired comparisons require identical initial
weights and validation points.

## Data boundaries

Large NPZ checkpoints, HDF5 GT arrays, NPY caches, training histories and individual
video-frame sequences are not included. Rebuild initial packs with the preparation
script and run training/evaluation to regenerate numerical arrays. Existing GPU
tests additionally require those generated packs; the video pipeline is a local
resume orchestrator that expects completed checkpoints and evaluation inputs at
its documented paths, not a fresh-clone one-command reproduction script.
`verify_analytic.py` also records the SHA256 of the external manuscript at
`../ICLR2027_Potential_Eigenfluids/ICLR2027_Potential_Eigenfluids.tex` relative to the
repository root; that manuscript is not part of this case snapshot.

Publication checks: all Python sources parse, included JSON files parse, the three
CPU DNS tests pass, and both published video SHA256 hashes match the saved
completion record. GPU training/tests were not rerun during publication to preserve
the active campaign's single-GPU ownership. The included final verification JSON
records the earlier experiment checks.

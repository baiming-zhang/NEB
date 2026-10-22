# Paper experiment map

| Paper result | Entry points | Supplied models/data |
|---|---|---|
| Periodic merger, seed accuracy, decay | `experiments/periodic/run.py`, `plot.py`; `experiments/2d/nkb/run.py` | K1024 Potential/Velocity/NKB, all three seeds; `periodic_merger.json` |
| Obstacle random fields and energy capture | `experiments/multihole/run.py`, `evaluate.py`; `reference/run.py` | Three topologies, K32 models across three seeds, K1024 allocations; random field/energy data |
| Six K1024 initial fields and trajectories | `experiments/multihole/learned_rom/run.py`, `compare.py` | Positive and harmonic checkpoints; `retained_k1024.json` |
| Large-vortex circulation and lift | `experiments.multihole.reference.run --mode vortex-trajectory`; learned ROM `--field-ensemble compact-vortex` | Twenty samples per topology; raw/scaled harmonic models |
| Metaball autonomous capacity | `experiments/metaball/prepare.py`, `evaluate.py`, `operators.py`, `run.py rom` | Velocity/NKB through K1024; capacity and coefficient data |
| Metaball independent-field reconstruction | `prepare.py random20`; `run.py paired20` | 460 paired measurements; Potential through K512; Velocity/NKB through K1024 |
| Matched finite gauge | `train.py --method Potential-FG`; `finite_gauge.py` | K32/64/128/256 models and configs |
| Augmented-Lagrangian gauge | `train.py --method Potential-AL` | K32/64/128 models, configs, multiplier state |
| Gauge and Poisson solvers | `experiments/gauge/inputs.py`, `run.py`, `particular.py` | `gauge_solver.json`; finite/Fourier/FMM settings |
| Transfer to four geometries | `heldout_samples.py`, `heldout.py`, `heldout_geometry4.py` | Four geometry configs, disjoint sample generator, transfer models |
| Basis quality and mode atlases | Case analytic-field evaluators; `neb figures` | `basis_quality.json`; PNG/PDF mode plates |
| Preparation and online cost | Case stage timers; `neb figures` | Measured times, online/assembly data, starred cost estimates |

Paths are relative to the repository root. Measurement JSON files are in `data/paper/`; paired CSV and preparation costs are in `data/paper_results/`. Each model's method, case, capacity, configuration, and endpoint are listed in `checkpoints/index.json`.

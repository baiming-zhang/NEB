# Periodic vortex merger

The paper case has separation 1.35, a 256-by-256 operator grid, 128-by-128 observations,
241 times through 18*pi and viscosity 0.0010742913669168762. The learned models
are the original epoch-2000 periodic bases. Changing the initial vortex pair
uses the same trained basis.

From the repository root:

```sh
python experiments/periodic/run.py dns --output outputs/periodic/gt
python experiments/periodic/run.py prepare --config experiments/periodic/configs/periodic-potential-k1024-seed0.json --output outputs/periodic/init
python experiments/periodic/run.py train --checkpoint outputs/periodic/init/initial_epoch0000.npz --output outputs/periodic/train
python experiments/periodic/run.py evaluate --checkpoint checkpoints/2d/periodic-potential-k1024-seed0/model.npz --gt outputs/periodic/gt --method Potential --seed 0 --output outputs/periodic/evaluation
```

Velocity uses the corresponding `periodic-velocity-k1024-seed0.json` and
`--method Velocity`. Repeat the seed 1/2 configurations for the three-seed
comparison. Reference checkpoints are optional downloadable models; fresh
initialization and full training are supported. Native CuPy is required for
training/learned evaluation. DNS is CPU Fourier ETDRK4.

The evaluator preserves full mass/curl-stiffness assembly, full-rank Ritz
coordinates, the original rotational loads, synchronized timing and both
0.005/0.0025 trajectories. All K1024 coordinates remain active. Training and evaluation costs are recorded separately. Outputs retain measured run costs.

NKB is the independent neural-vector baseline in `experiments/2d/nkb`. The plotting utility reads completed
trajectories and creates PNG/vector PDF without altering numerical outputs.

```sh
python experiments/periodic/run.py laplacian --output outputs/periodic/leb
python experiments/periodic/plot.py --gt outputs/periodic/gt/dns.npz --trajectory Potential=outputs/periodic/evaluation/potential_seed0_n256/fields_dt0.005.npz --trajectory Laplacian=outputs/periodic/leb/laplacian1024.npz --output outputs/periodic/figures
```

# Reviewed publication artifacts

This directory contains compact, version-controlled artifacts behind the
latest manuscript results. It is deliberately separate from case-local
runtime `results/` directories, which remain ignored because they contain
checkpoints, sampled fields, logs, and caches.

| Bundle | Contents | Provenance |
| --- | --- | --- |
| [`airfoil_k2048_rollout30`](airfoil_k2048_rollout30/README.md) | rollout table, audit metadata, PNG/PDF figures | paper branch `codex/anti-defensive-writing-revision` plus the latest local rollout code |
| [`2d_metaball_k1024_k2048`](2d_metaball_k1024_k2048/README.md) | complete mean held-out errors for K=1024 and K=2048 | latest local `square-jet3-ablation` experiment state |
| [`3d_metaball_k1024`](3d_metaball_k1024/README.md) | five configs, metrics, six-panel PNG | artifact commit `ecff940`; figure verified at paper branch `b11d90a` |

Every bundle has a SHA-256 manifest. Paths inside old evaluation JSONs that
record the machine on which a run was generated are provenance strings, not
required runtime paths.

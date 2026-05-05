# Tuning the LM-Huber refinement with Optuna

`experiments/lm_huber_tuning.py` is a sweep harness over the LM-Huber
refinement hyperparameters that flow through `lm_kwargs` into
`refine_homography_torch_lm_torch`. Use it to find a config that beats the
RANSAC init across your dataset on the metrics from the LM-Huber tuning
spec (median / p90 / mean corner error, with hard penalties on regression
and NaN rates).

## What gets tuned

| Param                | Bounds                | Sampling | Effect                             |
|----------------------|-----------------------|----------|------------------------------------|
| `init_damping`       | `1e-5 … 1e-1`         | log      | initial Marquardt λ                |
| `damping_up`         | `1.5 … 5.0`           | linear   | λ multiplier on rejected step      |
| `damping_down`       | `0.1 … 0.9`           | linear   | λ multiplier on accepted step      |
| `barrier_k`          | `0.1 … 10.0`          | log      | sRT bounds barrier strength        |
| `f_scale`            | `0.5 … 10.0`          | linear   | Huber scale (residual saturation)  |
| `max_iter`           | `20 … 200`            | int      | LM iteration cap                   |
| `abs_err_tolerance`  | `1e-12 … 1e-6`        | log      | absolute convergence cutoff        |
| `rel_err_tolerance`  | `1e-12 … 1e-6`        | log      | relative convergence cutoff        |

Source of truth: `_SEARCH_BOUNDS` in `experiments/lm_huber_tuning.py`.
`srt_bounds_low/high` and `model` are NOT swept (kept at the LM defaults
+ `model="sRT"`); change them in code if you need them in the search.

## Objective

Per config:

```
score = 0.6 · ce_refined_median  +  0.3 · ce_refined_p90  +  0.1 · ce_refined_mean
score = +∞   if regression_rate > 0.05  OR  nan_rate > 1e-3
```

Lower is better. Hard penalties cap any "good median, awful tail" trial so
Optuna's TPE prunes them naturally.

## Quick start

Three modes, all writing into `experiments/lm_huber_runs/<run_name>/`:

```bash
# Coarse grid only — fast triage, deterministic from --seed
.venv/bin/python -m experiments.lm_huber_tuning \
    --tensors-dir ./tensors \
    --mode grid \
    --max-grid-configs 32 \
    --seed 1234 \
    --run-name lm_huber_grid

# Optuna only — TPE on the same bounds, exploits prior trials
.venv/bin/python -m experiments.lm_huber_tuning \
    --tensors-dir ./tensors \
    --mode optuna \
    --optuna-trials 60 \
    --seed 1234 \
    --run-name lm_huber_optuna

# Grid + Optuna in one run — recommended; grid covers the bounds, Optuna
# narrows in. Both write to the same long-format JSONL so you can compare.
.venv/bin/python -m experiments.lm_huber_tuning \
    --tensors-dir ./tensors \
    --mode both \
    --max-grid-configs 32 --optuna-trials 60 \
    --seed 1234 \
    --run-name lm_huber_v1
```

CLI flags:

* `--backend {auto,torch_cpu,torch_cuda}` — `auto` picks CUDA when available.
* `--no-track-history` — skip `LMHistory` collection (faster but loses
  `cost_drop_ratio`, `accept_rate_mean`, `convergence_rate`, `n_iters_run`).
* `--seed` — seeds the grid sampler, the Optuna TPE, and
  `set_deterministic_seeds` inside `evaluate_config` so reruns reproduce.

## Outputs

```
experiments/lm_huber_runs/<run_name>/
├── per_sample.jsonl        # one row per (config_id, sample_id); long format
├── configs_ranked.csv      # one row per config; rank=1 is best
└── summary_report.md       # human-readable best config + top-5 leaderboard
```

Long-format `per_sample.jsonl` schema:

```
config_id, sample_id, n_corr,
ce_init_px, ce_refined_px, dce_px,
H_init_det, H_refined_det, H_diff_fro,
cost_init, cost_final, n_iters, converged,
mean_damping_final, accept_rate
```

This is by design — keeps post-hoc stratification (location, n_corr bucket,
init-quality quartile) doable without re-running the sweep. Load it with
pandas:

```python
import pandas as pd
df = pd.read_json("experiments/lm_huber_runs/lm_huber_v1/per_sample.jsonl",
                  lines=True)
df["delta_px"] = df["ce_init_px"] - df["ce_refined_px"]
# Improvement per config:
df.groupby("config_id")["delta_px"].agg(["median", "mean",
                                          lambda s: (s > 0).mean(),
                                          lambda s: (s < -0.5).mean()])
```

## Reading the result

Always sanity-check the LM actually moved before celebrating a low median:

* `improve_rate` ≈ 1.0 + `dce_px` median > 0  → LM is helping.
* `improve_rate < 0.5`                          → LM is mostly noise / hurting.
* `regression_rate > 0`                         → at least some samples got worse;
  if it's > 0.05 the score is `+∞` and the config is rejected.
* `H_diff_fro ≈ 0` for all samples              → LM did nothing
  (init_damping too large, or abs_err_tolerance too loose).
* `cost_drop_ratio ≈ 0`                         → same diagnosis.
* `accept_rate < 0.3`                           → λ ladder mistuned;
  damping_up too aggressive or damping_down too small.
* `convergence_rate < 0.5` AND `n_iters_run = max_iter` → bump `max_iter` or
  loosen `rel_err_tolerance`.

`summary_report.md` shows the top-5 with all of these. The CSV has the full
ranked list, including `+∞` rows so you can see WHAT got rejected (filter
on `score == inf`).

## Programmatic use

The harness is library-friendly when you want a custom objective or a
training-loop-integrated tuner:

```python
from experiments.lm_huber_tuning import (
    DEFAULT_LM_KWARGS, evaluate_config, score_from_metrics,
    suggest_optuna_lm_kwargs, load_samples,
)
import optuna

samples = load_samples("./tensors")

def objective(trial):
    cfg = suggest_optuna_lm_kwargs(trial)        # within _SEARCH_BOUNDS
    out = evaluate_config(samples, cfg, track_history=True)
    return score_from_metrics(out["metrics"])     # +∞ on penalty trip

study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=1234))
study.optimize(objective, n_trials=60)
print("best:", study.best_params)
```

You can also pin some axes (e.g. fix `max_iter=100`) by overriding the
trial-suggested cfg before calling `evaluate_config`:

```python
def objective(trial):
    cfg = suggest_optuna_lm_kwargs(trial)
    cfg["max_iter"] = 100  # don't sweep this axis
    out = evaluate_config(samples, cfg)
    return score_from_metrics(out["metrics"])
```

## Plugging into sat_roma val

`compute_multimodel_corner_errors_for_batch` accepts the same `lm_kwargs`
+ `track_history` + `return_details`. So once you've found a good config,
plumb it into the val loop:

```python
from romatch.eval.corner_error import compute_multimodel_corner_errors_for_batch

best = {"init_damping": 5e-3, "damping_up": 2.5, "damping_down": 0.4,
        "barrier_k": 1.0, "f_scale": 2.0, "max_iter": 100,
        "abs_err_tolerance": 1e-10, "rel_err_tolerance": 1e-10}

res = compute_multimodel_corner_errors_for_batch(
    gm_cls_b=gm_cls, gt_warp_b=gt_warp,
    im_a_hw=(h_a, w_a), im_b_hw=(h_b, w_b),
    pipeline_backend="batched_cuda", refine=True,
    lm_kwargs=best,
    track_history=True, return_details=True,
)
# res.corner_errors_init / res.corner_errors → log Δ + improve_rate
# res.history.cost / damping / accept / converged → log LM dynamics
```

The legacy `list[float]` return is preserved when you don't pass
`return_details=True`, so existing val-loop call sites in `train_roma_sat.py`
keep working unchanged.

## Divergence Guard for Production Deployment

When running on edge devices (RPi, NVIDIA Jetson) or in drone applications,
you may encounter LM refinement that diverges (cost increases, extreme H jumps,
degenerate determinant). The divergence guard detects these failure modes and
falls back to RANSAC init to preserve safety margins.

### When to use

- **Production drone autonomy**: Always use `divergence_guard=DEFAULT_GUARD_DRONE`.
  Fail-closed: any ambiguous sign of trouble triggers fallback.
- **High-volume cloud inference**: Use `divergence_guard=DEFAULT_GUARD_RESEARCH`
  for better tail behavior; catch only obvious blow-ups.
- **Research / tuning**: Omit guard (None by default) to see raw LM behavior.

### Configuration

```python
from ransac_multimodel.divergence_guard import (
    DivergenceGuardConfig, DEFAULT_GUARD_DRONE, DEFAULT_GUARD_RESEARCH
)
from ransac_multimodel.pipeline import estimate_homography_batched

# Strict for production:
res = estimate_homography_batched(
    logits, backend="torch_cuda", refine=True,
    return_result=True,
    divergence_guard=DEFAULT_GUARD_DRONE,  # Recommended for drone
)

# Permissive for research:
res = estimate_homography_batched(
    logits, backend="torch_cpu", refine=True,
    return_result=True,
    divergence_guard=DEFAULT_GUARD_RESEARCH,  # Permissive sweep mode
)

# Custom thresholds:
from ransac_multimodel.divergence_guard import DivergenceGuardConfig
custom = DivergenceGuardConfig(
    max_cost_ratio=1.5,     # allow 1.5x cost increase
    max_h_diff_fro=10.0,    # frobenius norm jump threshold
    det_min=0.02,           # singularity check
    det_max=50.0,           # degenerate scaling check
)
res = estimate_homography_batched(
    logits, backend="torch_cuda", refine=True,
    return_result=True,
    divergence_guard=custom,
)
```

### Reading the guard output

When `return_result=True`, the result includes:
- `mask_diverged`: (B,) bool array, True where guard fell back to H_init.
- `guard_reasons`: list of B dicts, one per sample, showing which checks tripped.

```python
res = estimate_homography_batched(..., divergence_guard=DEFAULT_GUARD_DRONE)
print(f"Fallback rate: {res.mask_diverged.mean():.2%}")
for b, reasons in enumerate(res.guard_reasons):
    if res.mask_diverged[b]:
        print(f"  Sample {b} diverged: {reasons}")
```

### Cross-runtime stability

Guard thresholds transfer 1:1 from torch (GPU, cloud) to scipy (CPU, edge):

```python
# Cloud: torch on CUDA
res_cloud = estimate_homography_batched(
    logits, backend="torch_cuda", refine=True,
    divergence_guard=DEFAULT_GUARD_DRONE,
)

# Edge (when deployed): pure scipy on CPU (edge.py wrapper)
# Same config works identically; guard logic is pure numpy:
from ransac_multimodel.edge import refine_homography_scipy
H_edge, history_np = refine_homography_scipy(
    pts_A, means_B, covs_B, H_init,
    f_scale=2.0, max_iter=100, lm_kwargs={},
)
H_returned, mask_div, reasons = apply_divergence_guard(
    H_init, H_edge, history_np, DEFAULT_GUARD_DRONE
)
```

The divergence guard is pure numpy (zero torch dependency) and stateless, so it
produces identical decisions on GPU clouds and resource-constrained edges.

## Caveats

* The bundled `./tensors/` set is 3 frames (samples 098, 122, 128) — fine for
  smoke runs, useless for picking a production config. Point `--tensors-dir`
  at a real dump.
* `evaluate_config` calls `set_deterministic_seeds(seed)` but cv2 RANSAC's
  internal RNG is not fully covered by torch/numpy seeding. Expect ~1e-3 px
  noise across reruns even at the same `--seed`.
* The scipy backend (`pipeline_backend="scipy"` in the sat_roma wrapper)
  does NOT honor `lm_kwargs` — it uses scipy's TRF, not the torch LM. The
  harness here drives the torch LM exclusively.
* `srt_bounds_low / srt_bounds_high` aren't in the search space. If your
  data has scale/translation outside the LM defaults, hardcode wider
  bounds in `evaluate_config` or the LM defaults file before sweeping.

## Coordinate space

The sweep's `corner_error` is computed in **im_A pixels with both H_pred and
H_gt expressed as im_A_px → im_B_px** — exactly the convention sat_roma's
`compute_multimodel_corner_errors_for_batch` uses. So a sweep-best config
transfers 1:1 to the production `corner_error/{mode}/multimodel/*` metric
without any coordinate-space rescaling.

Mechanically:

* H_pred (feature-grid) → im_A_px → im_B_px via `convert_to_pixel_homography`.
* H_gt comes from the 4 corners of `gt_warp` (numpy/cv2 mirror of sat_roma's
  `_h_gt_a2b_from_gt_warp`).
* `compute_corner_error(H_gt_pix, H_pred_pix, w=w_a, h=h_a)` → mean L2 corner
  reprojection error in im_A pixels.

The dataloader-stored `homography_gt` field is **ignored** — it sits in
B → A space with a centered crop offset and would mismatch production.

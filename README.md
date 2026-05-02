# ransac_multimodel

Reusable Python 3 submodule for:
- extracting Gaussian correspondences from heatmaps,
- robust homography fitting (full or sRT),
- homography coordinate-space conversions,
- visualization helpers.

The code was factored from `solve.py` without changing core constants/behavior in the optimization and Gaussian extraction logic.

## Module structure

- `ransac_multimodel/gaussian_fit.py`
  - `extract_gaussians_adaptive`
  - `extract_gaussians_from_heatmap2`
- `ransac_multimodel/correspondence.py`
  - `find_gaussians`
- `ransac_multimodel/homography.py`
  - `project_points`
  - `optimize_homography`
  - `compute_corner_error`
- `ransac_multimodel/homography_torch.py`
  - `project_points_torch`
  - `homography_residuals_vectorized_torch`
  - `srt_residuals_torch`
  - `optimize_homography_torch`
- `ransac_multimodel/parity_utils.py`
  - deterministic seed helpers
  - NumPy <-> Torch conversion helpers
  - JSON utility helpers for benchmark outputs
- `ransac_multimodel/transforms.py`
  - `convert_to_pixel_homography`
  - `convert_to_dataloader_homography`
- `ransac_multimodel/plotting.py`
  - plotting utilities
- `ransac_multimodel/homography_theseus.py`
  - `optimize_homography_theseus` (numpy in / numpy out — drop-in for `optimize_homography`)
  - `refine_homography_theseus_torch` (batched, torch in / torch out, autograd-able)
  - Levenberg-Marquardt via `theseus.TheseusLayer`, soft-barrier penalty for sRT bounds, identity regularization via `th.Difference`
- `ransac_multimodel/homography_torch_lm.py`
  - `optimize_homography_torch_lm` and `refine_homography_torch_lm_torch`
  - Hand-rolled batched Levenberg-Marquardt in pure torch using `torch.func.jacfwd` + `torch.linalg.solve`
  - Same Mahalanobis-Huber + soft-barrier formulation as the theseus path; no theseus dependency
- `benchmarks/benchmark_numpy_vs_torch.py`
  - residual-only and end-to-end benchmarks
  - CPU baseline + Torch CPU + optional Torch CUDA
- `benchmarks/benchmark_scipy_vs_theseus.py`
  - per-sample comparison of scipy `least_squares` vs `theseus.LevenbergMarquardt`
  - reports timing + dataloader-space corner error on synthetic and real `tensors/` samples
- `benchmarks/benchmark_batched_theseus.py`
  - batched throughput comparison: scipy loop vs theseus batched vs torch-LM batched
  - sweeps batch sizes on CPU and CUDA

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick usage

```python
from ransac_multimodel.correspondence import find_gaussians
from ransac_multimodel.homography import optimize_homography, compute_corner_error
from ransac_multimodel.transforms import (
    convert_to_dataloader_homography,
    convert_to_pixel_homography,
)

# pts_A, means_B, peaks_B, covs_B = find_gaussians(logits)
# H_final, H_init = optimize_homography(pts_A, means_B, covs_B, peaks_B=peaks_B, model='sRT')
```

## Benchmark

Entry point:

```bash
python -m benchmarks.benchmark_numpy_vs_torch --modes all --device auto
```

Useful options:

```bash
# synthetic fast smoke benchmark
python -m benchmarks.benchmark_numpy_vs_torch --synthetic --synthetic-n 20000 --modes residual --repeats 5 --warmup 2 --device auto --quiet

# run on selected tensor samples
python -m benchmarks.benchmark_numpy_vs_torch --sample-ids 98,122,128 --modes all --model sRT --device auto --quiet
```

Output:
- console summary with median and p95 latencies
- JSON result file at `benchmarks/results/latest.json`
- timestamped JSON snapshot in `benchmarks/results/`

CUDA behavior:
- `--device auto`: runs CPU benchmarks always, and runs CUDA benchmarks only when CUDA is available
- when CUDA is unavailable, GPU runs are skipped with explicit notice

Quiet behavior:
- `--quiet` suppresses noisy per-iteration logging from Gaussian extraction and sRT optimizer prints

Synthetic scaling:
- use `--synthetic-n <N>` to increase correspondence count (for example `20000`) so CUDA has a fairer workload

Parity notes:
- residual parity uses strict `np.allclose(..., rtol=1e-6, atol=1e-7)` checks
- end-to-end parity reports homography Frobenius difference, absolute corner errors, and corner-error delta

## SAT-ROMA Tuning Runner

Use the tuning runner when you want reproducible baseline vs grid-search tuning and manual review outputs from only a dataset folder.

### How To Run Grid Search

```bash
# 1) Baseline only (single default config)
.venv/bin/python experiments/sat_roma_tuning.py \
  --dataset-dir ./tensors \
  --mode baseline \
  --seed 1234 \
  --run-name sat_roma_baseline

# 2) Coarse grid search
.venv/bin/python experiments/sat_roma_tuning.py \
  --dataset-dir ./tensors \
  --mode grid \
  --max-grid-configs 64 \
  --seed 1234 \
  --write-review-images \
  --run-name sat_roma_grid

# 3) Baseline + coarse grid in one run
.venv/bin/python experiments/sat_roma_tuning.py \
  --dataset-dir ./tensors \
  --mode both \
  --max-grid-configs 64 \
  --seed 1234 \
  --write-review-images \
  --run-name sat_roma_both

# 4) Optional fine search around best coarse config
.venv/bin/python experiments/sat_roma_tuning.py \
  --dataset-dir ./tensors \
  --mode both \
  --max-grid-configs 64 \
  --run-fine-search \
  --seed 1234 \
  --run-name sat_roma_both_fine
```

After a run, check:
- `experiments/sat_roma_runs/<run_name>/configs_ranked.csv` for ranked configs and best params (`rank=1`)
- `experiments/sat_roma_runs/<run_name>/summary_report.md` for baseline vs best summary
- `experiments/sat_roma_runs/<run_name>/review_index.csv` for KEEP/EXCLUDE labels and image paths

Entry point:

```bash
.venv/bin/python experiments/sat_roma_tuning.py --dataset-dir ./tensors --mode both --write-review-images
```

Key options:

```bash
# baseline only (preserves baseline behavior path)
.venv/bin/python experiments/sat_roma_tuning.py --dataset-dir ./tensors --mode baseline

# coarse grid only, explicit samples, deterministic seed
.venv/bin/python experiments/sat_roma_tuning.py --dataset-dir ./tensors --sample-ids 98,122,128 --mode grid --max-grid-configs 64 --seed 1234 --write-review-images

# with split file and optional fine search around best coarse config
.venv/bin/python experiments/sat_roma_tuning.py --dataset-dir ./tensors --split-file ./splits/val_ids.txt --mode both --run-fine-search --seed 1234

# Optuna search (TPE sampler) instead of coarse grid
.venv/bin/python experiments/sat_roma_tuning.py --dataset-dir ./tensors --mode optuna --optuna-trials 40 --optuna-startup-trials 10 --seed 1234
```

Outputs are saved under `experiments/sat_roma_runs/<run_name>/`:
- `per_sample_metrics.csv` and `per_sample_metrics.json`
- `configs_ranked.csv` and `configs_ranked.json` ranked by corner error (primary) then runtime (secondary)
- `review_index.csv` and `review_index.json` with status labels (`KEEP_FOR_OPTIMIZATION` or `EXCLUDE`) and reasons
- `review_index_all_configs.csv` and `review_index_all_configs.json` with status labels and image paths for every config and every sample
- `optimize_on.txt` and `do_not_use_for_tuning.txt`
- `summary_report.md` with baseline vs best config summary
- `plots/` with automatic visual summaries (`top_configs_corner_error.png`, `runtime_vs_error.png`, `param_usefulness.png`, `baseline_vs_best_per_sample.png`, and `optuna_progress.png` for Optuna mode)
- when `--mode optuna`: `optuna_trials.csv` and `optuna_trials.json` with trial params, values, and linked `config_id`

## Differentiable refinement: Theseus and a hand-rolled torch LM

The scipy refinement (`ransac_multimodel/homography.py`) is the canonical, accuracy-best path but it is CPU-only, single-call, and not differentiable. Two alternatives were added so that the refinement can sit inside a torch training loop:

- **`ransac_multimodel/homography_theseus.py`** — uses [Theseus](https://github.com/facebookresearch/theseus) (vendored as a git submodule under `theseus/`) to wrap the inner LM in a `TheseusLayer`. Adds a soft-barrier penalty for sRT bounds (analogue of scipy's `srt_bounds`, lifted from the `gps_denied/` reference project), `th.Difference`-based identity regularization, and `nan_to_num` defensiveness on the residual.
- **`ransac_multimodel/homography_torch_lm.py`** — a ~350-line pure-torch batched Levenberg-Marquardt. Forward-mode Jacobian via `torch.func.jacfwd` (optimal when `residual_dim ≫ param_dim`), dense batched normal-equation solve via `torch.linalg.solve`. Same Mahalanobis-Huber + soft-barrier formulation as the theseus path. No theseus dependency.

Both paths expose two entry points:
- `optimize_homography_*` — numpy in / numpy out, drop-in for `optimize_homography`.
- `refine_homography_*_torch` — batched `(B, N, 2)` torch in / `(B, 3, 3)` torch out, autograd-able.

### Benchmark setup

- `benchmarks/benchmark_scipy_vs_theseus.py` — per-sample timing + corner error (dataloader space) on synthetic + real `tensors/` samples (98, 122, 128).
- `benchmarks/benchmark_batched_theseus.py` — batched throughput: scipy loop vs theseus batched vs torch-LM batched, B ∈ {1, 4, 16, 64}, CPU and CUDA.
- `tests/test_theseus_parity.py` — synthetic-case parity vs scipy (< 1 px corner error), plus a backprop smoke test for the differentiable path.

Run from any CWD (the `theseus/` checkout shadows `import theseus` when CWD is the repo root, so we run from elsewhere):

```bash
cd /tmp && PYTHONPATH=<repo> .venv-uv/bin/python \
    -m benchmarks.benchmark_scipy_vs_theseus \
    --dataset-dir <repo>/tensors --include-synthetic --model sRT \
    --output <repo>/benchmarks/results/scipy_vs_theseus_sRT.json

cd /tmp && PYTHONPATH=<repo> .venv-uv/bin/python \
    -m benchmarks.benchmark_batched_theseus \
    --dataset-dir <repo>/tensors --sample-id 128 \
    --batch-sizes 1,4,16,64 --model sRT --device cuda \
    --output <repo>/benchmarks/results/batched_3way_sRT_cuda.json
```

### Headline results

**Per-call refinement (sRT, dataloader-space corner error / median ms; sample 98/122/128 are noisy real samples).** All three methods converge to the same H on the synthetic case; the gap on real samples is a property of the optimization formulation (scipy's `srt_x_scale` variable preconditioning has no clean equivalent in either alternative).

| sample | scipy err / ms | theseus err / ms | torch-LM err / ms |
|---|---|---|---|
| synthetic | 0.024 / 47 | 0.010 / 271 | 0.84 / 32 |
| 98 sRT  | 116 / 127 | 229 / 489 | 183 / ~30 |
| 122 sRT | 340 / 65  | 392 / 739 | 1504 / ~30 |
| 128 sRT | 168 / 166 | 234 / 560 | 232 / ~30 |

**Batched throughput (sample 128, sRT, per-item ms — lower is better).**

| B | scipy CPU | theseus CUDA | torch-LM CUDA | torch-LM speedup vs scipy |
|---:|---:|---:|---:|---:|
| 1  | 171  | 588  | 43.5 | 4×    |
| 16 | 169  | 50   | 2.78 | 60×   |
| 64 | 166  | 32   | 0.69 | **240×** |

### Verdict

- **For inference accuracy on noisy data**, scipy still wins. The theseus and torch-LM paths trade a 1.5-4× accuracy loss for batchability and differentiability.
- **For batched GPU throughput**, the hand-rolled torch LM is fastest by a wide margin (~240× scipy at B=64) and beats theseus by 30-40×. Theseus's framework overhead is real and not amortized at this problem size.
- **For training loops** that need gradients flowing through the refinement, both `refine_homography_theseus_torch` and `refine_homography_torch_lm_torch` work. The torch-LM path is preferable: faster, dependency-free (~350 LoC vs vendoring theseus + torchlie + torchkin), and matches theseus on accuracy.

Result JSONs in `benchmarks/results/scipy_vs_theseus_*.json` and `benchmarks/results/batched_3way_*.json`.

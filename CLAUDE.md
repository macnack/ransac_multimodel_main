# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This repo is a **wrapper project** around the `ransac_multimodel/` git submodule (separate repo: `git@github.com:macnack/ransac_multimodel.git`). The submodule holds the reusable library; this top-level repo holds the entrypoint script (`solve.py`), tuning experiments (`experiments/`), benchmarks (`benchmarks/`), tests, sample tensors, and config JSONs.

When changing library code in `ransac_multimodel/`, remember it lives in its own repo — commits there must be pushed separately, and the outer repo only tracks the submodule pointer (see `5bd8493 Update ransac_multimodel submodule pointer`).

## Environment

A virtualenv is checked into `.venv/`. Many commands in the README invoke `.venv/bin/python` directly rather than activating it. Dependencies are pinned loosely in `requirements.txt` (`numpy scipy opencv-python matplotlib torch optuna`).

## Common commands

Single-sample end-to-end run (loads `tensors/sample_<id>_tensor.pt` + `tensors/input_sample_<id>.pt`, fits Gaussians, optimizes homography, plots):

```bash
.venv/bin/python solve.py --cfg ./solve_config.json
# or the tuned config
.venv/bin/python solve.py --cfg ./solve_config_tuned.json
```

Sample data is keyed by integer ID; only `098`, `122`, `128` are present under `tensors/`. The sample to load is hardcoded (`sample_nr`, `sample_index`) in `solve.py`.

Tests (stdlib `unittest`):

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest tests.test_torch_parity -v          # single module
.venv/bin/python -m unittest tests.test_torch_parity.TestTorchParity.test_residual_parity   # single test
```

Benchmarks (NumPy CPU vs Torch CPU vs Torch CUDA when available):

```bash
.venv/bin/python -m benchmarks.benchmark_numpy_vs_torch --modes all --device auto
.venv/bin/python -m benchmarks.benchmark_numpy_vs_torch --synthetic --synthetic-n 20000 --modes residual --repeats 5 --warmup 2 --device auto --quiet
```

Results land in `benchmarks/results/latest.json` plus a timestamped snapshot.

SAT-ROMA tuning runner (baseline / coarse grid / Optuna over gaussian + optimizer params):

```bash
.venv/bin/python experiments/sat_roma_tuning.py --dataset-dir ./tensors --mode both --max-grid-configs 64 --seed 1234 --write-review-images --run-name sat_roma_both
.venv/bin/python experiments/sat_roma_tuning.py --dataset-dir ./tensors --mode optuna --optuna-trials 40 --optuna-startup-trials 10 --seed 1234
```

Outputs land under `experiments/sat_roma_runs/<run_name>/` — `configs_ranked.csv` (rank=1 is best), `summary_report.md`, `review_index*.csv`, and `plots/`.

## Architecture

### Pipeline

The whole stack does one thing: given per-patch heatmap logits from a multi-model correspondence head, fit a single homography from image A's patch grid to image B's patch grid. `solve.py` is the canonical worked example of the four stages:

1. **Gaussian extraction** (`ransac_multimodel/correspondence.py` → `find_gaussians`): for each `(py, px)` patch in image A, softmax the per-patch logits over image B's flattened patch grid (`out_patch_dim * out_patch_dim`), reshape to a 2D heatmap, and fit one or more Gaussians via either `extract_gaussians_adaptive` (default — adaptive window expansion until ±n_sigma fits inside a half-window) or `extract_gaussians_from_heatmap2` (fixed window). Returns `pts_A`, `means_B`, `peaks_B`, `covs_B` arrays of correspondences.

2. **Homography optimization** (`ransac_multimodel/homography.py` → `optimize_homography`): `cv2.findHomography` with RANSAC for an initial estimate (using `peaks_B` by default, or `means_B` when `use_means_for_ransac=True`), then `scipy.optimize.least_squares` (TRF with a robust loss like Huber) refines either:
   - `model="full"` — 8 free homography elements with `homography_residuals_vectorized` (Mahalanobis distance using inverted per-correspondence covariances), or
   - `model="sRT"` — similarity transform parameterized as `[s, theta, tx, ty]` via `srt_to_matrix` + `srt_residuals` (adds scale/rotation regularization).

3. **Coordinate-space conversion** (`ransac_multimodel/transforms.py`): the optimizer works in *feature grid space* (e.g. `in_patch_dim=14`, `out_patch_dim=64`). Two helpers map results out:
   - `convert_to_pixel_homography(H_feat, in_patch_dim, out_patch_dim, crop_res, map_res)` — image-A pixels → image-B pixels.
   - `convert_to_dataloader_homography(...)` — back into the dataloader's GT homography space, which includes a centered crop offset (`(W_B - W_A) / 2`) and *inverts* the result (the dataloader's GT is `B → A`).
   These two helpers do *different* things; pick the right one for the comparison/visualization at hand. `compute_corner_error(H_gt, H_pred, w, h)` only makes sense when both matrices are in the same space.

4. **Visualization** (`ransac_multimodel/plotting.py`): correspondence arrows, heatmap-with-fit overlays, and image warps.

### Torch parity track

`ransac_multimodel/homography_torch.py` mirrors the NumPy `project_points`, residual functions, and `optimize_homography` so the pipeline can run on CUDA. The contract is **numerical parity with the NumPy version** — `tests/test_torch_parity.py` enforces `np.allclose(rtol=1e-6, atol=1e-7)` on residuals and bounds end-to-end corner-error drift to < 1.0 px. If you change either implementation, the other must move in lockstep (or the tests fail).

### Configs

`solve_config.json` and `solve_config_tuned.json` share the schema `{ "gausian_config": {...}, "optimize_param": {...} }` (note the typo `gausian` — `parity_utils.gaussian_config` accepts both spellings). The tuned config came out of `experiments/sat_roma_tuning.py`; treat its values as data, not as code-defined defaults.

`ransac_multimodel/parity_utils.py` provides `gaussian_config()` and `optimize_params()` that normalize/validate config dicts and `set_deterministic_seeds()` for reproducible runs (used heavily by the tuning runner).

## Conventions worth knowing

- The library was deliberately **factored out of `solve.py` without changing core constants/behavior** in the optimizer or Gaussian extraction. Don't silently retune defaults in `homography.py` / `gaussian_fit.py` — those values are part of the parity contract with the original script.
- `find_gaussians` prints `"No Gaussians found in patch (px, py)"` per missing patch. To silence in batch jobs, pass `log_missing_gaussians=False` (the benchmark runner and tuning runner do this via the `--quiet` plumbing).
- `out_patch_dim = int(loggits.shape[0] ** 0.5)` — the input logits' first dimension must be a perfect square (image B's flattened patch grid). Both `find_gaussians` and `solve.py` assert this.
- Images in `solve.py` are 224×224 (A) and 896×896 (B); `transforms.py` defaults are 256/1024. Always pass `crop_res` / `map_res` explicitly when the sizes don't match the defaults.

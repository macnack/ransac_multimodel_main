#!/usr/bin/env python3
"""LM-Huber fine-tuning sweep harness.

Tunes the differentiable LM refinement (``refine_homography_torch_lm_torch``)
hyperparameters via the ``lm_kwargs`` pass-through on
:func:`ransac_multimodel.estimate_homography_batched`. Targets the metrics
laid out in the LM-Huber tuning spec:

* H_init vs H_refined corner_error (im_A px)
* Δ corner_error, improve_rate, regression_rate
* refined-corner_error median / p90 / p95 / mean / AUC
* LM dynamics from LMHistory: cost_init, cost_final, cost_drop_ratio,
  accept_rate, n_iters_run, convergence_rate, mean_damping_final
* sanity / pathology: nan_rate, ||H_LM - H_init||_F, det(H) distribution

Two search modes:

* ``--mode grid``   — random-Latin-hypercube coarse grid (reproducible via seed)
* ``--mode optuna`` — TPE on the same bounds, jointly with explicit
  ``regression_rate`` / ``nan_rate`` hard penalties on the objective
* ``--mode both``   — grid first, seed Optuna with the top configs

Output (under ``experiments/lm_huber_runs/<run_name>/``):

* ``per_sample.jsonl``   — long format, one row per (config_id, sample_id),
                           safe for post-hoc stratification (location, n_corr,
                           init-quality bucket).
* ``configs_ranked.csv`` — rank=1 is best by the composite objective.
* ``summary_report.md``  — human-readable with the best config's per-sample
                           breakdown.

The objective is

    score = 0.6·median + 0.3·p90 + 0.1·mean
    +∞ if regression_rate > 0.05 OR nan_rate > 0.001

(see :func:`score_from_metrics`).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import cv2  # noqa: E402

from ransac_multimodel.homography import compute_corner_error  # noqa: E402
from ransac_multimodel.parity_utils import set_deterministic_seeds  # noqa: E402
from ransac_multimodel.pipeline import estimate_homography_batched  # noqa: E402
from ransac_multimodel.transforms import convert_to_pixel_homography  # noqa: E402


# --------------------------------------------------------------------------- #
# Defaults                                                                    #
# --------------------------------------------------------------------------- #


# Mirrors refine_homography_torch_lm_torch's signature defaults so a sweep
# trivially recovers "do nothing different" (sanity baseline).
DEFAULT_LM_KWARGS: dict[str, Any] = {
    "init_damping": 1e-3,
    "damping_up":   2.0,
    "damping_down": 0.5,
    "barrier_k":    1.0,
    "f_scale":      2.0,
    "max_iter":     100,
    "abs_err_tolerance": 1e-10,
    "rel_err_tolerance": 1e-10,
    # srt_bounds_low / srt_bounds_high are tuples — we expose them as separate
    # scalars below in the search-space code, but the default is what the LM
    # already uses.
}


# Default geometry; matches solve.py + tests/test_pipeline.py for the bundled
# tensors. Tune on the CLI when running on a different feature-grid.
DEFAULT_GEOMETRY: dict[str, Any] = {
    "in_patch_dim":  14,
    "out_patch_dim": 64,
    "crop_res":      (224, 224),
    "map_res":       (896, 896),
}


# --------------------------------------------------------------------------- #
# Sample loading                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class SampleData:
    sample_id: int
    logits: torch.Tensor       # (M, in_h, in_w) coarse-classification logits
    gt_warp: torch.Tensor      # (h_a, w_a, 2) normalized [-1,1] in im_B frame
    # Cached H_gt in im_A_px → im_B_px space, derived from the 4 corners of
    # gt_warp. Mirrors sat_roma's `_h_gt_a2b_from_gt_warp`. None when the
    # corners are non-finite (degenerate sample).
    H_gt_pix: np.ndarray | None


def _h_gt_pix_from_gt_warp(
    gt_warp: torch.Tensor, h_b: int, w_b: int,
) -> np.ndarray | None:
    """im_A_px → im_B_px homography from the 4 corners of ``gt_warp``.

    Numpy/cv2 mirror of
    ``romatch.eval.corner_error._h_gt_a2b_from_gt_warp`` (which uses kornia
    + torch). The mapping is exact (4 correspondences → unique H), so a
    plain ``cv2.findHomography`` works — no RANSAC needed.
    """
    if gt_warp is None:
        return None
    arr = gt_warp.detach().cpu().numpy() if hasattr(gt_warp, "detach") else np.asarray(gt_warp)
    if arr.ndim != 3 or arr.shape[-1] != 2:
        return None
    h_a, w_a = arr.shape[:2]
    corners_norm = np.stack([
        arr[0, 0],
        arr[0, w_a - 1],
        arr[h_a - 1, w_a - 1],
        arr[h_a - 1, 0],
    ]).astype(np.float64)
    if not np.isfinite(corners_norm).all():
        return None
    corners_a = np.array([
        [0.0, 0.0],
        [float(w_a - 1), 0.0],
        [float(w_a - 1), float(h_a - 1)],
        [0.0, float(h_a - 1)],
    ], dtype=np.float64)
    corners_b = np.empty_like(corners_norm)
    corners_b[..., 0] = (corners_norm[..., 0] + 1.0) * (w_b / 2.0) - 0.5
    corners_b[..., 1] = (corners_norm[..., 1] + 1.0) * (h_b / 2.0) - 0.5
    if not np.isfinite(corners_b).all():
        return None
    H, _ = cv2.findHomography(
        corners_a.astype(np.float32), corners_b.astype(np.float32),
    )
    if H is None or not np.isfinite(H).all():
        return None
    return H.astype(np.float64)


def load_samples(
    tensors_dir: str | Path, geom: dict[str, Any] | None = None,
) -> list[SampleData]:
    """Load every ``sample_<id>_tensor.pt`` + ``input_sample_<id>.pt`` pair
    in ``tensors_dir`` and return them in ascending-id order. Skips any pair
    where either file is missing or malformed.

    H_gt is derived from the 4 corners of ``gt_warp`` so the sweep metric
    matches sat_roma's production ``compute_multimodel_corner_errors_for_batch``
    (im_A_px → im_B_px space). The dataloader's ``homography_gt`` field is
    ignored — it sits in dataloader space (B → A with a centered crop offset)
    and would mismatch the production training-loop metric.
    """
    geom = geom or DEFAULT_GEOMETRY
    h_b, w_b = geom["map_res"]
    tensors_dir = Path(tensors_dir)
    out: list[SampleData] = []
    for sample_pt in sorted(tensors_dir.glob("sample_*_tensor.pt")):
        stem = sample_pt.stem  # e.g. sample_098_tensor
        sid_str = stem.split("_")[1]
        try:
            sid = int(sid_str)
        except ValueError:
            continue
        input_pt = tensors_dir / f"input_sample_{sid:06d}.pt"
        if not input_pt.exists():
            continue
        try:
            sample = torch.load(sample_pt, map_location="cpu", weights_only=False)
            inp = torch.load(input_pt, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if 16 not in sample or "gm_cls" not in sample[16]:
            continue
        logits = sample[16]["gm_cls"][0]
        gt_warp = inp.get("gt_warp")
        if gt_warp is None:
            continue
        H_gt_pix = _h_gt_pix_from_gt_warp(gt_warp, int(h_b), int(w_b))
        out.append(SampleData(
            sample_id=sid, logits=logits, gt_warp=gt_warp, H_gt_pix=H_gt_pix,
        ))
    return out


# --------------------------------------------------------------------------- #
# Per-sample metrics                                                          #
# --------------------------------------------------------------------------- #


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def per_sample_metrics(
    ce_init: list[float],
    ce_refined: list[float],
    *,
    regression_eps: float = 0.5,
) -> dict[str, Any]:
    """Compute per-sample diagnostics over a batch.

    delta_px        : ce_init - ce_refined per sample (float, with NaNs propagated)
    improve_rate    : fraction of samples with Δ > 0
    regression_rate : fraction of samples with Δ < -regression_eps
    nan_rate        : fraction of samples with non-finite ce_refined
    """
    n = len(ce_refined)
    deltas: list[float] = []
    for ci, cr in zip(ce_init, ce_refined):
        if _is_finite(ci) and _is_finite(cr):
            deltas.append(float(ci) - float(cr))
        else:
            deltas.append(float("nan"))
    improved = sum(1 for d in deltas if _is_finite(d) and d > 0.0)
    regressed = sum(1 for d in deltas if _is_finite(d) and d < -regression_eps)
    nan_count = sum(1 for v in ce_refined if not _is_finite(v))
    return {
        "delta_px":        deltas,
        "improve_rate":    (improved / n) if n else float("nan"),
        "regression_rate": (regressed / n) if n else float("nan"),
        "nan_rate":        (nan_count / n) if n else float("nan"),
    }


def aggregate_corner_errors(values: Iterable[float]) -> dict[str, float]:
    """Median / p90 / p95 / mean over the FINITE subset. Returns NaN when no
    finite values are present (avoids numpy warnings on all-NaN input)."""
    arr = np.asarray([v for v in values if _is_finite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"median": float("nan"), "p90": float("nan"),
                "p95": float("nan"), "mean": float("nan"), "n_finite": 0}
    return {
        "median": float(np.median(arr)),
        "p90":    float(np.percentile(arr, 90)),
        "p95":    float(np.percentile(arr, 95)),
        "mean":   float(np.mean(arr)),
        "n_finite": int(arr.size),
    }


def auc_at_thresholds(values: Iterable[float], thresholds=(1, 3, 5, 10, 20)) -> dict[str, float]:
    """Cumulative-fraction-below-threshold curve. Non-finite values count as
    above any threshold (failed)."""
    arr = np.asarray(list(values), dtype=np.float64)
    n = arr.size
    out: dict[str, float] = {}
    for th in thresholds:
        if n == 0:
            out[f"auc_lt_{th}"] = float("nan")
        else:
            ok = np.isfinite(arr) & (arr < float(th))
            out[f"auc_lt_{th}"] = float(ok.sum() / n)
    return out


# --------------------------------------------------------------------------- #
# Objective                                                                   #
# --------------------------------------------------------------------------- #


_DEFAULT_OBJECTIVE_WEIGHTS = {"median": 0.6, "p90": 0.3, "mean": 0.1}
_DEFAULT_OBJECTIVE_PENALTIES = {
    "max_regression_rate": 0.05,
    "max_nan_rate":        1e-3,
}


def score_from_metrics(metrics: dict[str, float], *,
                       weights: dict[str, float] | None = None,
                       penalties: dict[str, float] | None = None) -> float:
    """Composite objective; lower is better. Returns ``+inf`` when a hard
    penalty trips so Optuna naturally prunes the trial."""
    w = weights or _DEFAULT_OBJECTIVE_WEIGHTS
    p = penalties or _DEFAULT_OBJECTIVE_PENALTIES
    if metrics.get("regression_rate", 0.0) > p["max_regression_rate"]:
        return float("inf")
    if metrics.get("nan_rate", 0.0) > p["max_nan_rate"]:
        return float("inf")
    median = metrics.get("ce_refined_median", float("nan"))
    p90 = metrics.get("ce_refined_p90", float("nan"))
    mean = metrics.get("ce_refined_mean", float("nan"))
    if not (_is_finite(median) and _is_finite(p90) and _is_finite(mean)):
        return float("inf")
    return w["median"] * median + w["p90"] * p90 + w["mean"] * mean


# --------------------------------------------------------------------------- #
# Search space                                                                #
# --------------------------------------------------------------------------- #


# Bounds used by both the coarse grid and Optuna. log-uniform for damping
# scales, linear for the rest.
_SEARCH_BOUNDS = {
    "init_damping":        (1e-5, 1e-1, "log"),
    "damping_up":          (1.5,  5.0,  "linear"),
    "damping_down":        (0.1,  0.9,  "linear"),
    "barrier_k":           (0.1,  10.0, "log"),
    "f_scale":             (0.5,  10.0, "linear"),
    "max_iter":            (20,   200,  "int"),
    "abs_err_tolerance":   (1e-12, 1e-6, "log"),
    "rel_err_tolerance":   (1e-12, 1e-6, "log"),
}


def coarse_grid(seed: int = 0, max_configs: int = 32) -> list[dict[str, Any]]:
    """Random Latin-hypercube-ish sample over ``_SEARCH_BOUNDS`` -- not a
    full LHS (would need scipy.stats.qmc), but a reproducible random sweep
    that covers each axis. Bounded to ``max_configs`` rows for fast
    triage runs."""
    rng = random.Random(seed)
    configs: list[dict[str, Any]] = []
    for _ in range(max_configs):
        cfg: dict[str, Any] = {}
        for name, (lo, hi, kind) in _SEARCH_BOUNDS.items():
            if kind == "log":
                cfg[name] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
            elif kind == "linear":
                cfg[name] = float(rng.uniform(lo, hi))
            elif kind == "int":
                cfg[name] = int(rng.randint(int(lo), int(hi)))
            else:  # pragma: no cover
                raise ValueError(f"Unknown sampling kind {kind!r}")
        configs.append(cfg)
    return configs


def suggest_optuna_lm_kwargs(trial, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Optuna ``trial`` -> lm_kwargs dict, sampled over ``_SEARCH_BOUNDS``."""
    out = dict(base or {})
    for name, (lo, hi, kind) in _SEARCH_BOUNDS.items():
        if kind == "log":
            out[name] = trial.suggest_float(name, lo, hi, log=True)
        elif kind == "linear":
            out[name] = trial.suggest_float(name, lo, hi)
        elif kind == "int":
            out[name] = trial.suggest_int(name, int(lo), int(hi))
        else:  # pragma: no cover
            raise ValueError(f"Unknown sampling kind {kind!r}")
    return out


# --------------------------------------------------------------------------- #
# Eval driver                                                                 #
# --------------------------------------------------------------------------- #


def _resolve_device(prefer_cuda: bool) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "torch_cuda"
    return "torch_cpu"


def _h_pred_to_pixel(H_feat: np.ndarray, geom: dict[str, Any]) -> np.ndarray:
    """Feature-grid H → im_A_px → im_B_px. Same transform sat_roma's
    production metric uses, so a sweep-best config transfers directly to
    `corner_error/{mode}/multimodel/*` in the val loop."""
    H = H_feat / H_feat[2, 2] if abs(H_feat[2, 2]) > 1e-12 else H_feat
    return convert_to_pixel_homography(
        H, in_patch_dim=geom["in_patch_dim"],
        out_patch_dim=geom["out_patch_dim"],
        crop_res=geom["crop_res"], map_res=geom["map_res"],
    )


def _corner_error_safe(
    H_feat: np.ndarray, H_gt_pix: np.ndarray | None, geom: dict[str, Any],
) -> float:
    """Per-sample corner_error in im_A pixels. Both H_pred and H_gt are
    expressed in im_A_px → im_B_px so this is comparable to sat_roma's
    `compute_multimodel_corner_errors_for_batch` output."""
    if H_gt_pix is None or H_feat is None or not np.isfinite(H_feat).all():
        return float("nan")
    try:
        H_pix = _h_pred_to_pixel(H_feat, geom)
    except Exception:
        return float("nan")
    if not (isinstance(H_pix, np.ndarray) and np.isfinite(H_pix).all()):
        return float("nan")
    h_a, w_a = geom["crop_res"]
    return float(compute_corner_error(H_gt_pix, H_pix, w=w_a, h=h_a))


def evaluate_config(
    samples: list[SampleData],
    lm_kwargs: dict[str, Any],
    *,
    backend: str | None = None,
    geom: dict[str, Any] | None = None,
    track_history: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Run one config across all samples, return the per-sample records and
    aggregated metrics + the optional LMHistory.

    Returns dict::

        {
          "per_sample": [
              {sample_id, n_corr, ce_init_px, ce_refined_px, dce_px,
               H_init_det, H_refined_det, H_diff_fro,
               cost_init, cost_final, n_iters, converged,
               mean_damping_final, accept_rate}, ...
          ],
          "metrics": {ce_refined_median, ce_refined_p90, ce_refined_mean,
                      improve_rate, regression_rate, nan_rate, time_ms,
                      cost_drop_ratio, accept_rate_mean,
                      auc_lt_1 / auc_lt_3 / auc_lt_5 / auc_lt_10 / auc_lt_20},
          "history": LMHistory | None,
        }
    """
    if not samples:
        return {"per_sample": [], "metrics": {}, "history": None}
    set_deterministic_seeds(int(seed))
    geom = geom or DEFAULT_GEOMETRY
    backend = backend or _resolve_device(prefer_cuda=True)

    stacked = torch.stack([s.logits for s in samples], dim=0)

    # estimate_homography_batched takes ``f_scale`` and ``max_iter`` as
    # EXPLICIT kwargs (they pre-date lm_kwargs). When the sweep config
    # carries them in ``lm_kwargs`` we have to split them out — otherwise
    # the splat collides with the explicit arg and Python raises
    # "multiple values for keyword argument".
    lm_kwargs_local = dict(lm_kwargs)  # copy — caller may reuse the dict
    f_scale = lm_kwargs_local.pop("f_scale", 2.0)
    max_iter = lm_kwargs_local.pop("max_iter", 100)
    model = lm_kwargs_local.pop("model", "sRT")

    t0 = time.perf_counter()
    res = estimate_homography_batched(
        stacked, backend=backend,
        model=model,
        refine=True,
        f_scale=float(f_scale),
        max_iter=int(max_iter),
        lm_kwargs=lm_kwargs_local,
        track_history=track_history,
        return_result=True,
        return_per_frame=True,
    )
    t_ms = (time.perf_counter() - t0) * 1e3

    per_sample: list[dict[str, Any]] = []
    ce_inits: list[float] = []
    ce_refineds: list[float] = []
    for b, s in enumerate(samples):
        H_init = res.H_init[b]
        H_ref = res.H[b]
        ce_init = _corner_error_safe(H_init, s.H_gt_pix, geom)
        ce_ref = _corner_error_safe(H_ref, s.H_gt_pix, geom)
        n_corr = int(res.per_frame[b][0].shape[0]) if res.per_frame is not None else -1
        try:
            det_init = float(np.linalg.det(H_init))
        except Exception:
            det_init = float("nan")
        try:
            det_ref = float(np.linalg.det(H_ref))
        except Exception:
            det_ref = float("nan")
        try:
            h_diff = float(np.linalg.norm(H_ref - H_init))
        except Exception:
            h_diff = float("nan")
        rec: dict[str, Any] = {
            "sample_id":     s.sample_id,
            "n_corr":        n_corr,
            "ce_init_px":    ce_init,
            "ce_refined_px": ce_ref,
            "dce_px":        (ce_init - ce_ref) if (_is_finite(ce_init) and _is_finite(ce_ref)) else float("nan"),
            "H_init_det":    det_init,
            "H_refined_det": det_ref,
            "H_diff_fro":    h_diff,
        }
        if track_history and res.history is not None:
            cost_hist = res.history.cost   # (n_iters, B)
            damping_hist = res.history.damping
            accept_hist = res.history.accept
            try:
                rec["cost_init"]  = float(cost_hist[0, b].item())
                rec["cost_final"] = float(res.history.final_cost[b].item())
                rec["n_iters"]    = int(res.history.n_iters)
                rec["converged"]  = bool(res.history.converged[b].item())
                rec["mean_damping_final"] = float(damping_hist[-1, b].item())
                rec["accept_rate"] = float(accept_hist[:, b].float().mean().item())
            except Exception:
                pass
        per_sample.append(rec)
        ce_inits.append(ce_init)
        ce_refineds.append(ce_ref)

    diag = per_sample_metrics(ce_inits, ce_refineds)
    agg_ref = aggregate_corner_errors(ce_refineds)
    agg_init = aggregate_corner_errors(ce_inits)
    auc = auc_at_thresholds(ce_refineds)

    metrics: dict[str, float] = {
        "ce_refined_median": agg_ref["median"],
        "ce_refined_p90":    agg_ref["p90"],
        "ce_refined_p95":    agg_ref["p95"],
        "ce_refined_mean":   agg_ref["mean"],
        "ce_init_median":    agg_init["median"],
        "ce_init_mean":      agg_init["mean"],
        "improve_rate":      diag["improve_rate"],
        "regression_rate":   diag["regression_rate"],
        "nan_rate":          diag["nan_rate"],
        "time_ms":           float(t_ms),
        **{k: float(v) for k, v in auc.items()},
    }

    if track_history and res.history is not None:
        try:
            cost_init = res.history.cost[0].float()      # (B,)
            cost_final = res.history.final_cost.float()   # (B,)
            denom = cost_init.clamp_min(1e-30)
            metrics["cost_drop_ratio"] = float(((cost_init - cost_final) / denom).mean().item())
            metrics["accept_rate_mean"] = float(res.history.accept.float().mean().item())
            metrics["convergence_rate"] = float(res.history.converged.float().mean().item())
            metrics["n_iters_run"] = int(res.history.n_iters)
        except Exception:
            pass

    return {"per_sample": per_sample, "metrics": metrics, "history": res.history}


# --------------------------------------------------------------------------- #
# Run modes                                                                   #
# --------------------------------------------------------------------------- #


def _config_id_for(idx: int, prefix: str) -> str:
    return f"{prefix}_{idx:04d}"


def run_grid(
    samples: list[SampleData],
    *,
    seed: int = 0,
    max_configs: int = 32,
    track_history: bool = True,
    backend: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate ``coarse_grid(seed, max_configs)`` configs over ``samples``.

    Returns (per_sample_long, configs_summary). ``per_sample_long`` has one
    row per (config_id, sample_id) — JSONL/parquet ready. ``configs_summary``
    has one row per config with the aggregated metrics + score.
    """
    grid = [DEFAULT_LM_KWARGS] + coarse_grid(seed=seed, max_configs=max_configs)
    long_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for i, cfg in enumerate(grid):
        config_id = _config_id_for(i, "grid")
        out = evaluate_config(samples, cfg, track_history=track_history, backend=backend, seed=seed)
        score = score_from_metrics(out["metrics"])
        for s in out["per_sample"]:
            long_rows.append({"config_id": config_id, **s})
        summaries.append({
            "config_id": config_id,
            "config":    cfg,
            "score":     score,
            **out["metrics"],
        })
    return long_rows, summaries


def run_optuna(
    samples: list[SampleData],
    *,
    n_trials: int = 30,
    seed: int = 0,
    track_history: bool = True,
    backend: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Optuna TPE over ``_SEARCH_BOUNDS``. Stores the same long + summary
    rows as ``run_grid`` so output formats stay uniform."""
    try:
        import optuna
    except ImportError:
        raise SystemExit(
            "optuna not installed. `pip install optuna` or run with --mode grid."
        )
    long_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    def _objective(trial):
        cfg = suggest_optuna_lm_kwargs(trial)
        config_id = _config_id_for(trial.number, "optuna")
        out = evaluate_config(samples, cfg, track_history=track_history,
                              backend=backend, seed=seed)
        score = score_from_metrics(out["metrics"])
        for s in out["per_sample"]:
            long_rows.append({"config_id": config_id, **s})
        summaries.append({
            "config_id": config_id, "config": cfg, "score": score,
            **out["metrics"],
        })
        return score

    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_objective, n_trials=int(n_trials), show_progress_bar=False)
    return long_rows, summaries


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #


def _to_jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return v


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(_to_jsonable(r)))
            f.write("\n")


def write_ranked_csv(path: Path, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(summaries, key=lambda r: (r["score"] if math.isfinite(r["score"]) else float("inf")))
    for rank, r in enumerate(ranked, 1):
        r["rank"] = rank
    if not ranked:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("rank,config_id,score\n")
        return ranked
    field_names = ["rank", "config_id", "score"] + [
        k for k in ranked[0].keys() if k not in ("rank", "config_id", "score", "config")
    ] + ["config"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_names)
        w.writeheader()
        for r in ranked:
            row = {k: ("" if k == "config" else _to_jsonable(r.get(k, ""))) for k in field_names}
            row["config"] = json.dumps(_to_jsonable(r.get("config", {})))
            w.writerow(row)
    return ranked


def write_summary_md(path: Path, ranked: list[dict[str, Any]], top: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# LM-Huber tuning sweep — {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append(f"Configs evaluated: **{len(ranked)}**")
    finite = [r for r in ranked if math.isfinite(r["score"])]
    lines.append(f"Configs surviving hard penalties: **{len(finite)}**")
    lines.append("")
    if not finite:
        lines.append("_No config survived the regression_rate / nan_rate gates._")
    else:
        best = finite[0]
        lines.append("## Best config")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(_to_jsonable(best.get("config", {})), indent=2))
        lines.append("```")
        lines.append("")
        lines.append(f"- score = **{best['score']:.4f}**")
        for k in ("ce_refined_median", "ce_refined_p90", "ce_refined_mean",
                  "ce_init_median", "improve_rate", "regression_rate",
                  "nan_rate", "cost_drop_ratio", "accept_rate_mean",
                  "convergence_rate", "n_iters_run", "time_ms"):
            v = best.get(k)
            if v is None:
                continue
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.4f}")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append(f"## Top {min(top, len(finite))} configs")
        lines.append("")
        lines.append("| rank | config_id | score | median | p90 | mean | improve | regress |")
        lines.append("|------|-----------|-------|--------|-----|------|---------|---------|")
        for r in finite[:top]:
            lines.append(
                f"| {r['rank']} | {r['config_id']} | {r['score']:.3f} | "
                f"{r['ce_refined_median']:.3f} | {r['ce_refined_p90']:.3f} | "
                f"{r['ce_refined_mean']:.3f} | {r.get('improve_rate', float('nan')):.3f} | "
                f"{r.get('regression_rate', float('nan')):.3f} |"
            )
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tensors-dir", type=Path, default=REPO_ROOT / "tensors",
                   help="Directory with sample_<id>_tensor.pt + input_sample_<id>.pt pairs.")
    p.add_argument("--mode", choices=("grid", "optuna", "both"), default="both")
    p.add_argument("--max-grid-configs", type=int, default=32)
    p.add_argument("--optuna-trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--run-name", type=str, default=None,
                   help="Subdir under experiments/lm_huber_runs/. Default = timestamp.")
    p.add_argument("--backend", choices=("auto", "torch_cpu", "torch_cuda"), default="auto")
    p.add_argument("--no-track-history", action="store_true",
                   help="Skip LMHistory collection (faster but loses cost/accept/iter metrics).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    samples = load_samples(args.tensors_dir)
    if not samples:
        print(f"[lm_huber_tuning] No samples in {args.tensors_dir}; aborting.", flush=True)
        return 2
    print(f"[lm_huber_tuning] Loaded {len(samples)} samples from {args.tensors_dir}.", flush=True)

    backend = None if args.backend == "auto" else args.backend
    track_history = not args.no_track_history

    run_name = args.run_name or datetime.now(timezone.utc).strftime("lm_huber_%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "experiments" / "lm_huber_runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    if args.mode in ("grid", "both"):
        print(f"[lm_huber_tuning] Running coarse grid (max_configs={args.max_grid_configs}).", flush=True)
        l, s = run_grid(
            samples, seed=args.seed, max_configs=args.max_grid_configs,
            track_history=track_history, backend=backend,
        )
        long_rows.extend(l); summaries.extend(s)

    if args.mode in ("optuna", "both"):
        print(f"[lm_huber_tuning] Running Optuna ({args.optuna_trials} trials).", flush=True)
        l, s = run_optuna(
            samples, n_trials=args.optuna_trials, seed=args.seed,
            track_history=track_history, backend=backend,
        )
        long_rows.extend(l); summaries.extend(s)

    write_jsonl(out_dir / "per_sample.jsonl", long_rows)
    ranked = write_ranked_csv(out_dir / "configs_ranked.csv", summaries)
    write_summary_md(out_dir / "summary_report.md", ranked)
    print(f"[lm_huber_tuning] Wrote results to {out_dir}.", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

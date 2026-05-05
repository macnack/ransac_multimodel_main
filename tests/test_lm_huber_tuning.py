"""Unit tests for the LM-Huber fine-tuning harness.

Pure-Python checks on the metric / aggregation / scoring primitives so the
sweep driver can be trusted before we wire it to a long-running search.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestPerSampleMetrics(unittest.TestCase):

    def test_delta_and_improve_rate(self):
        from experiments.lm_huber_tuning import per_sample_metrics

        # 4 samples: 2 improved, 1 worse, 1 unchanged.
        ce_init = [10.0, 5.0, 2.0, 4.0]
        ce_refined = [3.0, 2.0, 5.0, 4.0]   # Δ = 7, 3, -3, 0
        m = per_sample_metrics(ce_init, ce_refined, regression_eps=0.5)
        self.assertEqual(m["delta_px"], [7.0, 3.0, -3.0, 0.0])
        self.assertAlmostEqual(m["improve_rate"], 0.5)        # Δ>0: 2/4
        self.assertAlmostEqual(m["regression_rate"], 0.25)    # Δ<-0.5: 1/4

    def test_nan_rate_counts_only_finite(self):
        from experiments.lm_huber_tuning import per_sample_metrics

        ce_init = [1.0, 2.0, 3.0, 4.0]
        ce_refined = [1.0, float("nan"), float("inf"), 0.5]
        m = per_sample_metrics(ce_init, ce_refined, regression_eps=0.5)
        self.assertAlmostEqual(m["nan_rate"], 0.5)            # 2 of 4 non-finite

    def test_aggregates_median_p90_mean(self):
        from experiments.lm_huber_tuning import aggregate_corner_errors

        rng = np.random.default_rng(0)
        vals = list(rng.uniform(0, 10, size=200))
        a = aggregate_corner_errors(vals)
        self.assertAlmostEqual(a["median"], float(np.median(vals)), places=5)
        self.assertAlmostEqual(a["p90"], float(np.percentile(vals, 90)), places=5)
        self.assertAlmostEqual(a["p95"], float(np.percentile(vals, 95)), places=5)
        self.assertAlmostEqual(a["mean"], float(np.mean(vals)), places=5)

    def test_aggregates_drop_non_finite(self):
        from experiments.lm_huber_tuning import aggregate_corner_errors

        a = aggregate_corner_errors([1.0, float("nan"), 2.0, float("inf")])
        # Median over the finite subset {1, 2} = 1.5.
        self.assertAlmostEqual(a["median"], 1.5)


class TestObjective(unittest.TestCase):

    def test_score_weighted_combination(self):
        from experiments.lm_huber_tuning import score_from_metrics

        m = {
            "ce_refined_median": 5.0,
            "ce_refined_p90": 12.0,
            "ce_refined_mean": 6.0,
            "regression_rate": 0.0,
            "nan_rate": 0.0,
        }
        s = score_from_metrics(m)
        # 0.6*5 + 0.3*12 + 0.1*6 = 3 + 3.6 + 0.6 = 7.2
        self.assertAlmostEqual(s, 7.2, places=6)

    def test_score_penalty_on_regression(self):
        from experiments.lm_huber_tuning import score_from_metrics

        m = {
            "ce_refined_median": 1.0, "ce_refined_p90": 1.0, "ce_refined_mean": 1.0,
            "regression_rate": 0.10,  # > 0.05
            "nan_rate": 0.0,
        }
        self.assertTrue(math.isinf(score_from_metrics(m)))

    def test_score_penalty_on_nan(self):
        from experiments.lm_huber_tuning import score_from_metrics

        m = {
            "ce_refined_median": 1.0, "ce_refined_p90": 1.0, "ce_refined_mean": 1.0,
            "regression_rate": 0.0,
            "nan_rate": 0.01,  # > 1e-3
        }
        self.assertTrue(math.isinf(score_from_metrics(m)))


class TestSearchSpace(unittest.TestCase):

    def test_coarse_grid_has_valid_keys_and_size(self):
        from experiments.lm_huber_tuning import coarse_grid

        grid = coarse_grid(seed=0, max_configs=16)
        self.assertGreater(len(grid), 0)
        self.assertLessEqual(len(grid), 16)
        for cfg in grid:
            for k in ("init_damping", "damping_up", "damping_down",
                      "barrier_k", "f_scale", "max_iter"):
                self.assertIn(k, cfg, msg=f"coarse grid config missing {k!r}: {cfg}")
            self.assertGreater(cfg["init_damping"], 0)
            self.assertGreaterEqual(cfg["damping_up"], 1.0)
            self.assertLess(cfg["damping_down"], 1.0)
            self.assertGreater(cfg["max_iter"], 0)


class TestCoordinateSpaceParity(unittest.TestCase):
    """The sweep's corner_error must match sat_roma's production metric
    coordinate-by-coordinate. Otherwise a sweep-best config would not
    transfer to the val loop (the original reviewer concern)."""

    def test_h_gt_pix_matches_sat_roma_derivation(self):
        import torch
        from experiments.lm_huber_tuning import _h_gt_pix_from_gt_warp

        # Build a synthetic gt_warp whose 4 corners map im_A_px → known im_B_px
        # via a known H. Forward the corners through that H, normalize, then
        # check the recovered H matches the synthetic one.
        h_a, w_a, h_b, w_b = 224, 224, 896, 896
        H_known = np.array([
            [3.5, 0.05, 12.0],
            [0.02, 3.6, -7.0],
            [1e-5, 2e-5, 1.0],
        ], dtype=np.float64)

        # Build gt_warp: at each (y, x) of the im_A grid, set the corresponding
        # normalized im_B coord. We only need the 4 corner entries to be valid
        # (the helper ignores the rest).
        gt = torch.zeros((h_a, w_a, 2), dtype=torch.float32)
        corners_a = [(0, 0), (0, w_a - 1), (h_a - 1, w_a - 1), (h_a - 1, 0)]
        for (yy, xx) in corners_a:
            v = np.array([float(xx), float(yy), 1.0])
            v_b = H_known @ v
            xb = v_b[0] / v_b[2]
            yb = v_b[1] / v_b[2]
            xn = (xb + 0.5) / (w_b / 2.0) - 1.0
            yn = (yb + 0.5) / (h_b / 2.0) - 1.0
            gt[yy, xx, 0] = float(xn)
            gt[yy, xx, 1] = float(yn)

        H_recovered = _h_gt_pix_from_gt_warp(gt, h_b, w_b)
        self.assertIsNotNone(H_recovered)
        H_recovered = H_recovered / H_recovered[2, 2]
        H_known_n = H_known / H_known[2, 2]
        self.assertTrue(
            np.allclose(H_recovered, H_known_n, atol=1e-3),
            msg=f"recovered={H_recovered}\nknown={H_known_n}",
        )


class TestEvalConfigSmoke(unittest.TestCase):
    """Smoke: evaluate_config on the bundled samples returns finite metrics."""

    def test_evaluate_default_config_runs(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")

        from experiments.lm_huber_tuning import (
            DEFAULT_LM_KWARGS, evaluate_config, load_samples,
        )
        samples = load_samples(os.path.join(_REPO_ROOT, "tensors"))
        self.assertGreater(len(samples), 0, "no sample tensors found")

        rec = evaluate_config(samples, DEFAULT_LM_KWARGS, track_history=True)
        self.assertEqual(len(rec["per_sample"]), len(samples))
        for s in rec["per_sample"]:
            self.assertIn("ce_init_px", s)
            self.assertIn("ce_refined_px", s)
        self.assertTrue(np.isfinite(rec["metrics"]["ce_refined_median"]))


if __name__ == "__main__":
    unittest.main()

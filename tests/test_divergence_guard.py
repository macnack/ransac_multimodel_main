"""Unit tests for ransac_multimodel.divergence_guard.

Covers:
  - Cost increase detection
  - Extreme H difference detection
  - Degenerate determinant detection
  - No-op when LM helped
  - NaN/Inf safety
  - Torch → numpy history conversion
  - Per-sample independence in batched evaluation
"""

import unittest

import numpy as np
import torch

from ransac_multimodel.divergence_guard import (
    DivergenceGuardConfig,
    NumpyLMHistory,
    apply_divergence_guard,
    DEFAULT_GUARD_DRONE,
    DEFAULT_GUARD_RESEARCH,
)
from ransac_multimodel.homography_torch_lm import (
    lm_history_to_numpy,
    LMHistory,
)


class TestDivergenceGuardBasics(unittest.TestCase):
    """Basic divergence guard functionality."""

    def test_guard_fallback_when_cost_increased(self):
        """Fallback to H_init when cost got worse."""
        B = 2
        H_init = np.tile(np.eye(3), (B, 1, 1)).astype(np.float64)
        H_refined = H_init.copy()
        H_refined[0] += 0.01 * np.random.randn(3, 3)  # perturb first sample

        history = NumpyLMHistory(
            cost_init=np.array([1.0, 1.0]),
            cost_final=np.array([1.5, 1.5]),  # cost_final > cost_init for both
            n_iters=10,
            converged=np.array([True, True]),
        )

        config = DivergenceGuardConfig(max_cost_ratio=1.0)
        H_ret, mask_div, reasons = apply_divergence_guard(H_init, H_refined, history, config)

        # Both samples diverged (cost increased).
        self.assertTrue(mask_div[0])
        self.assertTrue(mask_div[1])

        # Returned H should be H_init for both.
        np.testing.assert_array_almost_equal(H_ret, H_init)

        # Reason should indicate cost increase.
        self.assertTrue(reasons[0]["cost_increased"])
        self.assertTrue(reasons[1]["cost_increased"])

    def test_guard_fallback_on_extreme_h_diff(self):
        """Fallback when H jump is too extreme."""
        B = 2
        H_init = np.tile(np.eye(3), (B, 1, 1)).astype(np.float64)
        H_refined = H_init.copy()

        # Sample 0: small jump (OK)
        H_refined[0] += 0.1 * np.eye(3)

        # Sample 1: huge jump (diverged)
        H_refined[1] += 10.0 * np.eye(3)

        history = NumpyLMHistory(
            cost_init=np.array([1.0, 1.0]),
            cost_final=np.array([0.5, 0.5]),  # cost improved
            n_iters=10,
            converged=np.array([True, True]),
        )

        config = DivergenceGuardConfig(max_cost_ratio=2.0, max_h_diff_fro=5.0)
        H_ret, mask_div, reasons = apply_divergence_guard(H_init, H_refined, history, config)

        # Only sample 1 diverged.
        self.assertFalse(mask_div[0])
        self.assertTrue(mask_div[1])

        # Sample 0 should get refined; sample 1 should fall back.
        np.testing.assert_array_almost_equal(H_ret[0], H_refined[0])
        np.testing.assert_array_almost_equal(H_ret[1], H_init[1])

        # Reason.
        self.assertFalse(reasons[0].get("h_diff_extreme", False))
        self.assertTrue(reasons[1]["h_diff_extreme"])

    def test_guard_fallback_on_degenerate_det(self):
        """Fallback when determinant is out of range."""
        B = 3
        H_init = np.tile(np.eye(3), (B, 1, 1)).astype(np.float64)
        H_refined = H_init.copy()

        # Sample 0: near-singular (det ≈ 0.01, below det_min=0.05)
        H_refined[0] = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1.0]])

        # Sample 1: reasonable det ≈ 1
        H_refined[1] = np.eye(3)

        # Sample 2: huge det (above det_max=20)
        H_refined[2] = np.eye(3) * 30

        history = NumpyLMHistory(
            cost_init=np.ones(B),
            cost_final=np.ones(B) * 0.5,  # improved
            n_iters=10,
            converged=np.ones(B, dtype=bool),
        )

        config = DivergenceGuardConfig(
            max_cost_ratio=2.0, max_h_diff_fro=None, det_min=0.05, det_max=20.0
        )
        H_ret, mask_div, reasons = apply_divergence_guard(H_init, H_refined, history, config)

        self.assertTrue(mask_div[0])  # det too small
        self.assertFalse(mask_div[1])  # OK
        self.assertTrue(mask_div[2])  # det too large

        np.testing.assert_array_almost_equal(H_ret[0], H_init[0])
        np.testing.assert_array_almost_equal(H_ret[1], H_refined[1])
        np.testing.assert_array_almost_equal(H_ret[2], H_init[2])

    def test_guard_no_op_when_lm_helped(self):
        """No fallback when LM improved cost and no red flags."""
        B = 2
        H_init = np.tile(np.eye(3), (B, 1, 1)).astype(np.float64)
        H_refined = H_init + 0.01 * np.random.randn(B, 3, 3)

        history = NumpyLMHistory(
            cost_init=np.array([1.0, 2.0]),
            cost_final=np.array([0.5, 1.0]),  # cost improved
            n_iters=10,
            converged=np.array([True, True]),
        )

        config = DivergenceGuardConfig(
            max_cost_ratio=1.0,
            max_h_diff_fro=1.0,
            det_min=0.05,
            det_max=20.0,
        )
        H_ret, mask_div, reasons = apply_divergence_guard(H_init, H_refined, history, config)

        # Neither sample diverged.
        self.assertFalse(mask_div[0])
        self.assertFalse(mask_div[1])

        # Returned H should match refined.
        np.testing.assert_array_almost_equal(H_ret, H_refined)

    def test_guard_finite_check_always_on(self):
        """NaN/Inf → fallback (finite_only=True is default)."""
        B = 3
        H_init = np.tile(np.eye(3), (B, 1, 1)).astype(np.float64)
        H_refined = H_init.copy()

        # Sample 0: NaN
        H_refined[0, 0, 0] = np.nan

        # Sample 1: Inf
        H_refined[1, 1, 1] = np.inf

        # Sample 2: OK
        H_refined[2] += 0.001

        history = None  # finite check doesn't need history

        config = DivergenceGuardConfig(finite_only=True)
        H_ret, mask_div, reasons = apply_divergence_guard(H_init, H_refined, history, config)

        self.assertTrue(mask_div[0])  # NaN
        self.assertTrue(mask_div[1])  # Inf
        self.assertFalse(mask_div[2])  # OK

        np.testing.assert_array_equal(H_ret[0], H_init[0])
        np.testing.assert_array_equal(H_ret[1], H_init[1])
        np.testing.assert_array_almost_equal(H_ret[2], H_refined[2])

    def test_guard_per_sample_independence(self):
        """Mixed pass/fail batch: each sample evaluated independently."""
        B = 4
        H_init = np.tile(np.eye(3), (B, 1, 1)).astype(np.float64)
        H_refined = H_init.copy()

        # Sample 0: cost increased → diverged
        H_refined[0] += 0.001

        # Sample 1: NaN → diverged
        H_refined[1, 0, 0] = np.nan

        # Sample 2: all good → not diverged
        H_refined[2] += 0.001

        # Sample 3: extreme H diff → diverged
        H_refined[3] += 100 * np.eye(3)

        history = NumpyLMHistory(
            cost_init=np.array([1.0, 1.0, 1.0, 1.0]),
            cost_final=np.array([2.0, 0.5, 0.5, 0.5]),  # sample 0 worse
            n_iters=10,
            converged=np.ones(B, dtype=bool),
        )

        config = DivergenceGuardConfig(
            max_cost_ratio=1.0,
            max_h_diff_fro=5.0,
            det_min=0.05,
            det_max=20.0,
            finite_only=True,
        )
        H_ret, mask_div, reasons = apply_divergence_guard(H_init, H_refined, history, config)

        # Check mask
        expected_diverged = np.array([True, True, False, True])
        np.testing.assert_array_equal(mask_div, expected_diverged)

        # Check returned H
        for b in range(B):
            if expected_diverged[b]:
                np.testing.assert_array_equal(H_ret[b], H_init[b])
            else:
                np.testing.assert_array_almost_equal(H_ret[b], H_refined[b])


class TestDivergenceGuardHistoryConversion(unittest.TestCase):
    """Test conversion from torch LMHistory to numpy NumpyLMHistory."""

    def test_torch_to_numpy_conversion(self):
        """lm_history_to_numpy preserves cost and convergence info."""
        B = 3
        n_iters = 8

        # Simulate torch LMHistory
        history = LMHistory(
            cost=torch.randn(n_iters, B),  # (n_iters, B)
            damping=torch.randn(n_iters, B),
            accept=torch.randint(0, 2, (n_iters, B), dtype=torch.bool),
            n_iters=n_iters,
            converged=torch.tensor([True, False, True], dtype=torch.bool),
            final_cost=torch.tensor([0.5, 1.2, 0.8]),
        )

        # Convert
        np_hist = lm_history_to_numpy(history)

        # Check types and shapes
        self.assertIsInstance(np_hist, NumpyLMHistory)
        self.assertEqual(np_hist.cost_init.shape, (B,))
        self.assertEqual(np_hist.cost_final.shape, (B,))
        self.assertEqual(np_hist.converged.shape, (B,))
        self.assertEqual(np_hist.n_iters, n_iters)

        # Check values
        np.testing.assert_allclose(
            np_hist.cost_init,
            history.cost[0].numpy(),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np_hist.cost_final,
            history.final_cost.numpy(),
            rtol=1e-5,
        )
        np.testing.assert_array_equal(
            np_hist.converged,
            history.converged.numpy(),
        )

    def test_history_to_numpy_method(self):
        """LMHistory.to_numpy() method works."""
        history = LMHistory(
            cost=torch.ones(5, 2),
            damping=torch.ones(5, 2),
            accept=torch.ones(5, 2, dtype=torch.bool),
            n_iters=5,
            converged=torch.ones(2, dtype=torch.bool),
            final_cost=torch.tensor([0.5, 0.3]),
        )

        np_hist = history.to_numpy()

        self.assertIsInstance(np_hist, NumpyLMHistory)
        np.testing.assert_allclose(np_hist.cost_init, [1.0, 1.0])
        np.testing.assert_allclose(np_hist.cost_final, [0.5, 0.3])


class TestDivergenceGuardPresets(unittest.TestCase):
    """Test preset configurations."""

    def test_drone_vs_research_strictness(self):
        """DRONE is stricter than RESEARCH."""
        # DRONE should catch more divergence cases than RESEARCH.
        self.assertLess(
            DEFAULT_GUARD_DRONE.max_cost_ratio,
            DEFAULT_GUARD_RESEARCH.max_cost_ratio,
        )
        self.assertLess(
            DEFAULT_GUARD_DRONE.max_h_diff_fro,
            DEFAULT_GUARD_RESEARCH.max_h_diff_fro,
        )
        self.assertGreater(
            DEFAULT_GUARD_DRONE.det_min,
            DEFAULT_GUARD_RESEARCH.det_min,
        )
        self.assertLess(
            DEFAULT_GUARD_DRONE.det_max,
            DEFAULT_GUARD_RESEARCH.det_max,
        )

    def test_drone_config_on_borderline_cost(self):
        """DRONE rejects marginal cost increases; RESEARCH accepts."""
        B = 1
        H_init = np.eye(3, dtype=np.float64)[np.newaxis, :, :]
        H_refined = H_init.copy()

        # 1.5x cost increase (bad, but not catastrophic).
        history = NumpyLMHistory(
            cost_init=np.array([1.0]),
            cost_final=np.array([1.5]),
            n_iters=10,
            converged=np.array([True]),
        )

        _, mask_drone, _ = apply_divergence_guard(H_init, H_refined, history, DEFAULT_GUARD_DRONE)
        _, mask_research, _ = apply_divergence_guard(
            H_init, H_refined, history, DEFAULT_GUARD_RESEARCH
        )

        # DRONE rejects (max_cost_ratio=1.0).
        self.assertTrue(mask_drone[0])

        # RESEARCH accepts (max_cost_ratio=2.0).
        self.assertFalse(mask_research[0])


class TestDivergenceGuardEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_zero_cost_init_skips_ratio_check(self):
        """Zero initial cost → ratio check skipped."""
        B = 1
        H_init = np.eye(3, dtype=np.float64)[np.newaxis, :, :]
        H_refined = H_init.copy()

        history = NumpyLMHistory(
            cost_init=np.array([0.0]),  # zero
            cost_final=np.array([10.0]),  # large
            n_iters=10,
            converged=np.array([True]),
        )

        config = DivergenceGuardConfig(max_cost_ratio=1.0)
        _, mask_div, reasons = apply_divergence_guard(H_init, H_refined, history, config)

        # Should not diverge on ratio check (division by zero avoided).
        self.assertFalse(mask_div[0])
        self.assertFalse(reasons[0]["cost_increased"])

    def test_no_history_skips_cost_and_convergence(self):
        """history=None → cost and convergence checks skipped."""
        B = 1
        H_init = np.eye(3, dtype=np.float64)[np.newaxis, :, :]
        H_refined = H_init + 0.001  # small perturbation

        config = DivergenceGuardConfig(
            max_cost_ratio=1.0,
            require_converged=True,
            max_h_diff_fro=None,  # disable H diff to focus on history deps
            det_min=None,
            det_max=None,
        )

        _, mask_div, _ = apply_divergence_guard(H_init, H_refined, None, config)

        # Should not diverge (no history, so cost/convergence checks skipped).
        self.assertFalse(mask_div[0])

    def test_all_checks_disabled_via_none(self):
        """Setting all thresholds to None → only finite check active."""
        B = 1
        H_init = np.eye(3, dtype=np.float64)[np.newaxis, :, :]
        H_refined = H_init + 1000 * np.eye(3)  # huge jump, cost would increase, etc.

        history = NumpyLMHistory(
            cost_init=np.array([1.0]),
            cost_final=np.array([999.0]),
            n_iters=100,
            converged=np.array([False]),
        )

        config = DivergenceGuardConfig(
            max_cost_ratio=None,
            max_h_diff_fro=None,
            det_min=None,
            det_max=None,
            require_converged=False,
            finite_only=False,  # also disable finite
        )

        _, mask_div, _ = apply_divergence_guard(H_init, H_refined, history, config)

        # No thresholds active → no divergence detected.
        self.assertFalse(mask_div[0])

    def test_batch_size_one(self):
        """Guard works on single sample (B=1)."""
        H_init = np.eye(3, dtype=np.float64)[np.newaxis, :, :]
        H_refined = H_init + 0.001

        history = NumpyLMHistory(
            cost_init=np.array([1.0]),
            cost_final=np.array([0.5]),
            n_iters=10,
            converged=np.array([True]),
        )

        config = DivergenceGuardConfig()
        H_ret, mask_div, reasons = apply_divergence_guard(H_init, H_refined, history, config)

        self.assertEqual(H_ret.shape, (1, 3, 3))
        self.assertEqual(mask_div.shape, (1,))
        self.assertEqual(len(reasons), 1)

    def test_large_batch(self):
        """Guard scales to large batches."""
        B = 128
        H_init = np.tile(np.eye(3), (B, 1, 1)).astype(np.float64)
        H_refined = H_init + 0.001 * np.random.randn(B, 3, 3)

        history = NumpyLMHistory(
            cost_init=np.random.rand(B),
            cost_final=0.5 * np.random.rand(B),
            n_iters=10,
            converged=np.ones(B, dtype=bool),
        )

        config = DivergenceGuardConfig()
        H_ret, mask_div, reasons = apply_divergence_guard(H_init, H_refined, history, config)

        self.assertEqual(H_ret.shape, (B, 3, 3))
        self.assertEqual(mask_div.shape, (B,))
        self.assertEqual(len(reasons), B)


class TestDivergenceGuardReasonAudit(unittest.TestCase):
    """Test per-sample audit trail (reasons dict)."""

    def test_reasons_has_triggered_checks(self):
        """Reason dict includes keys for checks that were enabled and ran."""
        B = 1
        H_init = np.eye(3, dtype=np.float64)[np.newaxis, :, :]
        H_refined = H_init + 0.001  # no NaN

        history = NumpyLMHistory(
            cost_init=np.array([1.0]),
            cost_final=np.array([2.0]),
            n_iters=10,
            converged=np.array([True]),
        )

        config = DivergenceGuardConfig(
            max_cost_ratio=1.0,
            finite_only=True,
        )

        _, _, reasons = apply_divergence_guard(H_init, H_refined, history, config)

        # Both finite_only and cost_increased should be in reasons (no divergence before cost check).
        self.assertIn("finite_only", reasons[0])
        self.assertIn("cost_increased", reasons[0])
        # finite_only check should not trip (no NaN).
        self.assertFalse(reasons[0]["finite_only"])
        # cost_increased should trip.
        self.assertTrue(reasons[0]["cost_increased"])

    def test_reasons_skip_disabled_checks(self):
        """Reason dict omits keys for disabled checks."""
        B = 1
        H_init = np.eye(3, dtype=np.float64)[np.newaxis, :, :]
        H_refined = H_init.copy()

        history = None  # no history

        config = DivergenceGuardConfig(
            max_cost_ratio=None,  # disabled
            max_h_diff_fro=None,  # disabled
            det_min=None,  # disabled
            det_max=None,  # disabled
            require_converged=False,  # disabled
            finite_only=True,  # enabled
        )

        _, _, reasons = apply_divergence_guard(H_init, H_refined, history, config)

        # Only "finite_only" should be present.
        self.assertIn("finite_only", reasons[0])
        self.assertFalse(reasons[0]["finite_only"])


if __name__ == "__main__":
    unittest.main()

"""Test batch-size invariance of _batched_lm solver.

Critical test: identical inputs should produce identical results regardless
of batch size. This catches bugs in per-element independence.
"""

import unittest

import numpy as np
import torch

from ransac_multimodel.homography_torch_lm import (
    _batched_lm,
    _err_fn_factory,
)


class TestBatchSizeInvariance(unittest.TestCase):
    """Verify LM results are invariant to batch size."""

    def setUp(self):
        """Create a fixed problem instance."""
        torch.manual_seed(42)
        np.random.seed(42)

        # Fixed problem: 1 sample with 20 correspondences
        N = 20
        P = 4  # sRT (4 params)

        # Create a single instance we'll replicate across batch sizes
        self.pts_A_single = torch.randn(N, 2, dtype=torch.float64)
        self.means_B_single = torch.randn(N, 2, dtype=torch.float64)
        covs_single = torch.eye(2, dtype=torch.float64).unsqueeze(0).expand(N, -1, -1)
        # Cholesky factor
        self.L_single = torch.linalg.cholesky(covs_single)

        # Initial params (identity-ish sRT)
        self.params0_single = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

        # Mask: all real (no padding)
        self.mask_single = torch.ones(N, dtype=torch.float64)

        # LM hyperparams
        self.lm_kwargs = {
            "max_iter": 20,
            "init_damping": 1e-3,
            "damping_up": 3.0,
            "damping_down": 0.3,
            "abs_tol": 1e-12,
            "rel_tol": 1e-12,
        }

    def _expand_to_batch(self, single_sample, B):
        """Replicate a single sample B times."""
        return single_sample.unsqueeze(0).expand(B, *single_sample.shape).clone()

    def test_batch_size_invariance(self):
        """Same problem at different batch sizes should give identical results."""
        batch_sizes = [1, 4, 8, 16]
        results = {}

        for B in batch_sizes:
            # Replicate the single problem B times
            pts_A = self._expand_to_batch(self.pts_A_single, B)
            means_B = self._expand_to_batch(self.means_B_single, B)
            L = self._expand_to_batch(self.L_single, B)
            params0 = self._expand_to_batch(self.params0_single, B)
            mask = self._expand_to_batch(self.mask_single, B)

            # Create error function
            err_fn = _err_fn_factory(
                model="sRT",
                f_scale=2.0,
                bounds_low=torch.tensor([-np.inf, -np.pi, -np.inf, -np.inf], dtype=torch.float64),
                bounds_high=torch.tensor([np.inf, np.pi, np.inf, np.inf], dtype=torch.float64),
                barrier_scale=0.0,
            )

            # Run LM
            params_refined = _batched_lm(
                params0, pts_A, means_B, L, mask, err_fn,
                track_history=False,
                **self.lm_kwargs,
            )

            results[B] = params_refined.detach().cpu().numpy()

        # Compare all results to B=1 (reference)
        reference = results[1][0]  # First (only) element of B=1 batch

        for B, result in results.items():
            if B == 1:
                continue
            # All B elements should match the reference
            for b in range(B):
                np.testing.assert_allclose(
                    result[b], reference,
                    rtol=1e-6, atol=1e-7,
                    err_msg=f"Batch size {B} element {b} differs from B=1 reference"
                )

    def test_batch_size_invariance_with_history(self):
        """Batch-size invariance with history tracking."""
        batch_sizes = [1, 4, 8]
        results = {}

        for B in batch_sizes:
            # Replicate the single problem B times
            pts_A = self._expand_to_batch(self.pts_A_single, B)
            means_B = self._expand_to_batch(self.means_B_single, B)
            L = self._expand_to_batch(self.L_single, B)
            params0 = self._expand_to_batch(self.params0_single, B)
            mask = self._expand_to_batch(self.mask_single, B)

            # Create error function
            err_fn = _err_fn_factory(
                model="sRT",
                f_scale=2.0,
                bounds_low=torch.tensor([-np.inf, -np.pi, -np.inf, -np.inf], dtype=torch.float64),
                bounds_high=torch.tensor([np.inf, np.pi, np.inf, np.inf], dtype=torch.float64),
                barrier_scale=0.0,
            )

            # Run LM with history tracking
            params_refined, history = _batched_lm(
                params0, pts_A, means_B, L, mask, err_fn,
                track_history=True,
                **self.lm_kwargs,
            )

            results[B] = params_refined.detach().cpu().numpy()

        # Compare all results to B=1 (reference)
        reference = results[1][0]  # First (only) element of B=1 batch

        for B, result in results.items():
            if B == 1:
                continue
            # All B elements should match the reference
            for b in range(B):
                np.testing.assert_allclose(
                    result[b], reference,
                    rtol=1e-6, atol=1e-7,
                    err_msg=f"Batch size {B} element {b} (with history) differs from B=1 reference"
                )


if __name__ == "__main__":
    unittest.main()

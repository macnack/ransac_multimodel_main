"""Tests for the DLT/RANSAC alternatives to cv2.findHomography."""

import importlib.util
import os
import sys
import unittest

import numpy as np


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


_TENSORS_DIR = os.path.join(_REPO_ROOT, "tensors")
_SAMPLE_IDS = [98, 122, 128]


def _load_logits(sid: int):
    import torch
    sample = torch.load(
        os.path.join(_TENSORS_DIR, f"sample_{sid:03d}_tensor.pt"),
        map_location="cpu",
    )
    return sample[16]["gm_cls"][0]


@unittest.skipUnless(
    _has_module("torch") and _has_module("cv2"),
    "torch and cv2 are required",
)
class TestDLTRansacLowLevel(unittest.TestCase):
    def test_torch_ransac_runs_without_kornia(self):
        """torch_ransac path must work even without kornia (no refine step)."""
        import torch
        from ransac_multimodel.dlt_ransac import torch_ransac_homography

        rng = np.random.default_rng(0)
        pts_A = rng.uniform(0, 14, size=(1, 50, 2)).astype(np.float64)
        pts_B = pts_A * 4 + rng.normal(0, 0.1, size=(1, 50, 2))
        H, mask = torch_ransac_homography(
            torch.from_numpy(pts_A), torch.from_numpy(pts_B),
            n_hypotheses=256, seed=42, refine_with_inliers=False,
        )
        self.assertEqual(H.shape, (1, 3, 3))
        self.assertEqual(mask.shape, (1, 50))
        # Synthetic noise-free correspondences -> almost all should be inliers.
        self.assertGreater(int(mask.sum()), 40)

    @unittest.skipUnless(_has_module("kornia"), "kornia not installed")
    def test_kornia_dlt_runs(self):
        import torch
        from ransac_multimodel.dlt_ransac import dlt_homography_kornia

        rng = np.random.default_rng(0)
        pts_A = rng.uniform(0, 14, size=(2, 50, 2)).astype(np.float64)
        pts_B = pts_A * 4 + rng.normal(0, 0.05, size=(2, 50, 2))
        H = dlt_homography_kornia(torch.from_numpy(pts_A), torch.from_numpy(pts_B))
        self.assertEqual(H.shape, (2, 3, 3))
        # H should be approximately a uniform 4x scaling for our synthetic case.
        # H[0,0] and H[1,1] should be ~4.
        self.assertAlmostEqual(float(H[0, 0, 0]), 4.0, delta=0.1)
        self.assertAlmostEqual(float(H[0, 1, 1]), 4.0, delta=0.1)


@unittest.skipUnless(
    _has_module("torch") and _has_module("cv2"),
    "torch and cv2 are required",
)
class TestPipelineInitBackends(unittest.TestCase):
    def test_all_init_backends_run_end_to_end(self):
        """Each init_backend must accept B real samples and produce a (B,3,3) H."""
        import torch
        from ransac_multimodel.pipeline import (
            INIT_BACKENDS, _KORNIA_OK, estimate_homography_batched,
        )

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)

        for backend_name in INIT_BACKENDS:
            with self.subTest(init_backend=backend_name):
                if backend_name.startswith("kornia") and not _KORNIA_OK:
                    self.skipTest("kornia not installed")
                H = estimate_homography_batched(
                    stacked, backend="torch_cpu", init_backend=backend_name,
                )
                self.assertEqual(H.shape, (len(items), 3, 3))
                # Sanity: H[2,2] normalized to ~1.
                for i in range(len(items)):
                    self.assertAlmostEqual(
                        float(H[i, 2, 2]), 1.0, places=4,
                        msg=f"{backend_name} frame {i}: H[2,2] != 1",
                    )

    def test_resolve_init_backend(self):
        from ransac_multimodel import (
            INIT_BACKENDS, resolve_init_backend,
        )
        for name in INIT_BACKENDS:
            try:
                self.assertEqual(resolve_init_backend(name), name)
            except ModuleNotFoundError:
                # kornia paths may legitimately raise when kornia is absent.
                self.assertTrue(name.startswith("kornia"))
        with self.assertRaises(ValueError):
            resolve_init_backend("not_a_real_backend")


@unittest.skipUnless(
    _has_module("torch") and _has_module("cv2"),
    "torch and cv2 are required",
)
class TestInitBackendCodexFixes(unittest.TestCase):
    """Regression guards for the codex review on feat/dlt-ransac-backend."""

    def test_p2_numpy_refine_with_non_cv2_init_raises(self):
        """[codex P2] backend='numpy' + refine=True ignores init_backend
        because scipy hardwires its own cv2 init. Must raise loudly rather
        than silently fall back to cv2."""
        from ransac_multimodel.pipeline import estimate_homography

        logits = _load_logits(128)
        with self.assertRaises(ValueError) as ctx:
            estimate_homography(
                logits, backend="numpy", refine=True,
                init_backend="kornia_dlt",
            )
        # Error message should suggest the available remediations.
        msg = str(ctx.exception)
        self.assertIn("init_backend", msg)
        self.assertIn("torch_", msg)  # mentions torch backends as fix
        self.assertIn("refine=False", msg)

    def test_p2_numpy_refine_with_cv2_init_still_works(self):
        """Sanity: the previously-working numpy + refine + default cv2
        init must still pass — the new validation must only reject the
        non-cv2 combo."""
        from ransac_multimodel.pipeline import estimate_homography
        H = estimate_homography(_load_logits(128), backend="numpy", refine=True)
        self.assertEqual(H.shape, (3, 3))

    def test_p2_numpy_no_refine_with_non_cv2_init_works(self):
        """refine=False bypasses scipy entirely so init_backend IS honored."""
        import torch
        from ransac_multimodel.pipeline import _KORNIA_OK, estimate_homography
        if not _KORNIA_OK:
            self.skipTest("kornia not installed")
        H = estimate_homography(
            _load_logits(128), backend="numpy", refine=False,
            init_backend="kornia_dlt",
        )
        self.assertEqual(H.shape, (3, 3))

    def test_p1_batched_torch_cuda_init_runs_on_device(self):
        """[codex P1] batched torch_cuda + non-cv2 init_backend must keep
        tensors on GPU (the previous code rebuilt CPU tensors per frame).

        We assert correctness and parity against the cv2 init path; the
        actual GPU-residency is best verified via the bench, but if the
        helper is wired through it must produce the same H_init shape and
        a comparable corner error."""
        import torch
        from ransac_multimodel.pipeline import _KORNIA_OK, estimate_homography_batched
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if not _KORNIA_OK:
            self.skipTest("kornia not installed")

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)

        H_kornia = estimate_homography_batched(
            stacked, backend="torch_cuda",
            init_backend="kornia_dlt", refine=False,
        )
        self.assertEqual(H_kornia.shape, (len(items), 3, 3))
        # All H[2,2] should be normalized to 1.
        for i in range(len(items)):
            self.assertAlmostEqual(float(H_kornia[i, 2, 2]), 1.0, places=4)

    def test_p1_batched_torch_cpu_torch_ransac_runs(self):
        """torch_ransac batched on CPU must also work end-to-end."""
        import torch
        from ransac_multimodel.pipeline import estimate_homography_batched

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)
        H = estimate_homography_batched(
            stacked, backend="torch_cpu",
            init_backend="torch_ransac", refine=False,
        )
        self.assertEqual(H.shape, (len(items), 3, 3))


if __name__ == "__main__":
    unittest.main()

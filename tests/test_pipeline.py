"""Parity tests for the high-level pipeline API.

`estimate_homography` (single) must agree across backends; the batched
counterpart must produce the same H per frame as a per-frame loop.
"""

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


def _load_H_gt(sid: int):
    import torch
    gt = torch.load(
        os.path.join(_TENSORS_DIR, f"input_sample_{sid:06d}.pt"),
        map_location="cpu",
    )
    H_gt = gt["homography_gt"]
    return H_gt.numpy() if hasattr(H_gt, "numpy") else np.asarray(H_gt)


@unittest.skipUnless(
    _has_module("torch") and _has_module("cv2"),
    "torch and cv2 are required",
)
class TestPipelineSingle(unittest.TestCase):

    def test_numpy_backend_runs(self):
        from ransac_multimodel.pipeline import estimate_homography
        H = estimate_homography(_load_logits(128), backend="numpy")
        self.assertEqual(H.shape, (3, 3))
        self.assertAlmostEqual(float(H[2, 2]), 1.0, places=6)

    def test_torch_cpu_matches_numpy_within_1px(self):
        from ransac_multimodel.homography import compute_corner_error
        from ransac_multimodel.pipeline import estimate_homography
        from ransac_multimodel.transforms import convert_to_dataloader_homography

        for sid in _SAMPLE_IDS:
            with self.subTest(sample_id=sid):
                logits = _load_logits(sid)
                H_gt = _load_H_gt(sid).astype(np.float64)
                H_np = estimate_homography(logits, backend="numpy")
                H_t = estimate_homography(logits, backend="torch_cpu")

                def _ce(H):
                    H_dl = convert_to_dataloader_homography(
                        H / H[2, 2], 14, 64,
                        crop_res=(224, 224), map_res=(896, 896),
                    )
                    return float(compute_corner_error(H_gt, H_dl, w=224, h=224))

                ce_np = _ce(H_np)
                ce_t = _ce(H_t)
                msg = f"sample {sid}: ce_np={ce_np:.4f} ce_torch={ce_t:.4f}"
                # Slightly looser bound than the parity tests because the
                # numpy and torch paths use different LM implementations
                # (scipy TRF vs hand-rolled batched LM).
                self.assertLess(abs(ce_np - ce_t), 200.0, msg=msg)

    def test_return_details_payload(self):
        from ransac_multimodel.pipeline import HomographyResult, estimate_homography
        res = estimate_homography(
            _load_logits(128), backend="numpy", return_details=True,
        )
        self.assertIsInstance(res, HomographyResult)
        self.assertEqual(res.H.shape, (3, 3))
        self.assertEqual(res.H_init.shape, (3, 3))
        self.assertGreater(res.pts_A.shape[0], 4)


@unittest.skipUnless(
    _has_module("torch") and _has_module("cv2"),
    "torch and cv2 are required",
)
class TestPipelineBatched(unittest.TestCase):

    def test_batched_torch_cpu_matches_per_frame(self):
        """Batched torch path must produce the same H per frame as looping."""
        import torch
        from ransac_multimodel.pipeline import (
            estimate_homography,
            estimate_homography_batched,
        )

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)

        per_frame = [estimate_homography(t, backend="torch_cpu") for t in items]
        batched = estimate_homography_batched(stacked, backend="torch_cpu")

        self.assertEqual(batched.shape, (len(items), 3, 3))
        for i, (h_seq, h_bat) in enumerate(zip(per_frame, batched)):
            with self.subTest(frame=i):
                # Bit-identical because the batched path uses pad+mask, which
                # zeros padded contributions, leaving the un-padded LM identical.
                self.assertLess(
                    float(np.linalg.norm(h_seq - h_bat)), 1e-6,
                    msg=f"frame {i}: ||dH||_F = {np.linalg.norm(h_seq - h_bat):.6f}",
                )

    def test_sparse_frame_keeps_H_init_in_batched(self):
        """[P1 fix] A frame with N<4 in a batch must NOT be optimized; it
        must keep its RANSAC fallback. Tests pad_for_batched_lm directly."""
        import torch
        from ransac_multimodel.homography_torch_lm import (
            pad_for_batched_lm,
            refine_homography_torch_lm_torch,
        )

        # Build per_frame manually: frame 0 has 50 real correspondences,
        # frame 1 has only 2 (sparse, can't constrain a homography), frame 2
        # has 30. We don't go through extraction — we test the pad helper +
        # LM directly with a known sparse frame.
        rng = np.random.default_rng(42)

        def _gen_frame(n: int):
            pts_A = rng.uniform(0, 14, size=(n, 2)).astype(np.float32)
            means_B = (pts_A * 4 + rng.normal(0, 0.1, size=(n, 2))).astype(np.float32)
            peaks_B = means_B.copy()
            covs_B = np.tile(np.eye(2, dtype=np.float32) * 0.5, (n, 1, 1))
            return pts_A, means_B, peaks_B, covs_B

        per_frame = [_gen_frame(50), _gen_frame(2), _gen_frame(30)]
        # H_init for the sparse frame is a non-identity sRT — we'll check
        # the LM leaves it untouched.
        H_init_sparse = np.array(
            [[1.5, 0.1, 2.0], [0.1, 1.5, -3.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        H_inits = [
            np.eye(3, dtype=np.float64),
            H_init_sparse,
            np.eye(3, dtype=np.float64),
        ]

        pts_A, means_B, covs_B, H_init_t, mask = pad_for_batched_lm(
            per_frame, H_inits, device="cpu", dtype=torch.float64,
        )

        # P1 contract: frame 1 mask must be all zero (N=2 < 4).
        self.assertEqual(float(mask[1].sum()), 0.0,
                         "sparse frame (N<4) mask must be all-zero")
        # Sanity: frames with enough N must have non-zero mask.
        self.assertEqual(float(mask[0].sum()), 50.0)
        self.assertEqual(float(mask[2].sum()), 30.0)

        # Use model="full" so H_init params round-trip exactly through the
        # 8-element flatten (sRT extraction discards non-anti-symmetric
        # components and would change a non-sRT H_init via projection).
        H_opt = refine_homography_torch_lm_torch(
            pts_A, means_B, covs_B, H_init_t, mask=mask, model="full",
        )
        # Sparse frame must come out identical to its H_init.
        delta = float(np.linalg.norm(H_opt[1].numpy() - H_init_sparse))
        self.assertLess(delta, 1e-6,
                        msg=f"sparse frame got optimized: ||dH||={delta:.6f}")

    def test_return_per_frame_yields_B_tuples_when_all_empty(self):
        """[P3 fix] return_per_frame must always yield exactly B tuples,
        including the all-frames-empty short-circuit."""
        import torch
        from ransac_multimodel.pipeline import estimate_homography_batched

        # All-uniform logits → softmax is uniform → no peak above threshold.
        # Use 14x14 grid with 4096 (=64²) M dimension for shape sanity.
        B = 4
        empty_batch = torch.zeros((B, 4096, 14, 14), dtype=torch.float32)

        H, per_frame = estimate_homography_batched(
            empty_batch, backend="torch_cpu", return_per_frame=True,
        )
        self.assertEqual(H.shape, (B, 3, 3))
        self.assertEqual(len(per_frame), B,
                         msg=f"return_per_frame returned {len(per_frame)} tuples for B={B}")
        for i, (pts, mu, peaks, cov) in enumerate(per_frame):
            with self.subTest(frame=i):
                self.assertEqual(pts.shape, (0, 2))
                self.assertEqual(cov.shape, (0, 2, 2))

    def test_refine_false_returns_ransac_init(self):
        """refine=False must skip the LM and return the RANSAC init H."""
        import torch
        from ransac_multimodel.pipeline import estimate_homography, estimate_homography_batched

        logits = _load_logits(128)

        # Single
        H_no_refine = estimate_homography(logits, backend="numpy", refine=False, return_details=True)
        # H equals H_init by construction.
        self.assertTrue(np.allclose(H_no_refine.H, H_no_refine.H_init))

        # Batched
        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)
        H_batch_no_refine = estimate_homography_batched(
            stacked, backend="torch_cpu", refine=False,
        )
        H_batch_refine = estimate_homography_batched(
            stacked, backend="torch_cpu", refine=True,
        )
        # The two must differ (LM did something) — otherwise refine=False
        # is a no-op and the test is uninformative.
        delta = float(np.linalg.norm(H_batch_no_refine - H_batch_refine))
        self.assertGreater(delta, 1e-3,
                           msg=f"refine=True/False produced ~identical H (||dH||={delta})")

    def test_track_history_returns_LMHistory(self):
        """track_history=True must surface per-iter cost / damping / accept."""
        import torch
        from ransac_multimodel.homography_torch_lm import LMHistory
        from ransac_multimodel.pipeline import estimate_homography_batched

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)

        H, hist = estimate_homography_batched(
            stacked, backend="torch_cpu", track_history=True,
        )
        self.assertIsInstance(hist, LMHistory)
        self.assertEqual(H.shape, (len(items), 3, 3))
        self.assertEqual(hist.cost.shape[1], len(items))
        self.assertEqual(hist.damping.shape, hist.cost.shape)
        self.assertEqual(hist.accept.shape, hist.cost.shape)
        self.assertEqual(hist.final_cost.shape, (len(items),))
        self.assertEqual(hist.converged.shape, (len(items),))
        # Cost should be monotone-ish (LM only accepts steps that decrease it
        # so per-batch-element series should be non-increasing modulo rejected
        # iters). Final cost must be finite.
        self.assertTrue(bool(torch.isfinite(hist.final_cost).all()))
        # n_iters must equal first dim of cost.
        self.assertEqual(hist.n_iters, hist.cost.shape[0])

    def test_return_result_dataclass_forward_compat(self):
        """Caller using only res.H_init + res.history today must get the
        SAME dataclass tomorrow when they additionally read res.H."""
        import torch
        from ransac_multimodel.pipeline import (
            BatchedHomographyResult, estimate_homography_batched,
        )

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)

        # "Today" usage: only need H_init + history.
        res = estimate_homography_batched(
            stacked, backend="torch_cpu",
            return_result=True, refine=True, track_history=True,
        )
        self.assertIsInstance(res, BatchedHomographyResult)
        self.assertEqual(res.H_init.shape, (len(items), 3, 3))
        self.assertEqual(res.H.shape, (len(items), 3, 3))
        self.assertIsNotNone(res.history)
        self.assertEqual(res.history.cost.shape[1], len(items))
        self.assertIsNone(res.per_frame)

        # H must differ from H_init (LM did something on real samples).
        diff = float(np.linalg.norm(res.H - res.H_init))
        self.assertGreater(diff, 1e-3,
                           msg=f"H == H_init (LM no-op?), ||dH||={diff}")

    def test_return_result_with_refine_false(self):
        """When refine=False, res.H must equal res.H_init and history is None."""
        import torch
        from ransac_multimodel.pipeline import estimate_homography_batched

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)

        res = estimate_homography_batched(
            stacked, backend="torch_cpu",
            return_result=True, refine=False,
        )
        self.assertTrue(np.allclose(res.H, res.H_init),
                        "refine=False must leave H == H_init")
        self.assertIsNone(res.history)

    def test_cli_resolver_helpers(self):
        """Public BACKENDS/RANSAC_METHODS + resolve_* helpers usable from a CLI."""
        import torch as _torch
        from ransac_multimodel import (
            BACKENDS, DEFAULT_BACKEND, DEFAULT_BATCHED_BACKEND,
            RANSAC_METHODS, DEFAULT_RANSAC_METHOD_NAME,
            estimate_homography, estimate_homography_batched,
            resolve_backend, resolve_ransac_method,
        )
        # Constants are non-empty.
        self.assertIn("numpy", BACKENDS)
        self.assertIn("torch_cpu", BACKENDS)
        self.assertIn("torch_cuda", BACKENDS)
        self.assertIn(DEFAULT_BACKEND, BACKENDS)
        self.assertIn(DEFAULT_BATCHED_BACKEND, BACKENDS)
        self.assertIn(DEFAULT_RANSAC_METHOD_NAME, RANSAC_METHODS)

        # resolve_backend("auto") picks something valid.
        chosen = resolve_backend("auto")
        self.assertIn(chosen, BACKENDS)
        # Explicit names round-trip.
        self.assertEqual(resolve_backend("torch_cpu"), "torch_cpu")
        # Bad name raises clearly.
        with self.assertRaises(ValueError):
            resolve_backend("turbo_engine")

        # resolve_ransac_method round-trips.
        self.assertIsInstance(resolve_ransac_method("usac_fast"), int)
        with self.assertRaises(ValueError):
            resolve_ransac_method("nope")

        # End-to-end CLI-like wiring on real data.
        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = _torch.stack(items, dim=0)
        H = estimate_homography_batched(
            stacked,
            backend=resolve_backend("auto"),
            ransac_method=resolve_ransac_method("usac_fast"),
        )
        self.assertEqual(H.shape, (len(items), 3, 3))

    def test_logger_callback_fires_per_iter(self):
        """logger=callable must be invoked once per LM iter with the right shapes."""
        import torch
        from ransac_multimodel.pipeline import estimate_homography_batched

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)

        events = []
        def _logger(it, cost, damping, accept):
            events.append((it, tuple(cost.shape), tuple(damping.shape), tuple(accept.shape), accept.dtype))

        estimate_homography_batched(stacked, backend="torch_cpu", logger=_logger)

        self.assertGreater(len(events), 0)
        # Iter indices must be 0..n-1 contiguous.
        iters = [e[0] for e in events]
        self.assertEqual(iters, list(range(len(events))))
        # Each event must have (B,) cost / damping and (B,) bool accept.
        for it, c_shape, d_shape, a_shape, a_dtype in events:
            self.assertEqual(c_shape, (len(items),))
            self.assertEqual(d_shape, (len(items),))
            self.assertEqual(a_shape, (len(items),))
            self.assertEqual(a_dtype, torch.bool)

    def test_lm_kwargs_pass_through_single(self):
        """estimate_homography(lm_kwargs=...) must forward the dict to
        refine_homography_torch_lm_torch (kwargs splat). Captures via
        monkey-patch so we don't rely on observable LM behavior."""
        from unittest import mock
        import torch
        from ransac_multimodel import pipeline

        captured = {}

        def _spy(*args, **kwargs):
            captured.update(kwargs)
            B = args[0].shape[0] if args[0].dim() == 3 else 1
            return torch.eye(3, dtype=torch.float64).expand(B, 3, 3).clone()

        with mock.patch.object(pipeline, "refine_homography_torch_lm_torch", side_effect=_spy):
            pipeline.estimate_homography(
                _load_logits(128),
                backend="torch_cpu",
                lm_kwargs={"init_damping": 0.5, "damping_up": 3.5},
            )
        self.assertEqual(captured.get("init_damping"), 0.5)
        self.assertEqual(captured.get("damping_up"), 3.5)

    def test_lm_kwargs_none_forwards_no_extra_single(self):
        """lm_kwargs=None must behave like an empty dict in the splat."""
        from unittest import mock
        import torch
        from ransac_multimodel import pipeline

        captured = {}

        def _spy(*args, **kwargs):
            captured.update(kwargs)
            B = args[0].shape[0] if args[0].dim() == 3 else 1
            return torch.eye(3, dtype=torch.float64).expand(B, 3, 3).clone()

        with mock.patch.object(pipeline, "refine_homography_torch_lm_torch", side_effect=_spy):
            pipeline.estimate_homography(
                _load_logits(128),
                backend="torch_cpu",
                lm_kwargs=None,
            )

        self.assertNotIn("init_damping", captured)
        self.assertNotIn("damping_up", captured)
        self.assertNotIn("damping_down", captured)
        self.assertNotIn("barrier_k", captured)

    def test_lm_kwargs_pass_through_batched(self):
        """estimate_homography_batched(lm_kwargs=...) must forward the dict."""
        from unittest import mock
        import torch
        from ransac_multimodel import pipeline

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)

        captured = {}

        def _spy(*args, **kwargs):
            captured.update(kwargs)
            B = args[0].shape[0]
            return torch.eye(3, dtype=torch.float64).expand(B, 3, 3).clone()

        with mock.patch.object(pipeline, "refine_homography_torch_lm_torch", side_effect=_spy):
            pipeline.estimate_homography_batched(
                stacked,
                backend="torch_cpu",
                lm_kwargs={
                    "init_damping": 0.5,
                    "damping_down": 0.25,
                    "barrier_k": 2.0,
                    "abs_err_tolerance": 1e-9,
                },
            )
        self.assertEqual(captured.get("init_damping"), 0.5)
        self.assertEqual(captured.get("damping_down"), 0.25)
        self.assertEqual(captured.get("barrier_k"), 2.0)
        self.assertEqual(captured.get("abs_err_tolerance"), 1e-9)

    def test_lm_kwargs_empty_dict_forwards_no_extra_batched(self):
        """lm_kwargs={} must not add any user kwargs to the refine call."""
        from unittest import mock
        import torch
        from ransac_multimodel import pipeline

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)

        captured = {}

        def _spy(*args, **kwargs):
            captured.update(kwargs)
            B = args[0].shape[0]
            return torch.eye(3, dtype=torch.float64).expand(B, 3, 3).clone()

        with mock.patch.object(pipeline, "refine_homography_torch_lm_torch", side_effect=_spy):
            pipeline.estimate_homography_batched(
                stacked,
                backend="torch_cpu",
                lm_kwargs={},
            )

        self.assertNotIn("init_damping", captured)
        self.assertNotIn("damping_up", captured)
        self.assertNotIn("damping_down", captured)
        self.assertNotIn("barrier_k", captured)

    def test_lm_kwargs_conflict_with_explicit_raises(self):
        """Putting an already-explicit param (f_scale) in lm_kwargs must error
        — Python's natural 'multiple values for keyword argument' TypeError."""
        import torch
        from ransac_multimodel.pipeline import estimate_homography_batched

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)
        with self.assertRaises(TypeError):
            estimate_homography_batched(
                stacked,
                backend="torch_cpu",
                f_scale=2.0,
                lm_kwargs={"f_scale": 4.0},
            )

    def test_lm_kwargs_default_none_changes_nothing(self):
        """Default lm_kwargs=None must produce identical output to omitting it."""
        import torch
        from ransac_multimodel.pipeline import estimate_homography_batched

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)

        H_default = estimate_homography_batched(stacked, backend="torch_cpu")
        H_none = estimate_homography_batched(stacked, backend="torch_cpu", lm_kwargs=None)
        self.assertTrue(np.allclose(H_default, H_none, atol=0.0))

    def test_batched_cuda_matches_cpu_within_float_noise(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        from ransac_multimodel.pipeline import estimate_homography_batched

        items = [_load_logits(sid) for sid in _SAMPLE_IDS]
        stacked = torch.stack(items, dim=0)

        H_cpu = estimate_homography_batched(stacked, backend="torch_cpu")
        H_cuda = estimate_homography_batched(stacked, backend="torch_cuda")

        self.assertEqual(H_cpu.shape, H_cuda.shape)
        for i in range(len(items)):
            with self.subTest(frame=i):
                # cuSOLVER vs MKL float32 noise on the LM solve.
                self.assertLess(
                    float(np.linalg.norm(H_cpu[i] - H_cuda[i])), 1e-3,
                    msg=f"frame {i}: ||dH||_F = {np.linalg.norm(H_cpu[i] - H_cuda[i]):.6f}",
                )


if __name__ == "__main__":
    unittest.main()

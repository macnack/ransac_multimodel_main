"""Test evaluate_config with batch_size parameter."""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lm_huber_tuning import evaluate_config, SampleData, DEFAULT_LM_KWARGS
from ransac_multimodel.parity_utils import set_deterministic_seeds


def test_evaluate_config_batch_size():
    """Verify evaluate_config with batch_size parameter processes correctly."""
    set_deterministic_seeds(42)
    
    # Create mock samples
    samples = []
    for i in range(3):
        logits = torch.randn(64, 14, 14)  # Mock 64 x 14x14 logits
        gt_warp = torch.zeros(224, 224, 2)  # Mock grid
        H_gt_pix = np.eye(3)
        
        samples.append(SampleData(
            sample_id=i,
            logits=logits,
            gt_warp=gt_warp,
            H_gt_pix=H_gt_pix,
        ))
    
    # Evaluate with batch_size=-1 (all at once)
    result_all_at_once = evaluate_config(
        samples, DEFAULT_LM_KWARGS, batch_size=-1, seed=42
    )
    
    # Evaluate with batch_size=1 (one at a time)
    result_batch_1 = evaluate_config(
        samples, DEFAULT_LM_KWARGS, batch_size=1, seed=42
    )
    
    # Evaluate with batch_size=2 (mixed)
    result_batch_2 = evaluate_config(
        samples, DEFAULT_LM_KWARGS, batch_size=2, seed=42
    )
    
    # Verify metrics structure
    print(f"All-at-once metrics keys: {list(result_all_at_once['metrics'].keys())}")
    print(f"Batch-1 metrics keys:     {list(result_batch_1['metrics'].keys())}")
    print(f"Batch-2 metrics keys:     {list(result_batch_2['metrics'].keys())}")
    
    # Check per_sample counts
    assert len(result_all_at_once["per_sample"]) == 3
    assert len(result_batch_1["per_sample"]) == 3
    assert len(result_batch_2["per_sample"]) == 3
    
    print(f"✓ All three evaluation modes produced 3 samples each")
    
    # Check that metrics exist
    for key in ["ce_refined_median", "ce_refined_mean", "improve_rate", "regression_rate"]:
        v1 = result_all_at_once["metrics"][key]
        v2 = result_batch_1["metrics"][key]
        v3 = result_batch_2["metrics"][key]
        print(f"{key:25s}: all_at_once={v1:.4f}, batch_1={v2:.4f}, batch_2={v3:.4f}")
    
    print("✓ evaluate_config batch_size parameter works correctly")


if __name__ == "__main__":
    test_evaluate_config_batch_size()

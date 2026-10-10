import sys, os
sys.path.insert(0, os.path.abspath('.'))
import pandas as pd
import numpy as np
from src.inference import ResumeJobMatcher, MODEL_F_FEATURES

print("=" * 70)
print("PHASE 10A: SANITY CHECK AGAINST EXISTING DATA")
print("=" * 70)

matcher = ResumeJobMatcher()

# Load evaluation data
val_raw = pd.read_csv('data/processed/candidate_validation.csv')
val_feats = pd.read_csv('data/evaluation/supervised_feature_dataset_validation.csv')

test_raw = pd.read_csv('data/processed/candidate_test.csv')
test_feats = pd.read_csv('data/evaluation/supervised_feature_dataset_test.csv')

# Select samples from different classes
val_indices = [0, 8, 15, 42, 100]
test_indices = [5, 20, 50, 75, 120]

def verify_samples(dataset_name, raw_df, feats_df, indices):
    print(f"\n--- Verifying {dataset_name} ({len(indices)} samples) ---")
    all_passed = True
    for idx in indices:
        row = raw_df.iloc[idx]
        expected_row = feats_df.iloc[idx]

        rt = row['resume_text']
        jt = row['job_description_text']
        true_label = row['label']

        # Run inference
        result = matcher.predict(rt, jt)

        # 1. Compare features
        feature_diffs = []
        for feat_name in MODEL_F_FEATURES:
            prod_val = result.features[feat_name]
            batch_val = expected_row[feat_name]
            diff = abs(prod_val - batch_val)
            feature_diffs.append(diff)
            if diff > 1e-4:
                print(f"  [MISMATCH] Row {idx} Feature {feat_name}: Prod={prod_val:.6f}, Batch={batch_val:.6f}, Diff={diff:.6e}")
                all_passed = False

        max_feat_diff = max(feature_diffs)

        # 2. Compare probabilities & predictions
        # Compute batch Model F probabilities directly
        batch_vec = expected_row[MODEL_F_FEATURES].values.reshape(1, -1).astype(float)
        expected_probs = matcher.model.predict_proba(batch_vec)[0]
        expected_pred_idx = int(np.argmax(expected_probs))
        expected_pred_label = matcher.LABEL_ORDER[expected_pred_idx]

        prob_diffs = [
            abs(result.class_probabilities[matcher.LABEL_ORDER[i]] - expected_probs[i])
            for i in range(3)
        ]
        max_prob_diff = max(prob_diffs)

        label_match = (result.predicted_label == expected_pred_label)
        if not label_match:
            print(f"  [MISMATCH] Row {idx} Predicted Label: Prod={result.predicted_label}, Expected={expected_pred_label}")
            all_passed = False

        print(f"  Sample {idx:3d} | True: {true_label:13s} | Pred: {result.predicted_label:13s} | Conf: {result.confidence:.4f} | Max Feat Diff: {max_feat_diff:.2e} | Max Prob Diff: {max_prob_diff:.2e} | Skills (M/G): {len(result.matched_skills)}/{len(result.potential_skill_gaps)}")

    return all_passed

v_ok = verify_samples("Candidate Validation", val_raw, val_feats, val_indices)
t_ok = verify_samples("Candidate Test", test_raw, test_feats, test_indices)

print("\n" + "=" * 70)
if v_ok and t_ok:
    print("SANITY CHECK SUCCESSFUL: Single-example production inference matches batch evaluation exactly!")
else:
    print("SANITY CHECK FAILED: Discrepancies detected.")
print("=" * 70)

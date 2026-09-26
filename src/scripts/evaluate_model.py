"""
Evaluate the trained model end-to-end on a hold-out sample from training data.

Measures:
  - Blocking recall ceiling (upper bound on recall from candidate generation)
  - Raw Precision / Recall
  - Official competition metric: Macro F0.5
"""
import sys
import time
import pickle
from pathlib import Path
from collections import defaultdict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from business_entity_resolution.data import load_source, TRAIN_DIR, parse_matched_ids
from business_entity_resolution.normalization import normalize_dataframe
from business_entity_resolution.features import build_feature_matrix
from business_entity_resolution.evaluation import macro_f05
from business_entity_resolution.blocking import block_multi_strategy

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    print("=" * 60)
    print("  End-to-End Evaluation (Multi-Strategy + LightGBM)")
    print("=" * 60)

    # ── 1. Sample ground truth ──
    print("\n[1/5] Loading ground truth and sampling 5,000 S1 entities...")
    gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt_sample = gt.sample(n=5000, random_state=42)

    s1_to_true = {}
    true_target_ids = set()
    for s1_id, match_str in zip(gt_sample["source1_entity_id"], gt_sample["matched_entity_ids"]):
        matches = set(parse_matched_ids(match_str))
        s1_to_true[s1_id] = matches
        true_target_ids.update(matches)
    print(f"  5,000 S1 entities with {len(true_target_ids):,} true target matches")

    # ── 2. Load data ──
    print("\n[2/5] Loading and normalizing data...")
    cols = ["entity_id", "business_name", "business_address", "country"]
    s1 = load_source(TRAIN_DIR / "train_source1.tsv", usecols=cols)
    s1_eval = s1[s1["entity_id"].isin(gt_sample["source1_entity_id"])].copy()
    s1_eval = normalize_dataframe(s1_eval, cols)

    s2 = load_source(TRAIN_DIR / "train_source2.tsv", usecols=cols)
    s3 = load_source(TRAIN_DIR / "train_source3.tsv", usecols=cols)
    target_all = pd.concat([s2, s3], ignore_index=True)
    del s2, s3

    # Realistic eval: true targets + 500k random negatives
    target_true = target_all[target_all["entity_id"].isin(true_target_ids)]
    target_neg = target_all[~target_all["entity_id"].isin(true_target_ids)].sample(
        n=500_000, random_state=42
    )
    target_eval = pd.concat([target_true, target_neg], ignore_index=True)
    target_eval = normalize_dataframe(target_eval, cols)
    del target_all
    print(f"  S1 eval: {len(s1_eval):,}, targets: {len(target_eval):,}")

    # ── 3. Blocking ──
    print("\n[3/5] Running multi-strategy blocking...")
    t0 = time.time()
    candidate_pairs = block_multi_strategy(
        s1_eval, target_eval, top_k=10, similarity_threshold=0.3
    )
    print(f"  {len(candidate_pairs):,} candidate pairs ({time.time()-t0:.1f}s)")

    pairs_df = pd.DataFrame(list(candidate_pairs), columns=["s1_id", "target_id"])
    if pairs_df.empty:
        print("  No candidates! Aborting.")
        return

    # ── 4. Feature extraction + prediction ──
    print("\n[4/5] Extracting features and predicting...")
    s1_lookup = s1_eval.set_index("entity_id")
    target_lookup = target_eval.set_index("entity_id")
    X = build_feature_matrix(pairs_df, s1_lookup, target_lookup)

    with open(PROJECT_ROOT / "models" / "lgbm_model.pkl", "rb") as f:
        saved = pickle.load(f)

    model = saved["model"] if isinstance(saved, dict) else saved
    threshold = saved.get("threshold", 0.4) if isinstance(saved, dict) else 0.4

    y_prob = model.predict_proba(X)[:, 1]
    is_match = y_prob >= threshold
    match_rows = pairs_df[is_match]
    print(f"  Threshold: {threshold}, Matches: {is_match.sum():,}")

    # ── 5. Score ──
    predictions = defaultdict(set)
    for s1_id, t_id in zip(match_rows["s1_id"], match_rows["target_id"]):
        predictions[s1_id].add(t_id)

    f05 = macro_f05(predictions, s1_to_true)

    tp = fp = fn = 0
    for s1_id, actual in s1_to_true.items():
        pred = predictions.get(s1_id, set())
        tp += len(pred & actual)
        fp += len(pred - actual)
        fn += len(actual - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    # Blocking recall ceiling
    blocking_tp = 0
    total_true = sum(len(v) for v in s1_to_true.values())
    for s1_id, actual in s1_to_true.items():
        blocked = {t for (s, t) in candidate_pairs if s == s1_id}
        blocking_tp += len(actual & blocked)
    blocking_recall = blocking_tp / total_true if total_true else 0

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS (5,000 S1 entities)")
    print(f"{'='*60}")
    print(f"  Candidate pairs:         {len(pairs_df):,}")
    print(f"  Predicted matches:       {is_match.sum():,}")
    print(f"  True matches in sample:  {total_true:,}")
    print(f"  {'─'*40}")
    print(f"  Blocking recall ceiling: {blocking_recall:.4f}")
    print(f"  Global precision:        {precision:.4f}")
    print(f"  Global recall:           {recall:.4f}")
    print(f"  MACRO F0.5:              {f05:.4f}  <- competition metric")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

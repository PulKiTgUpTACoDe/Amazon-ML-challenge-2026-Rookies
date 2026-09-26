"""
Train the final LightGBM model for entity resolution.

Steps:
  1. Load training data + ground truth
  2. Generate hard negatives via multi-strategy blocking on a sample
  3. Inject positive pairs from ground truth
  4. Extract 27+ string-similarity features
  5. 5-fold GroupKFold cross-validation with F0.5 threshold optimization
  6. Retrain on 100% of data → save model + threshold
"""
import sys
import time
import pickle
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from business_entity_resolution.data import (
    load_source, load_ground_truth, parse_matched_ids, TRAIN_DIR,
)
from business_entity_resolution.normalization import normalize_dataframe
from business_entity_resolution.blocking import block_multi_strategy
from business_entity_resolution.features import build_feature_matrix
from business_entity_resolution.evaluation import f05_single, macro_f05


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True)

    cols = ["entity_id", "business_name", "business_address", "country"]

    print("=" * 60)
    print("  Training LightGBM Entity Resolution Model")
    print("=" * 60)

    # ── 1. Load data ──
    print("\n[1/7] Loading training datasets...")
    t0 = time.time()
    s1 = load_source(TRAIN_DIR / "train_source1.tsv", usecols=cols)
    s2 = load_source(TRAIN_DIR / "train_source2.tsv", usecols=cols)
    s3 = load_source(TRAIN_DIR / "train_source3.tsv", usecols=cols)
    print(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}  ({time.time()-t0:.1f}s)")

    # ── 2. Normalize ──
    print("\n[2/7] Normalizing text fields...")
    t0 = time.time()
    s1 = normalize_dataframe(s1, cols)
    target_df = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    target_df = normalize_dataframe(target_df, cols)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 3. Load ground truth ──
    print("\n[3/7] Loading ground truth...")
    gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt_with_matches = gt[gt["matched_entity_ids"].str.len() > 0]
    print(f"  {len(gt):,} total, {len(gt_with_matches):,} with matches")

    gt_dict_full = {}
    for s1_id, match_str in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        gt_dict_full[s1_id] = set(parse_matched_ids(match_str))

    # ── 4. Generate training pairs ──
    print("\n[4/7] Generating training pairs (hard negatives + positives)...")

    # 4a. Sample S1 and target for blocking to get hard negatives
    s1_sample = s1.sample(n=min(30_000, len(s1)), random_state=42)
    target_sample = target_df.sample(n=min(1_000_000, len(target_df)), random_state=42)

    t0 = time.time()
    candidate_pairs = block_multi_strategy(
        s1_sample, target_sample, top_k=10, similarity_threshold=0.3,
    )
    del target_sample
    print(f"  Blocking: {len(candidate_pairs):,} hard-negative candidates ({time.time()-t0:.1f}s)")

    # 4b. Inject ground-truth positives
    gt_pairs = set()
    sampled_gt = gt_with_matches.sample(n=min(300_000, len(gt_with_matches)), random_state=42)
    for s1_id, match_str in zip(sampled_gt["source1_entity_id"], sampled_gt["matched_entity_ids"]):
        for m in parse_matched_ids(match_str):
            gt_pairs.add((s1_id, m))

    all_pairs_set = candidate_pairs | gt_pairs
    print(f"  Total pairs (hard neg + positives): {len(all_pairs_set):,}")

    pairs_df = pd.DataFrame(list(all_pairs_set), columns=["s1_id", "target_id"])

    # ── 5. Label and extract features ──
    print("\n[5/7] Labelling pairs and extracting features...")
    t0 = time.time()

    pairs_df["label"] = pairs_df.apply(
        lambda r: int(r["target_id"] in gt_dict_full.get(r["s1_id"], set())), axis=1,
    )
    print(f"  Labels:\n{pairs_df['label'].value_counts().to_string()}")

    s1_lookup = s1.set_index("entity_id")
    target_lookup = target_df.set_index("entity_id")

    X = build_feature_matrix(pairs_df, s1_lookup, target_lookup)
    y = pairs_df["label"]
    print(f"  Features: {X.shape[1]} columns, {len(X):,} rows ({time.time()-t0:.1f}s)")

    # ── 6. 5-Fold GroupKFold CV + threshold optimisation ──
    print("\n[6/7] 5-Fold GroupKFold cross-validation...")
    from sklearn.model_selection import GroupKFold
    import lightgbm as lgb

    gkf = GroupKFold(n_splits=5)
    oof_probs = np.zeros(len(pairs_df))

    lgb_params = dict(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=8,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        n_jobs=-1,
        verbosity=-1,
    )

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=pairs_df["s1_id"])):
        fold_model = lgb.LGBMClassifier(**lgb_params, random_state=42 + fold)
        fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_probs[va_idx] = fold_model.predict_proba(X.iloc[va_idx])[:, 1]
        print(f"  Fold {fold+1}/5 done")

    # Threshold sweep
    print("\n  Optimizing threshold on macro F0.5...")
    val_s1_ids = set(pairs_df["s1_id"])
    actuals = {s1_id: gt_dict_full.get(s1_id, set()) for s1_id in val_s1_ids}

    best_thr, best_f05 = 0.5, -1.0
    for thr in np.arange(0.25, 0.86, 0.02):
        preds = defaultdict(set)
        mask = oof_probs >= thr
        for s1_id, tid in zip(pairs_df.loc[mask, "s1_id"], pairs_df.loc[mask, "target_id"]):
            preds[s1_id].add(tid)
        score = macro_f05(preds, actuals)
        print(f"    thr={thr:.2f}  →  F0.5={score:.4f}")
        if score > best_f05:
            best_f05 = score
            best_thr = thr

    print(f"\n  >> Best threshold: {best_thr:.2f}  ->  F0.5 = {best_f05:.4f}")

    # ── 7. Retrain on 100% + save ──
    print("\n[7/7] Retraining on 100% of data...")
    final_model = lgb.LGBMClassifier(**lgb_params, random_state=42)
    final_model.fit(X, y)

    model_path = MODELS_DIR / "lgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": final_model, "threshold": best_thr}, f)
    print(f"  Saved model + threshold to {model_path}")

    # Feature importance
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\n  Feature Importances:")
    print(imp.to_string(index=False))

    print(f"\n{'='*60}")
    print(f"  Training complete!  Model: {model_path}")
    print(f"  Optimal threshold: {best_thr:.2f}  (F0.5 = {best_f05:.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
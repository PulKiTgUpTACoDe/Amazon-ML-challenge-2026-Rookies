import sys
import os
import time
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from business_entity_resolution.data import load_source, load_ground_truth, parse_matched_ids, TRAIN_DIR
from business_entity_resolution.normalization import normalize_dataframe
from business_entity_resolution.blocking import block_multi_strategy
from business_entity_resolution.features import build_feature_matrix
from business_entity_resolution.evaluation import f05_single

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    
    cols = ['entity_id', 'business_name', 'business_address', 'country']
    
    print("Loading full train datasets (this may take a minute)...")
    t0 = time.time()
    s1 = load_source(TRAIN_DIR / "train_source1.tsv", usecols=cols)
    s2 = load_source(TRAIN_DIR / "train_source2.tsv", usecols=cols)
    # We load S3 as well because we want positive pairs to extract features properly
    # (otherwise positive pairs to S3 will get NaNs and break the model)
    s3 = load_source(TRAIN_DIR / "train_source3.tsv", usecols=cols)
    print(f"Loaded {len(s1):,} S1 rows, {len(s2):,} S2 rows, {len(s3):,} S3 rows in {time.time()-t0:.2f}s")
    
    print("Normalizing strings...")
    t0 = time.time()
    s1 = normalize_dataframe(s1, cols)
    
    # Concatenate S2 and S3 for blocking and lookup to make it simple
    target_df = pd.concat([s2, s3], ignore_index=True)
    del s2, s3 # Free memory
    target_df = normalize_dataframe(target_df, cols)
    
    print(f"Normalized in {time.time()-t0:.2f}s")
    
    print("Loading Ground Truth...")
    gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt_with_matches = gt[gt['matched_entity_ids'].str.len() > 0]
    
    print("Generating Training Pairs...")
    # 1. Sample 20,000 S1 IDs for multi-strategy blocking to get hard negatives
    s1_sample = s1.sample(n=20000, random_state=42)
    
    # Sample target_df down to 2 Million rows for hard-negative generation
    target_sample = target_df.sample(n=min(500000, len(target_df)), random_state=42)
    
    # 2. Block using MULTI-STRATEGY against the sampled target
    t0 = time.time()
    candidate_pairs = block_multi_strategy(s1_sample, target_sample, top_k=10, similarity_threshold=0.3)
    del target_sample # Free memory immediately
    print(f"Multi-strategy blocking generated {len(candidate_pairs):,} candidates in {time.time()-t0:.2f}s")
    
    # 3. Add positive Ground Truth pairs to ensure we have lots of true matches
    gt_pairs = set()
    # Sample 200,000 positive matches from all GT (more data = better model)
    sampled_gt = gt_with_matches.sample(n=min(200000, len(gt_with_matches)), random_state=42)
    
    for s1_id, match_str in zip(sampled_gt['source1_entity_id'], sampled_gt['matched_entity_ids']):
        matches = parse_matched_ids(match_str)
        for m in matches:
            gt_pairs.add((s1_id, m))
            
    all_pairs_set = candidate_pairs | gt_pairs
    print(f"Total pairs for training (Positives + Hard Negatives): {len(all_pairs_set):,}")
    
    pairs_list = list(all_pairs_set)
    pairs_df = pd.DataFrame(pairs_list, columns=['s1_id', 'target_id'])
    
    # Convert GT to a fast lookup dictionary to assign labels
    print("Assigning labels...")
    gt_dict_full = {}
    for s1_id, match_str in zip(gt['source1_entity_id'], gt['matched_entity_ids']):
        gt_dict_full[s1_id] = set(parse_matched_ids(match_str))
        
    def is_match(row):
        return int(row['target_id'] in gt_dict_full.get(row['s1_id'], set()))
        
    pairs_df['label'] = pairs_df.apply(is_match, axis=1)
    print(f"Label distribution:\n{pairs_df['label'].value_counts()}")
    
    print("Extracting features (this is CPU intensive)...")
    t0 = time.time()
    
    s1_lookup = s1.set_index('entity_id')
    target_lookup = target_df.set_index('entity_id')
    
    X = build_feature_matrix(pairs_df, s1_lookup, target_lookup)
    y = pairs_df['label']
    print(f"Feature extraction took {time.time()-t0:.2f}s")
    
    print("Training LightGBM Model with Cross-Validation...")
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import collections
    
    # Hold out 20% for threshold optimization
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pairs_train = pairs_df.iloc[X_train.index]
    pairs_val = pairs_df.iloc[X_val.index]
    
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Optimize threshold on F0.5 using validation set
    print("\nOptimizing prediction threshold on F0.5...")
    y_val_prob = model.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.5
    best_f05 = 0.0
    
    for threshold in np.arange(0.2, 0.8, 0.05):
        preds = collections.defaultdict(set)
        actuals = collections.defaultdict(set)
        
        match_mask = y_val_prob >= threshold
        for s1_id, target_id, is_true in zip(pairs_val['s1_id'], pairs_val['target_id'], y_val):
            actuals[s1_id]  # ensure key exists
        
        for s1_id, target_id in zip(pairs_val.loc[match_mask, 's1_id'], pairs_val.loc[match_mask, 'target_id']):
            preds[s1_id].add(target_id)
            
        # Build actuals from gt_dict_full for these s1 entities
        val_s1_ids = set(pairs_val['s1_id'])
        actuals = {s1_id: gt_dict_full.get(s1_id, set()) for s1_id in val_s1_ids}
        
        scores = []
        for s1_id, actual in actuals.items():
            predicted = preds.get(s1_id, set())
            scores.append(f05_single(predicted, actual))
        f05 = sum(scores) / len(scores) if scores else 0.0
        
        print(f"  Threshold {threshold:.2f} → F0.5 = {f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = threshold
    
    print(f"\n  ★ Best Threshold: {best_threshold:.2f} with F0.5 = {best_f05:.4f}")
    
    # Retrain on 100% of data with the full model
    print("\nRetraining on 100% of data for final submission model...")
    model.fit(X, y)
    
    # Save the model AND the optimal threshold
    model_path = MODELS_DIR / "lgbm_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'threshold': best_threshold}, f)
        
    print(f"Model + threshold saved to {model_path}!")
    
    # Print feature importance
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importances:")
    print(importances.to_string(index=False))
    
if __name__ == "__main__":
    main()

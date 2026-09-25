import sys
import os
import time
import pickle
from pathlib import Path
import pandas as pd
import collections

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from business_entity_resolution.data import load_source, TRAIN_DIR, parse_matched_ids
from business_entity_resolution.normalization import normalize_dataframe
from business_entity_resolution.features import build_feature_matrix
from business_entity_resolution.evaluation import macro_f05
from business_entity_resolution.blocking import block_multi_strategy

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
    print("="*60)
    print("  Evaluating Pre-Trained Model (Multi-Strategy + Rich Features)")
    print("="*60)
    
    # 1. Load Ground truth
    print("Loading Ground Truth...")
    gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    
    # Take a sample of 5,000 S1 entities
    gt_sample = gt.sample(n=5000, random_state=42)
    
    # 2. Extract true matches for these 5000
    true_target_ids = set()
    s1_to_true_matches = {}
    for s1_id, match_str in zip(gt_sample['source1_entity_id'], gt_sample['matched_entity_ids']):
        matches = set(parse_matched_ids(match_str))
        s1_to_true_matches[s1_id] = matches
        true_target_ids.update(matches)
        
    print(f"Sampled 5,000 S1 entities. They have {len(true_target_ids)} true matches combined.")
    
    # 3. Load Data
    print("Loading Target data...")
    cols = ['entity_id', 'business_name', 'business_address', 'country']
    s1 = load_source(TRAIN_DIR / "train_source1.tsv", usecols=cols)
    s1_eval = s1[s1['entity_id'].isin(gt_sample['source1_entity_id'])].copy()
    s1_eval = normalize_dataframe(s1_eval, cols)
    
    s2 = load_source(TRAIN_DIR / "train_source2.tsv", usecols=cols)
    s3 = load_source(TRAIN_DIR / "train_source3.tsv", usecols=cols)
    target_all = pd.concat([s2, s3], ignore_index=True)
    
    # 4. Build realistic evaluation set
    target_true = target_all[target_all['entity_id'].isin(true_target_ids)]
    target_false = target_all[~target_all['entity_id'].isin(true_target_ids)].sample(n=500000, random_state=42)
    target_eval = pd.concat([target_true, target_false], ignore_index=True)
    target_eval = normalize_dataframe(target_eval, cols)
    
    print(f"\nRunning MULTI-STRATEGY blocking on 5,000 S1 queries against {len(target_eval):,} targets...")
    t0 = time.time()
    candidate_pairs = block_multi_strategy(s1_eval, target_eval, top_k=10, similarity_threshold=0.3)
    print(f"Multi-strategy blocking took {time.time()-t0:.1f}s")
    
    pairs_list = list(candidate_pairs)
    pairs_df = pd.DataFrame(pairs_list, columns=['s1_id', 'target_id'])
    print(f"Total candidate pairs: {len(pairs_df):,}")
    
    if len(pairs_df) == 0:
        print("No candidates found!")
        return
        
    print("Extracting features...")
    s1_lookup = s1_eval.set_index('entity_id')
    target_lookup = target_eval.set_index('entity_id')
    X = build_feature_matrix(pairs_df, s1_lookup, target_lookup)
    
    print("Loading Pre-Trained Model and Predicting...")
    with open(PROJECT_ROOT / "models" / "lgbm_model.pkl", 'rb') as f:
        saved = pickle.load(f)
    
    if isinstance(saved, dict):
        model = saved['model']
        threshold = saved.get('threshold', 0.4)
    else:
        model = saved
        threshold = 0.4
    
    print(f"Using threshold: {threshold}")
    y_prob = model.predict_proba(X)[:, 1]
    is_match = y_prob >= threshold
    match_rows = pairs_df[is_match]
    
    # Format predictions for scoring
    predictions = collections.defaultdict(set)
    for s1_id, target_id in zip(match_rows['s1_id'], match_rows['target_id']):
        predictions[s1_id].add(target_id)
        
    # Calculate Macro F0.5
    f05 = macro_f05(predictions, s1_to_true_matches)
    
    # Calculate Raw Precision / Recall
    tp = 0
    fp = 0
    fn = 0
    for s1_id, actual in s1_to_true_matches.items():
        predicted = predictions.get(s1_id, set())
        tp += len(predicted & actual)
        fp += len(predicted - actual)
        fn += len(actual - predicted)
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Blocking recall ceiling
    blocking_recall_tp = 0
    for s1_id, actual in s1_to_true_matches.items():
        blocked_targets = {t for (s, t) in candidate_pairs if s == s1_id}
        blocking_recall_tp += len(actual & blocked_targets)
    
    total_true = sum(len(v) for v in s1_to_true_matches.values())
    blocking_recall = blocking_recall_tp / total_true if total_true > 0 else 0
    
    print("\n" + "="*60)
    print("  EVALUATION RESULTS (On 5,000 S1 Entity Sample)")
    print("="*60)
    print(f"Total S1 Entities Evaluated: {len(s1_to_true_matches):,}")
    print(f"Total True Matches in Sample: {total_true:,}")
    print(f"Candidate Pairs Generated: {len(pairs_df):,}")
    print(f"Pairs Predicted as Matches: {len(match_rows):,}")
    print("-" * 60)
    print(f"Blocking Recall Ceiling: {blocking_recall:.4f}  (upper bound on recall)")
    print(f"Global Precision: {precision:.4f}")
    print(f"Global Recall:    {recall:.4f}")
    print(f"MACRO F0.5 SCORE: {f05:.4f}  <-- (Official Competition Metric)")
    print("="*60)

if __name__ == "__main__":
    main()

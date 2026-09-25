import sys
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.data import load_source, load_ground_truth, parse_matched_ids, TRAIN_DIR
from business_entity_resolution.normalization import normalize_dataframe
from business_entity_resolution.blocking import block_exact_match
from business_entity_resolution.features import build_feature_matrix
from business_entity_resolution.evaluation import macro_f05

def main():
    MINI_DIR = TRAIN_DIR.parent / "mini"
    
    # 1. Load Data
    print("Loading Data...")
    gt = load_ground_truth(MINI_DIR / "train_ground_truth.tsv")
    ground_truth_dict = {}
    for s1_id, match_str in zip(gt['source1_entity_id'], gt['matched_entity_ids']):
        ground_truth_dict[s1_id] = set(parse_matched_ids(match_str))
        
    cols = ['entity_id', 'business_name', 'business_address', 'country']
    
    s1 = load_source(MINI_DIR / "train_source1.tsv", usecols=cols)
    s1 = normalize_dataframe(s1, cols)
    s1['name_prefix_10'] = s1['business_name'].apply(lambda x: x[:10] if isinstance(x, str) else "")
    
    s2 = load_source(MINI_DIR / "train_source2.tsv", usecols=cols)
    s2 = normalize_dataframe(s2, cols)
    s2['name_prefix_10'] = s2['business_name'].apply(lambda x: x[:10] if isinstance(x, str) else "")
    
    s3 = load_source(MINI_DIR / "train_source3.tsv", usecols=cols)
    s3 = normalize_dataframe(s3, cols)
    s3['name_prefix_10'] = s3['business_name'].apply(lambda x: x[:10] if isinstance(x, str) else "")
    
    target_df = pd.concat([s2, s3], ignore_index=True)
    
    # 2. Blocking
    print("Running Blocking (name_prefix_10)...")
    candidate_pairs = block_exact_match(s1, target_df, 'name_prefix_10')
    print(f"Generated {len(candidate_pairs):,} candidates.")
    
    # 3. Inject ALL Ground Truth pairs for training
    # This guarantees we have positive examples to learn from
    gt_pairs = set()
    for s1_id, matches in ground_truth_dict.items():
        for m in matches:
            gt_pairs.add((s1_id, m))
            
    print(f"Injecting {len(gt_pairs):,} ground truth pairs...")
    all_pairs_set = candidate_pairs | gt_pairs
    
    # Create DataFrame of pairs
    pairs_list = list(all_pairs_set)
    pairs_df = pd.DataFrame(pairs_list, columns=['s1_id', 'target_id'])
    
    # Labels
    # Efficiently lookup if a pair is in GT
    pairs_df['label'] = pairs_df.apply(
        lambda row: 1 if row['target_id'] in ground_truth_dict.get(row['s1_id'], set()) else 0,
        axis=1
    )
    print(f"Dataset ready. Total pairs: {len(pairs_df):,}. Positive: {pairs_df['label'].sum():,}")
    
    # 4. Feature Extraction
    print("Extracting features (RapidFuzz)...")
    t0 = time.time()
    X = build_feature_matrix(pairs_df, s1, target_df)
    y = pairs_df['label']
    print(f"Feature extraction took {time.time() - t0:.1f}s.")
    
    # 5. Train/Test Split (Grouped by s1_id)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=pairs_df['s1_id']))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    pairs_train, pairs_test = pairs_df.iloc[train_idx], pairs_df.iloc[test_idx]
    
    # 6. Train Model
    print("Training LightGBM Classifier...")
    model = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 7. Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_test)
    
    # Compute Precision, Recall on pairs
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"Pairwise Precision: {precision:.4f}")
    print(f"Pairwise Recall:    {recall:.4f}")
    
    # Reconstruct predictions to format {s1_id: [matches]} to compute macro F0.5
    pairs_test = pairs_test.copy()
    pairs_test['pred'] = y_pred
    
    pred_matches = pairs_test[pairs_test['pred'] == 1].groupby('s1_id')['target_id'].apply(set).to_dict()
    
    # The actual ground truth for these S1 IDs (we should only evaluate on S1 IDs in the test set)
    # But wait, an S1 ID might be split across train and test. To do this perfectly, we should group by S1_id for train/test split.
    # For now, let's just compute macro F0.5 using the full GT but only penalizing what we predict.
    
    test_s1_ids = set(pairs_test['s1_id'])
    test_gt = {k: set(v) for k, v in ground_truth_dict.items() if k in test_s1_ids}
    
    score = macro_f05(test_gt, pred_matches)
    print(f"\nFinal Macro F0.5 Score on Test Set S1 IDs: {score:.4f}")
    
    # Feature Importance
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importances:")
    print(importances.to_string(index=False))

if __name__ == "__main__":
    main()

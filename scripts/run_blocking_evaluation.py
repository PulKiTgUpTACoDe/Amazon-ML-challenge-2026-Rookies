import sys
import time
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.data import load_source, load_ground_truth, parse_matched_ids, TRAIN_DIR
from business_entity_resolution.normalization import normalize_dataframe
from business_entity_resolution.blocking import block_exact_match, block_tfidf

def evaluate_blocking(predicted_pairs: set[tuple[str, str]], ground_truth_dict: dict[str, set[str]]):
    total_gt_pairs = 0
    recovered_gt_pairs = 0
    
    for s1_id, matches in ground_truth_dict.items():
        total_gt_pairs += len(matches)
        for m in matches:
            if (s1_id, m) in predicted_pairs:
                recovered_gt_pairs += 1
                
    recall = recovered_gt_pairs / total_gt_pairs if total_gt_pairs > 0 else 0
    return recall, len(predicted_pairs), total_gt_pairs

def main():
    print("Loading GT...")
    t0 = time.time()
    MINI_DIR = TRAIN_DIR.parent / "mini"
    gt = load_ground_truth(MINI_DIR / "train_ground_truth.tsv")
    
    # Prepare GT
    ground_truth_dict = {}
    for s1_id, match_str in zip(gt['source1_entity_id'], gt['matched_entity_ids']):
        ground_truth_dict[s1_id] = set(parse_matched_ids(match_str))
        
    MINI_DIR = TRAIN_DIR.parent / "mini"
    
    cols = ['entity_id', 'business_name', 'business_address']
    
    print("Loading and normalizing S1...")
    s1 = load_source(MINI_DIR / "train_source1.tsv", usecols=cols)
    s1 = normalize_dataframe(s1, cols)
    s1['name_prefix_10'] = s1['business_name'].apply(lambda x: x[:10] if isinstance(x, str) else "")
    s1['name_first_word'] = s1['business_name'].apply(lambda x: x.split()[0] if isinstance(x, str) and x.split() else "")
    
    exact_strategies = [
        'business_name',
        'name_prefix_10',
        'name_first_word',
    ]
    
    # Process S2
    print("Loading and processing S2...")
    s2 = load_source(MINI_DIR / "train_source2.tsv", usecols=cols)
    s2 = normalize_dataframe(s2, cols)
    s2['name_prefix_10'] = s2['business_name'].apply(lambda x: x[:10] if isinstance(x, str) else "")
    s2['name_first_word'] = s2['business_name'].apply(lambda x: x.split()[0] if isinstance(x, str) and x.split() else "")
    
    pairs_s2_by_strat = {}
    for strategy in exact_strategies:
        pairs_s2_by_strat[strategy] = block_exact_match(s1, s2, strategy)
        
    print("Running TF-IDF blocking for S1 -> S2...")
    t_start = time.time()
    pairs_s2_by_strat['tfidf_business_name'] = block_tfidf(s1, s2, 'business_name', top_k=5, similarity_threshold=0.3)
    print(f"TF-IDF S2 took {time.time() - t_start:.2f}s")
        
    del s2 # Free memory
    
    # Process S3
    print("\nLoading and processing S3...")
    s3 = load_source(MINI_DIR / "train_source3.tsv", usecols=cols)
    s3 = normalize_dataframe(s3, cols)
    s3['name_prefix_10'] = s3['business_name'].apply(lambda x: x[:10] if isinstance(x, str) else "")
    s3['name_first_word'] = s3['business_name'].apply(lambda x: x.split()[0] if isinstance(x, str) and x.split() else "")
    
    all_strategies = exact_strategies + ['tfidf_business_name']
    
    for strategy in all_strategies:
        print(f"\nEvaluating strategy: {strategy}")
        if strategy == 'tfidf_business_name':
            t_start = time.time()
            pairs_s3 = block_tfidf(s1, s3, 'business_name', top_k=5, similarity_threshold=0.3)
            print(f"TF-IDF S3 took {time.time() - t_start:.2f}s")
        else:
            pairs_s3 = block_exact_match(s1, s3, strategy)
            
        all_pairs = pairs_s2_by_strat[strategy] | pairs_s3
        
        recall, num_candidates, num_gt = evaluate_blocking(all_pairs, ground_truth_dict)
        print(f"  Candidate Set Size: {num_candidates:,}")
        print(f"  Candidate Recall:   {recall:.4f} ({recall*100:.2f}%)")
        ratio = num_candidates/num_gt if num_gt > 0 else 0
        print(f"  Ratio (Pairs/GT):   {ratio:.2f}x")
        
        # Free memory
        del all_pairs
        del pairs_s3
        pairs_s2_by_strat[strategy] = None

if __name__ == "__main__":
    main()


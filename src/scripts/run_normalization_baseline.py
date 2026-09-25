import sys
import time
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.data import (
    load_source, load_ground_truth, parse_matched_ids
)
from business_entity_resolution.normalization import normalize_dataframe
from business_entity_resolution.evaluation import macro_f05

def main():
    print("Loading data...")
    t0 = time.time()
    from business_entity_resolution.data import load_train_sources
    s1, s2, s3 = load_train_sources()
    gt = load_ground_truth()
    print(f"Loaded in {time.time() - t0:.2f}s")
    
    # We only need the 'business_name' and 'entity_id' for this baseline
    s1 = s1[['entity_id', 'business_name']].copy()
    s2 = s2[['entity_id', 'business_name']].copy()
    s3 = s3[['entity_id', 'business_name']].copy()

    print("Normalizing names...")
    t0 = time.time()
    s1 = normalize_dataframe(s1, ['business_name'])
    s2 = normalize_dataframe(s2, ['business_name'])
    s3 = normalize_dataframe(s3, ['business_name'])
    print(f"Normalized in {time.time() - t0:.2f}s")
    
    # Create name -> list of S1 entity_ids mapping
    print("Building index...")
    t0 = time.time()
    name_to_s1 = defaultdict(list)
    # Only map names that are not empty
    for entity_id, name in zip(s1['entity_id'], s1['business_name']):
        if name:
            name_to_s1[name].append(entity_id)
            
    # Resolve matches from S2 and S3 to S1
    print("Finding matches...")
    predicted_matches = defaultdict(set)
    
    def process_source(df):
        for entity_id, name in zip(df['entity_id'], df['business_name']):
            if name and name in name_to_s1:
                # If the name maps to S1 entities, link them
                for s1_id in name_to_s1[name]:
                    predicted_matches[s1_id].add(entity_id)

    process_source(s2)
    process_source(s3)
    print(f"Matched in {time.time() - t0:.2f}s")
    
    print("Evaluating...")
    t0 = time.time()
    # Prepare ground truth dictionary
    ground_truth_dict = {}
    for s1_id, match_str in zip(gt['source1_entity_id'], gt['matched_entity_ids']):
        ground_truth_dict[s1_id] = set(parse_matched_ids(match_str))
        
    score = macro_f05(predicted_matches, ground_truth_dict)
    print(f"Evaluated in {time.time() - t0:.2f}s")
    
    print("\n--- RESULTS ---")
    print(f"Macro F0.5: {score:.4f}")
    
if __name__ == "__main__":
    main()

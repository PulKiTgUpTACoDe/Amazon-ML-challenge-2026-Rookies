import sys
import os
import time
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import collections

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from business_entity_resolution.data import load_source, TRAIN_DIR, TEST_DIR
from business_entity_resolution.normalization import normalize_dataframe
from business_entity_resolution.features import extract_features_for_pairs

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="test", help="Dataset to process")
    parser.add_argument("--threshold", type=float, default=0.4, help="Prediction threshold")
    args = parser.parse_args()
    
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODELS_DIR = PROJECT_ROOT / "models"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    
    data_dir = TEST_DIR if args.mode == "test" else TRAIN_DIR
    candidates_file = OUTPUT_DIR / "candidate_pairs.tsv"
    out_file = OUTPUT_DIR / "matching_results.tsv"
    
    if not candidates_file.exists():
        print(f"Error: {candidates_file} does not exist. Run generate_candidates.py first.")
        return
        
    print(f"Step 1: Loading Data into Memory ({args.mode})...")
    cols = ['entity_id', 'business_name', 'business_address', 'country']
    
    t0 = time.time()
    s1 = load_source(data_dir / f"{args.mode}_source1.tsv", usecols=cols)
    s2 = load_source(data_dir / f"{args.mode}_source2.tsv", usecols=cols)
    s3 = load_source(data_dir / f"{args.mode}_source3.tsv", usecols=cols)
    
    s1 = normalize_dataframe(s1, cols)
    s2 = normalize_dataframe(s2, cols)
    s3 = normalize_dataframe(s3, cols)
    
    target_df = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    
    s1_lookup = s1.set_index('entity_id')
    target_lookup = target_df.set_index('entity_id')
    print(f"Data loaded and indexed in {time.time()-t0:.2f}s")
    
    print("Step 2: Loading Model...")
    with open(MODELS_DIR / "lgbm_model.pkl", 'rb') as f:
        model = pickle.load(f)
        
    print("Step 3: Streaming Candidates and Predicting...")
    
    matches_dict = collections.defaultdict(list)
    chunk_size = 500000
    
    chunk_iter = pd.read_csv(candidates_file, sep="\t", chunksize=chunk_size, dtype=str)
    
    t_start = time.time()
    total_processed = 0
    total_matches = 0
    
    for i, chunk in enumerate(chunk_iter):
        t_chunk = time.time()
        
        # Extract features
        X = extract_features_for_pairs(chunk, s1_lookup, target_lookup)
        
        # Predict
        y_prob = model.predict_proba(X)[:, 1]
        
        # Filter matches
        is_match = y_prob >= args.threshold
        match_rows = chunk[is_match]
        
        # Accumulate matches
        for s1_id, target_id in zip(match_rows['s1_id'], match_rows['target_id']):
            matches_dict[s1_id].append(target_id)
            
        total_processed += len(chunk)
        total_matches += len(match_rows)
        
        print(f"  Chunk {i+1} ({total_processed:,} total pairs) - Found {len(match_rows):,} matches in {time.time()-t_chunk:.2f}s")
        
    print(f"Prediction complete in {time.time()-t_start:.2f}s. Total matches found: {total_matches:,}")
    
    print("Step 4: Writing Final Results...")
    
    # We must output EVERY ID from S1, even if it has no matches (empty string)
    out_lines = []
    out_lines.append("source1_entity_id\tmatched_entity_ids\n")
    
    for s1_id in s1_lookup.index:
        matched_ids = matches_dict.get(s1_id, [])
        joined_matches = "|".join(matched_ids) if matched_ids else ""
        out_lines.append(f"{s1_id}\t{joined_matches}\n")
        
    with open(out_file, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
        
    print(f"Success! Final predictions saved to {out_file}")

if __name__ == "__main__":
    main()

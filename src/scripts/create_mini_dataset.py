import sys
import os
from pathlib import Path
import pandas as pd
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.data import TRAIN_DIR, parse_matched_ids

def main():
    MINI_DIR = TRAIN_DIR.parent / "mini"
    MINI_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading GT...")
    gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    
    # Filter to only those with matches
    gt_with_matches = gt[gt['matched_entity_ids'].str.len() > 0]
    
    # Sample 10,000 S1 entities (about 10% of those with matches)
    sampled_gt = gt_with_matches.sample(n=10000, random_state=42)
    s1_target_ids = set(sampled_gt['source1_entity_id'])
    
    # Collect all targeted S2/S3 IDs
    target_s2_s3 = set()
    for match_str in sampled_gt['matched_entity_ids']:
        for m in parse_matched_ids(match_str):
            target_s2_s3.add(m)
            
    print(f"Sampled 10,000 S1 entities, containing {len(target_s2_s3)} S2/S3 matches.")
    
    # Add some noise S1, S2, S3
    # We will just stream through the files and pick the targets, plus some random chance.
    
    def process_file(source_filename, target_ids, keep_prob):
        path = TRAIN_DIR / source_filename
        out_path = MINI_DIR / source_filename
        print(f"Processing {source_filename}...")
        
        kept = 0
        with open(path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8', newline='\n') as fout:
            header = fin.readline()
            fout.write(header)
            
            for line in fin:
                eid = line.split('\t')[0]
                if eid in target_ids or random.random() < keep_prob:
                    fout.write(line)
                    kept += 1
        print(f"  Kept {kept} rows in {out_path}")

    # Process S1 (keep target matches + 1% noise)
    process_file("train_source1.tsv", s1_target_ids, 0.01)
    
    # Process S2 (keep target matches + 1% noise)
    process_file("train_source2.tsv", target_s2_s3, 0.01)
    
    # Process S3 (keep target matches + 1% noise)
    process_file("train_source3.tsv", target_s2_s3, 0.01)
    
    # Save the GT
    sampled_gt.to_csv(MINI_DIR / "train_ground_truth.tsv", sep="\t", index=False)
    print("Mini dataset creation complete.")

if __name__ == "__main__":
    main()

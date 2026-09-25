import sys
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.data import load_source, TRAIN_DIR, TEST_DIR
from business_entity_resolution.normalization import normalize_dataframe

def chunked_tfidf_blocking(s1: pd.DataFrame, target_df: pd.DataFrame, 
                          out_file: str, vectorizer: TfidfVectorizer,
                          column: str = 'business_name', 
                          top_k: int = 5, similarity_threshold: float = 0.4):
    """
    Performs memory-safe chunked sparse dot product blocking and appends to out_file.
    """
    s1_texts = s1[column].fillna("").astype(str)
    target_texts = target_df[column].fillna("").astype(str)
    
    print(f"  Transforming S1 (size {len(s1_texts)}) and Target (size {len(target_texts)}) to sparse matrices...")
    t0 = time.time()
    X1 = vectorizer.transform(s1_texts)
    X2 = vectorizer.transform(target_texts)
    print(f"  Shapes: X1={X1.shape}, X2={X2.shape}. Took {time.time()-t0:.2f}s")
    
    s1_ids = s1['entity_id'].values
    target_ids = target_df['entity_id'].values
    
    # 100 chunk size keeps it < 1GB
    chunk_size = 100 
    print(f"  Calculating sparse dot products (chunk size {chunk_size}) and writing to disk...")
    
    total_pairs_written = 0
    with open(out_file, 'a', encoding='utf-8') as f:
        for start_idx in range(0, X1.shape[0], chunk_size):
            t_chunk = time.time()
            end_idx = min(start_idx + chunk_size, X1.shape[0])
            X1_chunk = X1[start_idx:end_idx]
            
            similarity_chunk = X1_chunk.dot(X2.T)
            
            indptr = similarity_chunk.indptr
            indices = similarity_chunk.indices
            data = similarity_chunk.data
            
            pairs_chunk = []
            for i in range(similarity_chunk.shape[0]):
                start_ptr = indptr[i]
                end_ptr = indptr[i+1]
                if start_ptr == end_ptr:
                    continue
                    
                row_indices = indices[start_ptr:end_ptr]
                row_data = data[start_ptr:end_ptr]
                
                valid_mask = row_data >= similarity_threshold
                valid_indices = row_indices[valid_mask]
                valid_scores = row_data[valid_mask]
                
                if len(valid_indices) == 0:
                    continue
                    
                if len(valid_scores) > top_k:
                    idx = np.argpartition(-valid_scores, top_k - 1)[:top_k]
                    top_indices = valid_indices[idx]
                else:
                    top_indices = valid_indices
                
                s1_id = s1_ids[start_idx + i]
                for t_idx in top_indices:
                    pairs_chunk.append(f"{s1_id}\t{target_ids[t_idx]}\n")
            
            f.writelines(pairs_chunk)
            total_pairs_written += len(pairs_chunk)
            print(f"    Chunk {start_idx} to {end_idx} done in {time.time()-t_chunk:.2f}s. Wrote {len(pairs_chunk)} pairs.")
            
    return total_pairs_written

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="test", help="Dataset to process")
    args = parser.parse_args()
    
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    OUTPUT_DIR = PROJECT_ROOT / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    data_dir = TEST_DIR if args.mode == "test" else TRAIN_DIR
    out_file = OUTPUT_DIR / "candidate_pairs.tsv"
    
    # Clear the file if it exists
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("s1_id\ttarget_id\n")
        
    cols = ['entity_id', 'business_name', 'business_address', 'country']
    
    # 1. Fit Vectorizer on a sample of S1 to avoid holding all strings
    print(f"Step 1: Fitting TF-IDF Vectorizer on a sample of S1 ({args.mode})...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    
    # Load just 500k rows of S1 for fitting the vocabulary
    s1_sample = pd.read_csv(data_dir / f"{args.mode}_source1.tsv", sep="\t", usecols=['business_name'], nrows=500000, dtype=str, keep_default_na=False)
    vectorizer.fit(s1_sample['business_name'])
    del s1_sample
    print(f"Vectorizer fitted in {time.time()-t0:.2f}s")
    
    # 2. Process in Target Chunks
    # S1 is small enough (2-4M rows) that we can load it fully into Pandas (takes ~1GB).
    print(f"Step 2: Loading S1 ({args.mode})...")
    s1 = load_source(data_dir / f"{args.mode}_source1.tsv", usecols=cols)
    s1 = normalize_dataframe(s1, cols)
    
    total_candidates = 0
    
    # Process S2 and S3 one by one to save memory
    for source_idx in [2, 3]:
        filename = f"{args.mode}_source{source_idx}.tsv"
        print(f"\n--- Processing Target File: {filename} ---")
        
        # We read the target file in chunks using pandas
        chunk_iter = pd.read_csv(data_dir / filename, sep="\t", usecols=cols, dtype=str, keep_default_na=False, chunksize=1000000)
        
        for i, target_chunk in enumerate(chunk_iter):
            print(f"\n  >> Target Chunk {i+1} (1 Million Rows)")
            target_chunk = normalize_dataframe(target_chunk, cols)
            
            # Run blocking between ALL of S1 and THIS chunk of Target
            pairs_written = chunked_tfidf_blocking(
                s1, target_chunk, 
                out_file=out_file, 
                vectorizer=vectorizer,
                top_k=5, 
                similarity_threshold=0.4
            )
            total_candidates += pairs_written
            
            del target_chunk # Free memory eagerly
            
    print(f"\nSuccess! Wrote {total_candidates:,} candidate pairs to {out_file}")

if __name__ == "__main__":
    main()

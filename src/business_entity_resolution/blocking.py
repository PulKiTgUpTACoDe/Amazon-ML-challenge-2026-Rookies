import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

def block_exact_match(s1: pd.DataFrame, target_df: pd.DataFrame, column: str) -> set[tuple[str, str]]:
    """
    Blocks S1 against target_df based on exact match of `column`.
    Returns a set of tuples (s1_id, target_id).
    Skips empty/NaN values.
    """
    val_to_s1 = defaultdict(list)
    for eid, val in zip(s1['entity_id'], s1[column]):
        if val and pd.notna(val) and str(val).strip():
            val_to_s1[val].append(eid)
            
    candidate_pairs = set()
    for eid, val in zip(target_df['entity_id'], target_df[column]):
        if val and pd.notna(val) and val in val_to_s1:
            for s1_id in val_to_s1[val]:
                candidate_pairs.add((s1_id, eid))
                
    return candidate_pairs


def block_tfidf(s1: pd.DataFrame, target_df: pd.DataFrame, column: str, 
                top_k: int = 5, similarity_threshold: float = 0.5) -> set[tuple[str, str]]:
    """
    Blocks S1 against target_df using TF-IDF character n-grams and cosine similarity.
    Returns a set of tuples (s1_id, target_id).
    Skips empty/NaN values.
    """
    import time
    # 1. Prepare texts
    s1_texts = s1[column].fillna("").astype(str)
    target_texts = target_df[column].fillna("").astype(str)
    
    # 2. Fit vectorizer on a sample to save memory
    print("  Fitting TF-IDF Vectorizer on a sample...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    # A sample of 500k strings is more than enough to capture all common char n-grams
    sample_size = min(500000, len(s1_texts))
    sample_texts = s1_texts.sample(n=sample_size, random_state=42)
    vectorizer.fit(sample_texts)
    print(f"  Fitting took {time.time()-t0:.2f}s")
    
    print("  Transforming to sparse matrices...")
    t0 = time.time()
    X1 = vectorizer.transform(s1_texts)
    
    # Transform in chunks to prevent MemoryError from Scikit-learn internal dicts
    import scipy.sparse as sp
    X2_chunks = []
    chunk_size_transform = 500000
    for i in range(0, len(target_texts), chunk_size_transform):
        X2_chunks.append(vectorizer.transform(target_texts.iloc[i:i+chunk_size_transform]))
    X2 = sp.vstack(X2_chunks)
    
    print(f"  Shapes: X1={X1.shape}, X2={X2.shape}. Took {time.time()-t0:.2f}s")
    
    # Pre-fetch numpy arrays of IDs for fast lookup
    s1_ids = s1['entity_id'].values
    target_ids = target_df['entity_id'].values
    candidate_pairs = set()

    # Chunk over S1 to save memory during sparse dot product.
    # We must use a very small chunk size (100) 
    # produces 8.5 billion non-zeros, requiring ~68GB of RAM. 
    # A chunk size of 100 drops the peak memory for the dot product result to < 1.5GB.
    chunk_size = 100 
    print(f"  Calculating sparse dot products (chunk size {chunk_size})...")
    for start_idx in range(0, X1.shape[0], chunk_size):
        t_chunk = time.time()
        end_idx = min(start_idx + chunk_size, X1.shape[0])
        X1_chunk = X1[start_idx:end_idx]
        
        # Calculate cosine similarity
        similarity_chunk = X1_chunk.dot(X2.T)
        
        # Fast CSR iteration
        indptr = similarity_chunk.indptr
        indices = similarity_chunk.indices
        data = similarity_chunk.data
        
        for i in range(similarity_chunk.shape[0]):
            start_ptr = indptr[i]
            end_ptr = indptr[i+1]
            if start_ptr == end_ptr:
                continue
                
            row_indices = indices[start_ptr:end_ptr]
            row_data = data[start_ptr:end_ptr]
            
            # Filter by threshold
            valid_mask = row_data >= similarity_threshold
            valid_indices = row_indices[valid_mask]
            valid_scores = row_data[valid_mask]
            
            if len(valid_indices) == 0:
                continue
                
            # Take top_k efficiently using argpartition
            if len(valid_scores) > top_k:
                idx = np.argpartition(-valid_scores, top_k - 1)[:top_k]
                top_indices = valid_indices[idx]
            else:
                top_indices = valid_indices
            
            s1_id = s1_ids[start_idx + i]
            for t_idx in top_indices:
                candidate_pairs.add((s1_id, target_ids[t_idx]))
                
        print(f"    Chunk {start_idx} to {end_idx} done in {time.time()-t_chunk:.2f}s")
                
    return candidate_pairs

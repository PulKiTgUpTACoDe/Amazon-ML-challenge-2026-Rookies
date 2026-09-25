import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import time

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


GENERIC_BUSINESS_TOKENS = {
    'incorporated', 'corporation', 'company', 'limited', 'liability', 'partnership',
    'enterprises', 'services', 'solutions', 'international', 'national', 'industries',
    'pharmaceuticals', 'laboratories', 'construction', 'consulting', 'holdings', 'systems',
    'management', 'communications', 'distribution', 'private', 'public', 'group',
    'inc', 'llc', 'ltd', 'corp', 'co', 'llp', 'plc', 'pvt', 'intl', 'natl', 'svcs',
    'svc', 'mfg', 'assoc', 'assn', 'grp', 'hldgs', 'entr', 'tech', 'technol', 'soln',
    'solns', 'sys', 'mgmt', 'comm', 'dist', 'indus', 'pharm', 'pharma', 'lab', 'labs',
    'fin', 'finl', 'consult', 'constr', 'dba', 'and', 'the', 'for', 'with', 'store', 'shop'
}

def block_token_overlap(s1: pd.DataFrame, target_df: pd.DataFrame, column: str,
                        min_shared_tokens: int = 2) -> set[tuple[str, str]]:
    """
    Inverted-index blocking: builds a token → entity_id index for target,
    then for each S1 entity, finds targets sharing >= min_shared_tokens.
    Catches matches that TF-IDF misses due to short names or abbreviation differences.
    """
    # Build inverted index: token -> set of (target_idx)
    token_to_targets = defaultdict(set)
    target_ids = target_df['entity_id'].values
    target_names = target_df[column].fillna("").astype(str).values
    
    for idx, name in enumerate(target_names):
        tokens = set(name.split())
        # Only index meaningful non-generic tokens (length > 2 and not in stop-list)
        for token in tokens:
            if len(token) > 2 and token.lower() not in GENERIC_BUSINESS_TOKENS:
                token_to_targets[token].add(idx)
    
    candidate_pairs = set()
    s1_ids = s1['entity_id'].values
    s1_names = s1[column].fillna("").astype(str).values
    
    for s1_idx, name in enumerate(s1_names):
        tokens = set(t for t in name.split() if len(t) > 2 and t.lower() not in GENERIC_BUSINESS_TOKENS)
        if not tokens:
            continue
            
        # Count how many tokens each target shares with this S1 entity
        target_counts = defaultdict(int)
        for token in tokens:
            for t_idx in token_to_targets.get(token, set()):
                target_counts[t_idx] += 1
        
        s1_id = s1_ids[s1_idx]
        for t_idx, count in target_counts.items():
            if count >= min_shared_tokens:
                candidate_pairs.add((s1_id, target_ids[t_idx]))
    
    return candidate_pairs


def _soundex(name: str) -> str:
    """Simple Soundex implementation for phonetic blocking."""
    if not name or not isinstance(name, str):
        return ""
    name = name.upper()
    # Keep first letter
    # Remove non-alpha
    name = ''.join(c for c in name if c.isalpha())
    if not name:
        return ""
    
    first_letter = name[0]
    coding = {'B': '1', 'F': '1', 'P': '1', 'V': '1',
              'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
              'D': '3', 'T': '3',
              'L': '4',
              'M': '5', 'N': '5',
              'R': '6'}
    
    coded = first_letter
    prev_code = coding.get(first_letter, '0')
    for char in name[1:]:
        code = coding.get(char, '0')
        if code != '0' and code != prev_code:
            coded += code
        prev_code = code
        if len(coded) == 4:
            break
    
    return coded.ljust(4, '0')


def block_phonetic(s1: pd.DataFrame, target_df: pd.DataFrame, column: str) -> set[tuple[str, str]]:
    """
    Phonetic blocking using Soundex on the first word of the business name.
    Catches matches where names sound alike but are spelled differently.
    """
    # Build index: soundex_code -> list of target indices
    soundex_to_targets = defaultdict(list)
    target_ids = target_df['entity_id'].values
    target_names = target_df[column].fillna("").astype(str).values
    
    for idx, name in enumerate(target_names):
        first_word = name.split()[0] if name.split() else ""
        code = _soundex(first_word)
        if code:
            soundex_to_targets[code].append(idx)
    
    candidate_pairs = set()
    s1_ids = s1['entity_id'].values
    s1_names = s1[column].fillna("").astype(str).values
    
    for s1_idx, name in enumerate(s1_names):
        first_word = name.split()[0] if name.split() else ""
        code = _soundex(first_word)
        if not code:
            continue
        
        # Soundex blocks can be huge; limit to prevent memory explosion
        targets = soundex_to_targets.get(code, [])
        if len(targets) > 1000:
            continue  # Too generic a block, skip
            
        s1_id = s1_ids[s1_idx]
        for t_idx in targets:
            candidate_pairs.add((s1_id, target_ids[t_idx]))
    
    return candidate_pairs


def block_tfidf(s1: pd.DataFrame, target_df: pd.DataFrame, column: str, 
                top_k: int = 5, similarity_threshold: float = 0.5) -> set[tuple[str, str]]:
    """
    Blocks S1 against target_df using TF-IDF character n-grams and cosine similarity.
    Returns a set of tuples (s1_id, target_id).
    """
    # 1. Prepare texts
    s1_texts = s1[column].fillna("").astype(str)
    target_texts = target_df[column].fillna("").astype(str)
    
    # 2. Fit vectorizer on a sample to save memory
    print("  Fitting TF-IDF Vectorizer on a sample...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    sample_size = min(500000, len(s1_texts))
    sample_texts = s1_texts.sample(n=sample_size, random_state=42)
    vectorizer.fit(sample_texts)
    print(f"  Fitting took {time.time()-t0:.2f}s")
    
    print("  Transforming to sparse matrices...")
    t0 = time.time()
    X1 = vectorizer.transform(s1_texts)
    
    # Transform in chunks to prevent MemoryError
    import scipy.sparse as sp
    X2_chunks = []
    chunk_size_transform = 500000
    for i in range(0, len(target_texts), chunk_size_transform):
        X2_chunks.append(vectorizer.transform(target_texts.iloc[i:i+chunk_size_transform]))
    X2 = sp.vstack(X2_chunks)
    
    print(f"  Shapes: X1={X1.shape}, X2={X2.shape}. Took {time.time()-t0:.2f}s")
    
    s1_ids = s1['entity_id'].values
    target_ids = target_df['entity_id'].values
    candidate_pairs = set()

    chunk_size = 100 
    print(f"  Calculating sparse dot products (chunk size {chunk_size})...")
    for start_idx in range(0, X1.shape[0], chunk_size):
        t_chunk = time.time()
        end_idx = min(start_idx + chunk_size, X1.shape[0])
        X1_chunk = X1[start_idx:end_idx]
        
        similarity_chunk = X1_chunk.dot(X2.T)
        
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
                candidate_pairs.add((s1_id, target_ids[t_idx]))
                
        print(f"    Chunk {start_idx} to {end_idx} done in {time.time()-t_chunk:.2f}s")
                
    return candidate_pairs


def block_multi_strategy(s1: pd.DataFrame, target_df: pd.DataFrame,
                         top_k: int = 10, similarity_threshold: float = 0.3) -> set[tuple[str, str]]:
    """
    Union of multiple blocking strategies to maximize recall ceiling.
    Each strategy catches different types of matches that others miss.
    """
    all_pairs = set()
    
    # Strategy 1: TF-IDF on business_name (catches fuzzy name matches)
    print("\n[Strategy 1/4] TF-IDF blocking on business_name...")
    t0 = time.time()
    pairs_name = block_tfidf(s1, target_df, 'business_name', top_k=top_k, similarity_threshold=similarity_threshold)
    all_pairs |= pairs_name
    print(f"  → {len(pairs_name):,} pairs. Total: {len(all_pairs):,}. Took {time.time()-t0:.1f}s")
    
    # Strategy 2: TF-IDF on business_address (catches same-address different-name)
    has_address = target_df['business_address'].fillna("").str.len() > 3
    if has_address.sum() > 1000:
        print("\n[Strategy 2/4] TF-IDF blocking on business_address...")
        t0 = time.time()
        # Only block targets that actually have addresses
        target_with_addr = target_df[has_address].copy()
        s1_with_addr = s1[s1['business_address'].fillna("").str.len() > 3].copy()
        if len(s1_with_addr) > 0 and len(target_with_addr) > 0:
            pairs_addr = block_tfidf(s1_with_addr, target_with_addr, 'business_address', 
                                     top_k=5, similarity_threshold=0.4)
            all_pairs |= pairs_addr
            print(f"  → {len(pairs_addr):,} pairs. Total: {len(all_pairs):,}. Took {time.time()-t0:.1f}s")
    else:
        print("\n[Strategy 2/4] Skipped (target has no addresses)")
    
    # Strategy 3: Token overlap blocking on business_name (catches word-reordering)
    print("\n[Strategy 3/4] Token overlap blocking on business_name...")
    t0 = time.time()
    pairs_token = block_token_overlap(s1, target_df, 'business_name', min_shared_tokens=2)
    all_pairs |= pairs_token
    print(f"  → {len(pairs_token):,} pairs. Total: {len(all_pairs):,}. Took {time.time()-t0:.1f}s")
    
    # Strategy 4: Exact match on business_name (fastest, catches trivial matches)
    print("\n[Strategy 4/4] Exact match blocking on business_name...")
    t0 = time.time()
    pairs_exact = block_exact_match(s1, target_df, 'business_name')
    all_pairs |= pairs_exact
    print(f"  → {len(pairs_exact):,} pairs. Total: {len(all_pairs):,}. Took {time.time()-t0:.1f}s")
    
    return all_pairs


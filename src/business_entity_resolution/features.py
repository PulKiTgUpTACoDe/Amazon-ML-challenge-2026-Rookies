import pandas as pd
import numpy as np
from rapidfuzz import fuzz, distance

def compute_string_features(s1_series: pd.Series, s2_series: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Compute rich string similarity features between two Series of strings.
    Handles NaN values gracefully. Returns 11 features per field pair.
    """
    s1 = s1_series.fillna("").astype(str)
    s2 = s2_series.fillna("").astype(str)
    
    # 1. Exact match
    exact_match = (s1 == s2).astype(int)
    
    # 2. Jaro-Winkler similarity
    jw = [distance.JaroWinkler.normalized_similarity(a, b) for a, b in zip(s1, s2)]
    
    # 3. Levenshtein ratio
    lev = [fuzz.ratio(a, b) / 100.0 for a, b in zip(s1, s2)]
    
    # 4. Token set ratio (handles word reordering)
    token_set = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(s1, s2)]
    
    # 5. Token sort ratio (sorts tokens then compares)
    token_sort = [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1, s2)]
    
    # 6. Partial ratio (best substring match — critical for abbreviations)
    partial = [fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(s1, s2)]
    
    # 7. Length difference
    len_diff = np.abs(s1.str.len() - s2.str.len())
    
    # 8. Length ratio (handles very different length strings)
    len_s1 = s1.str.len().replace(0, 1)  # Avoid division by zero
    len_s2 = s2.str.len().replace(0, 1)
    len_ratio = np.minimum(len_s1, len_s2) / np.maximum(len_s1, len_s2)
    
    # 9. Token Jaccard similarity (word-level overlap)
    def token_jaccard(a, b):
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return intersection / union if union > 0 else 0.0
    
    jaccard = [token_jaccard(a, b) for a, b in zip(s1, s2)]
    
    # 10. Shared token count
    def shared_tokens(a, b):
        return len(set(a.split()) & set(b.split()))
    
    shared = [shared_tokens(a, b) for a, b in zip(s1, s2)]
    
    # 11. Containment ratio (is one string mostly inside the other?)
    def containment(a, b):
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))
    
    contain = [containment(a, b) for a, b in zip(s1, s2)]
    
    return pd.DataFrame({
        f"{prefix}_exact": exact_match,
        f"{prefix}_jw": jw,
        f"{prefix}_lev": lev,
        f"{prefix}_token_set": token_set,
        f"{prefix}_token_sort": token_sort,
        f"{prefix}_partial": partial,
        f"{prefix}_len_diff": len_diff,
        f"{prefix}_len_ratio": len_ratio,
        f"{prefix}_jaccard": jaccard,
        f"{prefix}_shared_tokens": shared,
        f"{prefix}_containment": contain,
    }, index=s1_series.index)


def build_feature_matrix(pairs: pd.DataFrame, s1_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of pairs (s1_id, target_id), join the raw features and compute similarities.
    Returns 24+ features per pair.
    """
    # Join s1 details
    df = pairs.merge(s1_df, left_on='s1_id', right_on='entity_id', how='left')
    df = df.rename(columns={
        'business_name': 's1_name',
        'business_address': 's1_address',
        'country': 's1_country'
    })
    
    # Join target details
    df = df.merge(target_df, left_on='target_id', right_on='entity_id', how='left')
    df = df.rename(columns={
        'business_name': 'target_name',
        'business_address': 'target_address',
        'country': 'target_country'
    })
    
    # Core string similarity features (11 each for name and address)
    name_feats = compute_string_features(df['s1_name'], df['target_name'], 'name')
    address_feats = compute_string_features(df['s1_address'], df['target_address'], 'addr')
    
    # Country features
    country_match = (df['s1_country'].fillna("") == df['target_country'].fillna("")).astype(int)
    country_missing = (df['s1_country'].isna() | df['target_country'].isna()).astype(int)
    
    # Address presence features (critical for S3 which has no addresses)
    s1_has_addr = (df['s1_address'].fillna("").str.len() > 2).astype(int)
    target_has_addr = (df['target_address'].fillna("").str.len() > 2).astype(int)
    both_have_addr = (s1_has_addr & target_has_addr).astype(int)
    
    feature_df = pd.concat([name_feats, address_feats], axis=1)
    feature_df['country_match'] = country_match
    feature_df['country_missing'] = country_missing
    feature_df['s1_has_addr'] = s1_has_addr
    feature_df['target_has_addr'] = target_has_addr
    feature_df['both_have_addr'] = both_have_addr
    
    return feature_df

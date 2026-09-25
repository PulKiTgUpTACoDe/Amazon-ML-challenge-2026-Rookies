import pandas as pd
import numpy as np
from rapidfuzz import fuzz, distance

def compute_string_features(s1_series: pd.Series, s2_series: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Compute string similarity features between two Series of strings.
    Handles NaN values gracefully.
    """
    # Fill NAs with empty string for string distances
    s1 = s1_series.fillna("").astype(str)
    s2 = s2_series.fillna("").astype(str)
    
    # Exact match
    exact_match = (s1 == s2).astype(int)
    
    # We use RapidFuzz which is highly optimized C++ under the hood
    # Jaro-Winkler
    jw = [distance.JaroWinkler.normalized_similarity(a, b) for a, b in zip(s1, s2)]
    
    # Levenshtein ratio
    lev = [fuzz.ratio(a, b) / 100.0 for a, b in zip(s1, s2)]
    
    # Token set ratio (handles out of order words)
    token_set = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(s1, s2)]
    
    # Length diff
    len_diff = np.abs(s1.str.len() - s2.str.len())
    
    return pd.DataFrame({
        f"{prefix}_exact": exact_match,
        f"{prefix}_jw": jw,
        f"{prefix}_lev": lev,
        f"{prefix}_token_set": token_set,
        f"{prefix}_len_diff": len_diff
    }, index=s1_series.index)

def build_feature_matrix(pairs: pd.DataFrame, s1_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of pairs (s1_id, target_id), join the raw features and compute similarities.
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
    
    # Features
    name_feats = compute_string_features(df['s1_name'], df['target_name'], 'name')
    address_feats = compute_string_features(df['s1_address'], df['target_address'], 'addr')
    
    # Country match
    country_match = (df['s1_country'].fillna("") == df['target_country'].fillna("")).astype(int)
    country_missing = (df['s1_country'].isna() | df['target_country'].isna()).astype(int)
    
    feature_df = pd.concat([name_feats, address_feats], axis=1)
    feature_df['country_match'] = country_match
    feature_df['country_missing'] = country_missing
    
    return feature_df

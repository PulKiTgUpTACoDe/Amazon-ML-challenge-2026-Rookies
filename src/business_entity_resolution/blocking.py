import pandas as pd
from collections import defaultdict

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

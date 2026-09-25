import re
import unicodedata
import pandas as pd

def normalize_text(text: str) -> str:
    """
    Apply conservative text normalization:
    1. Unicode NFKD (decompose, encode to ASCII to drop accents, decode)
    2. Lowercase
    3. Replace punctuation with space
    4. Remove extra whitespace
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # 1. Unicode normalize and remove accents
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Replace punctuation with space
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 4. Squish whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_dataframe(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Apply text normalization to a list of columns in a DataFrame in-place to save memory.
    """
    for col in columns:
        if col == 'entity_id' or col == 'country':
            continue
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)
    return df

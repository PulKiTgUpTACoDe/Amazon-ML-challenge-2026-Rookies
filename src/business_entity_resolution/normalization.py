import re
import unicodedata
import pandas as pd
from .abbreviations import expand_business_name, expand_address

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
    Apply text normalization + abbreviation expansion to a list of columns in a DataFrame.
    """
    for col in columns:
        if col == 'entity_id' or col == 'country':
            continue
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)
            # Apply abbreviation expansion after basic normalization
            if col == 'business_name':
                df[col] = df[col].apply(expand_business_name)
            elif col == 'business_address':
                df[col] = df[col].apply(expand_address)
    return df


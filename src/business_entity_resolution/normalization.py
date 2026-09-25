import pandas as pd
import unicodedata
from .abbreviations import BUSINESS_ABBREVIATIONS, ADDRESS_ABBREVIATIONS

def normalize_dataframe(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns or col in ['entity_id', 'country']:
            continue
            
        print(f"  Normalizing {col}...")
        
        # 1. Vectorized Lowercase & fast stripping
        df[col] = df[col].astype(str).str.lower().str.strip()
        
        # 2. Vectorized Punctuation removal (replace anything not word/space with space)
        df[col] = df[col].str.replace(r'[^\w\s]', ' ', regex=True)
        
        # 3. Vectorized Unicode normalization (NFKD to remove accents)
        df[col] = df[col].map(
            lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode('utf-8') 
            if pd.notna(x) else ""
        )
        
        # 4. Apply Abbreviations Column-Wise
        abbr_dict = BUSINESS_ABBREVIATIONS if col == 'business_name' else ADDRESS_ABBREVIATIONS
        if abbr_dict:
            for pattern, replacement in abbr_dict.items():
                df[col] = df[col].str.replace(pattern, replacement, regex=True)
                
        # 5. Collapse multiple spaces
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
        
    return df

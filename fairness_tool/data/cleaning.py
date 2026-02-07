import pandas as pd
import numpy as np

def handle_duplicate_columns(df):
    """Renames duplicate columns by appending a suffix."""
    if len(df.columns) == len(set(df.columns)):
        return df, []
        
    new_cols = []
    seen = {}
    renamed = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_name = f"{col}_dup_{seen[col]}"
            new_cols.append(new_name)
            renamed.append(new_name)
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df, renamed

def infer_numeric_types(df):
    """Automatically converts object/category columns to numeric if they contain mostly numbers."""
    converted_cols = []
    for col in df.columns:
        # Only check object or category columns
        if df[col].dtype == 'object' or (hasattr(df[col].dtype, 'name') and df[col].dtype.name == 'category'):
            # Try converting to numeric, non-convertible become NaN
            temp_col = pd.to_numeric(df[col], errors='coerce')
            
            # If the column is not all NaNs and we didn't lose too much data (e.g. < 20% loss)
            # Or if it's clearly a numeric column with just a few empty strings
            valid_count = temp_col.notna().sum()
            if valid_count > 0:
                original_non_na = df[col].notna().sum()
                # If we preserved at least 80% of the non-empty values after conversion
                if original_non_na > 0 and (valid_count / original_non_na) >= 0.8:
                    df[col] = temp_col
                    converted_cols.append(col)
                    
    return df, converted_cols

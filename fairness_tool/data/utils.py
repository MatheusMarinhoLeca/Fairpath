import pandas as pd

def get_sensitive_mapping(df, original_df, sensitive_col):
    """Detects if a column was encoded and returns a mapping from original to current values."""
    mapping = {}
    if sensitive_col not in df.columns or sensitive_col not in original_df.columns:
        return mapping
        
    orig_dtype = original_df[sensitive_col].dtype
    curr_dtype = df[sensitive_col].dtype
    
    if (orig_dtype == 'object' or isinstance(orig_dtype, pd.CategoricalDtype)) and pd.api.types.is_numeric_dtype(curr_dtype):
        # Improved mapping logic: Use unique pairs from the intersection of indices
        common_indices = df.index.intersection(original_df.index)
        if not common_indices.empty:
            temp_map = pd.DataFrame({
                'orig': original_df.loc[common_indices, sensitive_col],
                'curr': df.loc[common_indices, sensitive_col]
            }).dropna().drop_duplicates()
            
            if not temp_map.empty:
                for _, row in temp_map.iterrows():
                    mapping[str(row['orig'])] = row['curr']
    return mapping

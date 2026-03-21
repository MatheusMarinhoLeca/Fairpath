import pandas as pd
from typing import List, Tuple

def ensure_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Ensures that a column selection from a DataFrame returns a 1D Series,
    even if duplicate column names exist.
    """
    series = df[col_name]
    if isinstance(series, pd.DataFrame):
        return series.iloc[:, 0]
    return series

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

def parse_attribute_input(user_input: str, available_columns: List[str]) -> List[str]:
    """
    Parses a comma-separated string of column names into a list.
    Validates that each column exists in the available columns.
    
    Args:
        user_input (str): Comma-separated column names (e.g., "race, gender").
        available_columns (List[str]): List of valid column names in the DataFrame.
        
    Returns:
        List[str]: List of valid column names found in the input.
        
    Raises:
        ValueError: If any provided column name is invalid.
    """
    if not user_input:
        return []
        
    # Split by comma and strip whitespace
    cols = [c.strip() for c in user_input.split(',')]
    
    # Validate
    invalid_cols = [c for c in cols if c not in available_columns]
    if invalid_cols:
        raise ValueError(f"Columns not found in dataset: {', '.join(invalid_cols)}")
        
    return cols

def create_composite_attribute(df: pd.DataFrame, columns: List[str], separator: str = "_") -> Tuple[pd.DataFrame, str]:
    """
    Combines multiple columns into a single composite attribute.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (List[str]): List of column names to combine.
        separator (str): Separator string for the combined values.
        
    Returns:
        Tuple[pd.DataFrame, str]: 
            - The modified DataFrame with the new composite column.
            - The name of the new composite column.
    """
    if not columns:
        raise ValueError("No columns provided for composite attribute creation.")
    
    if len(columns) == 1:
        return df, columns[0]
    
    # Create composite column name
    composite_col_name = separator.join(columns)
    
    # Combine values efficiently
    # Convert all to string, fill missing with 'NA', and join
    composite_series = df[columns[0]].astype(str)
    for col in columns[1:]:
        composite_series = composite_series + separator + df[col].astype(str)
        
    df = df.copy()
    df[composite_col_name] = composite_series
    
    return df, composite_col_name

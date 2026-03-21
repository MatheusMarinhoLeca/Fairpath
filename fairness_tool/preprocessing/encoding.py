import pandas as pd
from sklearn.preprocessing import LabelEncoder

def one_hot_encode(df, target_col=None, columns=None):
    # Select columns that are categorical (object, category, bool)
    candidates = df.select_dtypes(include=['object', 'category', 'bool']).columns
    
    if columns is not None:
        columns_to_encode = [col for col in candidates if col in columns and col != target_col]
    else:
        columns_to_encode = [col for col in candidates if col != target_col]
    
    if not columns_to_encode:
        return df
        
    # Apply get_dummies. 
    # Drop original columns to avoid duplicates.
    dummies = pd.get_dummies(df[columns_to_encode], prefix=columns_to_encode, drop_first=False, dtype=int)
    df = pd.concat([df.drop(columns=columns_to_encode), dummies], axis=1)
    
    return df

def label_encode(df, target_col=None, columns=None):
    le = LabelEncoder()
    candidates = df.select_dtypes(include=['object', 'category']).columns
    
    if columns is not None:
        columns_to_encode = [col for col in candidates if col in columns and col != target_col]
    else:
        columns_to_encode = [col for col in candidates if col != target_col]
        
    for col in columns_to_encode:
        # We replace the column with encoded values. 
        # If we want to keep original, we could create a new column, 
        # but label encoding is often expected to be in-place for the primary column.
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Also encode target if it's categorical
    if target_col and df[target_col].dtype == 'object':
         df[target_col] = le.fit_transform(df[target_col].astype(str))
         
    return df

def binarize_attribute(df, col, privileged_value, pos_label=1, neg_label=0):
    """
    Transforms a column into a binary variable based on a privileged value.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        col (str): Column name to binarize.
        privileged_value (Any): The value to map to pos_label.
        pos_label (Any): Value for the privileged group (default 1).
        neg_label (Any): Value for the unprivileged group (default 0).
        
    Returns:
        pd.DataFrame: Dataframe with the column modified.
    """
    df = df.copy()
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataframe.")
        
    # Create mask
    mask = df[col] == privileged_value
    
    # Initialize with neg_label
    new_series = pd.Series([neg_label] * len(df), index=df.index, dtype=type(neg_label))
    
    # Set pos_label
    new_series[mask] = pos_label
    
    df[col] = new_series
    return df

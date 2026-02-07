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
    # To keep original columns, we generate dummies and then concat.
    dummies = pd.get_dummies(df[columns_to_encode], prefix=columns_to_encode, drop_first=False, dtype=int)
    df = pd.concat([df, dummies], axis=1)
    
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

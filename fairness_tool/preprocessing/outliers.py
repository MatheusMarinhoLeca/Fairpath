import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns=None):
    candidates = df.select_dtypes(include=['number']).columns
    
    if columns is not None:
        numeric_cols = [c for c in candidates if c in columns]
    else:
        numeric_cols = candidates
    
    if len(numeric_cols) == 0:
        return df
        
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    condition = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[condition]

def winsorize_outliers(df):
    # Placeholder for winsorization
    # Typically requires scipy.stats.mstats.winsorize or manual clipping
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        lower = df[col].quantile(0.05)
        upper = df[col].quantile(0.95)
        df[col] = np.clip(df[col], lower, upper)
    return df

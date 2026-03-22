import pandas as pd
import numpy as np

class OutlierHandler:
    """Stateful outlier handler that learns from train and applies to test."""
    
    def __init__(self, strategy='clip', columns=None):
        self.strategy = strategy
        self.columns = columns
        self.learned_bounds = {}

    def fit(self, df: pd.DataFrame):
        """Learns the outlier bounds from the training dataframe."""
        candidates = df.select_dtypes(include=['number']).columns
        if self.columns is not None:
            cols_to_process = [c for c in candidates if c in self.columns]
        else:
            cols_to_process = candidates

        for col in cols_to_process:
            # SAFETY CHECK: Skip outlier detection for binary or low-cardinality columns.
            # Applying IQR to these often incorrectly flags minority classes as outliers.
            if df[col].nunique() < 10:
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            # If IQR is 0 (e.g., highly skewed data), this method is unsafe. Skip.
            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.learned_bounds[col] = (lower, upper)
            
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the learned outlier handling strategy."""
        df_out = df.copy()
        
        for col, (lower, upper) in self.learned_bounds.items():
            if col not in df_out.columns:
                continue
                
            if self.strategy == 'remove':
                df_out = df_out[(df_out[col] >= lower) & (df_out[col] <= upper)]
            else: # default to 'clip'
                df_out[col] = np.clip(df_out[col], lower, upper)
                
        return df_out

# Backward compatibility functional wrappers
def remove_outliers_iqr(df, columns=None):
    return OutlierHandler(strategy='remove', columns=columns).fit(df).transform(df)

def clip_outliers(df):
    """Legacy helper for clipping outliers."""
    return OutlierHandler(strategy='clip').fit(df).transform(df)

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for IQR-based outlier removal.
    
    This transformer learns bounds during fit() and removes rows containing
    outliers outside these bounds during transform().
    """
    
    def __init__(self, strategy='remove', columns=None):
        self.strategy = strategy
        self.columns = columns
        self.learned_bounds = {}

    def fit(self, X, y=None):
        """Learns the outlier bounds from the provided data.

        Args:
            X: pd.DataFrame
        """
        # Ensure we work with a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        candidates = X.select_dtypes(include=['number']).columns
        cols_to_process = [c for c in candidates if c in self.columns] if self.columns else candidates

        for col in cols_to_process:
            # We only remove outliers if it has sufficient variance
            if X[col].nunique() < 10:
                continue

            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.learned_bounds[col] = (lower, upper)
            
        return self

    def transform(self, X):
        """Applies the learned outlier removal strategy.

        Args:
            X: pd.DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_out = X.copy()
        
        if self.strategy == 'remove':
            mask = pd.Series([True] * len(X_out))
            for col, (lower, upper) in self.learned_bounds.items():
                if col in X_out.columns:
                    mask &= (X_out[col] >= lower) & (X_out[col] <= upper)
            X_out = X_out[mask]
        else: # default to 'clip'
            for col, (lower, upper) in self.learned_bounds.items():
                if col in X_out.columns:
                    X_out[col] = np.clip(X_out[col], lower, upper)
                
        return X_out

# Backward compatibility functional wrapper
def remove_outliers_iqr(df, columns=None):
    remover = OutlierRemover(strategy='remove', columns=columns)
    return remover.fit_transform(df)

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class MissingValueImputer:
    """Stateful missing value imputer that learns from train and applies to test."""
    
    def __init__(self, strategy='mean', columns=None):
        self.strategy = strategy
        self.columns = columns
        self.imputer = None
        self.fitted_columns = None

    def fit(self, df: pd.DataFrame):
        """Learns the imputation values from the training dataframe."""
        if self.strategy == 'drop':
            return self # No fitting needed for drop
            
        # Determine candidate columns if not provided
        if self.columns is None:
            if self.strategy in ['mean', 'median']:
                self.fitted_columns = df.select_dtypes(include=['number']).columns.tolist()
            else:
                self.fitted_columns = df.columns.tolist()
        else:
            self.fitted_columns = [c for c in self.columns if c in df.columns]

        if not self.fitted_columns:
            return self

        # Initialize and fit the sklearn imputer
        sk_strategy = self.strategy if self.strategy != 'mode' else 'most_frequent'
        # keep_empty_features=True ensures the output has the same number of columns as input,
        # even if some features are all NaNs and can't be imputed with mean/mode.
        self.imputer = SimpleImputer(strategy=sk_strategy, keep_empty_features=True)
        self.imputer.fit(df[self.fitted_columns])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the learned imputation values to a dataframe."""
        df_out = df.copy()
        
        if self.strategy == 'drop':
            cols_to_check = self.columns if self.columns else df_out.columns
            return df_out.dropna(axis=0, subset=[c for c in cols_to_check if c in df_out.columns])

        if self.imputer is None or not self.fitted_columns:
            return df_out

        # Apply transformation
        # Ensure we only try to transform columns that exist in the input df
        cols_to_transform = [c for c in self.fitted_columns if c in df_out.columns]
        if not cols_to_transform:
            return df_out
            
        # Transform. SimpleImputer with keep_empty_features=True will return correct shape.
        df_out[self.fitted_columns] = self.imputer.transform(df_out[self.fitted_columns])
        
        # As a final safety measure, fill any columns that were all NaNs (and thus 
        # remained NaNs after mean/mode imputation) with 0.
        df_out[self.fitted_columns] = df_out[self.fitted_columns].fillna(0).infer_objects(copy=False)
        
        return df_out

# Backward compatibility functional wrapper
def impute_missing(df, strategy='mean', columns=None):
    """Functional wrapper for stateless imputation (not recommended for production)."""
    return MissingValueImputer(strategy=strategy, columns=columns).fit(df).transform(df)

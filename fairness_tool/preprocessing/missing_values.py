from sklearn.impute import SimpleImputer
import pandas as pd

def impute_missing(df, strategy='mean', columns=None):
    # Determine candidate columns based on types
    candidates_numeric = df.select_dtypes(include=['number']).columns
    candidates_categorical = df.select_dtypes(exclude=['number']).columns
    
    # Filter by user-provided columns if available
    if columns is not None:
        numeric_cols = [c for c in candidates_numeric if c in columns]
        categorical_cols = [c for c in candidates_categorical if c in columns]
    else:
        numeric_cols = candidates_numeric
        categorical_cols = candidates_categorical
    
    # helper to handle potential version differences for keep_empty_features
    def get_imputer(strat):
        try:
            return SimpleImputer(strategy=strat, keep_empty_features=True)
        except TypeError:
            return SimpleImputer(strategy=strat)

    if strategy in ['mean', 'median']:
        if len(numeric_cols) > 0:
            # Handle all-NaN numeric columns to prevent SimpleImputer from dropping them
            # This is critical for maintaining dataframe shape in downstream tasks
            all_nan_numeric = [c for c in numeric_cols if df[c].isnull().all()]
            if all_nan_numeric:
                df[all_nan_numeric] = 0
            
            imputer = get_imputer(strategy)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        if len(categorical_cols) > 0:
            cat_imputer = get_imputer('most_frequent')
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            
    elif strategy == 'mode':
        # Mode strategy usually applies to all, but we filter if columns provided
        imputer = get_imputer('most_frequent')
        cols_to_impute = list(numeric_cols) + list(categorical_cols)
        
        if len(cols_to_impute) > 0:
            # Handle all-NaN columns for mode imputation too if needed, though most_frequent usually handles it by returning first value or similar?
            # Actually, SimpleImputer(strategy='most_frequent') might also drop all-NaN cols.
            all_nan_cols = [c for c in cols_to_impute if df[c].isnull().all()]
            if all_nan_cols:
                # For categorical, 0 might not be valid, but 'Missing' or similar might be. 
                # Ideally we check type. For now, fill numeric with 0, object with 'Missing'
                for c in all_nan_cols:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        df[c] = 0
                    else:
                        df[c] = 'Missing'

            # Only transform the subset
            new_values = imputer.fit_transform(df[cols_to_impute])
            df[cols_to_impute] = new_values
        
    elif strategy == 'drop':
        # dropna(axis=0) drops rows with missing values.
        # If columns are provided, only drop rows where those specific columns are missing.
        df = df.dropna(axis=0, subset=columns)
        
    return df

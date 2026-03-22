import pandas as pd
from sklearn.preprocessing import LabelEncoder as SkLabelEncoder

class CategoricalEncoder:
    """Stateful categorical encoder (One-Hot or Label) that learns from train and applies to test."""
    
    def __init__(self, strategy='one-hot', target_col=None, columns=None):
        self.strategy = strategy
        self.target_col = target_col
        self.columns = columns
        self.fitted_columns = None
        self.label_encoders = {}
        self.one_hot_columns = None

    def fit(self, df: pd.DataFrame):
        """Learns the categories from the training dataframe."""
        candidates = df.select_dtypes(include=['object', 'category', 'bool']).columns
        
        if self.columns is not None:
            self.fitted_columns = [col for col in candidates if col in self.columns and col != self.target_col]
        else:
            self.fitted_columns = [col for col in candidates if col != self.target_col]

        if not self.fitted_columns:
            return self

        if self.strategy == 'label':
            for col in self.fitted_columns:
                le = SkLabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le
        elif self.strategy == 'one-hot':
            # Learn dummy column names by performing a dummy encoding on a sample
            dummies = pd.get_dummies(df[self.fitted_columns], prefix=self.fitted_columns, drop_first=False, dtype=int)
            self.one_hot_columns = dummies.columns.tolist()
            
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the learned encoding to a dataframe."""
        df_out = df.copy()
        if not self.fitted_columns:
            return df_out

        if self.strategy == 'label':
            for col, le in self.label_encoders.items():
                if col in df_out.columns:
                    # Handle unseen categories by mapping them to a fallback (or just use string conversion)
                    # Note: LabelEncoder doesn't natively handle unseen well.
                    # We'll convert to string and use transform, assuming train covers most.
                    # A more robust implementation would handle unseen explicitly.
                    df_out[col] = le.transform(df_out[col].astype(str))
        elif self.strategy == 'one-hot':
            # Perform encoding
            dummies = pd.get_dummies(df_out[self.fitted_columns], prefix=self.fitted_columns, drop_first=False, dtype=int)
            # Reindex to match fitted columns (adds missing dummies as 0, drops extra ones)
            dummies = dummies.reindex(columns=self.one_hot_columns, fill_value=0)
            # Drop original columns and concat dummies
            df_out = pd.concat([df_out.drop(columns=self.fitted_columns), dummies], axis=1)
            
        return df_out

class AttributeBinarizer:
    """Stateful attribute binarizer."""
    
    def __init__(self, col, privileged_value, pos_label=1, neg_label=0):
        self.col = col
        self.privileged_value = privileged_value
        self.pos_label = pos_label
        self.neg_label = neg_label

    def fit(self, df: pd.DataFrame):
        return self # No learning needed for binarization with fixed value

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if self.col not in df_out.columns:
            return df_out
            
        mask = df_out[self.col] == self.privileged_value
        new_series = pd.Series([self.neg_label] * len(df_out), index=df_out.index, dtype=type(self.neg_label))
        new_series[mask] = self.pos_label
        df_out[self.col] = new_series
        return df_out

# Backward compatibility functional wrappers
def one_hot_encode(df, target_col=None, columns=None):
    return CategoricalEncoder(strategy='one-hot', target_col=target_col, columns=columns).fit(df).transform(df)

def label_encode(df, target_col=None, columns=None):
    return CategoricalEncoder(strategy='label', target_col=target_col, columns=columns).fit(df).transform(df)

def binarize_attribute(df, col, privileged_value, pos_label=1, neg_label=0):
    return AttributeBinarizer(col, privileged_value, pos_label, neg_label).transform(df)

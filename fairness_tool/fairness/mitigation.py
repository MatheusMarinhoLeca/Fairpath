import pandas as pd
import numpy as np
import warnings
import logging
import os
import sys
import contextlib
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, Normalizer
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from sdv.sampling import Condition
from typing import Any, List, Optional
from core.interfaces import MitigationStrategy
from data.utils import ensure_series
from preprocessing.missing_values import impute_missing
from config.defaults import FAIRNESS_EVAL_COL

@contextlib.contextmanager
def silence_output():
    """
    Forcefully silence all stdout and stderr, including C-level calls 
    and diagnostic events from libraries like SDV.
    """
    # Open devnull
    null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
    # Save original fds
    save_fds = [os.dup(1), os.dup(2)]
    try:
        # Redirect stdout/stderr to devnull
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        yield
    finally:
        # Restore original fds
        os.dup2(save_fds[0], 1)
        os.dup2(save_fds[1], 2)
        # Close temp fds
        for fd in null_fds + save_fds:
            os.close(fd)

class ResamplingMitigation(MitigationStrategy):
    def __init__(self, strategy: str = 'oversample'):
        self.strategy = strategy

    def mitigate(self, df: pd.DataFrame, target_col: str, sensitive_col: str, 
                 privileged_group: Any, unprivileged_group: Any, **kwargs) -> pd.DataFrame:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        work_df = X.copy()
        work_df['target'] = y

        total_per_a = work_df.groupby(sensitive_col).size()
        max_total = total_per_a.max()
        # Ensure we drop NaNs for the target rate calculation
        target_pos_rate = pd.to_numeric(work_df['target'], errors='coerce').dropna().mean()

        df_resampled = []
        for a_val in work_df[sensitive_col].unique():
            group_a = work_df[work_df[sensitive_col] == a_val]
            n_a = len(group_a)

            target_n_pos = int(n_a * target_pos_rate) if self.strategy == 'undersample' else int(max_total * target_pos_rate)
            target_n_neg = (n_a - target_n_pos) if self.strategy == 'undersample' else (max_total - target_n_pos)

            pos_samples = group_a[group_a['target'] == 1]
            neg_samples = group_a[group_a['target'] == 0]

            if self.strategy == 'oversample':
                if len(pos_samples) > 0:
                    df_resampled.append(pos_samples.sample(target_n_pos, replace=True))
                if len(neg_samples) > 0:
                    df_resampled.append(neg_samples.sample(target_n_neg, replace=True))
            else:
                if len(pos_samples) > 0:
                    df_resampled.append(pos_samples.sample(min(len(pos_samples), target_n_pos), replace=False))
                if len(neg_samples) > 0:
                    df_resampled.append(neg_samples.sample(min(len(neg_samples), target_n_neg), replace=False))

        resampled_df = pd.concat(df_resampled).sample(frac=1).reset_index(drop=True)
        return resampled_df.rename(columns={'target': target_col}) 

class RelabelingMitigation(MitigationStrategy):
    def __init__(self, features_to_use: Optional[List[str]] = None):
        self.features_to_use = features_to_use

    def mitigate(self, df: pd.DataFrame, target_col: str, sensitive_col: str, 
                 privileged_group: Any, unprivileged_group: Any, **kwargs) -> pd.DataFrame:
        df_work = df.copy()

        if self.features_to_use is not None:
            X = df_work[[f for f in self.features_to_use if f in df_work.columns]]
            if target_col in X.columns:
                X = X.drop(columns=[target_col])
        else:
            X = df_work.drop(columns=[target_col])

        X_num = X.select_dtypes(include=[np.number])
        y = df_work[target_col]

        # Handle NaNs before LogisticRegression using robust impute_missing
        if X_num.isnull().sum().sum() > 0:
            X_num = impute_missing(X_num, strategy='mean')

        # Fix ConvergenceWarning by scaling and normalizing data
        scaler = StandardScaler()
        normalizer = Normalizer()
        X_scaled = scaler.fit_transform(X_num)
        X_scaled = normalizer.fit_transform(X_scaled)

        ranker = LogisticRegression(max_iter=1000)
        # LogisticRegression handles categorical y if it's 0/1, but we pass it as is.
        # If it fails, sklearn might need y.astype(int), but let's see.
        ranker.fit(X_scaled, y)
        probs = ranker.predict_proba(X_scaled)[:, 1]
        df_work['_prob'] = probs

        if isinstance(unprivileged_group, list):
            unpriv_mask = df_work[sensitive_col].isin(unprivileged_group)
        else:
            unpriv_mask = df_work[sensitive_col] == unprivileged_group

        priv_mask = df_work[sensitive_col] == privileged_group

        n_unpriv = unpriv_mask.sum()
        n_priv = priv_mask.sum()

        # Adhere to: "Only apply mean operations to numerical columns"
        if pd.api.types.is_numeric_dtype(y):
            target_rate = y.mean()
        else:
            # For categorical/object, compute rate manually without using .mean() on y itself
            target_rate = (y == 1).mean()

        # Manual calculation for current counts using boolean masks to avoid .sum() on non-numeric series
        current_unpriv_pos = (df_work.loc[unpriv_mask, target_col] == 1).sum()
        target_unpriv_pos = int(n_unpriv * target_rate)

        current_priv_pos = (df_work.loc[priv_mask, target_col] == 1).sum()
        target_priv_pos = int(n_priv * target_rate)

        # UNPRIVILEGED adjustment
        flips_up = 0
        flips_down = 0
        if current_unpriv_pos < target_unpriv_pos:
            num_to_flip = int(target_unpriv_pos - current_unpriv_pos)
            to_flip = df_work[unpriv_mask & (df_work[target_col] == 0)].nlargest(num_to_flip, '_prob').index
            df_work.loc[to_flip, target_col] = 1
            flips_up += len(to_flip)
        elif current_unpriv_pos > target_unpriv_pos:
            num_to_flip = int(current_unpriv_pos - target_unpriv_pos)
            to_flip = df_work[unpriv_mask & (df_work[target_col] == 1)].nsmallest(num_to_flip, '_prob').index
            df_work.loc[to_flip, target_col] = 0
            flips_down += len(to_flip)

        # PRIVILEGED adjustment
        if current_priv_pos > target_priv_pos:
            num_to_flip = int(current_priv_pos - target_priv_pos)
            to_flip = df_work[priv_mask & (df_work[target_col] == 1)].nsmallest(num_to_flip, '_prob').index
            df_work.loc[to_flip, target_col] = 0
            flips_down += len(to_flip)
        elif current_priv_pos < target_priv_pos:
            num_to_flip = int(target_priv_pos - current_priv_pos)
            to_flip = df_work[priv_mask & (df_work[target_col] == 0)].nlargest(num_to_flip, '_prob').index
            df_work.loc[to_flip, target_col] = 1
            flips_up += len(to_flip)

        if flips_up + flips_down > 0:
            print(f"✔ Relabeling successful: Flipped {flips_up} labels to 1 and {flips_down} labels to 0.")
        else:
            print("✔ Relabeling: No changes needed (dataset already balanced).")

        return df_work.drop(columns=['_prob'])

class SyntheticMitigation(MitigationStrategy):
    def __init__(self, method: str = 'smote'):
        self.method = method

    def mitigate(self, df: pd.DataFrame, target_col: str, sensitive_col: str, 
                 privileged_group: Any, unprivileged_group: Any, **kwargs) -> pd.DataFrame:

        # Identify columns that should NOT be treated as features for interpolation
        # but must be preserved or reconstructed (like the fairness evaluation column)
        fairness_eval_col = FAIRNESS_EVAL_COL

        if self.method == 'smote':
            resampled_df = self._smote(df, target_col, sensitive_col, fairness_eval_col)
        elif self.method == 'cda':
            X_res, y_res = self._cda(df, target_col, sensitive_col, privileged_group, unprivileged_groups=unprivileged_group)
            resampled_df = pd.DataFrame(X_res)
            y_values = y_res.values if hasattr(y_res, 'values') else y_res
            resampled_df = pd.concat([resampled_df, pd.Series(y_values, name=target_col)], axis=1)
        else:
            raise ValueError(f"Unknown synthetic method: {self.method}")

        if resampled_df is None:
            return df.copy()

        return resampled_df

    def _smote(self, df, target_col, sensitive_col, fairness_eval_col):
        # 1. Prepare data: SMOTE only works on numeric features.
        # We need to preserve the mapping for the sensitive attribute to reconstruct evaluation columns.
        df_work = df.copy()

        # Ensure we have 1D series for key columns even if duplicates exist
        s_target = ensure_series(df_work, target_col)
        
        # If sensitive_col is missing (could be one-hot encoded), we try to find it
        if sensitive_col not in df_work.columns and fairness_eval_col in df_work.columns:
            # Re-create sensitive_col from fairness_eval_col
            df_work[sensitive_col] = ensure_series(df_work, fairness_eval_col).copy()

        # If sensitive_col is still missing, we can't do SMOTE balancing on it
        if sensitive_col not in df_work.columns:
            return None
            
        s_sens = ensure_series(df_work, sensitive_col)

        # If sensitive_col is categorical, encode it temporarily
        is_sens_numeric = pd.api.types.is_numeric_dtype(s_sens)
        if not is_sens_numeric:
            from sklearn.preprocessing import LabelEncoder
            le_sens = LabelEncoder()
            # Use the Series directly to ensure 1D input to fit_transform
            df_work[sensitive_col] = le_sens.fit_transform(s_sens.astype(str))
            s_sens = df_work[sensitive_col] # Update reference

        # Build a mapping from sensitive column to fairness eval column
        mapping_sens_to_eval = {}
        if fairness_eval_col in df_work.columns:
            s_eval = ensure_series(df_work, fairness_eval_col)
            # Capture mapping from unique pairs
            pairs = pd.DataFrame({sensitive_col: s_sens, fairness_eval_col: s_eval}).drop_duplicates()
            mapping_sens_to_eval = dict(zip(pairs[sensitive_col], pairs[fairness_eval_col]))

        # Select numeric columns for X features
        df_numeric = df_work.select_dtypes(include=[np.number])

        # Handle NaNs before SMOTE using robust impute_missing
        if df_numeric.isnull().sum().sum() > 0:
            df_numeric = impute_missing(df_numeric, strategy='mean')

        # Create combined labels for balancing subgroups
        combined_col = s_sens.astype(str) + "_" + s_target.astype(str)
        y_combined = combined_col.rename('_combined')

        # Drop target and any special eval columns from features to avoid interpolation issues
        # Also drop sensitive_col if it's numeric to avoid duplicate columns in final output
        cols_to_drop = [target_col]
        if fairness_eval_col in df_numeric.columns:
            cols_to_drop.append(fairness_eval_col)
        if sensitive_col in df_numeric.columns:
            cols_to_drop.append(sensitive_col)

        X = df_numeric.drop(columns=cols_to_drop)

        # Dynamic neighbors check
        group_counts = y_combined.value_counts()
        min_samples = group_counts.min()
        n_neighbors = min(5, max(1, min_samples - 1))

        if min_samples < 2:
            print(f"⚠ Warning: Subgroup size {min_samples} too small for SMOTE. Skipping mitigation.")
            return None

        smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
        X_res, y_combined_res = smote.fit_resample(X, y_combined)

        # Reconstruct DataFrame with names
        if not isinstance(X_res, pd.DataFrame):
            X_res = pd.DataFrame(X_res, columns=X.columns).copy()
        else:
            X_res = X_res.copy()

        # 2. Reconstruct Target and Sensitive Attribute from combined labels
        # Combined label is "sens_target"
        res_combined_parts = y_combined_res.str.rsplit('_', n=1, expand=True)

        # Reconstruct all new columns as a dict first for a single concat
        new_cols = {}

        # Target reconstruction
        new_cols[target_col] = res_combined_parts[1].astype(ensure_series(df, target_col).dtype).values

        # Sensitive attribute reconstruction
        sens_res_raw = res_combined_parts[0]
        if is_sens_numeric:
            new_cols[sensitive_col] = sens_res_raw.astype(ensure_series(df, sensitive_col).dtype).values
        else:
            new_cols[sensitive_col] = le_sens.inverse_transform(sens_res_raw.astype(int))

        # 3. Reconstruct Fairness Evaluation Column
        if fairness_eval_col in df.columns:
            if mapping_sens_to_eval:
                # Use mapping to ensure categorical values are restored even if sensitive_col is numeric
                # If sens_res_raw (interpolated) has a value not in mapping (rounding error?), map to nearest
                # But SMOTE on label-encoded values usually returns the same integers.
                # However, res_combined_parts[0] is string, so we convert back to key type.
                key_type = ensure_series(df_work, sensitive_col).dtype
                new_cols[fairness_eval_col] = sens_res_raw.astype(key_type).map(mapping_sens_to_eval).values
            else:
                new_cols[fairness_eval_col] = new_cols[sensitive_col].copy()

        # Add all columns at once to prevent fragmentation
        X_res = pd.concat([X_res, pd.DataFrame(new_cols, index=X_res.index)], axis=1)

        return X_res

    def _cda(self, df, target_col, sensitive_col, privileged_group, unprivileged_groups):
        print("Applying Counterfactual Data Augmentation (CDA)...")
        df_cf = df.copy()

        priv_val = privileged_group[0] if isinstance(privileged_group, list) else privileged_group

        # 1. Flip the main sensitive column
        if isinstance(unprivileged_groups, list):
            mask_priv = df_cf[sensitive_col] == priv_val
            mask_unpriv = df_cf[sensitive_col].isin(unprivileged_groups)
            target_unpriv_val = unprivileged_groups[0]
            df_cf.loc[mask_priv, sensitive_col] = target_unpriv_val
            df_cf.loc[mask_unpriv, sensitive_col] = priv_val
        else:
            mask_priv = df_cf[sensitive_col] == priv_val
            mask_unpriv = df_cf[sensitive_col] == unprivileged_groups
            df_cf.loc[mask_priv, sensitive_col] = unprivileged_groups
            df_cf.loc[mask_unpriv, sensitive_col] = priv_val

        # 2. Synchronize One-Hot Encoded columns if they exist
        # We need to ensure that if 'race' became 'African-American', then 'race_African-American' becomes 1 
        # and 'race_Caucasian' becomes 0.
        
        # Identify potential one-hot columns for this sensitive attribute
        # We assume the standard format "{sensitive_col}_{value}"
        unique_vals = df[sensitive_col].unique()
        for val in unique_vals:
            # Construct the expected one-hot column name
            # Note: This simple check might need robustness if values contain special chars, 
            # but it covers standard pandas get_dummies behavior.
            oh_col = f"{sensitive_col}_{val}"
            
            if oh_col in df_cf.columns:
                # Update the one-hot column based on the NEW values in sensitive_col
                # If sensitive_col is now 'val', set oh_col to 1, else 0
                df_cf[oh_col] = (df_cf[sensitive_col] == val).astype(int)

        # Update the fairness evaluation column in the counterfactual data too
        fairness_eval_col = FAIRNESS_EVAL_COL
        if fairness_eval_col in df_cf.columns:
            df_cf[fairness_eval_col] = df_cf[sensitive_col].copy()

        df_final = pd.concat([df, df_cf], ignore_index=True)
        return df_final.drop(columns=[target_col]), df_final[target_col]
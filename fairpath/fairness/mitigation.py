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
from typing import Any, List, Optional, Tuple, Dict
from fairpath.core.interfaces import MitigationStrategy
from fairpath.data.utils import ensure_series
from fairpath.preprocessing.missing_values import impute_missing
from fairpath.config.defaults import FAIRNESS_EVAL_COL

@contextlib.contextmanager
def silence_output():
    """Forcefully silences all stdout and stderr, including C-level calls.

    This is used to suppress diagnostic events from libraries like SDV or 
    AIF360 that might clutter the CLI during mitigation steps.
    """
    null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
    save_fds = [os.dup(1), os.dup(2)]
    try:
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        yield
    finally:
        os.dup2(save_fds[0], 1)
        os.dup2(save_fds[1], 2)
        for fd in null_fds + save_fds:
            os.close(fd)

class ResamplingMitigation(MitigationStrategy):
    """Balances the dataset using group-aware resampling.
    
    This strategy addresses 'Representation Bias' by either oversampling the 
    underrepresented group or undersampling the overrepresented group until 
    base rates are equalized.
    """

    def __init__(self, strategy: str = 'oversample'):
        """Initializes the resampling strategy.

        Args:
            strategy: Either 'oversample' or 'undersample'.
        """
        self.strategy = strategy

    def mitigate(self, df: pd.DataFrame, target_col: str, sensitive_col: str, 
                 privileged_group: Any, unprivileged_group: Any, **kwargs) -> pd.DataFrame:
        """Applies resampling to the dataset.

        Args:
            df: The training dataset.
            target_col: The outcome column name.
            sensitive_col: The protected attribute column name.
            privileged_group: Value of the privileged group.
            unprivileged_group: Value(s) of the unprivileged group(s).

        Returns:
            A new DataFrame with balanced group distributions.
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]

        df_balancing = X.copy()
        df_balancing['target'] = y

        total_per_group = df_balancing.groupby(sensitive_col).size()
        max_group_size = total_per_group.max()
        target_pos_rate = pd.to_numeric(df_balancing['target'], errors='coerce').dropna().mean()

        balanced_chunks = []
        for group_val in df_balancing[sensitive_col].unique():
            group_df = df_balancing[df_balancing[sensitive_col] == group_val]
            n_group = len(group_df)

            target_n_pos = int(n_group * target_pos_rate) if self.strategy == 'undersample' else int(max_group_size * target_pos_rate)
            target_n_neg = (n_group - target_n_pos) if self.strategy == 'undersample' else (max_group_size - target_n_pos)

            pos_samples = group_df[group_df['target'] == 1]
            neg_samples = group_df[group_df['target'] == 0]

            if self.strategy == 'oversample':
                if len(pos_samples) > 0:
                    balanced_chunks.append(pos_samples.sample(target_n_pos, replace=True))
                if len(neg_samples) > 0:
                    balanced_chunks.append(neg_samples.sample(target_n_neg, replace=True))
            else:
                if len(pos_samples) > 0:
                    balanced_chunks.append(pos_samples.sample(min(len(pos_samples), target_n_pos), replace=False))
                if len(neg_samples) > 0:
                    balanced_chunks.append(neg_samples.sample(min(len(neg_samples), target_n_neg), replace=False))

        df_resampled = pd.concat(balanced_chunks).sample(frac=1).reset_index(drop=True)
        return df_resampled.rename(columns={'target': target_col}) 

class RelabelingMitigation(MitigationStrategy):
    """Mitigates bias by flipping labels near the decision boundary.
    
    This strategy targets 'Historical Bias' by ranking individuals by their 
    predicted probability and flipping labels to achieve group parity while 
    minimizing the impact on overall model utility.
    """

    def __init__(self, features_to_use: Optional[List[str]] = None):
        """Initializes with features used for ranking.

        Args:
            features_to_use: List of column names to use for the ranker model.
        """
        self.features_to_use = features_to_use

    def mitigate(self, df: pd.DataFrame, target_col: str, sensitive_col: str, 
                 privileged_group: Any, unprivileged_group: Any, **kwargs) -> pd.DataFrame:
        """Applies label flipping to the dataset.

        Args:
            df: The training dataset.
            target_col: The outcome column name.
            sensitive_col: The protected attribute column name.
            privileged_group: Value of the privileged group.
            unprivileged_group: Value(s) of the unprivileged group(s).

        Returns:
            A DataFrame with modified labels to improve fairness.
        """
        df_mitigation = df.copy()

        if self.features_to_use is not None:
            X = df_mitigation[[f for f in self.features_to_use if f in df_mitigation.columns]]
            if target_col in X.columns:
                X = X.drop(columns=[target_col])
        else:
            X = df_mitigation.drop(columns=[target_col])

        X_num = X.select_dtypes(include=[np.number])
        y = df_mitigation[target_col]

        if X_num.isnull().sum().sum() > 0:
            X_num = impute_missing(X_num, strategy='mean')

        scaler = StandardScaler()
        normalizer = Normalizer()
        X_scaled = scaler.fit_transform(X_num)
        X_scaled = normalizer.fit_transform(X_scaled)

        ranker = LogisticRegression(max_iter=1000)
        ranker.fit(X_scaled, y)
        probs = ranker.predict_proba(X_scaled)[:, 1]
        df_mitigation['_rank_prob'] = probs

        if isinstance(unprivileged_group, list):
            unpriv_mask = df_mitigation[sensitive_col].isin(unprivileged_group)
        else:
            unpriv_mask = df_mitigation[sensitive_col] == unprivileged_group

        priv_mask = df_mitigation[sensitive_col] == privileged_group

        n_unpriv = unpriv_mask.sum()
        n_priv = priv_mask.sum()

        if pd.api.types.is_numeric_dtype(y):
            target_rate = y.mean()
        else:
            target_rate = (y == 1).mean()

        current_unpriv_pos = (df_mitigation.loc[unpriv_mask, target_col] == 1).sum()
        target_unpriv_pos = int(n_unpriv * target_rate)

        current_priv_pos = (df_mitigation.loc[priv_mask, target_col] == 1).sum()
        target_priv_pos = int(n_priv * target_rate)

        # Flip labels for unprivileged group (promote candidates)
        flips_up = 0
        flips_down = 0
        if current_unpriv_pos < target_unpriv_pos:
            num_to_flip = int(target_unpriv_pos - current_unpriv_pos)
            to_flip = df_mitigation[unpriv_mask & (df_mitigation[target_col] == 0)].nlargest(num_to_flip, '_rank_prob').index
            df_mitigation.loc[to_flip, target_col] = 1
            flips_up += len(to_flip)
        elif current_unpriv_pos > target_unpriv_pos:
            num_to_flip = int(current_unpriv_pos - target_unpriv_pos)
            to_flip = df_mitigation[unpriv_mask & (df_mitigation[target_col] == 1)].nsmallest(num_to_flip, '_rank_prob').index
            df_mitigation.loc[to_flip, target_col] = 0
            flips_down += len(to_flip)

        # Flip labels for privileged group (demote candidates)
        if current_priv_pos > target_priv_pos:
            num_to_flip = int(current_priv_pos - target_priv_pos)
            to_flip = df_mitigation[priv_mask & (df_mitigation[target_col] == 1)].nsmallest(num_to_flip, '_rank_prob').index
            df_mitigation.loc[to_flip, target_col] = 0
            flips_down += len(to_flip)
        elif current_priv_pos < target_priv_pos:
            num_to_flip = int(target_priv_pos - current_priv_pos)
            to_flip = df_mitigation[priv_mask & (df_mitigation[target_col] == 0)].nlargest(num_to_flip, '_rank_prob').index
            df_mitigation.loc[to_flip, target_col] = 1
            flips_up += len(to_flip)

        if flips_up + flips_down > 0:
            print(f"✔ Relabeling successful: Flipped {flips_up} labels to 1 and {flips_down} labels to 0.")
        else:
            print("✔ Relabeling: No changes needed (dataset already balanced).")

        return df_mitigation.drop(columns=['_rank_prob'])

class SyntheticMitigation(MitigationStrategy):
    """Generates synthetic data to balance group representation.
    
    Supports SMOTE (interpolation-based) and CDA (counterfactual augmentation) 
    to create a more equitable training distribution.
    """

    def __init__(self, method: str = 'smote'):
        """Initializes with the synthetic generation method.

        Args:
            method: Either 'smote' or 'cda'.
        """
        self.method = method

    def mitigate(self, df: pd.DataFrame, target_col: str, sensitive_col: str, 
                 privileged_group: Any, unprivileged_group: Any, **kwargs) -> pd.DataFrame:
        """Applies synthetic data generation.

        Args:
            df: The training dataset.
            target_col: The outcome column name.
            sensitive_col: The protected attribute column name.
            privileged_group: Value of the privileged group.
            unprivileged_group: Value(s) of the unprivileged group(s).

        Returns:
            A DataFrame augmented with synthetic samples.
        """
        fairness_eval_col = FAIRNESS_EVAL_COL

        if self.method == 'smote':
            df_augmented = self._smote(df, target_col, sensitive_col, fairness_eval_col)
        elif self.method == 'cda':
            X_aug, y_aug = self._cda(df, target_col, sensitive_col, privileged_group, unprivileged_groups=unprivileged_group)
            df_augmented = pd.DataFrame(X_aug)
            y_values = y_aug.values if hasattr(y_aug, 'values') else y_aug
            df_augmented = pd.concat([df_augmented, pd.Series(y_values, name=target_col)], axis=1)
        else:
            raise ValueError(f"Unknown synthetic method: {self.method}")

        if df_augmented is None:
            return df.copy()

        return df_augmented

    def _smote(self, df: pd.DataFrame, target_col: str, sensitive_col: str, fairness_eval_col: str) -> Optional[pd.DataFrame]:
        """Interpolates new samples using SMOTE across group-outcome subgroups."""
        df_smote = df.copy()

        s_target = ensure_series(df_smote, target_col)
        
        if sensitive_col not in df_smote.columns and fairness_eval_col in df_smote.columns:
            df_smote[sensitive_col] = ensure_series(df_smote, fairness_eval_col).copy()

        if sensitive_col not in df_smote.columns:
            return None
            
        s_sens = ensure_series(df_smote, sensitive_col)

        is_sens_numeric = pd.api.types.is_numeric_dtype(s_sens)
        if not is_sens_numeric:
            from sklearn.preprocessing import LabelEncoder
            le_sens = LabelEncoder()
            df_smote[sensitive_col] = le_sens.fit_transform(s_sens.astype(str))
            s_sens = df_smote[sensitive_col]

        mapping_sens_to_eval = {}
        if fairness_eval_col in df_smote.columns:
            s_eval = ensure_series(df_smote, fairness_eval_col)
            pairs = pd.DataFrame({sensitive_col: s_sens, fairness_eval_col: s_eval}).drop_duplicates()
            mapping_sens_to_eval = dict(zip(pairs[sensitive_col], pairs[fairness_eval_col]))

        df_numeric = df_smote.select_dtypes(include=[np.number])

        if df_numeric.isnull().sum().sum() > 0:
            df_numeric = impute_missing(df_numeric, strategy='mean')

        y_combined = (s_sens.astype(str) + "_" + s_target.astype(str)).rename('_combined')

        cols_to_drop = [target_col]
        if fairness_eval_col in df_numeric.columns:
            cols_to_drop.append(fairness_eval_col)
        if sensitive_col in df_numeric.columns:
            cols_to_drop.append(sensitive_col)

        X = df_numeric.drop(columns=cols_to_drop)

        group_counts = y_combined.value_counts()
        min_samples = group_counts.min()
        n_neighbors = min(5, max(1, min_samples - 1))

        if min_samples < 2:
            print(f"⚠ Warning: Subgroup size {min_samples} too small for SMOTE. Skipping mitigation.")
            return None

        smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
        X_resampled, y_combined_resampled = smote.fit_resample(X, y_combined)

        if not isinstance(X_resampled, pd.DataFrame):
            df_resampled_features = pd.DataFrame(X_resampled, columns=X.columns).copy()
        else:
            df_resampled_features = X_resampled.copy()

        # Handle non-numeric columns that were dropped
        non_numeric_cols = df_smote.select_dtypes(exclude=[np.number]).columns.tolist()
        if target_col in non_numeric_cols: non_numeric_cols.remove(target_col)
        if sensitive_col in non_numeric_cols: non_numeric_cols.remove(sensitive_col)
        if fairness_eval_col in non_numeric_cols: non_numeric_cols.remove(fairness_eval_col)

        # Create the base for synthetic metadata
        n_resampled = len(df_resampled_features)
        n_original = len(df)
        
        res_combined_parts = y_combined_resampled.str.rsplit('_', n=1, expand=True)
        new_cols = {}
        new_cols[target_col] = res_combined_parts[1].astype(ensure_series(df, target_col).dtype).values

        sens_res_raw = res_combined_parts[0]
        if is_sens_numeric:
            new_cols[sensitive_col] = sens_res_raw.astype(ensure_series(df, sensitive_col).dtype).values
        else:
            new_cols[sensitive_col] = le_sens.inverse_transform(sens_res_raw.astype(int))

        if fairness_eval_col in df.columns:
            if mapping_sens_to_eval:
                key_type = ensure_series(df_smote, sensitive_col).dtype
                new_cols[fairness_eval_col] = sens_res_raw.astype(key_type).map(mapping_sens_to_eval).values
            else:
                new_cols[fairness_eval_col] = new_cols[sensitive_col].copy()

        df_final = pd.concat([df_resampled_features, pd.DataFrame(new_cols, index=df_resampled_features.index)], axis=1)

        # Re-attach non-numeric columns
        for col in non_numeric_cols:
            # For original indices, we can try to recover the data. 
            # However, SMOTE shuffled/shuffled indices. 
            # Safest is to fill original rows if indices match, or use placeholders for synthetic rows.
            # But SMOTE's fit_resample returns original rows first, then synthetic ones.
            orig_values = df[col].values
            synthetic_placeholder = "SYNTHETIC"
            
            new_values = np.empty(n_resampled, dtype=object)
            new_values[:n_original] = orig_values
            new_values[n_original:] = synthetic_placeholder
            df_final[col] = new_values

        return df_final

    def _cda(self, df: pd.DataFrame, target_col: str, sensitive_col: str, privileged_group: Any, unprivileged_groups: Any) -> Tuple[pd.DataFrame, pd.Series]:
        """Performs Counterfactual Data Augmentation by flipping the sensitive attribute."""
        print("Applying Counterfactual Data Augmentation (CDA)...")
        df_counterfactual = df.copy()

        priv_val = privileged_group[0] if isinstance(privileged_group, list) else privileged_group

        if isinstance(unprivileged_groups, list):
            mask_priv = df_counterfactual[sensitive_col] == priv_val
            mask_unpriv = df_counterfactual[sensitive_col].isin(unprivileged_groups)
            target_unpriv_val = unprivileged_groups[0]
            df_counterfactual.loc[mask_priv, sensitive_col] = target_unpriv_val
            df_counterfactual.loc[mask_unpriv, sensitive_col] = priv_val
        else:
            mask_priv = df_counterfactual[sensitive_col] == priv_val
            mask_unpriv = df_counterfactual[sensitive_col] == unprivileged_groups
            df_counterfactual.loc[mask_priv, sensitive_col] = unprivileged_groups
            df_counterfactual.loc[mask_unpriv, sensitive_col] = priv_val

        unique_vals = df[sensitive_col].unique()
        for val in unique_vals:
            oh_col = f"{sensitive_col}_{val}"
            if oh_col in df_counterfactual.columns:
                df_counterfactual[oh_col] = (df_counterfactual[sensitive_col] == val).astype(int)

        fairness_eval_col = FAIRNESS_EVAL_COL
        if fairness_eval_col in df_counterfactual.columns:
            df_counterfactual[fairness_eval_col] = df_counterfactual[sensitive_col].copy()

        df_augmented = pd.concat([df, df_counterfactual], ignore_index=True)
        return df_augmented.drop(columns=[target_col]), df_augmented[target_col]

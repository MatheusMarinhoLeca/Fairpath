import pandas as pd
import numpy as np
import warnings
import logging
import os
import sys
import contextlib
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from sdv.sampling import Condition
from typing import Any, List, Optional
from core.interfaces import MitigationStrategy

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
        target_pos_rate = pd.to_numeric(work_df['target'], errors='coerce').mean()

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

        ranker = LogisticRegression(max_iter=1000)
        ranker.fit(X_num, y)
        probs = ranker.predict_proba(X_num)[:, 1]
        df_work['_prob'] = probs

        if isinstance(unprivileged_group, list):
            unpriv_mask = df_work[sensitive_col].isin(unprivileged_group)
        else:
            unpriv_mask = df_work[sensitive_col] == unprivileged_group

        priv_mask = df_work[sensitive_col] == privileged_group

        n_unpriv = unpriv_mask.sum()
        n_priv = priv_mask.sum()

        target_rate = y.mean()

        current_unpriv_pos = df_work[unpriv_mask][target_col].sum()
        target_unpriv_pos = int(n_unpriv * target_rate)

        current_priv_pos = df_work[priv_mask][target_col].sum()
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
        if self.method == 'smote':
            X_res, y_res = self._smote(df, target_col, sensitive_col)
        elif self.method == 'cda':
            X_res, y_res = self._cda(df, target_col, sensitive_col, privileged_group, unprivileged_groups=unprivileged_group)
        else:
            raise ValueError(f"Unknown synthetic method: {self.method}")

        if X_res is None:
            return df.copy()

        resampled_df = pd.DataFrame(X_res)
        resampled_df[target_col] = y_res.values if hasattr(y_res, 'values') else y_res
        return resampled_df

    def _smote(self, df, target_col, sensitive_col):
        df_numeric = df.select_dtypes(include=[np.number]).copy()
        if sensitive_col not in df_numeric.columns:
            return None, None

        df_numeric['_combined'] = df_numeric[sensitive_col].astype(str) + "_" + df_numeric[target_col].astype(str)
        X = df_numeric.drop(columns=[target_col, '_combined'])
        y_combined = df_numeric['_combined']

        smote = SMOTE(random_state=42)
        X_res, y_combined_res = smote.fit_resample(X, y_combined)
        y_res = y_combined_res.apply(lambda x: int(float(x.rsplit('_', 1)[1])))
        return X_res, y_res

    def _cda(self, df, target_col, sensitive_col, privileged_group, unprivileged_groups):
        print("Applying Counterfactual Data Augmentation (CDA)...")
        df_cf = df.copy()

        priv_val = privileged_group[0] if isinstance(privileged_group, list) else privileged_group

        if isinstance(unprivileged_groups, list):
            for unpriv_val in unprivileged_groups:
                mask_priv = df_cf[sensitive_col] == priv_val
                mask_unpriv = df_cf[sensitive_col] == unpriv_val
                df_cf.loc[mask_priv, sensitive_col] = unpriv_val
                df_cf.loc[mask_unpriv, sensitive_col] = priv_val
        else:
            mask_priv = df_cf[sensitive_col] == priv_val
            mask_unpriv = df_cf[sensitive_col] == unprivileged_groups
            df_cf.loc[mask_priv, sensitive_col] = unprivileged_groups
            df_cf.loc[mask_unpriv, sensitive_col] = priv_val

        df_final = pd.concat([df, df_cf], ignore_index=True)
        return df_final.drop(columns=[target_col]), df_final[target_col]
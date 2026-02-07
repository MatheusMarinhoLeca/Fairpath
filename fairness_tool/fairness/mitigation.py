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

def mitigate_resampling(X, y, sensitive_col, strategy='oversample'):
    """
    Fairness-aware resampling. 
    Instead of just balancing the target Y, it balances each subgroup (A, Y) 
    to aim for Demographic Parity: P(Y=1 | A=0) = P(Y=1 | A=1).
    """
    df = X.copy()
    df['target'] = y
    
    # Identify unique groups (sensitive attribute, target)
    subgroups = df.groupby([sensitive_col, 'target']).size().reset_index(name='counts')
    
    # Calculate target counts per sensitive group to achieve parity
    # We aim to match the selection rate (P(Y=1|A)) across groups
    total_per_a = df.groupby(sensitive_col).size()
    max_total = total_per_a.max()
    
    # Calculate the global positive rate to use as a target
    target_pos_rate = df['target'].mean()
    
    df_resampled = []
    
    for a_val in df[sensitive_col].unique():
        group_a = df[df[sensitive_col] == a_val]
        n_a = len(group_a)
        
        # How many positives and negatives we WANT for this group
        target_n_pos = int(n_a * target_pos_rate) if strategy == 'undersample' else int(max_total * target_pos_rate)
        target_n_neg = (n_a - target_n_pos) if strategy == 'undersample' else (max_total - target_n_pos)
        
        pos_samples = group_a[group_a['target'] == 1]
        neg_samples = group_a[group_a['target'] == 0]
        
        if strategy == 'oversample':
            # Oversample/Adjust to reach exactly max_total
            # We use target_n_pos and target_n_neg strictly to ensure total size = max_total
            if len(pos_samples) > 0:
                df_resampled.append(pos_samples.sample(target_n_pos, replace=True))
            if len(neg_samples) > 0:
                df_resampled.append(neg_samples.sample(target_n_neg, replace=True))
        else:
            # Undersample to reach group's own total but with balanced ratio
            if len(pos_samples) > 0:
                df_resampled.append(pos_samples.sample(min(len(pos_samples), target_n_pos), replace=False))
            if len(neg_samples) > 0:
                df_resampled.append(neg_samples.sample(min(len(neg_samples), target_n_neg), replace=False))
                
    resampled_df = pd.concat(df_resampled).sample(frac=1).reset_index(drop=True)
    return resampled_df.drop(columns=['target']), resampled_df['target']

def mitigate_relabeling(df, target_col, sensitive_col, privileged_group, unprivileged_group, features_to_use=None):
    """
    Fairness-aware Relabeling (Massaging).
    Identifies samples near the decision boundary and flips their labels 
    to move both groups toward the global average positive rate.
    """
    df_work = df.copy()
    
    # Train a simple ranker
    if features_to_use is not None:
        X = df_work[[f for f in features_to_use if f in df_work.columns]]
        if target_col in X.columns:
            X = X.drop(columns=[target_col])
    else:
        X = df_work.drop(columns=[target_col])
        
    # Ensure numeric for ranking
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
        # Increase positives in unprivileged
        num_to_flip = int(target_unpriv_pos - current_unpriv_pos)
        to_flip = df_work[unpriv_mask & (df_work[target_col] == 0)].nlargest(num_to_flip, '_prob').index
        df_work.loc[to_flip, target_col] = 1
        flips_up += len(to_flip)
    elif current_unpriv_pos > target_unpriv_pos:
        # Decrease positives in unprivileged
        num_to_flip = int(current_unpriv_pos - target_unpriv_pos)
        to_flip = df_work[unpriv_mask & (df_work[target_col] == 1)].nsmallest(num_to_flip, '_prob').index
        df_work.loc[to_flip, target_col] = 0
        flips_down += len(to_flip)

    # PRIVILEGED adjustment
    if current_priv_pos > target_priv_pos:
        # Decrease positives in privileged
        num_to_flip = int(current_priv_pos - target_priv_pos)
        to_flip = df_work[priv_mask & (df_work[target_col] == 1)].nsmallest(num_to_flip, '_prob').index
        df_work.loc[to_flip, target_col] = 0
        flips_down += len(to_flip)
    elif current_priv_pos < target_priv_pos:
        # Increase positives in privileged
        num_to_flip = int(target_priv_pos - current_priv_pos)
        to_flip = df_work[priv_mask & (df_work[target_col] == 0)].nlargest(num_to_flip, '_prob').index
        df_work.loc[to_flip, target_col] = 1
        flips_up += len(to_flip)
    
    if flips_up + flips_down > 0:
        print(f"✔ Relabeling successful: Flipped {flips_up} labels to 1 and {flips_down} labels to 0.")
    else:
        print("✔ Relabeling: No changes needed (dataset already balanced).")
        
    return df_work.drop(columns=[target_col, '_prob']), df_work[target_col]

def mitigate_synthetic(df, target_col, sensitive_col, method='smote'):
    """
    Fairness-aware Synthetic Data Generation.
    Supports SMOTE or SDV (GaussianCopula).
    Focuses on balancing P(Y|A) by generating samples for disadvantaged subgroups.
    """
    if method == 'smote':
        return _mitigate_synthetic_smote(df, target_col, sensitive_col)
    elif method == 'sdv':
        return _mitigate_synthetic_sdv(df, target_col, sensitive_col)
    else:
        raise ValueError(f"Unknown synthetic method: {method}")

def _mitigate_synthetic_smote(df, target_col, sensitive_col):
    # SMOTE requires numeric data
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    if sensitive_col not in df_numeric.columns:
        return None, None
        
    # We create a synthetic combined label to balance subgroups
    df_numeric['_combined'] = df_numeric[sensitive_col].astype(str) + "_" + df_numeric[target_col].astype(str)
    
    X = df_numeric.drop(columns=[target_col, '_combined'])
    y_combined = df_numeric['_combined']
    
    # Apply SMOTE to balance the _combined groups
    smote = SMOTE(random_state=42)
    X_res, y_combined_res = smote.fit_resample(X, y_combined)
    
    # Extract original target from combined label
    # Using rsplit('_', 1) to correctly handle cases where the sensitive attribute value itself contains an underscore
    # Using float() before int() to handle strings like '1.0'
    y_res = y_combined_res.apply(lambda x: int(float(x.rsplit('_', 1)[1])))
    
    return X_res, y_res

def _mitigate_synthetic_sdv(df, target_col, sensitive_col):
    """
    Conditioned synthetic generation using SDV GaussianCopulaSynthesizer.
    """
    print("Training synthetic generator (this may take a minute)...")
    
    # Suppress all library outputs (SDV, Copulas, RDT, etc.)
    logging.getLogger('sdv').setLevel(logging.ERROR)
    logging.getLogger('copulas').setLevel(logging.ERROR)
    logging.getLogger('rdt').setLevel(logging.ERROR)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        # Absolute silence by redirecting system file descriptors
        with silence_output():
            df_work = df.copy()
            metadata = Metadata.detect_from_dataframe(data=df_work)
            
            synthesizer = GaussianCopulaSynthesizer(metadata)
            synthesizer.fit(df_work)
            
            # Goal: Equalize selection rates across sensitive groups.
            total_per_a = df_work.groupby(sensitive_col).size()
            max_total = int(total_per_a.max())
            target_pos_rate = float(df_work[target_col].mean())
            
            all_samples = [df_work]
            
            for a_val in df_work[sensitive_col].unique():
                target_n_pos = int(round(max_total * target_pos_rate))
                target_n_neg = int(max_total - target_n_pos)
                
                group_a = df_work[df_work[sensitive_col] == a_val]
                curr_pos = int((group_a[target_col] == 1).sum())
                curr_neg = int((group_a[target_col] == 0).sum())
                
                diff_pos = int(target_n_pos - curr_pos)
                diff_neg = int(target_n_neg - curr_neg)
                
                conditions = []
                a_val_std = a_val.item() if hasattr(a_val, 'item') else a_val
                
                if diff_pos > 0:
                    conditions.append(Condition(num_rows=diff_pos, column_values={sensitive_col: a_val_std, target_col: 1}))
                if diff_neg > 0:
                    conditions.append(Condition(num_rows=diff_neg, column_values={sensitive_col: a_val_std, target_col: 0}))
                    
                if conditions:
                    try:
                        samples = synthesizer.sample_from_conditions(conditions=conditions)
                        all_samples.append(samples)
                    except Exception as e:
                        pass 
            
            final_df = pd.concat(all_samples).sample(frac=1).reset_index(drop=True)
            return final_df.drop(columns=[target_col]), final_df[target_col]
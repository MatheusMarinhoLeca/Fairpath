import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition

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
    only as needed to improve statistical parity.
    1. Rank unprivileged samples with Y=0 by probability of being Y=1.
    2. Rank privileged samples with Y=1 by probability of being Y=0.
    3. Flip labels for the top candidates until parity is improved.
    """
    df_work = df.copy()
    
    # Train a simple ranker
    if features_to_use is not None:
        X = df_work[features_to_use]
        # Ensure target is not in X, though features_to_use shouldn't have it ideally
        if target_col in X.columns:
            X = X.drop(columns=[target_col])
    else:
        X = df_work.drop(columns=[target_col])
        
    # Ensure numeric
    X_num = X.select_dtypes(include=[np.number])
    y = df_work[target_col]
    
    ranker = LogisticRegression()
    ranker.fit(X_num, y)
    probs = ranker.predict_proba(X_num)[:, 1]
    df_work['_prob'] = probs
    
    # Calculate how many flips are needed for Statistical Parity
    # P(Y=1|unprivileged) should = P(Y=1|privileged)
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
    num_to_flip_up = max(0, target_unpriv_pos - current_unpriv_pos)
    
    current_priv_pos = df_work[priv_mask][target_col].sum()
    target_priv_pos = int(n_priv * target_rate)
    num_to_flip_down = max(0, current_priv_pos - target_priv_pos)
    
    # Flip unprivileged 0 -> 1 (those with highest prob of being 1)
    to_flip_up = df_work[unpriv_mask & (df_work[target_col] == 0)].nlargest(num_to_flip_up, '_prob').index
    df_work.loc[to_flip_up, target_col] = 1
    
    # Flip privileged 1 -> 0 (those with lowest prob of being 1)
    to_flip_down = df_work[priv_mask & (df_work[target_col] == 1)].nsmallest(num_to_flip_down, '_prob').index
    df_work.loc[to_flip_down, target_col] = 0
    
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
    y_res = y_combined_res.apply(lambda x: int(x.split('_')[1]))
    
    return X_res, y_res

def _mitigate_synthetic_sdv(df, target_col, sensitive_col):

    """

    Conditioned synthetic generation using SDV GaussianCopulaSynthesizer.

    """

    # Ensure standard types for SDV and metadata detection

    df_work = df.copy()

    

    metadata = SingleTableMetadata()

    metadata.detect_from_dataframe(df_work)

    

    # GaussianCopula is robust for general tabular data

    synthesizer = GaussianCopulaSynthesizer(metadata)

    synthesizer.fit(df_work)

    

    # Goal: Equalize selection rates across sensitive groups.

    # We upscale subgroups to reach demographic parity relative to the largest group.

    total_per_a = df_work.groupby(sensitive_col).size()

    max_total = int(total_per_a.max())

    target_pos_rate = float(df_work[target_col].mean())

    

    all_samples = [df_work]

    

    for a_val in df_work[sensitive_col].unique():

        # Calculate target counts for this group to reach max_total size

        target_n_pos = int(round(max_total * target_pos_rate))

        target_n_neg = int(max_total - target_n_pos)

        

        group_a = df_work[df_work[sensitive_col] == a_val]

        curr_pos = int((group_a[target_col] == 1).sum())

        curr_neg = int((group_a[target_col] == 0).sum())

        

        diff_pos = int(target_n_pos - curr_pos)

        diff_neg = int(target_n_neg - curr_neg)

        

        conditions = []

        # Standardize a_val to prevent type issues with SDV/Numpy

        a_val_std = a_val.item() if hasattr(a_val, 'item') else a_val

        

        if diff_pos > 0:

            conditions.append(Condition(

                num_rows=diff_pos,

                column_values={sensitive_col: a_val_std, target_col: 1}

            ))

            

        if diff_neg > 0:

            conditions.append(Condition(

                num_rows=diff_neg,

                column_values={sensitive_col: a_val_std, target_col: 0}

            ))

            

        if conditions:

            try:

                # sample_from_conditions expects a list of Condition objects

                samples = synthesizer.sample_from_conditions(conditions=conditions)

                all_samples.append(samples)

            except Exception as e:

                # Fallback: if sampling fails for a specific group, we continue with others

                print(f"  Warning: Sampling failed for group {a_val}: {e}")

            

    final_df = pd.concat(all_samples).sample(frac=1).reset_index(drop=True)

    return final_df.drop(columns=[target_col]), final_df[target_col]
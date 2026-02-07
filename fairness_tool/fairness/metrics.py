from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric, DatasetMetric
import pandas as pd
import numpy as np

def _encode_if_needed(df, protected_attribute, privileged_group, unprivileged_group):
    """
    Helper to encode categorical protected attributes to numeric (1/0) for AIF360.
    Returns: (encoded_df, priv_val, unpriv_val_list)
    """
    if pd.api.types.is_numeric_dtype(df[protected_attribute]):
        return df, privileged_group, unprivileged_group

    df_encoded = df.copy()
    
    # Create a mask for privileged and unprivileged
    mask_priv = df_encoded[protected_attribute] == privileged_group
    
    if isinstance(unprivileged_group, list):
        mask_unpriv = df_encoded[protected_attribute].isin(unprivileged_group)
    else:
        mask_unpriv = df_encoded[protected_attribute] == unprivileged_group
        
    # Assign numeric values: Privileged = 1, Unprivileged = 0
    # Note: If there are other groups not in either, they will be left as is (likely strings)
    # causing crash. So we should probably default everything else to 0 or -1.
    # Given the tool's flow, unprivileged_group usually covers "others".
    
    # Initialize with 0 (unprivileged default) or a distinct value? 
    # Let's start by forcing a numeric column.
    new_col = np.zeros(len(df_encoded), dtype=float)
    
    # Set privileged
    new_col[mask_priv] = 1.0
    
    # Set unprivileged (explicitly 0, though already 0 initialized, good for clarity)
    new_col[mask_unpriv] = 0.0
    
    df_encoded[protected_attribute] = new_col
    
    # New values
    new_priv_group = 1.0
    
    # For AIF360 unprivileged_groups param, if it was a list, we effectively mapped them all to 0.
    # So the new unprivileged target is just 0.
    if isinstance(unprivileged_group, list):
        new_unpriv_group = [0.0]
    else:
        new_unpriv_group = 0.0
        
    return df_encoded, new_priv_group, new_unpriv_group

def compute_group_fairness(df, target_col, protected_attribute, privileged_group, unprivileged_group):
    # Prepare AIF360 dataset for pre-prediction metrics (Demographic Parity)
    
    # Handle non-numeric protected attribute
    df_use, priv_val, unpriv_val = _encode_if_needed(df, protected_attribute, privileged_group, unprivileged_group)
    
    dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df_use[[target_col, protected_attribute]],
        label_names=[target_col],
        protected_attribute_names=[protected_attribute]
    )
    
    privileged_groups = [{protected_attribute: priv_val}]
    
    # Handle multiple values for unprivileged group (list)
    if isinstance(unpriv_val, list):
        unprivileged_groups = [{protected_attribute: val} for val in unpriv_val]
    else:
        unprivileged_groups = [{protected_attribute: unpriv_val}]
    
    metric = BinaryLabelDatasetMetric(
        dataset, 
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    results = {
        "Demographic Parity (Stat. Parity Diff.)": metric.mean_difference(),
        "Demographic Parity (Disparate Impact)": metric.disparate_impact()
    }
    
    return results

def compute_classification_fairness(df_true, df_pred, target_col, protected_attribute, privileged_group, unprivileged_group):
    # Handle non-numeric protected attribute
    # We assume df_true and df_pred have the same protected attribute values/types
    df_true_use, priv_val, unpriv_val = _encode_if_needed(df_true, protected_attribute, privileged_group, unprivileged_group)
    df_pred_use, _, _ = _encode_if_needed(df_pred, protected_attribute, privileged_group, unprivileged_group)

    # Prepare AIF360 datasets for post-prediction metrics (Equalized Odds)
    dataset_true = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df_true_use[[target_col, protected_attribute]],
        label_names=[target_col],
        protected_attribute_names=[protected_attribute]
    )
    
    dataset_pred = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df_pred_use[[target_col, protected_attribute]],
        label_names=[target_col],
        protected_attribute_names=[protected_attribute]
    )
    
    privileged_groups = [{protected_attribute: priv_val}]
    
    # Handle multiple values for unprivileged group (list)
    if isinstance(unpriv_val, list):
        unprivileged_groups = [{protected_attribute: val} for val in unpriv_val]
    else:
        unprivileged_groups = [{protected_attribute: unpriv_val}]
    
    metric = ClassificationMetric(
        dataset_true, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    results = {
        "Equalized Odds (Eq. Opp. Diff.)": metric.equal_opportunity_difference(),
        "Equalized Odds (Avg. Odds Diff.)": metric.average_odds_difference()
    }
    
    return results

def compute_individual_fairness(df, target_col, n_neighbors=5):
    # Consistency metric requires a dataset without protected attributes if we want pure individual fairness
    # but AIF360's DatasetMetric works on a BinaryLabelDataset
    
    # Consistency doesn't technically require protected attributes to be defined in 'protected_attribute_names'
    # BUT AIF360 might still try to cast the whole DF to float.
    # We should ensure everything passed to BinaryLabelDataset is numeric.
    
    # Filter only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df_numeric,
        label_names=[target_col],
        protected_attribute_names=[] # Individual fairness doesn't strictly need protected attrs
    )
    
    metric = DatasetMetric(dataset)
    consistency = metric.consistency(n_neighbors=n_neighbors)
    
    return {
        "Consistency Score": consistency[0]
    }

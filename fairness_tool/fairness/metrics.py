from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric, DatasetMetric
import pandas as pd
import numpy as np
from typing import Dict, Any
from core.interfaces import FairnessMetric

def _encode_if_needed(df, protected_attribute, privileged_group, unprivileged_group):
    """
    Helper to encode categorical protected attributes to numeric (1/0) for AIF360.
    """
    if pd.api.types.is_numeric_dtype(df[protected_attribute]):
        return df, privileged_group, unprivileged_group

    df_encoded = df.copy()
    mask_priv = df_encoded[protected_attribute] == privileged_group
    
    if isinstance(unprivileged_group, list):
        mask_unpriv = df_encoded[protected_attribute].isin(unprivileged_group)
    else:
        mask_unpriv = df_encoded[protected_attribute] == unprivileged_group
        
    new_col = np.zeros(len(df_encoded), dtype=float)
    new_col[mask_priv] = 1.0
    new_col[mask_unpriv] = 0.0
    
    df_encoded[protected_attribute] = new_col
    new_priv_group = 1.0
    
    if isinstance(unprivileged_group, list):
        new_unpriv_group = [0.0]
    else:
        new_unpriv_group = 0.0
        
    return df_encoded, new_priv_group, new_unpriv_group

class GroupFairnessMetric(FairnessMetric):
    def compute(self, df: pd.DataFrame, target_col: str, protected_attribute: str, 
               privileged_group: Any, unprivileged_group: Any) -> Dict[str, float]:
        df_use, priv_val, unpriv_val = _encode_if_needed(df, protected_attribute, privileged_group, unprivileged_group)
        
        dataset = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=df_use[[target_col, protected_attribute]],
            label_names=[target_col],
            protected_attribute_names=[protected_attribute]
        )
        
        privileged_groups = [{protected_attribute: priv_val}]
        unprivileged_groups = [{protected_attribute: val} for val in (unpriv_val if isinstance(unpriv_val, list) else [unpriv_val])]
        
        metric = BinaryLabelDatasetMetric(
            dataset, 
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        
        return {
            "Statistical Parity Difference": metric.statistical_parity_difference(),
            "Disparate Impact": metric.disparate_impact()
        }

class ClassificationFairnessMetric(FairnessMetric):
    def __init__(self, df_true: pd.DataFrame):
        self.df_true = df_true

    def compute(self, df_pred: pd.DataFrame, target_col: str, protected_attribute: str, 
               privileged_group: Any, unprivileged_group: Any) -> Dict[str, float]:
        # Reset indices to ensure positional alignment and avoid AIF360 reordering issues
        df_true_clean = self.df_true.reset_index(drop=True)
        df_pred_clean = df_pred.reset_index(drop=True)
        
        df_true_use, priv_val, unpriv_val = _encode_if_needed(df_true_clean, protected_attribute, privileged_group, unprivileged_group)
        df_pred_use, _, _ = _encode_if_needed(df_pred_clean, protected_attribute, privileged_group, unprivileged_group)

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
        unprivileged_groups = [{protected_attribute: val} for val in (unpriv_val if isinstance(unpriv_val, list) else [unpriv_val])]
        
        metric = ClassificationMetric(
            dataset_true, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        
        return {
            "Equal Opportunity Difference": metric.equal_opportunity_difference(),
            "Average Odds Difference": metric.average_odds_difference()
        }

def compute_classification_fairness(df_true, df_pred, target_col, protected_attribute, privileged_group, unprivileged_group):
    return ClassificationFairnessMetric(df_true).compute(df_pred, target_col, protected_attribute, privileged_group, unprivileged_group)

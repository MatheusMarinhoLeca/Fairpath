from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List, Tuple
from fairpath.core.interfaces import FairnessMetric
from fairpath.data.utils import ensure_series

def _encode_protected_attribute(
    df: pd.DataFrame, 
    protected_attribute: str, 
    privileged_group: Any, 
    unprivileged_group: Union[Any, List[Any]]
) -> Tuple[pd.DataFrame, float, Union[float, List[float]]]:
    """Encodes categorical protected attributes into a standardized binary numeric format.

    This function is a critical path for all fairness metrics. It ensures 
    compatibility with AIF360 metrics which strictly require numeric protected 
    attributes. 

    Encoding Mapping:
        - 1.0: Assigned to the 'privileged_group' value.
        - 0.0: Assigned to any value in 'unprivileged_group'.

    Note:
        If the attribute is already numeric, it is returned as-is to avoid 
        distorting existing numeric group definitions.

    Args:
        df: The DataFrame containing the protected attribute.
        protected_attribute: Name of the column to encode.
        privileged_group: The value representing the privileged group.
        unprivileged_group: The value(s) representing the unprivileged group(s).

    Returns:
        A tuple containing:
            - The DataFrame with the encoded protected attribute.
            - The numeric value for the privileged group (fixed at 1.0).
            - The numeric value(s) for the unprivileged group(s) (fixed at 0.0).
    """
    s_prot = ensure_series(df, protected_attribute)
    
    if pd.api.types.is_numeric_dtype(s_prot):
        return df, privileged_group, unprivileged_group

    df_encoded = df.copy()
    mask_priv = s_prot == privileged_group
    
    if isinstance(unprivileged_group, list):
        mask_unpriv = s_prot.isin(unprivileged_group)
    else:
        mask_unpriv = s_prot == unprivileged_group
        
    new_col = np.zeros(len(df_encoded), dtype=float)
    new_col[mask_priv] = 1.0
    new_col[mask_unpriv] = 0.0
    
    if isinstance(df[protected_attribute], pd.DataFrame):
        df_encoded = df_encoded.drop(columns=[protected_attribute])
        
    df_encoded[protected_attribute] = new_col
    new_priv_group = 1.0
    
    if isinstance(unprivileged_group, list):
        new_unpriv_group = [0.0]
    else:
        new_unpriv_group = 0.0
        
    return df_encoded, new_priv_group, new_unpriv_group

class GroupFairnessMetric(FairnessMetric):
    """Computes group-level fairness metrics based on dataset outcomes.
    
    This class focuses on parity metrics that do not require ground truth labels, 
    such as Statistical Parity Difference and Disparate Impact.
    """

    def compute(self, df: pd.DataFrame, target_col: str, protected_attribute: str, 
               privileged_group: Any, unprivileged_group: Any) -> Dict[str, float]:
        """Computes Statistical Parity and Disparate Impact.

        Args:
            df: The dataset or model predictions.
            target_col: The outcome column (e.g., predicted labels).
            protected_attribute: The sensitive feature.
            privileged_group: The value of the privileged group.
            unprivileged_group: The value(s) of the unprivileged group(s).

        Returns:
            A dictionary containing the calculated fairness metrics.
        """
        s_prot = ensure_series(df, protected_attribute)
        if isinstance(unprivileged_group, list):
            relevant_mask = (s_prot == privileged_group) | (s_prot.isin(unprivileged_group))
        else:
            relevant_mask = (s_prot == privileged_group) | (s_prot == unprivileged_group)
            
        df_relevant_groups = df[relevant_mask].copy()
        
        df_encoded, priv_val, unpriv_val = _encode_protected_attribute(
            df_relevant_groups, protected_attribute, privileged_group, unprivileged_group
        )
        
        s_target = ensure_series(df_encoded, target_col)
        df_dataset = pd.DataFrame({
            target_col: s_target,
            protected_attribute: ensure_series(df_encoded, protected_attribute)
        })

        dataset = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=df_dataset,
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
    """Computes fairness metrics based on classification performance.
    
    This class focuses on error-rate parity metrics that require ground truth labels, 
    such as Equal Opportunity Difference and Average Odds Difference.
    """

    def __init__(self, df_true: pd.DataFrame):
        """Initializes with ground truth data.

        Args:
            df_true: DataFrame containing ground truth labels.
        """
        self.df_true = df_true

    def compute(self, df_pred: pd.DataFrame, target_col: str, protected_attribute: str, 
               privileged_group: Any, unprivileged_group: Any) -> Dict[str, float]:
        """Computes Equal Opportunity and Average Odds Difference.

        Args:
            df_pred: DataFrame containing model predictions.
            target_col: The outcome column.
            protected_attribute: The sensitive feature.
            privileged_group: The value of the privileged group.
            unprivileged_group: The value(s) of the unprivileged group(s).

        Returns:
            A dictionary containing the calculated fairness metrics.
        """
        df_true_clean = self.df_true.reset_index(drop=True)
        df_pred_clean = df_pred.reset_index(drop=True)
        
        s_prot = ensure_series(df_true_clean, protected_attribute)
        if isinstance(unprivileged_group, list):
            relevant_mask = (s_prot == privileged_group) | (s_prot.isin(unprivileged_group))
        else:
            relevant_mask = (s_prot == privileged_group) | (s_prot == unprivileged_group)
            
        df_true_relevant = df_true_clean[relevant_mask].copy()
        df_pred_relevant = df_pred_clean[relevant_mask].copy()
        
        df_true_encoded, priv_val, unpriv_val = _encode_protected_attribute(
            df_true_relevant, protected_attribute, privileged_group, unprivileged_group
        )
        df_pred_encoded, _, _ = _encode_protected_attribute(
            df_pred_relevant, protected_attribute, privileged_group, unprivileged_group
        )

        df_true_dataset = pd.DataFrame({
            target_col: ensure_series(df_true_encoded, target_col),
            protected_attribute: ensure_series(df_true_encoded, protected_attribute)
        })
        df_pred_dataset = pd.DataFrame({
            target_col: ensure_series(df_pred_encoded, target_col),
            protected_attribute: ensure_series(df_pred_encoded, protected_attribute)
        })

        dataset_true = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=df_true_dataset,
            label_names=[target_col],
            protected_attribute_names=[protected_attribute]
        )
        
        dataset_pred = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=df_pred_dataset,
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

def compute_classification_fairness(
    df_true: pd.DataFrame, 
    df_pred: pd.DataFrame, 
    target_col: str, 
    protected_attribute: str, 
    privileged_group: Any, 
    unprivileged_group: Any
) -> Dict[str, float]:
    """Helper function to compute classification fairness metrics in one call."""
    return ClassificationFairnessMetric(df_true).compute(df_pred, target_col, protected_attribute, privileged_group, unprivileged_group)

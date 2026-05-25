import pandas as pd
import numpy as np

def get_basic_stats(df):
    stats = {
        'Total Samples': df.shape[0],
        'Missing Values': df.isnull().sum().sum(),
    }
    return stats

def get_comprehensive_stats(df, target_col, sensitive_col, original_feat_count=None, selected_features=None):
    """
    Computes detailed statistics including subgroup distributions and target rates.
    """
    stats = get_basic_stats(df)
    stats['Target Variable'] = target_col
    stats['Sensitive Attribute'] = sensitive_col
    
    if original_feat_count is not None:
        stats['Total Features'] = original_feat_count
    
    if selected_features is not None:
        stats['Features Selected'] = len(selected_features)
    else:
        # Fallback to total columns minus target if no selection was made
        stats['Features Selected'] = df.shape[1] - 1

    # Feature Types (of the full dataframe)
    stats['Numerical Features'] = len(df.select_dtypes(include=['number']).columns)
    stats['Categorical Features'] = len(df.select_dtypes(exclude=['number']).columns)
    
    # Global target rate
    pos_rate = df[target_col].mean()
    stats['Global Positive Rate'] = pos_rate
    
    # Subgroup statistics
    groups = df.groupby(sensitive_col)
    
    for name, group in groups:
        group_size = len(group)
        group_pos_rate = group[target_col].mean()
        
        stats[f"Group '{name}' Size"] = group_size
        stats[f"Group '{name}' Pos. Rate"] = group_pos_rate
        
    return stats

def get_target_distribution(df, target_col):
    return df[target_col].value_counts(normalize=True).to_dict()
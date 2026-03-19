import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

def save_plot(fig, filename, output_dir="outputs/reports/assets"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True) 
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path

def plot_class_distribution(df, target_col):
    fig = plt.figure(figsize=(8, 6))
    # Using data=df and x=target_col allows Seaborn to handle types internally
    sns.countplot(data=df, x=target_col)
    plt.title(f'Distribution of {target_col}')
    return save_plot(fig, 'class_distribution.png')

def plot_missing_heatmap(df):
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    return save_plot(fig, 'missing_values_heatmap.png')

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return save_plot(fig, filename)

def plot_metric_comparison(metrics_before, metrics_after):
    common_keys = [k for k in metrics_before.keys() if isinstance(metrics_before[k], (int, float))]
    
    data = {
        'Metric': common_keys,
        'Baseline': [metrics_before[k] for k in common_keys],
        'Mitigated': [metrics_after.get(k, 0) for k in common_keys]
    }
    
    df_plot = pd.DataFrame(data)
    df_melted = df_plot.melt(id_vars='Metric', var_name='Stage', value_name='Value')
    
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Value', hue='Stage', data=df_melted)
    plt.title('Performance & Fairness Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return save_plot(fig, 'metrics_comparison.png')

def plot_box_plots(df, numeric_cols, filename):
    n_cols = len(numeric_cols)
    if n_cols == 0: return None
    
    plot_cols = numeric_cols[:12] 
    rows = (len(plot_cols) + 2) // 3
    fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(15, 4 * rows))
    axes = axes.flatten() if rows > 1 or len(plot_cols) > 1 else [axes]
    
    for i, col in enumerate(plot_cols):
        sns.boxplot(data=df, y=col, ax=axes[i])
        axes[i].set_title(col)
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    return save_plot(fig, filename)

def plot_violin_plots(df, numeric_cols, target_col, filename):
    n_cols = len(numeric_cols)
    if n_cols == 0: return None
    
    plot_cols = numeric_cols[:12]
    rows = (len(plot_cols) + 2) // 3
    fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(15, 4 * rows))
    axes = axes.flatten() if rows > 1 or len(plot_cols) > 1 else [axes]
    
    for i, col in enumerate(plot_cols):
        sns.violinplot(data=df, x=target_col, y=col, ax=axes[i])
        axes[i].set_title(f"{col} by {target_col}")
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    return save_plot(fig, filename)

def plot_correlation_heatmap(df, numeric_cols, filename):
    if len(numeric_cols) < 2: return None
    
    corr_cols = numeric_cols[:20]
    corr = df[corr_cols].corr()
    
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
    plt.title('Feature Correlation Heatmap')
    return save_plot(fig, filename)

def plot_target_conditioned_bar_charts(df, cat_cols, target_col, filename):
    n_cols = len(cat_cols)
    if n_cols == 0: return None
    
    plot_cols = cat_cols[:9] 
    rows = (len(plot_cols) + 1) // 2
    fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(15, 5 * rows))
    axes = axes.flatten() if rows > 1 or len(plot_cols) > 1 else [axes]
    
    for i, col in enumerate(plot_cols):
        sns.countplot(data=df, x=col, hue=target_col, ax=axes[i])
        axes[i].set_title(f"{col} distribution by {target_col}")
        axes[i].tick_params(axis='x', rotation=45)
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    return save_plot(fig, filename)

def plot_precision_recall_curve(y_true, y_probs, title, filename):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', markersize=1, label=f'AP={avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return save_plot(fig, filename)

def plot_distribution_comparison(df_before, df_after, column, title, filename):
    """Plots side-by-side distribution of a column to show changes after mitigation."""
    df_b = df_before.copy()
    df_a = df_after.copy()
    
    df_b['Stage'] = 'Baseline'
    df_a['Stage'] = 'Mitigated'
    
    combined = pd.concat([df_b[[column, 'Stage']], df_a[[column, 'Stage']]])
    
    fig = plt.figure(figsize=(10, 6))
    sns.countplot(data=combined, x=column, hue='Stage', palette='muted')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return save_plot(fig, filename)

def plot_sensitive_target_comparison(df_before, df_after, sensitive_col, target_col, filename):
    """Plots how the relationship between sensitive attribute and target changes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Baseline
    sns.countplot(data=df_before, x=sensitive_col, hue=target_col, ax=axes[0], palette='viridis')
    axes[0].set_title(f'Baseline: {target_col} by {sensitive_col}')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Mitigated
    sns.countplot(data=df_after, x=sensitive_col, hue=target_col, ax=axes[1], palette='viridis')
    axes[1].set_title(f'Mitigated: {target_col} by {sensitive_col}')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return save_plot(fig, filename)

def plot_selection_rates(df_before, df_after, sensitive_col, target_col, filename):
    """
    Plots P(Y=1 | A) comparison. Selection rate per subgroup.
    """
    def get_rates(df):
        # Ensure target is numeric for calculation
        df_num = df.copy()
        df_num[target_col] = pd.to_numeric(df_num[target_col], errors='coerce')
        return df_num.groupby(sensitive_col)[target_col].mean().reset_index(name='Selection Rate')

    rates_before = get_rates(df_before)
    rates_before['Stage'] = 'Baseline'
    
    rates_after = get_rates(df_after)
    rates_after['Stage'] = 'Mitigated'
    
    combined = pd.concat([rates_before, rates_after])
    
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(data=combined, x=sensitive_col, y='Selection Rate', hue='Stage')
    
    # Global Rate
    global_rate = pd.to_numeric(df_before[target_col], errors='coerce').mean()
    plt.axhline(global_rate, color='r', linestyle='--', label='Global Rate (Baseline)')
    
    plt.title(f'Selection Rate P(Y=1 | {sensitive_col}) Comparison')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    return save_plot(fig, filename)

def plot_contingency_heatmap(df, sensitive_col, target_col, title, filename):
    """
    Heatmap of (Sensitive Attribute, Target) counts.
    """
    contingency = pd.crosstab(df[sensitive_col], df[target_col])
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(title)
    return save_plot(fig, filename)

def plot_kde_probabilities(y_probs, sensitive_attr, filename):
    """KDE Plots of Predicted Probabilities by sensitive group."""
    df = pd.DataFrame({'Probability': y_probs, 'Group': sensitive_attr})
    fig = plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Probability', hue='Group', fill=True, common_norm=False, alpha=0.5)
    plt.title('KDE of Predicted Probabilities by Group')
    plt.xlim(0, 1)
    return save_plot(fig, filename)

def plot_subgroup_confusion_matrices(y_true, y_pred, sensitive_attr, filename):
    """Subgroup-Specific Confusion Matrices."""
    groups = np.unique(sensitive_attr)
    n_groups = len(groups)
    cols = 2
    rows = (n_groups + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten()
    
    for i, group in enumerate(groups):
        mask = (sensitive_attr == group)
        if not any(mask):
            axes[i].axis('off')
            continue
        cm = confusion_matrix(y_true[mask], y_pred[mask])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Group: {group}')
        axes[i].set_ylabel('True')
        axes[i].set_xlabel('Pred')
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    return save_plot(fig, filename)

def plot_fairness_utility_tradeoff(metrics_list, fairness_metric, utility_metric, filename):
    """
    Fairness–Utility Trade-off Scatter Plot.
    metrics_list: list of dicts with metrics.
    """
    df = pd.DataFrame(metrics_list)
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=fairness_metric, y=utility_metric, hue='Stage', s=100)
    plt.title(f'Trade-off: {utility_metric} vs {fairness_metric}')
    plt.grid(True, linestyle='--', alpha=0.6)
    return save_plot(fig, filename)

def plot_grouped_bar_charts(df, sensitive_col, target_col, filename):
    """Grouped Bar Charts by Subgroup for categorical target rates."""
    df_num = df.copy()
    df_num[target_col] = pd.to_numeric(df_num[target_col], errors='coerce')
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(data=df_num, x=sensitive_col, y=target_col, errorbar=None)
    plt.title(f'Positive Rate by {sensitive_col}')
    plt.ylabel('Positive Rate (Mean of Target)')
    return save_plot(fig, filename)

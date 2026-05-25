from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    balanced_accuracy_score,
    roc_auc_score
)
import numpy as np

def evaluate_classification(y_true, y_pred, y_prob=None):
    """
    Computes standard accuracy and performance metrics directly from scikit-learn.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (positive class), optional
        
    Returns:
        dict: Contains metrics with direct library equivalents.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        try:
            # Check if binary classification
            if len(np.unique(y_true)) == 2:
                metrics['ROC AUC'] = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
            
    return metrics

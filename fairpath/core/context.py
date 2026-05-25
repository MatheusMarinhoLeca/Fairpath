import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class ProjectContext:
    """Encapsulates the current state of the fairness project.
    
    This context object acts as the primary data store for the interactive 
    workflow, holding everything from raw datasets to final experiment results.
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the context to its initial state."""
        self.df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.target_col: Optional[str] = None
        self.sensitive_col: Optional[str] = None
        self.privileged_group: Any = None
        self.unprivileged_group: Any = None
        self.privileged_group_name: Optional[str] = None
        self.task_type: Optional[str] = None
        self.model: Any = None
        self.model_choice: str = 'logistic'
        self.metrics_baseline: Dict[str, float] = {}
        self.metrics_mitigated: Dict[str, float] = {}
        self.stats_baseline: Dict[str, Any] = {}
        self.stats_mitigated: Dict[str, Any] = {}
        self.stats_raw: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}
        
        # Baseline results
        self.baseline_test_labels: Optional[pd.Series] = None
        self.baseline_predictions: Optional[np.ndarray] = None
        self.baseline_probabilities: Optional[np.ndarray] = None
        
        # Mitigated results
        self.mitigated_test_labels: Optional[pd.Series] = None
        self.mitigated_predictions: Optional[np.ndarray] = None
        self.mitigated_probabilities: Optional[np.ndarray] = None
        
        self.df_improved: Optional[pd.DataFrame] = None
        self.inverse_sensitive_mapping: Optional[Dict[Any, Any]] = None
        self.selected_features: Optional[List[str]] = None
        self.comparison_mode: str = 'combined'
        self.selections: Dict[str, Any] = {
            'preprocessing': {},
            'fairness': {},
            'mitigation': {},
            'model': {}
        }
        self.metric_choice: Optional[str] = None

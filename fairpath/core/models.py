from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for data cleaning and preprocessing steps."""
    feature_selection_method: Optional[str] = None
    selected_features: List[str] = field(default_factory=list)
    missing_strategy: Optional[str] = None
    outlier_strategy: Optional[str] = None
    encoding_strategy: Optional[str] = None
    remaining_categorical_strategy: Optional[str] = None

@dataclass(frozen=True)
class FairnessConfig:
    """Configuration for fairness audit and mitigation."""
    sensitive_col: str
    selected_attributes: List[str]
    privileged_group: Any
    privileged_group_name: str
    unprivileged_group: List[Any]
    comparison_mode: str = 'combined'
    metric_choice: str = '1'
    model_choice: str = 'logistic'
    inverse_mapping: Optional[Dict[Any, Any]] = None

@dataclass
class TrainingResult:
    """Detailed results from a model training session."""
    model: Any
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    y_prob_val: Optional[np.ndarray]
    y_prob_test: Optional[np.ndarray]
    train_metrics: Dict[str, float]

@dataclass
class ExperimentResult:
    """Results from a single model training and evaluation run."""
    metrics: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    y_test: Optional[pd.Series] = None
    y_pred: Optional[np.ndarray] = None
    y_prob: Optional[np.ndarray] = None
    model: Any = None

@dataclass(frozen=True)
class Recommendation:
    """Standardized DTO for evidence-based fairness recommendations."""
    category: str
    description: str
    evidence: str
    action: str
    confidence_level: str = "Medium"
    p_value: Optional[float] = None

@dataclass
class AuditReportData:
    """Aggregated data required for generating a full fairness report."""
    baseline: ExperimentResult
    mitigated: Optional[ExperimentResult] = None
    df_baseline: Optional[pd.DataFrame] = None
    df_mitigated: Optional[pd.DataFrame] = None
    preprocessing_selections: Dict[str, Any] = field(default_factory=dict)
    fairness_selections: Dict[str, Any] = field(default_factory=dict)
    mitigation_selections: Dict[str, Any] = field(default_factory=dict)
    target_col: Optional[str] = None
    sensitive_col: Optional[str] = None

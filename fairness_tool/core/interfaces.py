from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class DataPreprocessor(ABC):
    """Abstract base class for all data cleaning and preprocessing steps."""
    @abstractmethod
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass

class FairnessMetric(ABC):
    """Abstract base class for fairness metrics."""
    @abstractmethod
    def compute(self, df: pd.DataFrame, target_col: str, sensitive_col: str, 
               privileged_group: Any, unprivileged_group: Any) -> Dict[str, float]:
        pass

class MitigationStrategy(ABC):
    """Abstract base class for bias mitigation techniques."""
    @abstractmethod
    def mitigate(self, df: pd.DataFrame, target_col: str, sensitive_col: str, 
                 privileged_group: Any, unprivileged_group: Any, **kwargs) -> pd.DataFrame:
        pass

class ModelTrainer(ABC):
    """Abstract base class for model training and evaluation."""
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> Any:
        pass
    
    @abstractmethod
    def evaluate(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        pass

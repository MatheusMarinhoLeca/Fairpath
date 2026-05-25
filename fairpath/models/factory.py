from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, Any, Callable, Type

class ModelFactory:
    """Factory for creating and registering machine learning models."""
    
    _models: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, model_name: str, constructor: Callable[..., Any]):
        """Registers a new model constructor."""
        cls._models[model_name] = constructor

    @classmethod
    def get_model(cls, model_name: str, **kwargs) -> Any:
        """Creates a model instance based on the registered name."""
        if model_name not in cls._models:
            raise ValueError(f"Model '{model_name}' is not registered in ModelFactory.")
        return cls._models[model_name](**kwargs)

# Register default models with original hyperparameters
ModelFactory.register('logistic', lambda **kwargs: LogisticRegression(
    C=1.0, solver='lbfgs', max_iter=10000, random_state=42, **kwargs
))

ModelFactory.register('random_forest', lambda **kwargs: RandomForestClassifier(
    n_estimators=100, max_depth=8, min_samples_split=5, min_samples_leaf=2, 
    random_state=42, n_jobs=-1, **kwargs
))

ModelFactory.register('gbm', lambda **kwargs: HistGradientBoostingClassifier(
    max_iter=1000, 
    learning_rate=0.05,
    max_depth=4, 
    l2_regularization=0.5,
    n_iter_no_change=20,
    random_state=42,
    **kwargs
))

def create_svm(**kwargs):
    base_model = LinearSVC(dual="auto", C=0.8, max_iter=30000, random_state=42, **kwargs)
    return CalibratedClassifierCV(base_model)

ModelFactory.register('svm', create_svm)
ModelFactory.register('linear_svc', create_svm)

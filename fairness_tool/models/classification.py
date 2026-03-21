from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
import pandas as pd
from typing import Dict, Any, Tuple
from core.interfaces import ModelTrainer
from evaluation.performance import evaluate_classification

class DefaultModelTrainer(ModelTrainer):
    def train(self, X, y, model_type='logistic') -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
        
        # Scale and Normalize data
        # LinearSVC and LogisticRegression are sensitive to feature scaling.
        # Standardizing (StandardScaler) followed by Normalizing (Normalizer) ensures
        # all features have zero mean, unit variance, and the sample vectors have unit norm,
        # which significantly improves convergence for LinearSVC.
        scaler = StandardScaler()
        normalizer = Normalizer()
        
        # Capture columns and indices if input is a DataFrame
        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            X_train_cols, X_train_idx = X_train.columns, X_train.index
            X_val_cols, X_val_idx = X_val.columns, X_val.index
            X_test_cols, X_test_idx = X_test.columns, X_test.index

        X_train = scaler.fit_transform(X_train)
        X_train = normalizer.fit_transform(X_train)
        
        X_val = scaler.transform(X_val)
        X_val = normalizer.transform(X_val)
        
        X_test = scaler.transform(X_test)
        X_test = normalizer.transform(X_test)
        
        # Restore as DataFrames if necessary
        if is_df:
            X_train = pd.DataFrame(X_train, columns=X_train_cols, index=X_train_idx)
            X_val = pd.DataFrame(X_val, columns=X_val_cols, index=X_val_idx)
            X_test = pd.DataFrame(X_test, columns=X_test_cols, index=X_test_idx)
        
        if model_type == 'logistic':
            model = LogisticRegression(solver='lbfgs', max_iter=5000, random_state=42)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
        elif model_type == 'gbm':
            # HistGradientBoostingClassifier is significantly faster and often more accurate 
            # than GradientBoostingClassifier for medium to large datasets.
            model = HistGradientBoostingClassifier(
                max_iter=1000, 
                learning_rate=0.1,
                max_depth=5,
                l2_regularization=0.1,
                n_iter_no_change=15,
                random_state=42
            )
        elif model_type == 'svm' or model_type == 'linear_svc':
            # LinearSVC does not support predict_proba by default, so we use CalibratedClassifierCV
            # Increasing max_iter significantly to ensure convergence on all datasets/subgroups
            base_model = LinearSVC(dual="auto", max_iter=20000, random_state=42)
            model = CalibratedClassifierCV(base_model)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train, y_train)
        
        # Get probabilities for test set
        y_prob_test = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        y_prob_val = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
        
        return model, X_val, y_val, X_test, y_test, y_prob_val, y_prob_test

    def evaluate(self, model, X, y) -> Dict[str, float]:
        y_pred = model.predict(X)
        return evaluate_classification(y, y_pred)

# Backward compatibility wrapper
def train_classifier(X, y, model_type='logistic'):
    res = DefaultModelTrainer().train(X, y, model_type)
    # Return all 7 values
    return res

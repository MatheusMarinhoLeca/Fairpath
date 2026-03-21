from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, Normalizer
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from core.interfaces import ModelTrainer
from evaluation.performance import evaluate_classification

class DefaultModelTrainer(ModelTrainer):
    def train(self, X, y, model_type='logistic', X_val=None, y_val=None, X_test=None, y_test=None) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Dict[str, float]]:
        """
        Trains a model and returns (model, X_val, y_val, X_test, y_test, y_prob_val, y_prob_test, train_metrics)
        """
        if X_test is not None and y_test is not None:
            # Pre-split data provided. X and y are treated as X_train, y_train.
            X_train = X
            y_train = y
            if X_val is None or y_val is None:
                # Create validation split from training data if not provided
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        else:
            # Standard split
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
        
        # Scale and Normalize data
        scaler = StandardScaler()
        normalizer = Normalizer()
        
        # Capture columns and indices if input is a DataFrame
        is_df = isinstance(X_train, pd.DataFrame)
        if is_df:
            X_train_cols, X_train_idx = X_train.columns, X_train.index
            X_val_cols, X_val_idx = X_val.columns, X_val.index
            X_test_cols, X_test_idx = X_test.columns, X_test.index

        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = normalizer.fit_transform(X_train_scaled)
        
        X_val_scaled = scaler.transform(X_val)
        X_val_scaled = normalizer.transform(X_val_scaled)
        
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = normalizer.transform(X_test_scaled)
        
        # Restore as DataFrames if necessary
        if is_df:
            X_train_final = pd.DataFrame(X_train_scaled, columns=X_train_cols, index=X_train_idx)
            X_val_final = pd.DataFrame(X_val_scaled, columns=X_val_cols, index=X_val_idx)
            X_test_final = pd.DataFrame(X_test_scaled, columns=X_test_cols, index=X_test_idx)
        else:
            X_train_final = X_train_scaled
            X_val_final = X_val_scaled
            X_test_final = X_test_scaled
        
        if model_type == 'logistic':
            # Explicitly set L2 regularization and high max_iter for stability
            model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=10000, random_state=42)
        elif model_type == 'random_forest':
            # Added n_jobs=-1 to parallelize tree building
            model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
        elif model_type == 'gbm':
            model = HistGradientBoostingClassifier(
                max_iter=1000, 
                learning_rate=0.05, # Slower learning rate for better generalization
                max_depth=4, # Reduced depth
                l2_regularization=0.5, # Increased regularization
                n_iter_no_change=20,
                random_state=42
            )
        elif model_type == 'svm' or model_type == 'linear_svc':
            base_model = LinearSVC(dual="auto", C=0.8, max_iter=30000, random_state=42)
            model = CalibratedClassifierCV(base_model)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Perform Cross-Validation on the training set in parallel (n_jobs=-1)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        try:
            cv_scores = cross_val_score(model, X_train_final, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception:
            cv_mean = 0.0
            cv_std = 0.0

        model.fit(X_train_final, y_train)
        
        # Calculate training metrics for overfitting detection
        y_pred_train = model.predict(X_train_final)
        y_prob_train = model.predict_proba(X_train_final)[:, 1] if hasattr(model, "predict_proba") else None
        
        train_metrics = evaluate_classification(y_train, y_pred_train, y_prob_train)
        train_metrics['CV Mean Accuracy'] = cv_mean
        train_metrics['CV Std Accuracy'] = cv_std
        
        # Get probabilities
        y_prob_test = model.predict_proba(X_test_final)[:, 1] if hasattr(model, "predict_proba") else None
        y_prob_val = model.predict_proba(X_val_final)[:, 1] if hasattr(model, "predict_proba") else None
        
        return model, X_val_final, y_val, X_test_final, y_test, y_prob_val, y_prob_test, train_metrics

    def evaluate(self, model, X, y) -> Dict[str, float]:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
        return evaluate_classification(y, y_pred, y_prob)

# Backward compatibility wrapper
def train_classifier(X, y, model_type='logistic'):
    res = DefaultModelTrainer().train(X, y, model_type)
    # Return all 8 values if expected, but many old calls expect 7.
    # To be safe, we return the full tuple. 
    # If the user only unpacks 7, it will throw an error. 
    # I will update the known callers now.
    return res

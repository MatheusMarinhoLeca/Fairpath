from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple
from core.interfaces import ModelTrainer
from evaluation.performance import evaluate_classification

class DefaultModelTrainer(ModelTrainer):
    def train(self, X, y, model_type='logistic') -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
        
        # Scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        if model_type == 'logistic':
            model = LogisticRegression(solver='lbfgs', max_iter=5000, random_state=42)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
        elif model_type == 'gbm':
            model = GradientBoostingClassifier(
                n_estimators=500, 
                max_depth=3, 
                subsample=0.8, 
                n_iter_no_change=10, 
                validation_fraction=0.1,
                random_state=42
            )
        elif model_type == 'svm':
            model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
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

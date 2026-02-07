from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

def train_classifier(X, y, model_type='logistic'):
    # Initial split: 80% for training + validation, 20% for testing
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Second split: 75% of train_val for training, 25% for validation
    # (0.8 * 0.75 = 0.6 total train, 0.8 * 0.25 = 0.2 total validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
    
    if model_type == 'logistic':
        model = LogisticRegression(solver='liblinear', C=1.0, random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
    elif model_type == 'gbm':
        # Using early stopping with validation set to prevent overfitting
        model = GradientBoostingClassifier(
            n_estimators=500, 
            max_depth=3, 
            subsample=0.8, 
            n_iter_no_change=10, 
            validation_fraction=0.1, # Scikit-learn's internal val for early stopping
            random_state=42
        )
    
    model.fit(X_train, y_train)
    return model, X_val, y_val, X_test, y_test

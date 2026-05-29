import pandas as pd
import numpy as np
import sys

# To run this test: PYTHONPATH=fairpath python3 fairpath/tests/test_svm_integration.py

from fairpath.models.classification import DefaultModelTrainer

def test_svm_integration():
    print("Testing SVM integration in DefaultModelTrainer...")
    
    # Create dummy data
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feat_{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    
    trainer = DefaultModelTrainer()
    
    try:
        # Attempt to train SVM (LinearSVC)
        res = trainer.train(X, y, model_type='linear_svc')
        
        print(f"Success: Model type trained: {type(res.model)}")
        # Expecting CalibratedClassifierCV because we wrap LinearSVC
        assert 'CalibratedClassifierCV' in str(type(res.model)) or 'LinearSVC' in str(type(res.model)), f"Expected CalibratedClassifierCV or LinearSVC, got {type(res.model)}"
        
        # Check if probability estimates are available
        assert res.y_prob_test is not None, "Probability estimates should not be None"
        print(f"Success: Probability estimates generated. Shape: {res.y_prob_test.shape}")
        
        # Check for CV metrics
        assert 'CV Mean Accuracy' in res.train_metrics, "CV Mean Accuracy missing from train_metrics"
        print(f"Success: CV Mean Accuracy: {res.train_metrics['CV Mean Accuracy']:.4f}")

        # Test evaluation
        metrics = trainer.evaluate(res.model, res.X_test, res.y_test)
        print(f"Success: Evaluation metrics: {metrics}")
        
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_svm_integration()

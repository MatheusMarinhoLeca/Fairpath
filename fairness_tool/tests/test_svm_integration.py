import pandas as pd
import numpy as np
import sys
import os

# To run this test: PYTHONPATH=fairness_tool python3 fairness_tool/tests/test_svm_integration.py

from models.classification import DefaultModelTrainer

def test_svm_integration():
    print("Testing SVM integration in DefaultModelTrainer...")
    
    # Create dummy data
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feat_{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    
    trainer = DefaultModelTrainer()
    
    try:
        # Attempt to train SVM
        model, X_val, y_val, X_test, y_test, y_prob_val, y_prob_test = trainer.train(X, y, model_type='svm')
        
        print(f"Success: Model type trained: {type(model)}")
        assert 'SVC' in str(type(model)), f"Expected SVC, got {type(model)}"
        
        # Check if probability estimates are available
        assert y_prob_test is not None, "Probability estimates should not be None when probability=True"
        print(f"Success: Probability estimates generated. Shape: {y_prob_test.shape}")
        
        # Test evaluation
        metrics = trainer.evaluate(model, X_test, y_test)
        print(f"Success: Evaluation metrics: {metrics}")
        
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_svm_integration()

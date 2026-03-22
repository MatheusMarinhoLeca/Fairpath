import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from models.classification import DefaultModelTrainer
from fairness.metrics import GroupFairnessMetric, ClassificationFairnessMetric

class ExperimentRunner:
    """Consolidates the logic for a single fairness experiment run."""
    
    def __init__(self, trainer: Optional[DefaultModelTrainer] = None):
        self.trainer = trainer or DefaultModelTrainer()

    def run(self, 
            df_train: pd.DataFrame, 
            df_test: pd.DataFrame, 
            target_col: str, 
            sensitive_col: str, 
            privileged_group: Any, 
            unprivileged_group: Any,
            model_type: str,
            mitigation_strategy: Optional[Any] = None,
            metric_choice: str = '1', 
            selected_features: Optional[List[str]] = None,
            comparison_mode: str = 'combined',
            group_names: Optional[Dict[Any, str]] = None) -> Dict[str, Any]:
        """
        Executes the Split -> Mitigate -> Train -> Evaluate pipeline.
        
        Returns a dictionary containing metrics and predictions.
        """
        # 1. Mitigation (TRAIN ONLY)
        if mitigation_strategy:
            df_train_mitigated = mitigation_strategy.mitigate(
                df_train, target_col, sensitive_col, privileged_group, unprivileged_group
            )
        else:
            df_train_mitigated = df_train.copy()

        # 2. Data Preparation
        def prepare_xy(df_in):
            X_temp = df_in[selected_features] if selected_features else df_in.drop(columns=[target_col])
            # Ensure only numeric columns are passed to the trainer
            X_num = X_temp.select_dtypes(include=['number'])
            
            # Final safety check: if any columns are still NaNs (e.g. all-NaN columns),
            # fill them with 0 to prevent model training crashes.
            if X_num.isnull().sum().sum() > 0:
                X_num = X_num.fillna(0)
                
            y_temp = df_in[target_col]
            return X_num, y_temp

        X_train, y_train = prepare_xy(df_train_mitigated)
        X_test, y_test = prepare_xy(df_test)
        
        # Ensure column consistency
        missing_cols = set(X_train.columns) - set(X_test.columns)
        for c in missing_cols:
            X_test[c] = 0
        X_test = X_test[X_train.columns]

        # 3. Model Training
        model, X_val_final, y_val, X_test_final, y_test, y_prob_val, y_prob_test, train_metrics = self.trainer.train(
            X_train, y_train, model_type, X_test=X_test, y_test=y_test
        )
        
        y_pred_test = model.predict(X_test_final)
        perf_metrics = self.trainer.evaluate(model, X_test_final, y_test)
        
        # 4. Fairness Evaluation
        sens_test = df_test.loc[X_test_final.index, sensitive_col]
        
        df_true = pd.DataFrame({target_col: y_test.values, sensitive_col: sens_test.values}, index=y_test.index)
        df_pred = pd.DataFrame(X_test_final, columns=X_train.columns, index=X_test_final.index)
        df_pred[target_col] = y_pred_test
        df_pred[sensitive_col] = sens_test.values
        
        fair_metrics = {}
        
        # Determine comparison targets
        # If individual mode, we compute metrics for each subgroup in unprivileged_group
        if comparison_mode == 'individual' and isinstance(unprivileged_group, list):
            targets = unprivileged_group
        else:
            targets = [unprivileged_group] # Treat as a single combined group
            
        for target in targets:
            suffix = ""
            if comparison_mode == 'individual' and isinstance(unprivileged_group, list):
                # Attempt to get descriptive name from group_names mapping
                name = group_names.get(target, str(target)) if group_names else str(target)
                suffix = f" (Group: {name})"
            
            target_metrics = {}
            if metric_choice == '1' or metric_choice == 'all':
                group_metric_impl = GroupFairnessMetric()
                m = group_metric_impl.compute(df_pred, target_col, sensitive_col, privileged_group, target)
                target_metrics.update(m)
            
            if metric_choice == '2' or metric_choice == 'all':
                class_metric_impl = ClassificationFairnessMetric(df_true)
                m = class_metric_impl.compute(df_pred, target_col, sensitive_col, privileged_group, target)
                target_metrics.update(m)
                
            # Add results to the master dict with optional suffix
            for k, v in target_metrics.items():
                fair_metrics[f"{k}{suffix}"] = v

        # 5. Result Aggregation
        results = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': perf_metrics,
            'fairness_metrics': fair_metrics,
            'y_test': y_test,
            'y_pred': y_pred_test,
            'y_prob': y_prob_test,
            'X_test': X_test_final,
            'df_train_mitigated': df_train_mitigated
        }
        
        return results

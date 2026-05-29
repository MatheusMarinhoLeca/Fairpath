import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from fairpath.models.classification import DefaultModelTrainer
from fairpath.fairness.metrics import GroupFairnessMetric, ClassificationFairnessMetric

class ExperimentRunner:
    """Consolidates the logic for a single fairness experiment run.
    
    This class handles the end-to-end pipeline: training data mitigation, 
    model training, and a comprehensive fairness/performance evaluation on the 
    test set.
    """
    
    def __init__(self, trainer: Optional[DefaultModelTrainer] = None):
        """Initializes the runner with a model trainer.

        Args:
            trainer: An optional ModelTrainer instance. Defaults to DefaultModelTrainer.
        """
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
        """Executes the Split -> Mitigate -> Train -> Evaluate pipeline.

        Args:
            df_train: Training DataFrame.
            df_test: Testing DataFrame (should be untouched by mitigation).
            target_col: Name of the outcome column.
            sensitive_col: Name of the protected attribute column.
            privileged_group: Value representing the privileged group.
            unprivileged_group: Value(s) representing the unprivileged group(s).
            model_type: Key for the model factory (e.g., 'logistic').
            mitigation_strategy: Optional object implementing the mitigate() method.
            metric_choice: '1' for Group, '2' for Classification, 'all' for both.
            selected_features: Optional subset of features to include.
            comparison_mode: 'combined' (one vs all) or 'individual' (one vs each).
            group_names: Optional mapping from numeric values to human-readable names.

        Returns:
            A dictionary containing training metrics, test metrics, fairness metrics, 
            and prediction arrays.
        """
        # 1. Mitigation (Applied ONLY to the training set)
        if mitigation_strategy:
            df_train_mitigated = mitigation_strategy.mitigate(
                df_train, target_col, sensitive_col, privileged_group, unprivileged_group
            )
        else:
            df_train_mitigated = df_train.copy()

        # 2. Data Preparation helper
        def prepare_features_and_target(df_source: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
            X_raw = df_source[selected_features] if selected_features else df_source.drop(columns=[target_col])
            
            # Ensure only numeric columns reach the trainer (models assume encoded data)
            X_numeric = X_raw.select_dtypes(include=['number'])
            
            # Final safety fill for missing values to prevent training crashes
            if X_numeric.isnull().sum().sum() > 0:
                X_numeric = X_numeric.fillna(0)
                
            y_source = df_source[target_col]
            return X_numeric, y_source

        X_train, y_train = prepare_features_and_target(df_train_mitigated)
        X_test, y_test = prepare_features_and_target(df_test)
        
        # Ensure column consistency across splits
        missing_in_test = set(X_train.columns) - set(X_test.columns)
        for col in missing_in_test:
            X_test[col] = 0
        X_test = X_test[X_train.columns]

        # 3. Model Training and Performance Evaluation
        train_res = self.trainer.train(
            X_train, y_train, model_type, X_test=X_test, y_test=y_test
        )
        
        y_pred_test = train_res.model.predict(train_res.X_test)
        performance_metrics = self.trainer.evaluate(train_res.model, train_res.X_test, train_res.y_test)
        
        # 4. Fairness Evaluation setup
        s_sensitive_test = df_test.loc[train_res.X_test.index, sensitive_col]
        
        # DataFrames for metric computation (AIF360 compatible)
        df_ground_truth = pd.DataFrame({
            target_col: train_res.y_test.values, 
            sensitive_col: s_sensitive_test.values
        }, index=train_res.y_test.index)
        
        df_predictions = pd.DataFrame(train_res.X_test, columns=X_train.columns, index=train_res.X_test.index)
        df_predictions[target_col] = y_pred_test
        df_predictions[sensitive_col] = s_sensitive_test.values
        
        fairness_metrics_aggregated = {}
        
        # Determine unprivileged target groups for comparison
        target_groups = unprivileged_group if (comparison_mode == 'individual' and isinstance(unprivileged_group, list)) else [unprivileged_group]
            
        for group_val in target_groups:
            label_suffix = ""
            if comparison_mode == 'individual' and isinstance(unprivileged_group, list):
                readable_name = group_names.get(group_val, str(group_val)) if group_names else str(group_val)
                label_suffix = f" (Group: {readable_name})"
            
            group_metrics = {}
            if metric_choice == '1' or metric_choice == 'all':
                parity_calc = GroupFairnessMetric()
                m = parity_calc.compute(df_predictions, target_col, sensitive_col, privileged_group, group_val)
                group_metrics.update(m)
            
            if metric_choice == '2' or metric_choice == 'all':
                error_rate_calc = ClassificationFairnessMetric(df_ground_truth)
                m = error_rate_calc.compute(df_predictions, target_col, sensitive_col, privileged_group, group_val)
                group_metrics.update(m)
                
            for metric_name, metric_val in group_metrics.items():
                fairness_metrics_aggregated[f"{metric_name}{label_suffix}"] = metric_val

        # 5. Final Result Aggregation
        return {
            'model': train_res.model,
            'train_metrics': train_res.train_metrics,
            'test_metrics': performance_metrics,
            'fairness_metrics': fairness_metrics_aggregated,
            'y_test': train_res.y_test,
            'y_pred': y_pred_test,
            'y_prob': train_res.y_prob_test,
            'X_test': train_res.X_test,
            'df_train_mitigated': df_train_mitigated
        }

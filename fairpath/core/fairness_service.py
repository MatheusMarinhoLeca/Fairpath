from typing import Any, List, Optional, Tuple
import pandas as pd
from fairpath.core.experiment import ExperimentRunner
from fairpath.core.models import FairnessConfig, ExperimentResult
from fairpath.models.classification import DefaultModelTrainer
from sklearn.model_selection import train_test_split

class FairnessService:
    """Service for orchestrating fairness audits and mitigation experiments.
    
    This service manages the lifecycle of fairness experiments, including 
    baseline evaluation and the application of pre-processing mitigation strategies.
    """

    def __init__(self, trainer: Optional[DefaultModelTrainer] = None):
        """Initializes the service with a model trainer.

        Args:
            trainer: An optional ModelTrainer instance. Defaults to DefaultModelTrainer.
        """
        self.trainer = trainer or DefaultModelTrainer()
        self.experiment_runner = ExperimentRunner(self.trainer)

    def _get_stratified_splits(self, df: pd.DataFrame, target_col: str, sensitive_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Standardizes the train/test split logic with multi-level stratification.

        Attempts to stratify by both the target variable and the sensitive attribute 
        to ensure group representation in both splits. Falls back to target-only 
        or random splitting if groups are too small.

        Args:
            df: The DataFrame to split.
            target_col: The outcome column name.
            sensitive_col: The protected attribute column name.

        Returns:
            A tuple containing (df_train, df_test).
        """
        stratify_key = df[target_col].astype(str) + "_" + df[sensitive_col].astype(str)
        try:
            return train_test_split(df, test_size=0.2, random_state=42, stratify=stratify_key)
        except ValueError:
            # Fallback 1: Stratify by target only
            try:
                return train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
            except ValueError:
                # Fallback 2: Random split
                return train_test_split(df, test_size=0.2, random_state=42)

    def run_experiment(self, df: pd.DataFrame, target_col: str, config: FairnessConfig, 
                       selected_features: Optional[List[str]] = None,
                       mitigation_strategy: Optional[Any] = None) -> Tuple[ExperimentResult, Optional[pd.DataFrame]]:
        """Executes a fairness experiment (Baseline Audit or Mitigation).

        Args:
            df: The dataset.
            target_col: The outcome column name.
            config: An object containing fairness audit configurations.
            selected_features: Optional list of features to use in the model.
            mitigation_strategy: Optional strategy to apply before training.

        Returns:
            A tuple containing:
                - The structured ExperimentResult (metrics, predictions, etc.).
                - The transformed training DataFrame if mitigation was applied.
        """
        df_train, df_test = self._get_stratified_splits(df, target_col, config.sensitive_col)
        
        experiment_results = self.experiment_runner.run(
            df_train, df_test,
            target_col,
            config.sensitive_col,
            config.privileged_group,
            config.unprivileged_group,
            config.model_choice,
            mitigation_strategy=mitigation_strategy,
            metric_choice=config.metric_choice,
            selected_features=selected_features,
            comparison_mode=config.comparison_mode,
            group_names=config.inverse_mapping
        )

        # Build the structured result object
        exp_result = ExperimentResult(
            metrics={
                **{f"Val {k}": v for k, v in experiment_results['train_metrics'].items() if k not in ['CV Mean Accuracy', 'CV Std Accuracy']},
                **{f"Test {k}": v for k, v in experiment_results['test_metrics'].items()},
                **experiment_results['fairness_metrics']
            },
            y_test=experiment_results['y_test'],
            y_pred=experiment_results['y_pred'],
            y_prob=experiment_results['y_prob'],
            model=experiment_results['model']
        )
        
        # Include Cross-Validation metadata if available
        if 'CV Mean Accuracy' in experiment_results['train_metrics']:
            exp_result.metrics['CV Mean Accuracy'] = experiment_results['train_metrics']['CV Mean Accuracy']
        
        return exp_result, experiment_results.get('df_train_mitigated')

import pandas as pd
from typing import List, Tuple, Dict, Any
from fairpath.core.models import (
    PreprocessingConfig, FairnessConfig, ExperimentResult, AuditReportData
)
from fairpath.core.data_service import DataService
from fairpath.core.fairness_service import FairnessService
from fairpath.core.preprocessing_service import PreprocessingService

class FairnessEngine:
    """Core business logic for fairness analysis, decoupled from any UI.
    
    This engine orchestrates the interaction between data services, 
    preprocessing, model training, and fairness auditing.
    """

    def __init__(self):
        self.data_service = DataService()
        self.fairness_service = FairnessService()
        self.preprocessing_service = PreprocessingService()

    def _prepare_config(self, df_processed: pd.DataFrame, fair_config: FairnessConfig, 
                        sensitive_col: str) -> FairnessConfig:
        """Enriches the fairness config with derived information from processed data."""
        is_binarized = fair_config.comparison_mode == 'combined'
        
        # Determine the actual values used for groups in the processed dataframe
        actual_priv_val = 1 if is_binarized else fair_config.privileged_group
        
        # Derive unprivileged groups if empty or if binarized (overwrite to 0)
        if is_binarized:
            unprivileged = [0]
        else:
            unprivileged = fair_config.unprivileged_group
            if not unprivileged:
                unprivileged = [x for x in df_processed[sensitive_col].unique() if str(x) != str(actual_priv_val)]
            
        # Create a new config with the actual sensitive column and numeric identifiers if binarized
        return FairnessConfig(
            sensitive_col=sensitive_col,
            selected_attributes=fair_config.selected_attributes,
            privileged_group=actual_priv_val,
            privileged_group_name=fair_config.privileged_group_name,
            unprivileged_group=unprivileged,
            comparison_mode=fair_config.comparison_mode,
            metric_choice=fair_config.metric_choice,
            model_choice=fair_config.model_choice,
            inverse_mapping=fair_config.inverse_mapping
        )

    def _get_display_mapping(self, enriched_config: FairnessConfig) -> Dict[Any, str]:
        """Generates a mapping for human-readable group names in reports."""
        mapping = {}
        if enriched_config.comparison_mode == 'combined':
            mapping[1] = f"{enriched_config.privileged_group_name} (Privileged)"
            mapping[0] = "Other (Unprivileged)"
        return mapping

    def run_baseline_audit(self, 
                          df: pd.DataFrame, 
                          target_col: str, 
                          prep_config: PreprocessingConfig,
                          fair_config: FairnessConfig) -> Tuple[ExperimentResult, pd.DataFrame, List[str]]:
        """Executes a baseline fairness audit.

        Args:
            df: The raw dataset.
            target_col: Outcome column.
            prep_config: Preprocessing settings.
            fair_config: Fairness settings.

        Returns:
            A tuple of (ExperimentResult, ProcessedDataFrame, FinalFeatures).
        """
        # 1. Preprocess
        df_processed, final_features, sensitive_col = self.preprocessing_service.run_pipeline(
            df, target_col, prep_config, 
            sensitive_attrs=fair_config.selected_attributes,
            privileged_val=fair_config.privileged_group,
            binarize_sensitive=(fair_config.comparison_mode == 'combined')
        )

        # 2. Enrich Config
        enriched_config = self._prepare_config(df_processed, fair_config, sensitive_col)

        # 3. Run Experiment
        exp_result, _ = self.fairness_service.run_experiment(
            df_processed, target_col, enriched_config,
            selected_features=final_features
        )
        
        # 4. Add stats with human-readable group mapping
        exp_result.stats = self.data_service.get_audit_stats(
            df_processed, target_col, sensitive_col, 
            original_feat_count=df.shape[1]-1, 
            selected_features=final_features,
            display_mapping=self._get_display_mapping(enriched_config)
        )
        
        # 5. Final Reordering: Original columns first, new columns (encoded) later
        original_cols = df.columns.tolist()
        surviving_cols = [c for c in original_cols if c in df_processed.columns]
        new_cols = [c for c in df_processed.columns if c not in original_cols]
        df_processed = df_processed[surviving_cols + new_cols]
        
        return exp_result, df_processed, final_features

    def run_mitigation_audit(self, 
                            df: pd.DataFrame, 
                            target_col: str, 
                            prep_config: PreprocessingConfig,
                            fair_config: FairnessConfig,
                            mitigation_strategy: Any) -> Tuple[ExperimentResult, pd.DataFrame]:
        """Executes a fairness audit after applying mitigation.

        Args:
            df: The raw dataset.
            target_col: Outcome column.
            prep_config: Preprocessing settings.
            fair_config: Fairness settings.
            mitigation_strategy: The strategy to apply.

        Returns:
            A tuple of (ExperimentResult, MitigatedDataFrame).
        """
        # 1. Preprocess (Reuse same logic as baseline)
        df_processed, final_features, sensitive_col = self.preprocessing_service.run_pipeline(
            df, target_col, prep_config, 
            sensitive_attrs=fair_config.selected_attributes,
            privileged_val=fair_config.privileged_group,
            binarize_sensitive=(fair_config.comparison_mode == 'combined')
        )

        # 2. Enrich Config
        enriched_config = self._prepare_config(df_processed, fair_config, sensitive_col)

        # 3. Run Experiment with Mitigation (Audit uses Split)
        exp_result, _ = self.fairness_service.run_experiment(
            df_processed, target_col, enriched_config,
            selected_features=final_features,
            mitigation_strategy=mitigation_strategy
        )
        
        # 4. Generate Full Enhanced Dataset for Export
        df_full_mitigated = df_processed.copy()
        if mitigation_strategy:
            # We apply mitigation to the WHOLE dataset for the saved output
            df_full_mitigated = mitigation_strategy.mitigate(
                df_processed, target_col, enriched_config.sensitive_col,
                enriched_config.privileged_group, enriched_config.unprivileged_group
            )

        # 5. Add stats with human-readable group mapping
        exp_result.stats = self.data_service.get_audit_stats(
            df_full_mitigated, 
            target_col, sensitive_col, 
            original_feat_count=df.shape[1]-1, 
            selected_features=final_features,
            display_mapping=self._get_display_mapping(enriched_config)
        )
        
        # 6. Final Reordering: Original columns first, new columns (encoded/synthetic) later
        original_cols = df.columns.tolist()
        surviving_cols = [c for c in original_cols if c in df_full_mitigated.columns]
        new_cols = [c for c in df_full_mitigated.columns if c not in original_cols]
        df_full_mitigated = df_full_mitigated[surviving_cols + new_cols]
        
        return exp_result, df_full_mitigated

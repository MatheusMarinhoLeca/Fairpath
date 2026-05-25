import os
import pandas as pd
from typing import Dict, Any, List, Optional
from fairpath.core.models import AuditReportData, ExperimentResult
from fairpath.reporting.report_builder import generate_pdf_report
from fairpath.eda.visualizations import (
    plot_confusion_matrix, plot_subgroup_confusion_matrices,
    plot_kde_probabilities, plot_metric_comparison,
    plot_fairness_utility_tradeoff, plot_distribution_comparison,
    plot_selection_rates, plot_grouped_bar_charts,
    plot_contingency_heatmap
)

from fairpath.core.recommendation_service import RecommendationService

class ReportingService:
    """Service for generating fairness reports and decision-support visualizations.
    
    This service aggregates results from multiple experiment stages and produces 
    both visual assets (PNGs) and the final PDF audit report.
    """

    def __init__(self, output_dir: str = "outputs"):
        """Initializes the service and ensures output directories exist.

        Args:
            output_dir: The root directory for all generated outputs.
        """
        self.output_dir = output_dir
        self.reports_dir = os.path.join(output_dir, "reports")
        self.datasets_dir = os.path.join(output_dir, "datasets")
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        self.recommendation_service = RecommendationService()

    def save_mitigated_dataset(self, df: pd.DataFrame, filename: str = "fairness_improved.csv") -> str:
        """Saves the transformed dataset to the outputs folder.

        Args:
            df: The DataFrame to save.
            filename: The target filename.

        Returns:
            The full path to the saved file.
        """
        save_path = os.path.join(self.datasets_dir, filename)
        df.to_csv(save_path, index=False)
        return save_path

    def generate_full_audit_report(self, audit_data: AuditReportData, filename: str = "fairness_report.pdf") -> str:
        """Orchestrates visualization generation and PDF building.

        Args:
            audit_data: A DTO containing baseline and mitigated experiment results.
            filename: The name of the final PDF report.

        Returns:
            The path to the generated PDF report.
        """
        visualizations = self._generate_visualizations(audit_data)
        
        # Generate data-driven recommendations
        recommendations = self.recommendation_service.generate_recommendations(
            selections={
                'preprocessing': audit_data.preprocessing_selections,
                'fairness': audit_data.fairness_selections,
                'mitigation': audit_data.mitigation_selections
            },
            stats_before=audit_data.baseline.stats,
            metrics_before=audit_data.baseline.metrics,
            metrics_after=audit_data.mitigated.metrics if audit_data.mitigated else {}
        )
        
        generate_pdf_report(
            filename, 
            audit_data.baseline.stats, 
            audit_data.baseline.metrics, 
            audit_data.mitigated.metrics if audit_data.mitigated else {}, 
            recommendations=recommendations,
            plots=visualizations, 
            stats_after=audit_data.mitigated.stats if audit_data.mitigated else None, 
            selections={
                'preprocessing': audit_data.preprocessing_selections,
                'fairness': audit_data.fairness_selections,
                'mitigation': audit_data.mitigation_selections
            }
        )
        return os.path.join(self.reports_dir, filename)

    def _generate_visualizations(self, data: AuditReportData) -> Dict[str, str]:
        """Generates all comparison plots for the fairness report.

        Args:
            data: The aggregated audit data.

        Returns:
            A dictionary mapping plot titles to their local file paths.
        """
        plots = {}
        baseline_res = data.baseline
        mitigated_res = data.mitigated
        
        # We need local copies to avoid modifying the original data objects
        df_baseline = data.df_baseline.copy() if data.df_baseline is not None else None
        df_mitigated = data.df_mitigated.copy() if data.df_mitigated is not None else None
        
        # Determine mapping for sensitive attribute display names
        fair_sel = data.fairness_selections
        display_mapping = {}
        if fair_sel.get('privileged_group'):
            # Mapping for binarized data (1=Privileged, 0=Unprivileged)
            display_mapping[1] = f"{fair_sel['privileged_group']} (Privileged)"
            display_mapping[0] = "Other Groups (Unprivileged)"
            display_mapping[1.0] = display_mapping[1]
            display_mapping[0.0] = display_mapping[0]

        # Apply mapping to the full dataframes if they are binarized
        def apply_display_mapping(df: pd.DataFrame):
            if df is not None and data.sensitive_col in df.columns:
                if display_mapping and set(df[data.sensitive_col].unique()).issubset({0, 1, 0.0, 1.0}):
                    df[data.sensitive_col] = df[data.sensitive_col].map(display_mapping)
            return df

        df_baseline = apply_display_mapping(df_baseline)
        df_mitigated = apply_display_mapping(df_mitigated)

        # Helper to extract sensitive attribute series for the test set
        def extract_test_sensitive(df: pd.DataFrame, target_series: pd.Series):
            if df is not None and target_series is not None:
                return df.loc[target_series.index, data.sensitive_col]
            return None

        # 1. Baseline Visualizations
        if baseline_res.y_test is not None:
            print("Generating Baseline Confusion Matrix...")
            plots["Baseline Confusion Matrix"] = plot_confusion_matrix(
                baseline_res.y_test, baseline_res.y_pred, 
                "Baseline Confusion Matrix", "cm_baseline.png"
            )
            s_test_baseline = extract_test_sensitive(df_baseline, baseline_res.y_test)
            if s_test_baseline is not None:
                print("Generating Baseline Subgroup Plots...")
                plots["Baseline Subgroup Confusion Matrices"] = plot_subgroup_confusion_matrices(
                    baseline_res.y_test.values, baseline_res.y_pred, 
                    s_test_baseline.values, "cm_subgroups_baseline.png"
                )
                if baseline_res.y_prob is not None:
                    plots["Baseline KDE Predicted Probabilities"] = plot_kde_probabilities(
                        baseline_res.y_prob, s_test_baseline.values, "kde_baseline.png"
                    )

        # 2. Mitigated Visualizations
        if mitigated_res and mitigated_res.y_test is not None:
            print("Generating Mitigated Confusion Matrix...")
            plots["Mitigated Confusion Matrix"] = plot_confusion_matrix(
                mitigated_res.y_test, mitigated_res.y_pred, 
                "Mitigated Confusion Matrix", "cm_mitigated.png"
            )
            s_test_mitigated = extract_test_sensitive(df_baseline, mitigated_res.y_test) # Use baseline as feature ref
            if s_test_mitigated is not None:
                print("Generating Mitigated Subgroup Plots...")
                plots["Mitigated Subgroup Confusion Matrices"] = plot_subgroup_confusion_matrices(
                    mitigated_res.y_test.values, mitigated_res.y_pred, 
                    s_test_mitigated.values, "cm_subgroups_mitigated.png"
                )
                if mitigated_res.y_prob is not None:
                    plots["Mitigated KDE Predicted Probabilities"] = plot_kde_probabilities(
                        mitigated_res.y_prob, s_test_mitigated.values, "kde_mitigated.png"
                    )

        # 3. Comparative Visualizations
        print("Generating Performance Comparisons...")
        plots["Performance & Fairness Metrics Comparison"] = plot_metric_comparison(
            baseline_res.metrics, mitigated_res.metrics if mitigated_res else {}
        )
        
        # Fairness-Utility Trade-off
        fairness_key = "Statistical Parity Difference" if "Statistical Parity Difference" in baseline_res.metrics else "Disparate Impact"
        utility_key = "Test Accuracy"
        tradeoff_points = [
            {fairness_key: baseline_res.metrics.get(fairness_key, 0), utility_key: baseline_res.metrics.get(utility_key, 0), 'Stage': 'Baseline'}
        ]
        if mitigated_res:
            tradeoff_points.append({
                fairness_key: mitigated_res.metrics.get(fairness_key, 0), 
                utility_key: mitigated_res.metrics.get(utility_key, 0), 
                'Stage': 'Mitigated'
            })
        
        plots["Fairness-Utility Trade-off"] = plot_fairness_utility_tradeoff(tradeoff_points, fairness_key, utility_key, "tradeoff.png")

        # 4. Distributional Comparisons (if mitigation was applied)
        if df_mitigated is not None:
            print("Generating Distributional Comparisons...")
            plots["Class Distribution Comparison"] = plot_distribution_comparison(
                df_baseline, df_mitigated, data.target_col, 
                "Class Distribution: Before vs After", "class_dist_comparison.png"
            )
            plots["Sensitive Attribute Distribution Comparison"] = plot_distribution_comparison(
                df_baseline, df_mitigated, data.sensitive_col, 
                "Sensitive Attribute Distribution: Before vs After", "sensitive_dist_comparison.png"
            )
            plots["Selection Rate Comparison P(Y=1|A)"] = plot_selection_rates(
                df_baseline, df_mitigated, data.sensitive_col, data.target_col, "selection_rates_comparison.png"
            )
            plots["Grouped Bar Charts (Positive Rate)"] = plot_grouped_bar_charts(
                df_mitigated, data.sensitive_col, data.target_col, "grouped_bars_mitigated.png"
            )
            print("Generating Subgroup Heatmaps...")
            plots["Baseline Subgroup Heatmap"] = plot_contingency_heatmap(
                df_baseline, data.sensitive_col, data.target_col, 
                "Baseline: Subgroup Counts (A, Y)", "heatmap_baseline.png"
            )
            plots["Mitigated Subgroup Heatmap"] = plot_contingency_heatmap(
                df_mitigated, data.sensitive_col, data.target_col, 
                "Mitigated: Subgroup Counts (A, Y)", "heatmap_mitigated.png"
            )

        print("All plots generated.")
        return plots

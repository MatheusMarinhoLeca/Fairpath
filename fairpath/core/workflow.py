import os
import pandas as pd
import numpy as np
from fairpath.core.context import ProjectContext
from fairpath.ui.terminal import TerminalUI
from fairpath.utils.logging import log_action
from fairpath.data import (
    validate_file_path, validate_dataset, get_sensitive_mapping,
    parse_attribute_input
)
from fairpath.data.validator import check_duplicates
from fairpath.eda.visualizations import (
    plot_class_distribution, plot_missing_heatmap
)
from fairpath.core.reporting_service import ReportingService
from fairpath.core.preprocessing_service import PreprocessingService
from fairpath.core.fairness_engine import FairnessEngine
from fairpath.core.models import (
    PreprocessingConfig, FairnessConfig, ExperimentResult, AuditReportData
)
from fairpath.config.defaults import POTENTIAL_SENSITIVE_ATTRIBUTES, DEFAULT_PRIORITY_SENSITIVE
from fairpath.fairness.mitigation import ResamplingMitigation, RelabelingMitigation, SyntheticMitigation

class WorkflowController:
    """Orchestrates the fairness analysis workflow via the Terminal UI.
    
    This class handles UI state and user interaction, delegating all business 
    logic to specialized services and the FairnessEngine. It maintains the 
    overall lifecycle of an audit, from data loading to report generation.
    """
    
    def __init__(self, ui: TerminalUI, context: ProjectContext):
        """Initializes the controller with UI and context.

        Args:
            ui: The TerminalUI instance for user interaction.
            context: The ProjectContext instance for state management.
        """
        self.ui = ui
        self.context = context
        self.reporting_service = ReportingService()
        self.preprocessing_service = PreprocessingService()
        self.engine = FairnessEngine()

    def run(self):
        """Starts the main application loop."""
        while True:
            self.ui.display_welcome()
            choice = self.ui.get_main_menu_choice()
            if choice == '1':
                self.load_data_workflow()
            elif choice == '2':
                self.benchmark_workflow()
            elif choice == '3':
                self.run_visualization_workflow()
            elif choice == '4':
                self.ui.exit()

    def run_visualization_workflow(self):
        """Handles the visualization of existing benchmark results."""
        try:
            from fairpath.reporting.benchmark_viz import BenchmarkVisualizer
            file_path = self.ui.get_benchmark_file_path()
            if not file_path or not os.path.exists(file_path):
                 self.ui.display_message("\nInvalid file path or file not found.")
                 self.ui.wait_for_user()
                 return
                 
            self.ui.display_message(f"\nLoading benchmark results from {file_path}...")
            viz = BenchmarkVisualizer(file_path)
            viz.generate_full_report()
            self.ui.display_message(f"\nVisualization complete. Check the 'viz_report' folder.")
            self.ui.wait_for_user()
        except Exception as e:
            self.ui.display_message(f"\nError during visualization: {e}")
            self.ui.wait_for_user()

    def benchmark_workflow(self):
        """Handles the automated benchmark configuration and execution."""
        try:
            from fairpath.core.benchmark import BenchmarkEngine
            n_runs = self.ui.get_benchmark_runs()
            
            from fairpath.utils.menus import get_user_input
            limit_input = get_user_input("Enter sample limit for debugging (Enter to skip)", lambda x: x == "" or x.isdigit(), allow_empty=True)
            debug_limit = int(limit_input) if limit_input else None
            
            engine = BenchmarkEngine(n_runs, debug_sample_limit=debug_limit)
            self.ui.display_message("\nStarting automated benchmark...")
            engine.run()
            self.ui.display_message("\nBenchmark complete. Results saved in 'outputs/reports/'.")
            self.ui.wait_for_user()
        except Exception as e:
            self.ui.display_message(f"Error during benchmark: {e}")
            self.ui.wait_for_user()

    def load_data_workflow(self):
        """Orchestrates the data ingestion and initial validation phase."""
        self.context.reset()
        path = self.ui.get_dataset_path(validate_file_path)
        try:
            self.context.df, renamed = self.engine.data_service.load_initial_data(path)
            if renamed:
                self.ui.display_message(f"\nWarning: Renamed columns: {renamed}")

            self.context.original_df = self.context.df.copy()
            valid, msg = validate_dataset(self.context.df)
            if not valid:
                self.ui.display_message(f"Error: {msg}")
                return
            
            self.context.df, converted_cols = self.engine.data_service.prepare_raw_data(self.context.df)
            if converted_cols:
                self.ui.display_message(f"✔ Auto-converted to numeric: {', '.join(converted_cols)}")
            
            self.ui.display_message(f"\nDataset loaded. Rows: {self.context.df.shape[0]}, Columns: {self.context.df.shape[1]}")
            
            has_dups, dup_count = check_duplicates(self.context.df)
            if has_dups and self.ui.confirm_action(f"Remove {dup_count} duplicate rows?"):
                self.context.df = self.context.df.drop_duplicates()
            
            self.ui.display_message("\nAvailable columns: " + ", ".join(self.context.df.columns))
            detected_target = self.context.df.columns[-1]
            if self.ui.confirm_target_col(detected_target):
                self.context.target_col = detected_target
            else:
                self.context.target_col = self.ui.get_target_col(self.context.df.columns.tolist())
            
            log_action(f"Loaded dataset {path}. Target: {self.context.target_col}")
            self.ui.wait_for_user("\nPress Enter for EDA...")
            self.eda_workflow()
            
        except Exception as e:
            self.ui.display_message(f"Error loading dataset: {e}")
            self.ui.wait_for_user()

    def eda_workflow(self):
        """Generates initial statistics and visualizations."""
        self.ui.display_message("\n--- Exploratory Data Analysis ---")
        self.context.stats = self.engine.data_service.get_eda_stats(self.context.df)
        plot_class_distribution(self.context.df, self.context.target_col)
        plot_missing_heatmap(self.context.df)
        self.ui.display_message("✔ EDA Visualizations saved")
        self.ui.wait_for_user("\nPress Enter for Preprocessing...")
        self.preprocessing_workflow()

    def preprocessing_workflow(self):
        """Configures the data cleaning and encoding pipeline based on user input."""
        self.ui.display_message("\n--- Preprocessing ---")
        
        # 1. Feature Selection UI
        selection_method = self.ui.get_feature_selection_method()
        methods_labels = {'1': 'Manual Selection', '2': 'Skip'}
        self.context.selections['preprocessing']['feature_selection'] = methods_labels.get(selection_method, 'N/A')

        if selection_method == '1':
            selected = self.ui.get_columns_to_keep(self.context.df.columns.tolist())
            if self.context.target_col not in selected: selected.append(self.context.target_col)
            self.context.selected_features = [c for c in selected if c != self.context.target_col]
        else:
            self.context.selected_features = [c for c in self.context.df.columns if c != self.context.target_col]

        # 2. Cleaning & Encoding Choices
        missing_choice = self.ui.ask_handle_missing_values() if self.context.df.isnull().sum().sum() > 0 else 'none'
        outlier_choice = self.ui.ask_handle_outliers()
        
        cat_cols = list(self.context.df.select_dtypes(include=['object', 'category', 'bool']).columns)
        if self.context.target_col in cat_cols: cat_cols.remove(self.context.target_col)
        features_to_encode = [c for c in cat_cols if c in self.context.selected_features] if self.context.selected_features else cat_cols
        
        encoding_info = self.ui.ask_encode_categorical(features_to_encode, self.context.selected_features)
        if encoding_info is None:
            encoding_info = {"choice": "none"}
        
        # Mapping for better report readability
        missing_labels = {'1': 'Mean/Median', '2': 'Mode', '3': 'Drop', 'none': 'Skip'}
        outlier_labels = {'1': 'IQR-based removal', '2': 'Skip', 'none': 'Skip'}
        encoding_labels = {'1': 'One-Hot Encoding', '2': 'Label Encoding', 'none': 'Skip'}

        # 3. Store choices in context for reporting (Human Readable)
        self.context.selections['preprocessing'].update({
            'missing_values': missing_labels.get(missing_choice, 'Skip'),
            'outliers': outlier_labels.get(outlier_choice, 'Skip'),
            'encoding': encoding_labels.get(encoding_info.get('choice', 'none'), 'Skip')
        })

        # 4. Define the PreprocessingConfig
        self.context.prep_config = PreprocessingConfig(
            selected_features=self.context.selected_features,
            missing_strategy=missing_choice,
            outlier_strategy=outlier_choice,
            encoding_strategy=encoding_info.get('choice', 'none')
        )
        
        # 5. Feedback Message
        is_raw = (missing_choice == 'none' and outlier_choice in ['2', 'none'] and encoding_info.get('choice') == 'none')
        if is_raw:
            self.ui.display_message("\n⚠️ Note: Raw data can hide real bias behind messy results and may cause the computer to wrongly treat some groups as 'better' or 'worse' than others.")
        else:
            self.ui.display_message("\nPreprocessing configuration complete.")

        log_action("Preprocessing configuration set")
        self.fairness_setup_workflow()

    def fairness_setup_workflow(self):
        """Configures sensitive attributes and groups for the fairness audit."""
        available_cols = self.context.original_df.columns.tolist()
        potential = [c for c in available_cols if c.lower() in POTENTIAL_SENSITIVE_ATTRIBUTES]
        default_sens = next((c for c in potential if c.lower() == DEFAULT_PRIORITY_SENSITIVE.lower()), None) or (potential[0] if potential else None)

        user_input = self.ui.get_sensitive_attributes(available_cols, default=default_sens)
        selected_attrs = parse_attribute_input(user_input, available_cols)

        mapping = {} if len(selected_attrs) > 1 else get_sensitive_mapping(self.context.df, self.context.original_df, selected_attrs[0])
        unique_vals = self.context.df[selected_attrs[0]].unique() if len(selected_attrs) == 1 else ["Intersectional"] 
        
        # Get privileged group via UI
        user_val = self.ui.get_privileged_group(unique_vals.tolist() if len(selected_attrs) == 1 else ["Custom"], mapping)
        privileged_group = mapping.get(user_val, user_val)

        un_choice = self.ui.get_unprivileged_comparison_mode()
        self.context.comparison_mode = 'combined' if un_choice == '1' else 'individual'
        
        # Model Selection
        m_choice = self.ui.get_model_choice()
        model_map = {'1': 'logistic', '2': 'random_forest', '3': 'gbm', '4': 'linear_svc'}
        self.context.model_choice = model_map[m_choice]
        
        metric_choice = self.ui.get_specific_fairness_metric()
        metric_labels = {'1': 'Demographic Parity', '2': 'Equalized Odds'}

        # Store fairness selections for reporting
        self.context.selections['fairness'] = {
            'sensitive_column': ", ".join(selected_attrs),
            'privileged_group': str(user_val),
            'specific_metric': metric_labels.get(metric_choice, 'N/A')
        }

        self.context.fair_config = FairnessConfig(
            sensitive_col=", ".join(selected_attrs), # Initial name, engine refines this
            selected_attributes=selected_attrs,
            privileged_group=privileged_group,
            privileged_group_name=str(user_val),
            unprivileged_group=[], 
            comparison_mode=self.context.comparison_mode,
            metric_choice=metric_choice,
            model_choice=self.context.model_choice,
            inverse_mapping={privileged_group: str(user_val)} 
        )

        self.ui.wait_for_user("\nPress Enter to compute baseline...")
        self.baseline_evaluation()

    def baseline_evaluation(self):
        """Executes the baseline (no-mitigation) audit."""
        self.ui.display_message("\n--- Baseline Evaluation ---")
        
        res, df_proc, feats = self.engine.run_baseline_audit(
            self.context.df, self.context.target_col, 
            self.context.prep_config, self.context.fair_config
        )
        
        self.context.baseline_res = res
        self.context.df_processed = df_proc
        self.context.selected_features = feats
        self.context.sensitive_col = res.stats.get('Sensitive Attribute', 'sensitive_attribute')
        
        # Update renamed context members
        self.context.baseline_test_labels = res.y_test
        self.context.baseline_predictions = res.y_pred
        self.context.baseline_probabilities = res.y_prob
        
        self.ui.display_metrics(res.metrics, "Baseline Metrics")
        self.ui.wait_for_user("\nPress Enter for Mitigation...")
        self.mitigation_workflow()

    def mitigation_workflow(self):
        """Configures and executes the bias mitigation phase."""
        method_choice = self.ui.get_mitigation_method()
        strategy = None
        mit_info = {'method': "None"}

        if method_choice == '1':
            res_type = self.ui.get_resampling_type()
            strategy = ResamplingMitigation('oversample' if res_type == '1' else 'undersample')
            mit_info['method'] = f"Resampling ({'Over' if res_type == '1' else 'Under'})"
        elif method_choice == '2':
            strategy = RelabelingMitigation(self.context.selected_features)
            mit_info['method'] = "Relabeling"
        elif method_choice == '3':
            synth = 'smote' if self.ui.get_synthetic_method() == '1' else 'cda'
            strategy = SyntheticMitigation(synth)
            mit_info['method'] = f"Synthetic ({synth.upper()})"
            
        self.context.selections['mitigation'] = mit_info
        
        self.ui.display_message("Executing mitigation experiment...")
        res_mit, df_mit = self.engine.run_mitigation_audit(
            self.context.df, self.context.target_col,
            self.context.prep_config, self.context.fair_config,
            mitigation_strategy=strategy
        )
        
        self.context.mitigated_res = res_mit
        self.context.df_improved = df_mit
        
        # Update renamed context members
        self.context.mitigated_test_labels = res_mit.y_test
        self.context.mitigated_predictions = res_mit.y_pred
        self.context.mitigated_probabilities = res_mit.y_prob
        
        self.ui.display_metrics(res_mit.metrics, "Mitigated Metrics")
        self.ui.wait_for_user("\nPress Enter to generate reports...")
        self.output_workflow()

    def output_workflow(self):
        """Finalizes the audit by generating reports and saving datasets."""
        audit_data = AuditReportData(
            baseline=self.context.baseline_res,
            mitigated=self.context.mitigated_res,
            df_baseline=self.context.df_processed,
            df_mitigated=self.context.df_improved,
            preprocessing_selections=self.context.selections['preprocessing'],
            fairness_selections=self.context.selections['fairness'],
            mitigation_selections=self.context.selections['mitigation'],
            target_col=self.context.target_col,
            sensitive_col=self.context.sensitive_col
        )
        
        if self.context.df_improved is not None:
            out_path = self.reporting_service.save_mitigated_dataset(self.context.df_improved)
            self.ui.display_message(f"✔ Improved dataset saved to {out_path}")
        
        self.ui.display_message("Generating audit report...")
        self.reporting_service.generate_full_audit_report(audit_data)
        
        self.ui.display_message("\nAnalysis Complete!")
        self.ui.wait_for_user("Press Enter to return to main menu...")

import os
import pandas as pd
import numpy as np
from core.context import ProjectContext
from ui.terminal import TerminalUI
from utils.logging import log_action
from data import (
    load_dataset, validate_file_path, validate_dataset, 
    handle_duplicate_columns, infer_numeric_types, get_sensitive_mapping,
    create_composite_attribute, parse_attribute_input
)
from eda.statistics import get_basic_stats, get_comprehensive_stats
from eda.visualizations import (
    plot_class_distribution, plot_missing_heatmap, plot_confusion_matrix, 
    plot_metric_comparison, plot_distribution_comparison,
    plot_selection_rates, plot_contingency_heatmap,
    plot_kde_probabilities, plot_subgroup_confusion_matrices,
    plot_fairness_utility_tradeoff, plot_grouped_bar_charts
)
from preprocessing.missing_values import impute_missing
from preprocessing.outliers import remove_outliers_iqr
from preprocessing.encoding import one_hot_encode, label_encode, binarize_attribute
from models.classification import DefaultModelTrainer
from fairness.metrics import GroupFairnessMetric, ClassificationFairnessMetric
from fairness.mitigation import ResamplingMitigation, RelabelingMitigation, SyntheticMitigation
from reporting.report_builder import generate_pdf_report
from config.defaults import POTENTIAL_SENSITIVE_ATTRIBUTES, DEFAULT_PRIORITY_SENSITIVE

class WorkflowController:
    """Orchestrates the fairness analysis workflow."""
    
    def __init__(self, ui: TerminalUI, context: ProjectContext):
        self.ui = ui
        self.context = context
        self.trainer = DefaultModelTrainer()

    def run(self):
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
        try:
            from reporting.benchmark_viz import BenchmarkVisualizer
            
            file_path = self.ui.get_benchmark_file_path()
            if not file_path or not os.path.exists(file_path):
                 self.ui.display_message("\nInvalid file path or file not found.")
                 self.ui.wait_for_user()
                 return
                 
            self.ui.display_message(f"\nLoading benchmark results from {file_path}...")
            viz = BenchmarkVisualizer(file_path)
            
            self.ui.display_message("\nGenerating standard report (Scatter Plots, Bar Charts, Heatmaps)...")
            viz.generate_full_report()
            
            self.ui.display_message(f"\nVisualization complete. Check the 'viz_report' folder in {os.path.dirname(file_path)}.")
            self.ui.wait_for_user()
        except Exception as e:
            self.ui.display_message(f"\nError during visualization: {e}")
            import traceback
            traceback.print_exc()
            self.ui.wait_for_user()

    def benchmark_workflow(self):
        try:
            from core.benchmark import BenchmarkEngine
            n_runs = self.ui.get_benchmark_runs()
            
            # Prompt for debug sample limit
            from utils.menus import get_user_input
            limit_input = get_user_input("Enter sample limit for debugging (press Enter to use full dataset)", lambda x: x == "" or x.isdigit(), allow_empty=True)
            debug_limit = int(limit_input) if limit_input else None
            
            engine = BenchmarkEngine(n_runs, debug_sample_limit=debug_limit)
            self.ui.display_message("\nStarting automated benchmark. This may take a while...")
            engine.run()
            self.ui.display_message("\nBenchmark complete. Results saved in 'outputs/reports/'.")
            self.ui.wait_for_user()
        except Exception as e:
            self.ui.display_message(f"Error during benchmark: {e}")
            self.ui.wait_for_user()

    def load_data_workflow(self):
        path = self.ui.get_dataset_path(validate_file_path)
        try:
            self.context.df = load_dataset(path)
            self.context.df, renamed = handle_duplicate_columns(self.context.df)
            if renamed:
                self.ui.display_message(f"\nWarning: Duplicate column names detected. Renamed columns: {renamed}")

            self.context.original_df = self.context.df.copy()
            valid, msg = validate_dataset(self.context.df)
            if not valid:
                self.ui.display_message(f"Error: {msg}")
                return
            
            self.ui.display_message("\nChecking for numeric columns hidden as text...")
            self.context.df, converted_cols = infer_numeric_types(self.context.df)
            if converted_cols:
                self.ui.display_message(f"✔ Auto-converted to numeric: {', '.join(converted_cols)}")
            
            self.ui.display_message(f"\nDataset loaded successfully. Rows: {self.context.df.shape[0]}, Columns: {self.context.df.shape[1]}")
            self.ui.display_message("\nAvailable columns: " + ", ".join(self.context.df.columns))
            
            detected_target = self.context.df.columns[-1]
            self.ui.display_message(f"\nTarget column detected: {detected_target}")
            if self.ui.confirm_target_col(detected_target):
                self.context.target_col = detected_target
            else:
                self.context.target_col = self.ui.get_target_col(self.context.df.columns.tolist())
            
            log_action(f"Loaded dataset {path}. Target: {self.context.target_col}")
            self.ui.wait_for_user("\nPress Enter to continue to EDA...")
            self.eda_workflow()
            
        except Exception as e:
            self.ui.display_message(f"Error loading dataset: {e}")
            self.ui.wait_for_user("Press Enter to return...")

    def eda_workflow(self):
        self.ui.display_message("\n--- Exploratory Data Analysis ---")
        self.ui.display_message("Generating exploratory statistics...")
        self.context.stats = get_basic_stats(self.context.df)
        plot_class_distribution(self.context.df, self.context.target_col)
        plot_missing_heatmap(self.context.df)
        self.ui.display_message("✔ Class distribution plot saved")
        self.ui.display_message("✔ Missing value report generated")
        self.ui.display_message("✔ Summary statistics computed")
        self.ui.wait_for_user("\nPress Enter to continue to Preprocessing...")
        self.preprocessing_workflow()

    def preprocessing_workflow(self):
        self.ui.display_message("\n--- Preprocessing ---")
        self.ui.display_message("Step 1: Feature Selection")
        selection_method = self.ui.get_feature_selection_method()
        
        methods_labels = {
            '1': 'Keep specific features (Manual Selection)',
            '2': 'Drop specific features (e.g. IDs, Names)',
            '3': 'Skip feature selection'
        }
        self.context.selections['preprocessing']['feature_selection'] = methods_labels[selection_method]

        if selection_method == '1':
            selected_cols = self.ui.get_columns_to_keep(self.context.df.columns.tolist())
            if self.context.target_col not in selected_cols:
                selected_cols.append(self.context.target_col)
            self.context.selected_features = [c for c in selected_cols if c != self.context.target_col]
            self.ui.display_message(f"✔ Features selected. Model will use {len(self.context.selected_features)} features (Target: '{self.context.target_col}').")
        elif selection_method == '2':
            to_drop = self.ui.get_columns_to_drop(self.context.df.columns.tolist())
            if to_drop:
                self.context.selected_features = [c for c in self.context.df.columns if c not in to_drop and c != self.context.target_col]
                self.ui.display_message(f"✔ Dropped {len(to_drop)} columns from model training.")
        else:
            self.context.selected_features = [c for c in self.context.df.columns if c != self.context.target_col]
        
        self.context.selections['preprocessing']['selected_features_list'] = self.context.selected_features

        self.ui.display_message("\nStep 2: Data Cleaning & Encoding")
        missing_choice = self.ui.ask_handle_missing_values()
        if missing_choice:
            strat_labels = {'1': 'Mean/Median', '2': 'Mode', '3': 'Drop'}
            self.context.selections['preprocessing']['missing_values'] = strat_labels[missing_choice]
            strategy_map = {'1': 'mean', '2': 'mode', '3': 'drop'}
            
            if strategy_map[missing_choice] == 'drop':
                cols_to_check = (self.context.selected_features + [self.context.target_col]) if self.context.selected_features else None
                self.context.df = impute_missing(self.context.df, 'drop', columns=cols_to_check)
                self.context.df = impute_missing(self.context.df, 'mode', columns=None) 
            else:
                self.context.df = impute_missing(self.context.df, strategy_map[missing_choice], columns=None)
            self.ui.display_message(f"✔ Missing values handled. Rows remaining: {len(self.context.df)}")
        else:
            self.context.selections['preprocessing']['missing_values'] = "Skipped"

        outlier_choice = self.ui.ask_handle_outliers()
        if outlier_choice == '1':
            self.context.selections['preprocessing']['outliers'] = "IQR-based removal"
            cols_for_outliers = (self.context.selected_features + [self.context.target_col]) if self.context.selected_features else None
            self.context.df = remove_outliers_iqr(self.context.df, columns=cols_for_outliers)
            self.ui.display_message(f"✔ Outliers removed. Rows remaining: {len(self.context.df)}")
        else:
            self.context.selections['preprocessing']['outliers'] = "Skipped"

        if len(self.context.df) == 0:
            self.ui.display_message("\nCritical Warning: Preprocessing has removed ALL rows from the dataset.")
            self.ui.wait_for_user("Press Enter to return to main menu...")
            return

        cat_cols = list(self.context.df.select_dtypes(include=['object', 'category', 'bool']).columns)
        if self.context.target_col in cat_cols: cat_cols.remove(self.context.target_col)
        
        encoding_info = self.ui.ask_encode_categorical(cat_cols, self.context.selected_features or [])
        if encoding_info and encoding_info['choice'] != 'none':
            strat = encoding_info['choice']
            cols_to_encode = encoding_info['columns']
            strat_labels = {'1': 'One-hot encoding', '2': 'Label encoding'}
            self.context.selections['preprocessing']['encoding'] = strat_labels[strat]
            col_count_before = len(self.context.df.columns)
            
            if strat == '1':
                self.context.df = one_hot_encode(self.context.df, self.context.target_col, columns=cols_to_encode)
                if self.context.selected_features:
                    new_features = []
                    for feat in self.context.selected_features:
                        dummies = [c for c in self.context.df.columns if c.startswith(f"{feat}_") and c not in self.context.df.columns[:col_count_before]]
                        if dummies: new_features.extend(dummies)
                        else: new_features.append(feat) 
                    self.context.selected_features = new_features
            else:
                self.context.df = label_encode(self.context.df, self.context.target_col, columns=cols_to_encode)
            self.ui.display_message(f"✔ Encoding applied. Columns: {col_count_before} -> {len(self.context.df.columns)}")
        else:
            self.context.selections['preprocessing']['encoding'] = "Skipped"

        if self.context.selected_features:
            remaining_cat = [c for c in self.context.df[self.context.selected_features].select_dtypes(exclude=['number']).columns]
            if remaining_cat:
                rem_choice = self.ui.get_remaining_categorical_handling(remaining_cat)
                rem_labels = {
                    '1': 'Label Encode automatically',
                    '2': 'Drop from model training',
                    '3': 'Keep as-is'
                }
                self.context.selections['preprocessing']['remaining_categoricals'] = rem_labels.get(rem_choice, "Skipped")
                if rem_choice == '1':
                    self.context.df = label_encode(self.context.df, self.context.target_col, columns=remaining_cat)
                    self.ui.display_message(f"✔ Automatically encoded remaining features.")
                elif rem_choice == '2':
                    self.context.selected_features = [f for f in self.context.selected_features if f not in remaining_cat]
                    self.ui.display_message(f"✔ Features dropped from model selection.")
            else:
                self.context.selections['preprocessing']['remaining_categoricals'] = "None required (all features numeric)"
        
        if self.context.df[self.context.target_col].dtype == 'object':
             from sklearn.preprocessing import LabelEncoder
             le = LabelEncoder()
             self.context.df[self.context.target_col] = le.fit_transform(self.context.df[self.context.target_col].astype(str))
            
        log_action("Preprocessing completed")
        self.ui.wait_for_user("\nPress Enter to continue to Fairness Analysis...")
        self.fairness_setup_workflow()

    def fairness_setup_workflow(self):
        potential = [c for c in self.context.df.columns if c.lower() in POTENTIAL_SENSITIVE_ATTRIBUTES]
        default_sens = next((c for c in potential if c.lower() == DEFAULT_PRIORITY_SENSITIVE.lower()), None) or (potential[0] if potential else None)

        while True:
            user_input = self.ui.get_sensitive_attributes(self.context.df.columns.tolist(), default=default_sens)
            try:
                selected_cols = parse_attribute_input(user_input, self.context.df.columns.tolist())
                if selected_cols: break
                self.ui.display_message("Error: No valid columns selected.")
            except ValueError as e:
                self.ui.display_message(f"Error: {e}")

        self.context.df, self.context.sensitive_col = create_composite_attribute(self.context.df, selected_cols)
        self.context.selections['fairness']['sensitive_column'] = self.context.sensitive_col
        self.context.selections['fairness']['selected_sensitive_attributes'] = selected_cols

        mapping = {} if len(selected_cols) > 1 else get_sensitive_mapping(self.context.df, self.context.original_df, self.context.sensitive_col)
        unique_vals = self.context.df[self.context.sensitive_col].unique()
        
        if len(unique_vals) == 0:
            self.ui.display_message(f"\nError: Column '{self.context.sensitive_col}' has no values.")
            return

        user_val = self.ui.get_privileged_group(unique_vals.tolist(), mapping)
        if mapping and user_val in mapping:
            self.context.privileged_group = mapping[user_val]
            self.context.privileged_group_name = user_val
        else:
            def cast_value(val, target_series):
                if pd.api.types.is_numeric_dtype(target_series):
                    try: return type(target_series.iloc[0])(val)
                    except: 
                        try: return float(val)
                        except: return val
                return val
            self.context.privileged_group = cast_value(user_val, self.context.df[self.context.sensitive_col])
            if self.context.privileged_group not in unique_vals:
                self.context.privileged_group = unique_vals[0]
            self.context.privileged_group_name = str(self.context.privileged_group)

        self.context.selections['fairness']['privileged_group'] = self.context.privileged_group_name
        self.context.inverse_sensitive_mapping = {v: k for k, v in mapping.items()} if mapping else None

        un_choice = self.ui.get_unprivileged_comparison_mode()
        self.context.unprivileged_group = [x for x in unique_vals if x != self.context.privileged_group]
        if un_choice == '1':
            self.context.comparison_mode = 'combined'
            if self.ui.confirm_binary_transformation():
                self.context.df = binarize_attribute(self.context.df, self.context.sensitive_col, self.context.privileged_group)
                old_priv_name = self.context.privileged_group_name
                self.context.privileged_group = 1
                self.context.unprivileged_group = [0]
                self.context.privileged_group_name = f"Privileged ({old_priv_name})"
                self.context.inverse_sensitive_mapping = {1: self.context.privileged_group_name, 0: "Unprivileged (Rest)"}
        else:
            self.context.comparison_mode = 'individual'

        self.context.stats_raw = get_comprehensive_stats(self.context.df, self.context.target_col, self.context.sensitive_col, original_feat_count=self.context.original_df.shape[1], selected_features=self.context.selected_features)
        self.context.stats_baseline = self.context.stats_raw.copy()

        self.context.metric_choice = self.ui.get_specific_fairness_metric()
        metric_labels = {'1': 'Demographic Parity', '2': 'Equalized Odds'}
        self.context.selections['fairness']['specific_metric'] = metric_labels.get(self.context.metric_choice, "Skipped")
        
        m_choice = self.ui.get_model_choice()
        model_map = {'1': 'logistic', '2': 'random_forest', '3': 'gbm', '4': 'linear_svc'}
        self.context.model_choice = model_map[m_choice]
        
        self.ui.wait_for_user("\nPress Enter to compute baseline...")
        self.baseline_evaluation()

    def baseline_evaluation(self):
        self.ui.display_message("\n--- Baseline Evaluation ---")
        X = self.context.df[self.context.selected_features] if self.context.selected_features else self.context.df.drop(columns=[self.context.target_col])
        y = self.context.df[self.context.target_col]
        
        X = X.select_dtypes(include=[np.number])
        if X.isnull().sum().sum() > 0: X = impute_missing(X, strategy='mean')

        model, X_val, y_val, X_test, y_test, y_prob_val, y_prob_test = self.trainer.train(X, y, self.context.model_choice)
        self.context.y_test_bl = y_test
        self.context.y_pred_bl = model.predict(X_test)
        self.context.y_prob_bl = y_prob_test
        
        self.context.metrics_baseline.update({f"Val {k}": v for k, v in self.trainer.evaluate(model, X_val, y_val).items()})
        self.context.metrics_baseline.update({f"Test {k}": v for k, v in self.trainer.evaluate(model, X_test, y_test).items()})
        
        sens_test = self.context.df.loc[X_test.index, self.context.sensitive_col]
        temp_df = X_test.copy(); temp_df[self.context.target_col] = self.context.y_pred_bl; temp_df[self.context.sensitive_col] = sens_test.values
        
        # Group Fairness
        metric_impl = GroupFairnessMetric() if self.context.metric_choice == '1' else ClassificationFairnessMetric(pd.DataFrame({self.context.target_col: y_test.values, self.context.sensitive_col: sens_test.values}))
        
        if self.context.comparison_mode == 'combined':
            self.context.metrics_baseline.update(metric_impl.compute(temp_df, self.context.target_col, self.context.sensitive_col, self.context.privileged_group, self.context.unprivileged_group))
        else:
            for val in self.context.unprivileged_group:
                group_name = self.context.inverse_sensitive_mapping.get(val, str(val)) if self.context.inverse_sensitive_mapping else str(val)
                res = metric_impl.compute(temp_df, self.context.target_col, self.context.sensitive_col, self.context.privileged_group, val)
                self.context.metrics_baseline.update({f"[{group_name}] {k}": v for k, v in res.items()})
             
        self.ui.display_metrics(self.context.metrics_baseline, "Baseline Metrics")
        self.ui.wait_for_user("\nPress Enter to proceed to Mitigation...")
        self.mitigation_workflow()

    def mitigation_workflow(self):
        method_choice = self.ui.get_mitigation_method()
        self.context.metrics_mitigated = self.context.metrics_baseline.copy()
        
        mit_info = {}
        if method_choice == '4':
            self.context.df_improved = self.context.df.copy()
            mit_info['method'] = "None (Baseline)"
        else:
            strategy = None
            if method_choice == '1':
                res_type = self.ui.get_resampling_type()
                strategy = ResamplingMitigation('oversample' if res_type == '1' else 'undersample')
                mit_info['method'] = f"Resampling ({'Oversampling' if res_type == '1' else 'Undersampling'})"
            elif method_choice == '2':
                strategy = RelabelingMitigation(self.context.selected_features)
                mit_info['method'] = "Relabeling"
            elif method_choice == '3':
                synth_method = 'smote' if self.ui.get_synthetic_method() == '1' else 'cda'
                strategy = SyntheticMitigation(synth_method)
                mit_info['method'] = f"Synthetic ({synth_method.upper()})"
            
            self.context.selections['mitigation'] = mit_info
            
            self.context.df_improved = strategy.mitigate(self.context.df, self.context.target_col, self.context.sensitive_col, self.context.privileged_group, self.context.unprivileged_group)
            
            X_model = self.context.df_improved[self.context.selected_features] if self.context.selected_features else self.context.df_improved.drop(columns=[self.context.target_col])
            X_model_numeric = X_model.select_dtypes(include=['number'])
            y_res = self.context.df_improved[self.context.target_col]
            
            model, X_val, y_val, X_test, y_test, y_prob_val, y_prob_test = self.trainer.train(X_model_numeric, y_res, self.context.model_choice)
            self.context.y_test_mit = y_test
            self.context.y_pred_mit = model.predict(X_test)
            self.context.y_prob_mit = y_prob_test
            
            self.context.metrics_mitigated.update({f"Val {k}": v for k, v in self.trainer.evaluate(model, X_val, y_val).items()})
            self.context.metrics_mitigated.update({f"Test {k}": v for k, v in self.trainer.evaluate(model, X_test, y_test).items()})
            
            sens_test = self.context.df_improved.loc[X_test.index, self.context.sensitive_col]
            temp_df = X_test.copy(); temp_df[self.context.target_col] = self.context.y_pred_mit; temp_df[self.context.sensitive_col] = sens_test.values
            
            # Group Fairness
            metric_impl = GroupFairnessMetric() if self.context.metric_choice == '1' else ClassificationFairnessMetric(pd.DataFrame({self.context.target_col: y_test.values, self.context.sensitive_col: sens_test.values}))
            self.context.metrics_mitigated.update(metric_impl.compute(temp_df, self.context.target_col, self.context.sensitive_col, self.context.privileged_group, self.context.unprivileged_group))

        if self.context.df_improved is not None:
            self.context.stats_mitigated = get_comprehensive_stats(self.context.df_improved, self.context.target_col, self.context.sensitive_col, selected_features=self.context.selected_features)

        self.ui.display_metrics(self.context.metrics_mitigated, "Mitigated Metrics")
        self.ui.wait_for_user("\nPress Enter to generate reports...")
        self.output_workflow()

    def output_workflow(self):
        os.makedirs("outputs/datasets", exist_ok=True)
        os.makedirs("outputs/reports", exist_ok=True)
        out_path = os.path.join("outputs/datasets", "fairness_improved.csv")
        self.context.df_improved.to_csv(out_path, index=False)
        self.ui.display_message(f"✔ Improved dataset saved to {out_path}")
        
        df_display_bl = self.context.df.copy()
        df_display_mit = self.context.df_improved.copy() if self.context.df_improved is not None else None
        
        if self.context.inverse_sensitive_mapping:
            df_display_bl[self.context.sensitive_col] = df_display_bl[self.context.sensitive_col].map(self.context.inverse_sensitive_mapping)
            if df_display_mit is not None:
                df_display_mit[self.context.sensitive_col] = df_display_mit[self.context.sensitive_col].map(self.context.inverse_sensitive_mapping)

        self.context.stats_baseline = get_comprehensive_stats(df_display_bl, self.context.target_col, self.context.sensitive_col, original_feat_count=self.context.original_df.shape[1], selected_features=self.context.selected_features)
        if df_display_mit is not None:
            self.context.stats_mitigated = get_comprehensive_stats(df_display_mit, self.context.target_col, self.context.sensitive_col, original_feat_count=self.context.original_df.shape[1], selected_features=self.context.selected_features)

        plots = {}
        if self.context.y_test_bl is not None:
            plots["Baseline Confusion Matrix"] = plot_confusion_matrix(self.context.y_test_bl, self.context.y_pred_bl, "Baseline Confusion Matrix", "cm_baseline.png")
            sens_test_bl = df_display_bl.loc[self.context.y_test_bl.index, self.context.sensitive_col]
            plots["Baseline Subgroup Confusion Matrices"] = plot_subgroup_confusion_matrices(self.context.y_test_bl.values, self.context.y_pred_bl, sens_test_bl.values, "cm_subgroups_baseline.png")
            if self.context.y_prob_bl is not None:
                plots["Baseline KDE Predicted Probabilities"] = plot_kde_probabilities(self.context.y_prob_bl, sens_test_bl.values, "kde_baseline.png")

        if self.context.y_test_mit is not None:
            plots["Mitigated Confusion Matrix"] = plot_confusion_matrix(self.context.y_test_mit, self.context.y_pred_mit, "Mitigated Confusion Matrix", "cm_mitigated.png")
            sens_test_mit = df_display_mit.loc[self.context.y_test_mit.index, self.context.sensitive_col]
            plots["Mitigated Subgroup Confusion Matrices"] = plot_subgroup_confusion_matrices(self.context.y_test_mit.values, self.context.y_pred_mit, sens_test_mit.values, "cm_subgroups_mitigated.png")
            if self.context.y_prob_mit is not None:
                plots["Mitigated KDE Predicted Probabilities"] = plot_kde_probabilities(self.context.y_prob_mit, sens_test_mit.values, "kde_mitigated.png")
            
        plots["Performance & Fairness Metrics Comparison"] = plot_metric_comparison(self.context.metrics_baseline, self.context.metrics_mitigated)
        
        fairness_key = "Statistical Parity Difference" if "Statistical Parity Difference" in self.context.metrics_baseline else "Disparate Impact"
        utility_key = "Test Accuracy"
        tradeoff_data = [
            {fairness_key: self.context.metrics_baseline.get(fairness_key, 0), utility_key: self.context.metrics_baseline.get(utility_key, 0), 'Stage': 'Baseline'},
            {fairness_key: self.context.metrics_mitigated.get(fairness_key, 0), utility_key: self.context.metrics_mitigated.get(utility_key, 0), 'Stage': 'Mitigated'}
        ]
        plots["Fairness-Utility Trade-off"] = plot_fairness_utility_tradeoff(tradeoff_data, fairness_key, utility_key, "tradeoff.png")

        if df_display_mit is not None:
            plots["Class Distribution Comparison"] = plot_distribution_comparison(df_display_bl, df_display_mit, self.context.target_col, "Class Distribution: Before vs After", "class_dist_comparison.png")
            plots["Sensitive Attribute Distribution Comparison"] = plot_distribution_comparison(df_display_bl, df_display_mit, self.context.sensitive_col, "Sensitive Attribute Distribution: Before vs After", "sensitive_dist_comparison.png")
            plots["Selection Rate Comparison P(Y=1|A)"] = plot_selection_rates(df_display_bl, df_display_mit, self.context.sensitive_col, self.context.target_col, "selection_rates_comparison.png")
            plots["Grouped Bar Charts (Positive Rate)"] = plot_grouped_bar_charts(df_display_mit, self.context.sensitive_col, self.context.target_col, "grouped_bars_mitigated.png")
            plots["Baseline Subgroup Heatmap"] = plot_contingency_heatmap(df_display_bl, self.context.sensitive_col, self.context.target_col, "Baseline: Subgroup Counts (A, Y)", "heatmap_baseline.png")
            plots["Mitigated Subgroup Heatmap"] = plot_contingency_heatmap(df_display_mit, self.context.sensitive_col, self.context.target_col, "Mitigated: Subgroup Counts (A, Y)", "heatmap_mitigated.png")

        generate_pdf_report("fairness_report.pdf", self.context.stats_baseline, self.context.metrics_baseline, self.context.metrics_mitigated, plots=plots, stats_after=self.context.stats_mitigated, selections=self.context.selections)
        self.ui.display_message("\nAnalysis Complete!")
        self.ui.wait_for_user("Press Enter to return to main menu...")

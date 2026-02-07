import sys
import os
import warnings
import pandas as pd
import numpy as np

# Suppress FutureWarning from inFairness until they update to torch.vmap
warnings.filterwarnings("ignore", category=FutureWarning, module="inFairness")
# Suppress RuntimeWarning from aif360 when subgroups are too small for certain metrics
warnings.filterwarnings("ignore", category=RuntimeWarning, module="aif360.metrics.classification_metric")

from utils.menus import clear_screen, print_header, get_user_choice, get_user_confirmation, get_user_input
from utils.logging import setup_logging, log_action
from config.defaults import POTENTIAL_SENSITIVE_ATTRIBUTES, DEFAULT_PRIORITY_SENSITIVE
from data import (
    load_dataset, validate_file_path, validate_dataset, 
    handle_duplicate_columns, infer_numeric_types, get_sensitive_mapping
)
from eda.statistics import get_basic_stats, get_comprehensive_stats
from eda.visualizations import (
    plot_class_distribution, plot_missing_heatmap, plot_confusion_matrix, 
    plot_metric_comparison, plot_distribution_comparison,
    plot_selection_rates, plot_contingency_heatmap
)
from preprocessing.missing_values import impute_missing
from preprocessing.outliers import remove_outliers_iqr
from preprocessing.encoding import one_hot_encode, label_encode
from models.classification import train_classifier
from fairness.metrics import compute_group_fairness, compute_classification_fairness, compute_individual_fairness
from fairness.mitigation import mitigate_resampling, mitigate_relabeling, mitigate_synthetic
from evaluation.performance import evaluate_classification
from reporting.report_builder import generate_pdf_report

class FairnessApp:
    def __init__(self):
        self.df = None
        self.original_df = None
        self.target_col = None
        self.sensitive_col = None
        self.privileged_group = None
        self.unprivileged_group = None
        self.task_type = None
        self.model = None
        self.model_choice = 'logistic'
        self.metrics_baseline = {}
        self.metrics_mitigated = {}
        self.stats_baseline = {}
        self.stats_mitigated = {}
        self.stats = {}
        self.y_test_bl = None
        self.y_pred_bl = None
        self.y_probs_bl = None
        self.y_test_mit = None
        self.y_pred_mit = None
        self.y_probs_mit = None
        self.df_improved = None
        self.inverse_sensitive_mapping = None
        self.selected_features = None  # List of columns selected for the model
        self.comparison_mode = 'combined' # 'combined' or 'individual'
        self.initial_missing_count = 0
        self.selections = {
            'preprocessing': {},
            'fairness': {},
            'mitigation': {},
            'model': {}
        }
        
        setup_logging()

    def run(self):
        while True:
            clear_screen()
            print_header("Fairness Toolkit for Tabular Datasets")
            options = {'1': 'Load dataset', '2': 'Exit'}
            choice = get_user_choice(options)
            
            if choice == '1':
                self.load_data_workflow()
            elif choice == '2':
                sys.exit()

    def display_stats(self, stats, title="Dataset Statistics"):
        print(f"\n--- {title} ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    def load_data_workflow(self):
        print_header("Dataset Loading")
        path = get_user_input("Enter dataset path (.csv or .xlsx)", validate_file_path)
        
        try:
            self.df = load_dataset(path)
            
            # --- Handle Duplicate Columns ---
            self.df, renamed = handle_duplicate_columns(self.df)
            if renamed:
                print(f"\nWarning: Duplicate column names detected. Renamed columns: {renamed}")

            self.original_df = self.df.copy()
            valid, msg = validate_dataset(self.df)
            if not valid:
                print(f"Error: {msg}")
                return
            
            # --- Robust Type Inference ---
            print("\nChecking for numeric columns hidden as text...")
            self.df, converted_cols = infer_numeric_types(self.df)
            
            if converted_cols:
                print(f"✔ Auto-converted to numeric: {', '.join(converted_cols)}")
            
            print(f"\nDataset loaded successfully. Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
            
            # List all columns
            print("\nAvailable columns:")
            print(", ".join(self.df.columns))
            
            # Auto-detect target? For now ask user or last column
            detected_target = self.df.columns[-1]
            print(f"\nTarget column detected: {detected_target}")
            if get_user_confirmation(f"Is '{detected_target}' the target column?"):
                self.target_col = detected_target
            else:
                self.target_col = get_user_input("Enter target column name", lambda x: x in self.df.columns)
            
            log_action(f"Loaded dataset {path}. Target: {self.target_col}")
            input("\nPress Enter to continue to EDA...")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            input("Press Enter to return...")
            return

        self.eda_workflow()

    def eda_workflow(self):
        print_header("Exploratory Data Analysis")
        print("Generating exploratory statistics...")
        
        self.stats = get_basic_stats(self.df)
        plot_class_distribution(self.df, self.target_col)
        plot_missing_heatmap(self.df)
        
        print("✔ Class distribution plot saved")
        print("✔ Missing value report generated")
        print("✔ Summary statistics computed")
        
        input("\nPress Enter to continue to Preprocessing...")
        self.preprocessing_workflow()

    def preprocessing_workflow(self):
        print_header("Preprocessing")
        
        # Feature Selection Step
        print("Step 1: Feature Selection")
        feat_options = {
            '1': 'Keep specific features (Manual Selection)',
            '2': 'Drop specific features (e.g. IDs, Names)',
            '3': 'Skip feature selection'
        }
        selection_method = get_user_choice(feat_options, "Select Feature Selection Method")
        self.selections['preprocessing']['feature_selection'] = feat_options[selection_method]

        if selection_method == '1':
            print(f"\nAvailable columns: {', '.join(self.df.columns)}")
            print(f"Target column ('{self.target_col}') will be kept automatically.")
            cols_input = get_user_input("Enter column names to KEEP (comma separated)", lambda x: True)
            selected_cols = [c.strip() for c in cols_input.split(',') if c.strip() in self.df.columns]
            if self.target_col not in selected_cols:
                selected_cols.append(self.target_col)
            
            # Store selected features for model training, but keep full df
            self.selected_features = [c for c in selected_cols if c != self.target_col]
            print(f"✔ Features selected. Model will use {len(self.selected_features)} features (Target: '{self.target_col}').")

        elif selection_method == '2':
            print(f"\nAvailable columns: {', '.join(self.df.columns)}")
            cols_input = get_user_input("Enter column names to DROP (comma separated)", lambda x: True)
            to_drop = [c.strip() for c in cols_input.split(',') if c.strip() in self.df.columns]
            if to_drop:
                # Store kept features
                self.selected_features = [c for c in self.df.columns if c not in to_drop and c != self.target_col]
                print(f"✔ Dropped {len(to_drop)} columns from model training.")
            else:
                print("No valid columns entered to drop.")
        else:
            self.selected_features = [c for c in self.df.columns if c != self.target_col]
        
        self.selections['preprocessing']['selected_features_list'] = self.selected_features

        # Data Cleaning Steps
        print("\nStep 2: Data Cleaning & Encoding")
        if get_user_confirmation("Do you want to handle missing values?"):
            strat_options = {
                '1': 'Mean/Median (Pros: Retains data size; Cons: Can distort distributions, ignores correlations)',
                '2': 'Mode (Pros: Works for categorical; Cons: Can bias towards majority class)',
                '3': 'Drop (Pros: Removes noise; Cons: Reduces data size, potential bias if missingness is systematic)'
            }
            strat = get_user_choice(strat_options, "Select Missing Value Strategy")
            self.selections['preprocessing']['missing_values'] = strat_options[strat]
            strategy_map = {'1': 'mean', '2': 'mode', '3': 'drop'}
            
            if strategy_map[strat] == 'drop':
                # If dropping, we only check selected columns to avoid over-deleting
                cols_to_check = None
                if self.selected_features:
                    cols_to_check = self.selected_features + [self.target_col]
                self.df = impute_missing(self.df, 'drop', columns=cols_to_check)
                
                # IMPORTANT: After dropping rows with critical NaNs, we fill any NaNs in the REMAINING columns
                # so the final enhanced dataset is clean and oversampling doesn't duplicate them.
                self.df = impute_missing(self.df, 'mode', columns=None) 
            else:
                # If imputing, we clean the entire dataset (columns=None)
                self.df = impute_missing(self.df, strategy_map[strat], columns=None)
            
            print(f"✔ Missing values handled. Rows remaining: {len(self.df)}")
        else:
            self.selections['preprocessing']['missing_values'] = "Skipped"

        if get_user_confirmation("Do you want to handle outliers?"):
            strat_options = {
                '1': 'IQR-based removal (Pros: Removes extreme values that might drive bias; Cons: Might remove minority group outliers)',
                '2': 'Skip (Pros: Retains all data; Cons: Models might be sensitive to outliers)'
            }
            strat = get_user_choice(strat_options, "Select Outlier Strategy")
            self.selections['preprocessing']['outliers'] = strat_options[strat]
            if strat == '1':
                # Apply outlier removal ONLY to selected features + sensitive candidates to avoid dropping all rows 
                # based on metadata like unique IDs that might look like outliers
                cols_for_outliers = None
                if self.selected_features:
                    cols_for_outliers = self.selected_features + [self.target_col]
                
                self.df = remove_outliers_iqr(self.df, columns=cols_for_outliers)
                print(f"✔ Outliers removed. Rows remaining: {len(self.df)}")
        else:
            self.selections['preprocessing']['outliers'] = "Skipped"

        if len(self.df) == 0:
            print("\nCritical Warning: Preprocessing has removed ALL rows from the dataset.")
            print("Please restart and choose less aggressive cleaning strategies (e.g., don't drop rows).")
            input("Press Enter to return to main menu...")
            return

        if get_user_confirmation("Do you want to encode categorical variables?"):
            # Check for categorical columns (including bool)
            cat_cols = list(self.df.select_dtypes(include=['object', 'category', 'bool']).columns)
            if self.target_col in cat_cols:
                cat_cols.remove(self.target_col)
            
            if not cat_cols:
                print("\nNo categorical columns detected to encode (excluding target).")
                cols_to_encode = []
                self.selections['preprocessing']['encoding'] = "No categorical columns detected"
            else:
                # Highlight which of the SELECTED features need encoding
                if self.selected_features:
                    selected_cat = [c for c in cat_cols if c in self.selected_features]
                    if selected_cat:
                        print(f"\nCRITICAL: The following SELECTED features are categorical and MUST be encoded to be used in the model: {', '.join(selected_cat)}")
                    else:
                        print(f"\nNote: All selected features are already numeric. Other categorical columns in the dataset: {', '.join(cat_cols[:10])}...")
                else:
                    print(f"\nDetected categorical columns: {', '.join(cat_cols[:20])}...")
                
                # Check for high cardinality columns to warn user
                high_card_cols = [c for c in cat_cols if self.df[c].nunique() > 20]
                if high_card_cols:
                    print(f"Warning: High cardinality (>20 unique values) detected in: {', '.join(high_card_cols[:5])}...")
                
                cols_input = get_user_input("Enter categorical column names to ENCODE (comma separated, leave empty for all)", lambda x: True, allow_empty=True)
                if cols_input.strip():
                    cols_to_encode = [c.strip() for c in cols_input.split(',') if c.strip() in self.df.columns]
                else:
                    cols_to_encode = cat_cols
            
            if not cols_to_encode and cat_cols:
                 print("No valid columns selected for manual encoding. Skipping manual step.")
                 self.selections['preprocessing']['encoding'] = "Manual selection skipped"
            elif not cols_to_encode:
                 pass 
            else:
                strat_options = {
                    '1': 'One-hot encoding (Pros: No ordinal assumption, good for fairness; Cons: High dimensionality)',
                    '2': 'Label encoding (Pros: Low dimensionality; Cons: Impose arbitrary order, bad for linear fairness)'
                }
                strat = get_user_choice(strat_options, "Select Encoding Method")
                self.selections['preprocessing']['encoding'] = strat_options[strat]
                col_count_before = len(self.df.columns)
                
                if strat == '1':
                    # Preserve selected features list before dummies are added
                    old_features = self.selected_features if self.selected_features else [c for c in self.df.columns if c != self.target_col]
                    
                    self.df = one_hot_encode(self.df, self.target_col, columns=cols_to_encode)
                    
                    # Update selected_features to include the new dummy columns
                    if self.selected_features:
                        new_features = []
                        for feat in self.selected_features:
                            dummies = [c for c in self.df.columns if c.startswith(f"{feat}_") and c not in self.df.columns[:col_count_before]]
                            if dummies:
                                new_features.extend(dummies)
                            else:
                                new_features.append(feat) 
                        self.selected_features = new_features
                else:
                    self.df = label_encode(self.df, self.target_col, columns=cols_to_encode)
                
                print(f"✔ Encoding applied. Columns: {col_count_before} -> {len(self.df.columns)}")

            # --- NEW: Handle Remaining Categoricals in Selected Features ---
            if self.selected_features:
                remaining_cat = [c for c in self.df[self.selected_features].select_dtypes(exclude=['number']).columns]
                if remaining_cat:
                    print(f"\nNotice: The following selected features are still categorical: {remaining_cat}")
                    print("To 'calculate all metrics', these must be numeric. How should they be handled?")
                    rem_options = {
                        '1': 'Label Encode automatically (Keeps column structure, converts to numbers)',
                        '2': 'Drop from model training (Features will be ignored by the model)',
                        '3': 'Keep as-is (Warning: They will be dropped automatically during model training)'
                    }
                    rem_choice = get_user_choice(rem_options)
                    self.selections['preprocessing']['remaining_categoricals'] = rem_options[rem_choice]
                    if rem_choice == '1':
                        self.df = label_encode(self.df, self.target_col, columns=remaining_cat)
                        print(f"✔ Automatically encoded remaining features.")
                    elif rem_choice == '2':
                        self.selected_features = [f for f in self.selected_features if f not in remaining_cat]
                        print(f"✔ Features dropped from model selection.")
                else:
                    self.selections['preprocessing']['remaining_categoricals'] = "None remaining"
        else:
            self.selections['preprocessing']['encoding'] = "Skipped"

        if self.df[self.target_col].dtype == 'object':
             from sklearn.preprocessing import LabelEncoder
             le = LabelEncoder()
             self.df[self.target_col] = le.fit_transform(self.df[self.target_col].astype(str))
            
        log_action("Preprocessing completed")
        input("\nPress Enter to continue to Fairness Analysis...")
        self.fairness_setup_workflow()

    def fairness_setup_workflow(self):
        print_header("Fairness Setup")
        
        print("Available columns:")
        print(", ".join(self.df.columns))
        
        potential = [c for c in self.df.columns if c.lower() in POTENTIAL_SENSITIVE_ATTRIBUTES]
        if potential:
            print(f"\nDetected potential sensitive attributes: {potential}")
        
        # Determine default choice (prioritize 'Race' if configured)
        default_sens = next((c for c in potential if c.lower() == DEFAULT_PRIORITY_SENSITIVE.lower()), None)
        # If no priority default found but potential exists, use the first one
        if not default_sens and potential:
            default_sens = potential[0]

        self.sensitive_col = get_user_input("Enter sensitive column name", lambda x: x in self.df.columns, default=default_sens)
        self.selections['fairness']['sensitive_column'] = self.sensitive_col
        
        mapping = get_sensitive_mapping(self.df, self.original_df, self.sensitive_col)
                        
        if mapping:
            self.inverse_sensitive_mapping = {v: k for k, v in mapping.items()}
            print(f"Values in '{self.sensitive_col}': {list(mapping.keys())}")
            user_val = get_user_input("Enter value for privileged group (by name)")
            
            if user_val in mapping:
                self.privileged_group = mapping[user_val]
                self.privileged_group_name = user_val
                print(f"Mapped '{user_val}' to encoded value {self.privileged_group}")
            else:
                print(f"Warning: '{user_val}' not found. Available: {list(mapping.keys())}")
                print("Defaulting to first available value.")
                self.privileged_group_name = list(mapping.keys())[0]
                self.privileged_group = mapping[self.privileged_group_name]
                
            unique_vals = self.df[self.sensitive_col].unique()
            
        else:
            unique_vals = self.df[self.sensitive_col].unique()
            if len(unique_vals) == 0:
                print(f"\nError: Column '{self.sensitive_col}' has no values. This might be because all rows were dropped during preprocessing.")
                input("Press Enter to return...")
                return

            print(f"Values in '{self.sensitive_col}': {unique_vals}")
            self.privileged_group = get_user_input("Enter value for privileged group")
            
            def cast_value(val, target_series):
                if pd.api.types.is_numeric_dtype(target_series):
                    try:
                        return type(target_series.iloc[0])(val)
                    except:
                        try: return float(val)
                        except: return val
                return val

            self.privileged_group = cast_value(self.privileged_group, self.df[self.sensitive_col])
            
            if self.privileged_group not in unique_vals:
                print(f"Warning: {self.privileged_group} not found in {unique_vals}. Defaulting to first value.")
                self.privileged_group = unique_vals[0]
            
            self.privileged_group_name = str(self.privileged_group)

        self.selections['fairness']['privileged_group'] = self.privileged_group_name

        # Simplified Unprivileged Group Selection (Shared for both Encoded and Raw)
        other_vals = [x for x in unique_vals if x != self.privileged_group]
        print("\nSelect Unprivileged Group Comparison:")
        un_options = {
            '1': 'All other groups combined',
            '2': 'Against each group individually'
        }
        un_choice = get_user_choice(un_options)
        self.selections['fairness']['unprivileged_comparison'] = un_options[un_choice]
        
        self.unprivileged_group = other_vals # List of all other values
        if un_choice == '1':
            self.comparison_mode = 'combined'
        else:
            self.comparison_mode = 'individual'

        unpriv_names = [self.inverse_sensitive_mapping.get(x, str(x)) for x in self.unprivileged_group] if self.inverse_sensitive_mapping else self.unprivileged_group
        print(f"Privileged: {self.privileged_group_name}, Unprivileged: {unpriv_names}")

        # Capture RAW statistics for the report comparison (Before any cleaning/mitigation)
        self.stats_raw = get_comprehensive_stats(self.df, self.target_col, self.sensitive_col, original_feat_count=self.original_df.shape[1], selected_features=self.selected_features)
        self.stats_baseline = self.stats_raw.copy()

        # 5. Metric Selection
        print("\nFairness Metric Type:")
        m_type_options = {'1': 'Group Fairness', '2': 'Individual Fairness'}
        self.m_type = get_user_choice(m_type_options)
        self.selections['fairness']['metric_type'] = m_type_options[self.m_type]
        
        if self.m_type == '1': # Group Fairness
            print("\nAvailable Group Fairness Metrics:")
            metric_options = {
                '1': 'Demographic Parity (Pros: Ensures equal acceptance rates; Cons: Ignores qualified candidates if base rates differ)',
                '2': 'Equalized Odds (Pros: Balances error rates/quality; Cons: Harder to achieve, allows unequal outcomes)'
            }
            self.metric_choice = get_user_choice(metric_options, "Select a metric")
            self.selections['fairness']['specific_metric'] = metric_options[self.metric_choice]
        else: # Individual Fairness
            print("\nAvailable Individual Fairness Metrics:")
            metric_options = {
                '1': 'Consistency (Pros: Similar individuals treated similarly; Cons: Depends heavily on distance metric)',
                '2': 'Similarity-based Fairness (Pros: Context-aware; Cons: Requires defining "similarity" matrix)'
            }
            self.metric_choice = get_user_choice(metric_options, "Select a metric")
            self.selections['fairness']['specific_metric'] = metric_options[self.metric_choice]
        
        # 6. Model Selection
        print("\nSelect Model Type:")
        model_options = {'1': 'Logistic Regression', '2': 'Random Forest', '3': 'Gradient Boosting (GBM)'}
        m_choice = get_user_choice(model_options)
        self.selections['model']['type'] = model_options[m_choice]
        model_map = {'1': 'logistic', '2': 'random_forest', '3': 'gbm'}
        self.model_choice = model_map[m_choice]
        
        input("\nPress Enter to compute baseline...")
        self.baseline_evaluation()

    def baseline_evaluation(self):
        print_header("Baseline Evaluation")
        print(f"Computing baseline fairness and performance metrics using {self.model_choice}...")
        
        # Use selected features if available, otherwise use all columns except target
        if self.selected_features:
            valid_feats = [f for f in self.selected_features if f in self.df.columns]
            X = self.df[valid_feats]
        else:
            X = self.df.drop(columns=[self.target_col])
            
        y = self.df[self.target_col]
        
        # Safety check: Drop non-numeric columns that were not encoded
        non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if non_numeric_cols:
            print(f"\nWarning: The following columns are non-numeric and were not encoded: {non_numeric_cols}")
            X = X.drop(columns=non_numeric_cols)

        # Safety check: Handle missing values if any remain
        if X.isnull().sum().sum() > 0:
            print("\nWarning: Input contains missing values (NaNs). Imputing...")
            X = impute_missing(X, strategy='mean')

        # PREVENT OVERFITTING: General Data Leakage Detection
        try:
            temp_check = X.copy()
            temp_check['__target__'] = y
            correlations = temp_check.corr()['__target__'].drop('__target__')
            leakage_cols = correlations[abs(correlations) > 0.95].index.tolist()
            if leakage_cols:
                print(f"\nWarning: High correlation detected (>0.95): {leakage_cols}. Dropping...")
                X = X.drop(columns=leakage_cols)
        except Exception as e:
            pass

        if X.empty:
             print("Error: No features remaining. Please check your dataset.")
             return
        
        # Train Model with 3-way split (60/20/20)
        model, X_val, y_val, X_test, y_test = train_classifier(X, y, model_type=self.model_choice)
        
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        self.y_test_bl = y_test
        self.y_pred_bl = y_test_pred
        
        # Performance Metrics
        val_perf = evaluate_classification(y_val, y_val_pred)
        test_perf = evaluate_classification(y_test, y_test_pred)
        
        for k, v in val_perf.items(): self.metrics_baseline[f"Val {k}"] = v
        for k, v in test_perf.items(): self.metrics_baseline[f"Test {k}"] = v
        
        # Compute Fairness on Test Set
        sens_test = self.df.loc[X_test.index, self.sensitive_col]
        temp_df = pd.DataFrame(X_test, columns=X.columns)
        temp_df[self.target_col] = y_test_pred
        temp_df[self.sensitive_col] = sens_test.values
        
        if self.m_type == '1': # Group Fairness
            if self.comparison_mode == 'combined':
                if self.metric_choice == '1': # Demographic Parity
                    f_metrics = compute_group_fairness(temp_df, self.target_col, self.sensitive_col, self.privileged_group, self.unprivileged_group)
                else: # Equalized Odds
                    true_df = temp_df.copy(); true_df[self.target_col] = y_test.values
                    f_metrics = compute_classification_fairness(true_df, temp_df, self.target_col, self.sensitive_col, self.privileged_group, self.unprivileged_group)
                self.metrics_baseline.update(f_metrics)
            else: # Individual Group Comparisons
                for val in self.unprivileged_group:
                    group_name = self.inverse_sensitive_mapping.get(val, str(val)) if self.inverse_sensitive_mapping else str(val)
                    if self.metric_choice == '1':
                        f_metrics = compute_group_fairness(temp_df, self.target_col, self.sensitive_col, self.privileged_group, val)
                    else:
                        true_df = temp_df.copy(); true_df[self.target_col] = y_test.values
                        f_metrics = compute_classification_fairness(true_df, temp_df, self.target_col, self.sensitive_col, self.privileged_group, val)
                    
                    prefixed = {f"[{group_name}] {k}": v for k, v in f_metrics.items()}
                    self.metrics_baseline.update(prefixed)
        else: # Individual Fairness
            f_metrics = compute_individual_fairness(temp_df, self.target_col)
            self.metrics_baseline.update(f_metrics)
             
        print("✔ Metrics computed")
        for k, v in self.metrics_baseline.items():
            print(f"  {k}: {v}")
            
        input("\nPress Enter to proceed to Mitigation...")
        self.mitigation_workflow()

    def mitigation_workflow(self):
        print_header("Bias Mitigation")
        options = {
            '1': 'Resampling (Pros: Simple, balances classes; Cons: Overfitting (oversample) or data loss (undersample))',
            '2': 'Relabeling (Pros: Modifies labels to be fairer; Cons: Changes ground truth, legally questionable in some contexts)',
            '3': 'Synthetic (Pros: Generates diverse data; Cons: Quality depends on generator, can hallucinate)',
            '4': 'Skip (Baseline)'
        }
        method = get_user_choice(options, "Select Mitigation Method")
        self.selections['mitigation']['method'] = options[method]
        
        self.metrics_mitigated = self.metrics_baseline.copy()
        
        if method == '4':
            self.df_improved = self.df.copy()
        else:
            if method == '1':
                res_options = {'1': 'Random Oversampling', '2': 'Undersampling'}
                resample_type = get_user_choice(res_options)
                self.selections['mitigation']['sub_method'] = res_options[resample_type]
                strategy = 'oversample' if resample_type == '1' else 'undersample'
                X_res, y_res = mitigate_resampling(self.df.drop(columns=[self.target_col]), self.df[self.target_col], self.sensitive_col, strategy=strategy)
            elif method == '2':
                X_res, y_res = mitigate_relabeling(self.df, self.target_col, self.sensitive_col, self.privileged_group, self.unprivileged_group, features_to_use=self.selected_features)
            elif method == '3':
                synth_options = {'1': 'SMOTE', '2': 'SDV'}
                synth_choice = get_user_choice(synth_options, "Select Synthetic Method")
                self.selections['mitigation']['sub_method'] = synth_options[synth_choice]
                synth_method = 'smote' if synth_choice == '1' else 'sdv'
                X_res, y_res = mitigate_synthetic(self.df, self.target_col, self.sensitive_col, method=synth_method)
                if X_res is None: return

            self.df_improved = pd.DataFrame(X_res)
            self.df_improved[self.target_col] = y_res.values if hasattr(y_res, 'values') else y_res
            
            print("Recomputing metrics...")
            X_model = X_res[self.selected_features] if self.selected_features else X_res
            X_model_numeric = X_model.select_dtypes(include=['number'])
            
            model, X_val, y_val, X_test, y_test = train_classifier(X_model_numeric, y_res, model_type=self.model_choice)
            
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            self.y_test_mit = y_test
            self.y_pred_mit = y_test_pred
            
            val_perf = evaluate_classification(y_val, y_val_pred)
            test_perf = evaluate_classification(y_test, y_test_pred)
            
            for k, v in val_perf.items(): self.metrics_mitigated[f"Val {k}"] = v
            for k, v in test_perf.items(): self.metrics_mitigated[f"Test {k}"] = v
            
            sens_test = self.df_improved.loc[X_test.index, self.sensitive_col]
            temp_df = pd.DataFrame(X_test, columns=X_model_numeric.columns)
            temp_df[self.target_col] = y_test_pred
            temp_df[self.sensitive_col] = sens_test.values
            
            if self.m_type == '1':
                if self.metric_choice == '1':
                    f_metrics = compute_group_fairness(temp_df, self.target_col, self.sensitive_col, self.privileged_group, self.unprivileged_group)
                else:
                    true_df = temp_df.copy(); true_df[self.target_col] = y_test.values
                    f_metrics = compute_classification_fairness(true_df, temp_df, self.target_col, self.sensitive_col, self.privileged_group, self.unprivileged_group)
            else:
                f_metrics = compute_individual_fairness(temp_df, self.target_col)
                
            self.metrics_mitigated.update(f_metrics)

        if self.df_improved is not None:
            self.stats_mitigated = get_comprehensive_stats(self.df_improved, self.target_col, self.sensitive_col, selected_features=self.selected_features)

        print("✔ Mitigation Eval complete")
        for k, v in self.metrics_mitigated.items():
            print(f"  {k}: {v}")

        input("\nPress Enter to generate reports...")
        self.output_workflow()

    def output_workflow(self):
        print_header("Output Generation")
        
        os.makedirs("outputs/datasets", exist_ok=True)
        os.makedirs("outputs/reports", exist_ok=True)
        
        out_path = os.path.join("outputs/datasets", "fairness_improved.csv")
        self.df_improved.to_csv(out_path, index=False)
        print(f"✔ Improved dataset saved to {out_path}")
        
        df_display_bl = self.df.copy()
        df_display_mit = self.df_improved.copy() if self.df_improved is not None else None
        
        if self.inverse_sensitive_mapping:
            df_display_bl[self.sensitive_col] = df_display_bl[self.sensitive_col].map(self.inverse_sensitive_mapping)
            if df_display_mit is not None:
                df_display_mit[self.sensitive_col] = df_display_mit[self.sensitive_col].map(self.inverse_sensitive_mapping)

        self.stats_baseline = get_comprehensive_stats(df_display_bl, self.target_col, self.sensitive_col, original_feat_count=self.original_df.shape[1], selected_features=self.selected_features)
        if df_display_mit is not None:
            self.stats_mitigated = get_comprehensive_stats(df_display_mit, self.target_col, self.sensitive_col, original_feat_count=self.original_df.shape[1], selected_features=self.selected_features)

        plots = {}
        if self.y_test_bl is not None:
            plots["Baseline Confusion Matrix"] = plot_confusion_matrix(self.y_test_bl, self.y_pred_bl, "Baseline Confusion Matrix", "cm_baseline.png")
        if self.y_test_mit is not None:
            plots["Mitigated Confusion Matrix"] = plot_confusion_matrix(self.y_test_mit, self.y_pred_mit, "Mitigated Confusion Matrix", "cm_mitigated.png")
            
        plots["Performance & Fairness Metrics Comparison"] = plot_metric_comparison(self.metrics_baseline, self.metrics_mitigated)
        
        if df_display_mit is not None:
            plots["Class Distribution Comparison"] = plot_distribution_comparison(df_display_bl, df_display_mit, self.target_col, "Class Distribution: Before vs After", "class_dist_comparison.png")
            plots["Sensitive Attribute Distribution Comparison"] = plot_distribution_comparison(df_display_bl, df_display_mit, self.sensitive_col, "Sensitive Attribute Distribution: Before vs After", "sensitive_dist_comparison.png")
            plots["Selection Rate Comparison P(Y=1|A)"] = plot_selection_rates(df_display_bl, df_display_mit, self.sensitive_col, self.target_col, "selection_rates_comparison.png")
            plots["Baseline Subgroup Heatmap"] = plot_contingency_heatmap(df_display_bl, self.sensitive_col, self.target_col, "Baseline: Subgroup Counts (A, Y)", "heatmap_baseline.png")
            plots["Mitigated Subgroup Heatmap"] = plot_contingency_heatmap(df_display_mit, self.sensitive_col, self.target_col, "Mitigated: Subgroup Counts (A, Y)", "heatmap_mitigated.png")

        generate_pdf_report("fairness_report.pdf", self.stats_baseline, self.metrics_baseline, self.metrics_mitigated, plots=plots, stats_after=self.stats_mitigated, selections=self.selections)
        
        print("\nAnalysis Complete!")
        input("Press Enter to return to main menu...")

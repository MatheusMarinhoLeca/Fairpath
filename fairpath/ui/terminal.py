import sys
from typing import Dict, Any, List, Optional, Callable
from fairpath.utils.menus import clear_screen, print_header, get_user_choice, get_user_confirmation, get_user_input


class TerminalUI:
    """Handles all terminal-based user interactions.
    
    This class provides a high-level API for presenting menus, gathering 
    validated user input, and displaying experiment results to the console.
    """
    
    def display_welcome(self):
        """Displays the welcome screen and project header."""
        clear_screen()
        print_header("FairPath Toolkit for Tabular Datasets")

    def get_main_menu_choice(self) -> str:
        """Presents the main navigation menu.

        Returns:
            str: The user's selection ('1', '2', '3', or '4').
        """
        options = {'1': 'Load dataset', '2': 'Run Automated Benchmark', '3': 'Visualize Benchmark Results', '4': 'Exit'}
        return get_user_choice(options, title="Main Menu")

    def get_benchmark_runs(self) -> int:
        """Gathers the number of iterations for the benchmark engine.

        Returns:
            int: The number of runs per configuration.
        """
        print_header("Automated Benchmark")
        val = get_user_input("Enter number of runs per configuration", lambda x: x.isdigit())
        return int(val)

    def get_benchmark_file_path(self, default_dir: str = "outputs/reports") -> str:
        """Provides a selection menu for existing benchmark result files.

        Args:
            default_dir: Directory to scan for benchmark CSV/Excel files.

        Returns:
            str: The absolute or relative path to the selected file.
        """
        print_header("Benchmark Visualization")
        import os
        
        # List available files
        files = []
        if os.path.exists(default_dir):
            files = [f for f in os.listdir(default_dir) if f.startswith("benchmark_results") and (f.endswith(".csv") or f.endswith(".xlsx"))]
            files.sort(reverse=True)
            
        if files:
            print(f"Found recent benchmark files in {default_dir}:")
            for i, f in enumerate(files[:5]):
                print(f"  {i+1}. {f}")
            print("  0. Enter custom path")
            
            choice = get_user_input("Select a file number or 0 for custom path", lambda x: x.isdigit() and 0 <= int(x) <= len(files[:5]))
            if choice != '0':
                return os.path.join(default_dir, files[int(choice)-1])
        
        return get_user_input("Enter path to benchmark results file (.csv or .xlsx)", lambda x: os.path.exists(x) and (x.endswith(".csv") or x.endswith(".xlsx") or x.endswith(".xls")))

    def get_dataset_path(self, validator: Callable) -> str:
        """Prompts for and validates the input dataset path.

        Args:
            validator: A callback function to verify file existence and format.

        Returns:
            str: Validated path to the dataset.
        """
        print_header("Dataset Loading")
        return get_user_input("Enter dataset path (.csv or .xlsx)", validator)

    def confirm_target_col(self, detected: str) -> bool:
        """Asks the user to confirm the auto-detected target column.

        Args:
            detected: The name of the column detected by the engine.

        Returns:
            bool: True if confirmed, False otherwise.
        """
        return get_user_confirmation(f"Is '{detected}' the target column?")

    def get_target_col(self, columns: List[str]) -> str:
        """Prompts the user to manually select the target column.

        Args:
            columns: List of available column names in the dataframe.

        Returns:
            str: The selected target column name.
        """
        return get_user_input("Enter target column name", lambda x: x in columns)

    def display_message(self, message: str):
        """Prints a general message to the console.

        Args:
            message: The text to display.
        """
        print(message)

    def wait_for_user(self, message: str = "\nPress Enter to continue..."):
        """Pauses execution until the user presses Enter.

        Args:
            message: Optional custom prompt message.
        """
        input(message)

    def get_feature_selection_method(self) -> str:
        """Prompts for the preferred feature selection strategy.

        Returns:
            str: Selection key ('1' or '2').
        """
        feat_options = {
            '1': 'Keep specific features (Manual Selection)',
            '2': 'Skip feature selection'
        }
        return get_user_choice(feat_options, title="Select Feature Selection Method")

    def get_columns_to_keep(self, all_columns: List[str]) -> List[str]:
        """Gathers list of features to retain for training.

        Args:
            all_columns: Total set of columns in the dataset.

        Returns:
            List[str]: Filtered list of feature names.
        """
        print(f"\nAvailable columns: {', '.join(all_columns)}")
        cols_input = get_user_input("Enter column names to KEEP (comma separated)", lambda x: True)
        return [c.strip() for c in cols_input.split(',') if c.strip() in all_columns]

    def get_columns_to_drop(self, all_columns: List[str]) -> List[str]:
        """Gathers list of features to exclude from training.

        Args:
            all_columns: Total set of columns in the dataset.

        Returns:
            List[str]: List of column names to remove.
        """
        print(f"\nAvailable columns: {', '.join(all_columns)}")
        cols_input = get_user_input("Enter column names to DROP (comma separated)", lambda x: True)
        return [c.strip() for c in cols_input.split(',') if c.strip() in all_columns]

    def ask_handle_missing_values(self) -> Optional[str]:
        """Prompts for missing value imputation strategy.

        Returns:
            Optional[str]: Selection key or None if skipped.
        """
        if get_user_confirmation("Do you want to handle missing values?"):
            strat_options = {
                '1': 'Mean/Median (Pros: Retains data size; Cons: Can distort distributions, ignores correlations)',
                '2': 'Mode (Pros: Works for categorical; Cons: Can bias towards majority class)',
                '3': 'Drop (Pros: Removes noise; Cons: Reduces data size, potential bias if missingness is systematic)'
            }
            return get_user_choice(strat_options, title="Select Missing Value Strategy")
        return None

    def ask_handle_outliers(self) -> Optional[str]:
        """Prompts for outlier removal strategy.

        Returns:
            Optional[str]: Selection key or None if skipped.
        """
        if get_user_confirmation("Do you want to handle outliers?"):
            strat_options = {
                '1': 'IQR-based removal (Pros: Removes extreme values that might drive bias; Cons: Might remove minority group outliers)',
                '2': 'Skip (Pros: Retains all data; Cons: Models might be sensitive to outliers)'
            }
            return get_user_choice(strat_options, title="Select Outlier Strategy")
        return None

    def ask_encode_categorical(self, cat_cols: List[str], selected_features: List[str]) -> Optional[Dict[str, Any]]:
        """Handles the configuration of categorical encoding.

        Args:
            cat_cols: All categorical columns found in the dataset.
            selected_features: Features selected for model training.

        Returns:
            Optional[Dict]: Dictionary with 'choice' (method) and 'columns' to encode.
        """
        if get_user_confirmation("Do you want to encode categorical variables?"):
            if not cat_cols:
                return {"choice": "none"}
            
            if selected_features:
                selected_cat = [c for c in cat_cols if c in selected_features]
                if selected_cat:
                    print(f"\nCRITICAL: The following SELECTED features are categorical and MUST be encoded to be used in the model: {', '.join(selected_cat)}")
                else:
                    print(f"\nNote: All selected features are already numeric. Other categorical columns in the dataset: {', '.join(cat_cols[:10])}...")
            else:
                print(f"\nDetected categorical columns: {', '.join(cat_cols[:20])}...")
            
            cols_input = get_user_input("Enter categorical column names to ENCODE (comma separated, leave empty for all)", lambda x: True, allow_empty=True)
            cols_to_encode = [c.strip() for c in cols_input.split(',') if c.strip() in cat_cols] if cols_input.strip() else cat_cols
            
            if not cols_to_encode:
                return {"choice": "none"}

            strat_options = {
                '1': 'One-hot encoding (Pros: No ordinal assumption, good for fairness; Cons: High dimensionality)',
                '2': 'Label encoding (Pros: Low dimensionality; Cons: Impose arbitrary order, bad for linear fairness)'
            }
            strat = get_user_choice(strat_options, title="Select Encoding Method")
            return {"choice": strat, "columns": cols_to_encode}
        return None

    def get_remaining_categorical_handling(self, remaining_cat: List[str]) -> str:
        """Asks how to handle categorical features that were not explicitly encoded.

        Args:
            remaining_cat: List of features that remain categorical.

        Returns:
            str: Selection key for automated handling.
        """
        print(f"\nNotice: The following selected features are still categorical: {remaining_cat}")
        print("To 'calculate all metrics', these must be numeric. How should they be handled?")
        rem_options = {
            '1': 'Label Encode automatically (Keeps column structure, converts to numbers)',
            '2': 'Drop from model training (Features will be ignored by the model)',
            '3': 'Keep as-is (Warning: They will be dropped automatically during model training)'
        }
        return get_user_choice(rem_options, title="Select Handling Method")

    def get_sensitive_attributes(self, columns: List[str], default: Optional[str] = None) -> str:
        """Prompts for one or more sensitive attribute columns.

        Args:
            columns: Available columns for selection.
            default: Recommended sensitive attribute.

        Returns:
            str: User input (comma-separated string of columns).
        """
        print_header("Fairness Setup")
        print("To analyze intersectional fairness (e.g., Race AND Gender), enter multiple columns separated by commas.")
        return get_user_input("Enter sensitive column name(s)", lambda x: True, default=default)

    def get_privileged_group(self, values: List[Any], mapping: Optional[Dict[Any, Any]] = None) -> str:
        """Prompts for the value representing the privileged group.

        Args:
            values: Unique values found in the sensitive column.
            mapping: Optional human-readable mapping for categorical attributes.

        Returns:
            str: The selected value or name.
        """
        if mapping:
            print(f"Values in sensitive column: {list(mapping.keys())}")
            return get_user_input("Enter value for privileged group (by name)")
        else:
            print(f"Values in sensitive column: {values}")
            return get_user_input("Enter value for privileged group")

    def get_unprivileged_comparison_mode(self) -> str:
        """Prompts for the unprivileged group comparison strategy.

        Returns:
            str: Selection key ('1' for combined, '2' for individual).
        """
        un_options = {
            '1': 'All other groups combined',
            '2': 'Against each group individually'
        }
        return get_user_choice(un_options, title="Select Unprivileged Group Comparison")

    def confirm_binary_transformation(self) -> bool:
        """Confirms the transformation of the sensitive attribute to binary.

        Returns:
            bool: True if confirmed.
        """
        print("\nOption: Transform dataset to binary sensitive attribute?")
        print("This will replace the sensitive column with 1 (Privileged) and 0 (Unprivileged).")
        return get_user_confirmation("Apply binary transformation?")

    def get_specific_fairness_metric(self) -> str:
        """Prompts for the primary fairness metric category to audit.

        Returns:
            str: Selection key ('1' or '2').
        """
        metric_options = {
            '1': 'Demographic Parity (Statistical Parity Difference, Disparate Impact)',
            '2': 'Equalized Odds (Equal Opportunity Difference, Average Odds Difference)'
        }
        return get_user_choice(metric_options, title="Select Fairness Metric")

    def get_model_choice(self) -> str:
        """Prompts for the machine learning model architecture.

        Returns:
            str: Selection key ('1' through '4').
        """
        model_options = {'1': 'Logistic Regression', '2': 'Random Forest', '3': 'Gradient Boosting (GBM)', '4': 'Linear Support Vector Machine (LinearSVC)'}
        return get_user_choice(model_options, title="Select Model Type")

    def get_mitigation_method(self) -> str:
        """Prompts for the bias mitigation strategy.

        Returns:
            str: Selection key ('1' through '4').
        """
        print_header("Bias Mitigation")
        options = {
            '1': 'Resampling',
            '2': 'Relabeling',
            '3': 'Synthetic',
            '4': 'Skip (Baseline)'
        }
        return get_user_choice(options, title="Select Mitigation Method")

    def confirm_action(self, message: str) -> bool:
        """Generic confirmation prompt.

        Args:
            message: Question to ask the user.

        Returns:
            bool: User response.
        """
        return get_user_confirmation(message)

    def get_resampling_type(self) -> str:
        """Prompts for the type of resampling to apply.

        Returns:
            str: Selection key ('1' or '2').
        """
        res_options = {'1': 'Random Oversampling', '2': 'Undersampling'}
        return get_user_choice(res_options, title="Select Resampling Type")

    def get_synthetic_method(self) -> str:
        """Prompts for the synthetic data generation technique.

        Returns:
            str: Selection key ('1' or '2').
        """
        synth_options = {'1': 'Synthetic Minority Over-sampling Technique (SMOTE)', '2': 'Counterfactual Data Augmentation (CDA)'}
        return get_user_choice(synth_options, title="Select Synthetic Method")

    def display_metrics(self, metrics: Dict[str, float], title: str = "Metrics"):
        """Renders experiment metrics in a readable list.

        Args:
            metrics: Dictionary of metric names and values.
            title: Section title.
        """
        print(f"\n--- {title} ---")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    def display_stats(self, stats: Dict[str, Any], title: str = "Statistics"):
        """Renders dataset statistics in a readable list.

        Args:
            stats: Dictionary of statistic names and values.
            title: Section title.
        """
        print(f"\n--- {title} ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    def exit(self):
        """Terminates the application."""
        sys.exit()

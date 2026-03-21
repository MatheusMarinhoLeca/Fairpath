import pandas as pd
import numpy as np
import os
import itertools
import time
import sys
import warnings
from typing import Dict, Any, List, Tuple, Optional

# Define custom warning handler to pause execution
def pause_on_warning(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write(f"\n\n!!! WARNING CAUGHT !!!\n{category.__name__}: {message}\nLocation: {filename}:{lineno}\n")
    # Print line content if available
    if line:
        sys.stderr.write(f"  {line.strip()}\n")
    
    try:
        input("\nExecution paused. Press Enter to continue...")
    except EOFError:
        pass # Handle non-interactive environments gracefully

# Apply warning settings
warnings.simplefilter('default')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress expected aif360 warnings about small subgroups during large sweeps
warnings.filterwarnings("ignore", category=RuntimeWarning, module="aif360")
# Suppress performance warnings about fragmentation (common with many-column datasets like Adult)
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings("ignore", category=PerformanceWarning)
except ImportError:
    pass

# Override the showwarning function
warnings.showwarning = pause_on_warning

# Use aif360.sklearn fetchers for automatic downloads
from aif360.sklearn.datasets import fetch_adult, fetch_compas, fetch_german
from preprocessing.missing_values import impute_missing
from preprocessing.outliers import remove_outliers_iqr
from preprocessing.encoding import one_hot_encode, label_encode, binarize_attribute
from models.classification import DefaultModelTrainer
from fairness.metrics import GroupFairnessMetric, ClassificationFairnessMetric
from fairness.mitigation import ResamplingMitigation, RelabelingMitigation, SyntheticMitigation
from data import create_composite_attribute

class BenchmarkEngine:
    def __init__(self, num_runs: int, debug_sample_limit: Optional[int] = None):
        self.num_runs = num_runs
        self.debug_sample_limit = debug_sample_limit
        self.results = []
        # Reverting to DefaultModelTrainer to ensure benchmark and user mode are identical
        self.trainer = DefaultModelTrainer()
        
    def get_datasets(self):
        """Returns a list of AIF360 datasets to benchmark using the automatic sklearn fetchers."""
        datasets = []
        
        # 1. Adult Dataset
        try:
            print("Fetching Adult Census Income dataset...")
            adult = fetch_adult()
            # The fetcher returns MultiIndex where names collide with columns. 
            # We use the columns and ignore the index for loading.
            df = adult.X.reset_index(drop=True)
            target_col = adult.y.name
            df[target_col] = adult.y.values
            # Standardize labels: >50K is 1 (favorable), <=50K is 0
            df[target_col] = df[target_col].apply(lambda x: 1 if '>50K' in str(x) else 0).astype(int)
            
            # Legacy AIF360 names for sensitive attributes
            sensitive_attrs = ['race', 'sex']
            privileged_groups = {'race': 'White', 'sex': 'Male'}
            
            datasets.append(('Adult Census Income', df, target_col, sensitive_attrs, privileged_groups))
        except Exception as e:
            print(f"Error loading Adult Census Income: {e}")
            raise e 
            
        # 2. COMPAS Dataset
        try:
            print("Fetching ProPublica COMPAS dataset...")
            compas = fetch_compas()
            df = compas.X.reset_index(drop=True)
            target_col = compas.y.name
            df[target_col] = compas.y.values
            # Standardize labels: Survived is 1 (favorable), Recidivated is 0
            df[target_col] = df[target_col].apply(lambda x: 1 if str(x).lower() == 'survived' else 0)
            
            sensitive_attrs = ['sex', 'race']
            privileged_groups = {'sex': 'Female', 'race': 'Caucasian'}
            
            datasets.append(('ProPublica COMPAS', df, target_col, sensitive_attrs, privileged_groups))
        except Exception as e:
            print(f"Error loading ProPublica COMPAS: {e}")
            raise e
            
        # 3. German Credit Dataset
        try:
            print("Fetching German Credit dataset...")
            german = fetch_german()
            df = german.X.reset_index(drop=True)
            target_col = german.y.name
            df[target_col] = german.y.values
            # Standardize labels: good is 1 (favorable), bad is 0
            df[target_col] = df[target_col].apply(lambda x: 1 if str(x).lower() == 'good' else 0)
            
            # AGE: German dataset 'age' is numeric. We need to categorize it if we want to use 'aged'.
            # Common benchmark split for German Credit: aged (>25) vs young (<=25)
            df['age_cat'] = df['age'].apply(lambda x: 'aged' if x > 25 else 'young')
            
            sensitive_attrs = ['sex', 'age_cat']
            privileged_groups = {'sex': 'male', 'age_cat': 'aged'}
            
            datasets.append(('German Credit', df, target_col, sensitive_attrs, privileged_groups))
        except Exception as e:
            print(f"Error loading German Credit: {e}")
            raise e
            
        return datasets

    def generate_configurations(self):
        """Generates all combinations of preprocessing, models, and mitigation."""
        missing_strategies = ['mode']
        outlier_strategies = ['iqr']
        encoding_strategies = ['one-hot']
        models = ['logistic', 'random_forest', 'gbm', 'linear_svc']
        mitigations = [
            ('none', None),
            ('resampling_over', 'oversample'),
            ('resampling_under', 'undersample'),
            ('relabeling', None),
            ('synthetic_smote', 'smote'),
            ('synthetic_cda', 'cda')
        ]
        unprivileged_strategies = ['combined']
        binary_trans = [True]

        configs = list(itertools.product(
            missing_strategies, 
            outlier_strategies, 
            encoding_strategies, 
            models, 
            mitigations, 
            unprivileged_strategies,
            binary_trans
        ))
        return configs

    def run(self):
        datasets_to_run = self.get_datasets()
        if not datasets_to_run:
            print("No datasets loaded. Benchmark cannot proceed.")
            return

        configs = self.generate_configurations()
        
        start_time = time.time()
        
        for ds_name, df_original, target_col, sensitive_attrs, privileged_groups_meta in datasets_to_run:
            # New requirement: Do each sensitive attribute individually, THEN combined
            analysis_targets = []
            # 1. Individual attributes
            for attr in sensitive_attrs:
                analysis_targets.append(([attr], {attr: privileged_groups_meta[attr]}))
            
            # 2. Combined attribute (only if there's more than one)
            if len(sensitive_attrs) > 1:
                analysis_targets.append((sensitive_attrs, privileged_groups_meta))

            # Apply debug sample limit if specified
            if self.debug_sample_limit and len(df_original) > self.debug_sample_limit:
                print(f"--- DEBUG MODE: Limiting {ds_name} to {self.debug_sample_limit} samples ---")
                df_original_sampled = df_original.sample(n=self.debug_sample_limit, random_state=42).reset_index(drop=True)
            else:
                df_original_sampled = df_original.copy()

            total_configs_for_ds = len(configs) * len(analysis_targets)
            current_ds_config_idx = 0

            for current_attrs, current_priv_meta in analysis_targets:
                attr_display_name = ", ".join(current_attrs)
                
                # Pre-calculate group info for display
                if len(current_attrs) == 1:
                    primary_sens = current_attrs[0]
                    priv_val = current_priv_meta[primary_sens]
                    all_vals = df_original_sampled[primary_sens].unique()
                    unpriv_vals = [str(v) for v in all_vals if str(v) != str(priv_val)]
                else:
                    # Combined case
                    priv_val = "_".join([str(current_priv_meta[attr]) for attr in current_attrs])
                    # For unprivileged display in combined mode, we can show a few unique combinations that aren't the privileged one
                    df_temp, composite_col = create_composite_attribute(df_original_sampled, current_attrs)
                    all_vals = df_temp[composite_col].unique()
                    unpriv_vals = [str(v) for v in all_vals if str(v) != str(priv_val)]
                
                unpriv_display = ", ".join(unpriv_vals[:3]) + ("..." if len(unpriv_vals) > 3 else "")

                for config_idx, config in enumerate(configs):
                    current_ds_config_idx += 1
                    (miss_strat, out_strat, enc_strat, model_name, 
                     mit_tuple, unpriv_strat, bin_trans) = config
                    
                    mit_name, mit_sub = mit_tuple
                    
                    for run_num in range(1, self.num_runs + 1):
                        self._print_progress(
                            f"{ds_name} ({attr_display_name})", 
                            current_ds_config_idx, total_configs_for_ds, 
                            model_name, mit_name, run_num,
                            miss_strat, out_strat, priv_val, unpriv_display
                        )
                        
                        try:
                            # Execute single run with specific attributes
                            res = self.execute_single_run(
                                df_original_sampled.copy(), target_col, current_attrs, 
                                current_priv_meta, config, run_num, ds_name
                            )
                            
                            # VALIDATION: Ensure all 4 required fairness metrics are present
                            required_metrics = [
                                "Statistical Parity Difference", 
                                "Disparate Impact", 
                                "Equal Opportunity Difference", 
                                "Average Odds Difference"
                            ]
                            missing_metrics = [m for m in required_metrics if m not in res or pd.isna(res[m])]
                            
                            if missing_metrics:
                                print(f"\n\n!!! CRITICAL ALERT: MISSING FAIRNESS METRICS !!!")
                                print(f"Run {run_num} failed to produce: {', '.join(missing_metrics)}")
                                print(f"Configuration: {config}")
                                input("\nExecution paused. Please check logs. Press Enter to continue anyway...")

                            self.results.append(res)
                        except Exception as e:
                            print(f"\n[!] Error in {ds_name} [{attr_display_name}] Config {config_idx + 1}: {e}")
                            error_res = {
                                'Dataset': ds_name,
                                'Sensitive Attributes': attr_display_name,
                                'Model Type': model_name,
                                'Mitigation Technique': mit_name,
                                'Run Number': run_num,
                                'Error': str(e)
                            }
                            self.results.append(error_res)
                            time.sleep(1)

        self.save_results()
        print(f"\nBenchmark completed in {time.time() - start_time:.2f} seconds.")

    def _print_progress(self, dataset, config_idx, total_configs, model, mitigation, run, miss_strat, out_strat, priv_group, unpriv_group):
        # Clear screen and print status
        sys.stdout.write("\033[H\033[J") 
        print(f"[Dataset: {dataset}]")
        if self.debug_sample_limit:
            print(f"!!! DEBUG MODE ACTIVE: Sample Limit = {self.debug_sample_limit} !!!")
        print(f"Configuration {config_idx} / {total_configs}")
        print(f"Model: {model}")
        print(f"Bias Mitigation: {mitigation}")
        print(f"Fairness Metric: Exhaustive (All Calculated)")
        print(f"Missing Values: {miss_strat} | Outliers: {out_strat}")
        print(f"Privileged Group: {priv_group} | Unprivileged: {unpriv_group}")
        print(f"Run: {run} / {self.num_runs}")
        print(f"\nStatus: {'Training...' if run > 0 else 'Initializing...'}")
        
        overall_progress = (config_idx / total_configs) * 100
        print(f"Overall Progress: {overall_progress:.2f}%")
        sys.stdout.flush()

    def execute_single_run(self, df, target_col, sensitive_attrs, privileged_groups_meta, config, run_num, ds_name):
        (miss_strat, out_strat, enc_strat, model_name, 
         mit_tuple, unpriv_strat, bin_trans) = config
        
        mit_name, mit_sub = mit_tuple
        
        # 1. Preprocessing: Missing Values
        if miss_strat != 'skip':
            df = impute_missing(df, miss_strat)
            
        # 2. Preprocessing: Outliers
        if out_strat == 'iqr':
            df = remove_outliers_iqr(df)
            
        # 3. Fairness Setup (BEFORE Encoding to preserve original labels)
        df, composite_sens_col = create_composite_attribute(df, sensitive_attrs)
        
        if len(sensitive_attrs) == 1:
            privileged_val_raw = privileged_groups_meta[sensitive_attrs[0]]
        else:
            privileged_val_raw = "_".join([str(privileged_groups_meta[attr]) for attr in sensitive_attrs])
            
        # Create a dedicated column for fairness evaluation that will NOT be encoded
        fairness_eval_col = "_fairness_eval_sens_attr"
        df[fairness_eval_col] = df[composite_sens_col].copy()
        
        if bin_trans:
            df = binarize_attribute(df, composite_sens_col, privileged_val_raw)
            # Also binarize the evaluation column for consistency
            df = binarize_attribute(df, fairness_eval_col, privileged_val_raw)
            eval_priv_val = 1
            eval_unpriv_vals = [0]
            # Use the raw descriptive name for the report display
            display_priv_val = privileged_val_raw
        else:
            eval_priv_val = privileged_val_raw
            eval_unpriv_vals = [x for x in df[fairness_eval_col].unique() if str(x) != str(eval_priv_val)]
            display_priv_val = privileged_val_raw

        # 4. Preprocessing: Encoding
        # Exclude fairness_eval_col from encoding candidates
        cat_cols = list(df.select_dtypes(exclude=['number']).columns)
        if target_col in cat_cols: cat_cols.remove(target_col)
        if fairness_eval_col in cat_cols: cat_cols.remove(fairness_eval_col)
        
        if enc_strat == 'one-hot' and cat_cols:
            df = one_hot_encode(df, target_col, columns=cat_cols)
        elif enc_strat == 'label' and cat_cols:
            df = label_encode(df, target_col, columns=cat_cols)
            
        if df[target_col].dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            df[target_col] = LabelEncoder().fit_transform(df[target_col].astype(str))

        # 5. Bias Mitigation
        # Mitigation should use the composite_sens_col (which might be encoded/binarized already)
        if mit_name == 'resampling_over' or mit_name == 'resampling_under':
            mitigator = ResamplingMitigation(mit_sub)
            df = mitigator.mitigate(df, target_col, composite_sens_col, 
                                   1 if bin_trans else privileged_val_raw, 
                                   [0] if bin_trans else eval_unpriv_vals)
        elif mit_name == 'relabeling':
            mitigator = RelabelingMitigation()
            df = mitigator.mitigate(df, target_col, composite_sens_col, 
                                   1 if bin_trans else privileged_val_raw, 
                                   [0] if bin_trans else eval_unpriv_vals)
        elif mit_name.startswith('synthetic'):
            mitigator = SyntheticMitigation(mit_sub)
            df = mitigator.mitigate(df, target_col, composite_sens_col, 
                                   1 if bin_trans else privileged_val_raw, 
                                   [0] if bin_trans else eval_unpriv_vals)
        
        # 6. Model Training & Evaluation
        # Drop target and fairness_eval_col from features
        X = df.drop(columns=[target_col, fairness_eval_col])
        y = df[target_col]
        
        # Ensure all columns in X are numeric
        cat_cols_remaining = list(X.select_dtypes(exclude=['number']).columns)
        if cat_cols_remaining:
             X = one_hot_encode(X, None, columns=cat_cols_remaining)
             
        X_num = X.select_dtypes(include=[np.number])
        if X_num.isnull().sum().sum() > 0:
            X_num = X_num.fillna(X_num.mean())
            
        from sklearn.model_selection import train_test_split
        _, _, _, _, _, test_idx = train_test_split(
            X_num, y, df.index, test_size=0.2, random_state=42
        )
        
        model, X_val, y_val, X_test_scaled, y_test, _, _ = self.trainer.train(X_num, y, model_name)
        
        y_pred = model.predict(X_test_scaled)
        perf_metrics = self.trainer.evaluate(model, X_test_scaled, y_test)
        
        # Fairness Metrics (using fairness_eval_col)
        sens_test = df.loc[test_idx, fairness_eval_col]
        
        df_pred = pd.DataFrame(X_test_scaled, columns=X_num.columns)
        df_pred[target_col] = y_pred
        df_pred[fairness_eval_col] = sens_test.values
        
        df_true = pd.DataFrame({target_col: y_test.values, fairness_eval_col: sens_test.values})
        
        # Compute BOTH metric types
        group_metric_impl = GroupFairnessMetric()
        class_metric_impl = ClassificationFairnessMetric(df_true)
            
        fair_metrics = {}
        if unpriv_strat == 'combined':
            res_group = group_metric_impl.compute(df_pred, target_col, fairness_eval_col, eval_priv_val, eval_unpriv_vals)
            fair_metrics.update(res_group)
            res_class = class_metric_impl.compute(df_pred, target_col, fairness_eval_col, eval_priv_val, eval_unpriv_vals)
            fair_metrics.update(res_class)

        # 7. Result Collection
        result = {
            'Dataset': ds_name,
            'Missing Values Strategy': miss_strat,
            'Outlier Strategy': out_strat,
            'Encoding Strategy': enc_strat,
            'Model Type': model_name,
            'Mitigation Technique': mit_name,
            'Mitigation Detail': mit_sub,
            'Fairness Metric Goal': 'Exhaustive (All)',
            'Unprivileged Comparison': unpriv_strat,
            'Binary Sensitive Attr': bin_trans,
            'Run Number': run_num,
            'Sensitive Attributes': ", ".join(sensitive_attrs),
            'Chosen Sensitive Attribute': composite_sens_col,
            'Privileged Group Value': display_priv_val
        }
        result.update(perf_metrics)
        result.update(fair_metrics)
        
        return result

    def save_results(self):
        os.makedirs("outputs/reports", exist_ok=True)
        filename = f"benchmark_results_{int(time.time())}.csv"
        filepath = os.path.join("outputs/reports", filename)
        
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(filepath, index=False)
        
        try:
            excel_path = filepath.replace(".csv", ".xlsx")
            df_results.to_excel(excel_path, index=False)
            print(f"Results saved to {filepath} and {excel_path}")
        except Exception as e:
            print(f"Results saved to {filepath} (Excel export failed: {e})")

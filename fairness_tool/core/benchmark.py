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
            
            sensitive_attrs = ['sex', 'age'] # We pick the two primary ones
            privileged_groups = {'sex': 'male', 'age': 'aged'}
            
            datasets.append(('German Credit', df, target_col, sensitive_attrs, privileged_groups))
        except Exception as e:
            print(f"Error loading German Credit: {e}")
            raise e
            
        return datasets

    def generate_configurations(self):
        """Generates all combinations of preprocessing, models, and mitigation."""
        missing_strategies = ['skip', 'mean', 'mode', 'drop']
        outlier_strategies = ['skip', 'iqr']
        encoding_strategies = ['one-hot', 'label']
        models = ['logistic', 'random_forest', 'gbm', 'svm']
        mitigations = [
            ('none', None),
            ('resampling_over', 'oversample'),
            ('resampling_under', 'undersample'),
            ('relabeling', None),
            ('synthetic_smote', 'smote'),
            ('synthetic_cda', 'cda')
        ]
        metrics = ['1', '2'] # 1: Demographic Parity, 2: Equalized Odds
        unprivileged_strategies = ['combined', 'individual']
        binary_trans = [True, False]

        configs = list(itertools.product(
            missing_strategies, 
            outlier_strategies, 
            encoding_strategies, 
            models, 
            mitigations, 
            metrics,
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
        total_configs = len(configs)
        
        start_time = time.time()
        
        for ds_name, df_original, target_col, sensitive_attrs, privileged_groups_meta in datasets_to_run:
            # Apply debug sample limit if specified
            if self.debug_sample_limit and len(df_original) > self.debug_sample_limit:
                print(f"--- DEBUG MODE: Limiting {ds_name} to {self.debug_sample_limit} samples ---")
                df_original = df_original.sample(n=self.debug_sample_limit, random_state=42).reset_index(drop=True)

            for config_idx, config in enumerate(configs):
                (miss_strat, out_strat, enc_strat, model_name, 
                 mit_tuple, metric_choice, unpriv_strat, bin_trans) = config
                
                mit_name, mit_sub = mit_tuple
                
                # Preliminary Progress reporting
                self._print_progress(
                    ds_name, config_idx + 1, total_configs, 
                    model_name, mit_name, metric_choice, 0
                )
                
                for run_num in range(1, self.num_runs + 1):
                    self._print_progress(
                        ds_name, config_idx + 1, total_configs, 
                        model_name, mit_name, metric_choice, run_num
                    )
                    
                    try:
                        # Execute single run
                        res = self.execute_single_run(
                            df_original.copy(), target_col, sensitive_attrs, 
                            privileged_groups_meta, config, run_num, ds_name
                        )
                        self.results.append(res)
                    except Exception as e:
                        print(f"\n[!] Error in Configuration {config_idx + 1}: {e}")
                        # Optionally record the failure in results
                        error_res = {
                            'Dataset': ds_name,
                            'Model Type': model_name,
                            'Mitigation Technique': mit_name,
                            'Run Number': run_num,
                            'Error': str(e)
                        }
                        self.results.append(error_res)
                        # Small pause to allow user to see error before progress screen clears
                        time.sleep(1)

        self.save_results()
        print(f"\nBenchmark completed in {time.time() - start_time:.2f} seconds.")

    def _print_progress(self, dataset, config_idx, total_configs, model, mitigation, metric, run):
        metric_label = 'Demographic Parity' if metric == '1' else 'Equalized Odds'
        # Clear screen and print status
        sys.stdout.write("\033[H\033[J") 
        print(f"[Dataset: {dataset}]")
        if self.debug_sample_limit:
            print(f"!!! DEBUG MODE ACTIVE: Sample Limit = {self.debug_sample_limit} !!!")
        print(f"Configuration {config_idx} / {total_configs}")
        print(f"Model: {model}")
        print(f"Bias Mitigation: {mitigation}")
        print(f"Fairness Metric: {metric_label}")
        print(f"Run: {run} / {self.num_runs}")
        print(f"\nStatus: {'Training...' if run > 0 else 'Initializing...'}")
        
        overall_progress = (config_idx / total_configs) * 100
        print(f"Overall Progress: {overall_progress:.2f}%")
        sys.stdout.flush()

    def execute_single_run(self, df, target_col, sensitive_attrs, privileged_groups_meta, config, run_num, ds_name):
        (miss_strat, out_strat, enc_strat, model_name, 
         mit_tuple, metric_choice, unpriv_strat, bin_trans) = config
        
        mit_name, mit_sub = mit_tuple
        
        # 1. Preprocessing: Missing Values
        if miss_strat != 'skip':
            df = impute_missing(df, miss_strat)
            
        # 2. Preprocessing: Outliers
        if out_strat == 'iqr':
            df = remove_outliers_iqr(df)
            
        # 3. Preprocessing: Encoding
        cat_cols = list(df.select_dtypes(exclude=['number']).columns)
        if target_col in cat_cols: cat_cols.remove(target_col)
        
        if enc_strat == 'one-hot' and cat_cols:
            df = one_hot_encode(df, target_col, columns=cat_cols)
        elif enc_strat == 'label' and cat_cols:
            df = label_encode(df, target_col, columns=cat_cols)
            
        if df[target_col].dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            df[target_col] = LabelEncoder().fit_transform(df[target_col].astype(str))

        # 4. Fairness Setup
        df, composite_sens_col = create_composite_attribute(df, sensitive_attrs)
        
        if len(sensitive_attrs) == 1:
            privileged_val = privileged_groups_meta[sensitive_attrs[0]]
        else:
            privileged_val = "_".join([str(privileged_groups_meta[attr]) for attr in sensitive_attrs])
        
        if bin_trans:
            df = binarize_attribute(df, composite_sens_col, privileged_val)
            privileged_val = 1
            unprivileged_vals = [0]
        else:
            unprivileged_vals = [x for x in df[composite_sens_col].unique() if str(x) != str(privileged_val)]

        # 5. Bias Mitigation
        if mit_name == 'resampling_over' or mit_name == 'resampling_under':
            mitigator = ResamplingMitigation(mit_sub)
            df = mitigator.mitigate(df, target_col, composite_sens_col, privileged_val, unprivileged_vals)
        elif mit_name == 'relabeling':
            mitigator = RelabelingMitigation()
            df = mitigator.mitigate(df, target_col, composite_sens_col, privileged_val, unprivileged_vals)
        elif mit_name.startswith('synthetic'):
            mitigator = SyntheticMitigation(mit_sub)
            df = mitigator.mitigate(df, target_col, composite_sens_col, privileged_val, unprivileged_vals)
        
        # 6. Model Training & Evaluation
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Ensure all columns in X are numeric via preprocessing if not already
        cat_cols = list(X.select_dtypes(exclude=['number']).columns)
        if cat_cols:
             X = one_hot_encode(X, None, columns=cat_cols)
             
        X_num = X.select_dtypes(include=[np.number])
        if X_num.isnull().sum().sum() > 0:
            X_num = X_num.fillna(X_num.mean())
            
        # The trainer splits data inside: X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # We need to recreate this split to get the correct indices for aligning predictions with sensitive attributes.
        from sklearn.model_selection import train_test_split
        # We only need test_idx from this split to align sensitive attributes
        _, _, _, _, _, test_idx = train_test_split(
            X_num, y, df.index, test_size=0.2, random_state=42
        )
        
        # Use the trainer for training, but pass X_num directly as requested for consistency
        # Capture the scaled X_test returned by the trainer
        model, X_val, y_val, X_test_scaled, y_test, _, _ = self.trainer.train(X_num, y, model_name)
        
        # We must use the model on the X_test we split here (but using the scaled version from trainer)
        y_pred = model.predict(X_test_scaled)
        perf_metrics = self.trainer.evaluate(model, X_test_scaled, y_test)
        
        # Fairness Metrics
        # Align sens_test with the test_idx
        sens_test = df.loc[test_idx, composite_sens_col]
        
        # Construct df_pred using scaled features (as numpy array) but with original column names for structure
        # Note: The values are scaled, but metrics only care about target and sensitive columns
        df_pred = pd.DataFrame(X_test_scaled, columns=X_num.columns)
        df_pred[target_col] = y_pred
        df_pred[composite_sens_col] = sens_test.values
        
        df_true = pd.DataFrame({target_col: y_test.values, composite_sens_col: sens_test.values})
        
        if metric_choice == '1':
            metric_impl = GroupFairnessMetric()
        else:
            metric_impl = ClassificationFairnessMetric(df_true)
            
        fair_metrics = {}
        if unpriv_strat == 'combined':
            res = metric_impl.compute(df_pred, target_col, composite_sens_col, privileged_val, unprivileged_vals)
            fair_metrics.update(res)
        else:
            for uv in unprivileged_vals:
                group_label = str(uv)
                res = metric_impl.compute(df_pred, target_col, composite_sens_col, privileged_val, [uv])
                for k, v in res.items():
                    fair_metrics[f"[{group_label}] {k}"] = v
            
            if unprivileged_vals:
                for k in res.keys():
                    fair_metrics[f"Mean {k}"] = np.mean([fair_metrics[f"[{str(uv)}] {k}"] for uv in unprivileged_vals])

        # 7. Result Collection
        result = {
            'Dataset': ds_name,
            'Missing Values Strategy': miss_strat,
            'Outlier Strategy': out_strat,
            'Encoding Strategy': enc_strat,
            'Model Type': model_name,
            'Mitigation Technique': mit_name,
            'Mitigation Detail': mit_sub,
            'Fairness Metric Goal': 'Demographic Parity' if metric_choice == '1' else 'Equalized Odds',
            'Unprivileged Comparison': unpriv_strat,
            'Binary Sensitive Attr': bin_trans,
            'Run Number': run_num,
            'Sensitive Attributes': ", ".join(sensitive_attrs)
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

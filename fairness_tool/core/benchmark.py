import pandas as pd
import numpy as np
import os
import itertools
import time
import sys
import warnings
import contextlib
from typing import Dict, Any, List, Tuple, Optional
from joblib import Parallel, delayed
from tqdm import tqdm

# Define custom warning handler to pause execution
def pause_on_warning(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write(f"\n\n!!! WARNING CAUGHT !!!\n{category.__name__}: {message}\nLocation: {filename}:{lineno}\n")
    if line:
        sys.stderr.write(f"  {line.strip()}\n")
    
    try:
        input("\nExecution paused. Press Enter to continue...")
    except EOFError:
        pass

# Apply warning settings
warnings.simplefilter('default')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="aif360")
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings("ignore", category=PerformanceWarning)
except ImportError:
    pass

warnings.showwarning = pause_on_warning

from aif360.sklearn.datasets import fetch_adult, fetch_compas, fetch_german
from preprocessing.missing_values import impute_missing
from preprocessing.outliers import remove_outliers_iqr
from preprocessing.encoding import one_hot_encode, label_encode, binarize_attribute
from models.classification import DefaultModelTrainer
from core.experiment import ExperimentRunner
from fairness.mitigation import ResamplingMitigation, RelabelingMitigation, SyntheticMitigation
from data import create_composite_attribute
from data.validator import check_duplicates

class BenchmarkEngine:
    def __init__(self, num_runs: int, debug_sample_limit: Optional[int] = None):
        self.num_runs = num_runs
        self.debug_sample_limit = debug_sample_limit
        self.results = []
        self.trainer = DefaultModelTrainer()
        self.experiment_runner = ExperimentRunner(self.trainer)
        
    def get_datasets(self):
        """Returns a list of AIF360 datasets to benchmark using the automatic sklearn fetchers."""
        datasets = []
        
        # 1. Adult Dataset
        try:
            print("Fetching Adult Census Income dataset...")
            adult = fetch_adult()
            df = adult.X.reset_index(drop=True)
            target_col = adult.y.name
            df[target_col] = adult.y.values
            df[target_col] = df[target_col].apply(lambda x: 1 if '>50K' in str(x) else 0).astype(int)
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
            df[target_col] = df[target_col].apply(lambda x: 1 if str(x).lower() == 'good' else 0)
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
        
        all_tasks = []
        for ds_name, df_original, target_col, sensitive_attrs, privileged_groups_meta in datasets_to_run:
            has_dups, dup_count = check_duplicates(df_original)
            if has_dups:
                print(f"Warning: Dataset '{ds_name}' contains {dup_count} duplicate rows.")

            analysis_targets = []
            for attr in sensitive_attrs:
                analysis_targets.append(([attr], {attr: privileged_groups_meta[attr]}))
            if len(sensitive_attrs) > 1:
                analysis_targets.append((sensitive_attrs, privileged_groups_meta))

            if self.debug_sample_limit and len(df_original) > self.debug_sample_limit:
                df_original_sampled = df_original.sample(n=self.debug_sample_limit, random_state=42).reset_index(drop=True)
            else:
                df_original_sampled = df_original.copy()

            for current_attrs, current_priv_meta in analysis_targets:
                for config in configs:
                    for run_num in range(1, self.num_runs + 1):
                        all_tasks.append((df_original_sampled, target_col, current_attrs, current_priv_meta, config, run_num, ds_name))

        print(f"\nStarting benchmark with {len(all_tasks)} total runs in parallel...")
        
        # Execute tasks in parallel using joblib
        results_list = Parallel(n_jobs=-1)(
            delayed(self.execute_single_run)(*task) for task in tqdm(all_tasks, desc="Benchmark Progress")
        )
        
        self.results = results_list
        self.save_results()
        print(f"\nBenchmark completed in {time.time() - start_time:.2f} seconds.")

    def execute_single_run(self, df, target_col, sensitive_attrs, privileged_groups_meta, config, run_num, ds_name):
        # Silence all output from worker threads to maintain a clean UI
        with contextlib.redirect_stdout(None):
            (miss_strat, out_strat, enc_strat, model_name, 
             mit_tuple, unpriv_strat, bin_trans) = config
            
            mit_name, mit_sub = mit_tuple
            
            # 1. Preprocessing: Missing Values
            if miss_strat != 'skip':
                df = impute_missing(df, miss_strat)
                
            # 2. Preprocessing: Outliers
            if out_strat == 'iqr':
                # OutlierHandler now internally skips binary/low-cardinality columns
                df = remove_outliers_iqr(df)
                
            # 3. Fairness Setup (BEFORE Encoding to preserve original labels)
            df, composite_sens_col = create_composite_attribute(df, sensitive_attrs)
            
            if len(sensitive_attrs) == 1:
                privileged_val_raw = privileged_groups_meta[sensitive_attrs[0]]
            else:
                privileged_val_raw = "_".join([str(privileged_groups_meta[attr]) for attr in sensitive_attrs])
                
            fairness_eval_col = "_fairness_eval_sens_attr"
            df[fairness_eval_col] = df[composite_sens_col].copy()
            
            if bin_trans:
                df = binarize_attribute(df, composite_sens_col, privileged_val_raw)
                df = binarize_attribute(df, fairness_eval_col, privileged_val_raw)
                eval_priv_val = 1
                eval_unpriv_vals = [0]
                display_priv_val = privileged_val_raw
            else:
                eval_priv_val = privileged_val_raw
                eval_unpriv_vals = [x for x in df[fairness_eval_col].unique() if str(x) != str(eval_priv_val)]
                display_priv_val = privileged_val_raw

            # 4. Preprocessing: Encoding
            cat_cols = list(df.select_dtypes(exclude=['number']).columns)
            if target_col in cat_cols: cat_cols.remove(target_col)
            if fairness_eval_col in cat_cols: cat_cols.remove(fairness_eval_col)
            if composite_sens_col in cat_cols: cat_cols.remove(composite_sens_col)
            
            if enc_strat == 'one-hot' and cat_cols:
                df = one_hot_encode(df, target_col, columns=cat_cols)
            elif enc_strat == 'label' and cat_cols:
                df = label_encode(df, target_col, columns=cat_cols)
                
            if df[target_col].dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                df[target_col] = LabelEncoder().fit_transform(df[target_col].astype(str))

            # 5. Split BEFORE Mitigation
            from sklearn.model_selection import train_test_split
            try:
                df_train, df_test = train_test_split(df, test_size=0.2, random_state=42 + run_num, stratify=df[target_col])
            except ValueError:
                df_train, df_test = train_test_split(df, test_size=0.2, random_state=42 + run_num)

            # Check for insufficient classes
            if df_train[target_col].nunique() < 2:
                return {
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
                    'Privileged Group Value': display_priv_val,
                    'CV Mean Accuracy': np.nan,
                    'CV Std Accuracy': np.nan,
                    'Error': "Insufficient classes in training set"
                }

            # 6. Bias Mitigation (TRAIN ONLY)
            mitigator = None
            if mit_name == 'resampling_over' or mit_name == 'resampling_under':
                mitigator = ResamplingMitigation(mit_sub)
            elif mit_name == 'relabeling':
                mitigator = RelabelingMitigation()
            elif mit_name.startswith('synthetic'):
                mitigator = SyntheticMitigation(mit_sub)
            
            # 7. Execute Core Pipeline via ExperimentRunner
            runner_res = self.experiment_runner.run(
                df_train, df_test, 
                target_col, fairness_eval_col, 
                eval_priv_val, eval_unpriv_vals,
                model_name,
                mitigation_strategy=mitigator,
                metric_choice='all'
            )

            # 8. Result Collection
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
                'Privileged Group Value': display_priv_val,
                'CV Mean Accuracy': runner_res['train_metrics'].get('CV Mean Accuracy', 0),
                'CV Std Accuracy': runner_res['train_metrics'].get('CV Std Accuracy', 0)
            }
            result.update(runner_res['test_metrics'])
            result.update(runner_res['fairness_metrics'])
            
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

import pandas as pd
import os
import itertools
import time
import contextlib
from typing import Dict, Any, List, Tuple, Optional
from joblib import Parallel, delayed
from tqdm import tqdm

from aif360.sklearn.datasets import fetch_adult, fetch_compas, fetch_german
from fairpath.models.classification import DefaultModelTrainer
from fairpath.core.experiment import ExperimentRunner
from fairpath.fairness.mitigation import RelabelingMitigation, SyntheticMitigation

from fairpath.utils.warnings_config import configure_warnings
configure_warnings()

from fairpath.core.preprocessing_service import PreprocessingService
from fairpath.core.models import PreprocessingConfig

from fairpath.core.metadata import DATASET_REGISTRY

class BenchmarkEngine:
    """Automates large-scale fairness audits across multiple datasets and models.
    
    This engine executes a grid search over preprocessing strategies, machine 
    learning models, and bias mitigation techniques to identify optimal 
    fairness-utility trade-offs.
    """

    def __init__(self, num_runs: int, debug_sample_limit: Optional[int] = None):
        """Initializes the benchmark engine.

        Args:
            num_runs: Number of iterations per configuration (to account for variance).
            debug_sample_limit: Optional limit on dataset size for faster debugging.
        """
        self.num_runs = num_runs
        self.debug_sample_limit = debug_sample_limit
        self.results = []
        self.trainer = DefaultModelTrainer()
        self.experiment_runner = ExperimentRunner(self.trainer)
        self.preprocessing_service = PreprocessingService()
        
    def get_benchmark_datasets(self) -> List[Tuple[str, pd.DataFrame, str, List[str], Dict[str, Any]]]:
        """Fetches standard fairness datasets (Adult, COMPAS, German Credit).

        Returns:
            A list of tuples containing:
                - Dataset name.
                - Loaded DataFrame.
                - Target column name.
                - List of sensitive attributes.
                - Dictionary of privileged group metadata.
        """
        datasets = []
        
        # 1. Adult Census Income
        try:
            print("Fetching Adult Census Income dataset...")
            adult = fetch_adult()
            meta = DATASET_REGISTRY['Adult Census Income']
            df_adult = adult.X.reset_index(drop=True)
            df_adult[meta.target_col] = adult.y.values
            
            # Map labels to 0/1 based on metadata (favorable outcome = 1)
            df_adult[meta.target_col] = df_adult[meta.target_col].apply(
                lambda x: 1 if str(x).lower() == str(meta.positive_label).lower() else 0
            ).astype(int)
            
            datasets.append((meta.name, df_adult, meta.target_col, meta.sensitive_cols, meta.privileged_values))
        except Exception as e:
            print(f"Error loading Adult: {e}")
            
        # 2. ProPublica COMPAS
        try:
            print("Fetching ProPublica COMPAS dataset...")
            compas = fetch_compas()
            meta = DATASET_REGISTRY['ProPublica COMPAS']
            df_compas = compas.X.reset_index(drop=True)
            df_compas[meta.target_col] = compas.y.values
            
            # Map labels to 0/1 based on metadata (favorable outcome = 1)
            df_compas[meta.target_col] = df_compas[meta.target_col].apply(
                lambda x: 1 if str(x).lower() == str(meta.positive_label).lower() else 0
            ).astype(int)
            
            datasets.append((meta.name, df_compas, meta.target_col, meta.sensitive_cols, meta.privileged_values))
        except Exception as e:
            print(f"Error loading COMPAS: {e}")
            
        # 3. German Credit
        try:
            print("Fetching German Credit dataset...")
            german = fetch_german()
            meta = DATASET_REGISTRY['German Credit']
            df_german = german.X.reset_index(drop=True)
            df_german[meta.target_col] = german.y.values
            
            # Map labels to 0/1 based on metadata (favorable outcome = 1)
            df_german[meta.target_col] = df_german[meta.target_col].apply(
                lambda x: 1 if str(x).lower() == str(meta.positive_label).lower() else 0
            ).astype(int)
            
            if 'age' in df_german.columns:
                df_german['age_cat'] = df_german['age'].apply(lambda x: 'aged' if x > 25 else 'young')
                
            datasets.append((meta.name, df_german, meta.target_col, meta.sensitive_cols, meta.privileged_values))
        except Exception as e:
            print(f"Error loading German Credit: {e}")
            
        return datasets

    def _generate_grid_configs(self) -> List[Tuple]:
        """Generates the full cartesian product of experiment parameters."""
        models = ['logistic', 'random_forest', 'gbm', 'linear_svc']
        mitigations = [
            ('none', None),
            ('resampling_over', 'oversample'),
            ('resampling_under', 'undersample'),
            ('relabeling', None),
            ('synthetic_smote', 'smote'),
            ('synthetic_cda', 'cda')
        ]
        
        # Grid simplified for standard benchmarking
        configs = list(itertools.product(
            ['mode'],            # Missing value strategy
            ['iqr'],             # Outlier strategy
            ['one-hot'],         # Encoding strategy
            models, 
            mitigations, 
            ['combined'],        # Unprivileged comparison mode
            [True]               # Binary sensitive transformation
        ))
        return configs

    def run(self):
        """Executes the parallelized benchmark suite."""
        benchmark_datasets = self.get_benchmark_datasets()
        if not benchmark_datasets:
            print("No datasets loaded. Benchmark aborted.")
            return

        experiment_configs = self._generate_grid_configs()
        start_time = time.time()
        
        tasks = []
        for name, df_raw, target, sens_attrs, priv_meta in benchmark_datasets:
            # Sampling for debug/speed if requested
            df_to_use = df_raw.sample(n=self.debug_sample_limit, random_state=42).reset_index(drop=True) if self.debug_sample_limit else df_raw

            for attrs in sens_attrs: # Individual attributes
                for config in experiment_configs:
                    for run_idx in range(1, self.num_runs + 1):
                        tasks.append((df_to_use, target, [attrs], {attrs: priv_meta[attrs]}, config, run_idx, name))
            
            if len(sens_attrs) > 1: # Intersectional attribute
                for config in experiment_configs:
                    for run_idx in range(1, self.num_runs + 1):
                        tasks.append((df_to_use, target, sens_attrs, priv_meta, config, run_idx, name))

        print(f"\nStarting benchmark: {len(tasks)} runs in parallel...")
        
        # Parallel execution via joblib
        results_raw = Parallel(n_jobs=-1)(
            delayed(self._execute_single_run)(*t) for t in tqdm(tasks, desc="Auditing Progress")
        )
        
        self.results = [r for r in results_raw if r is not None]
        self._save_results_to_disk()
        print(f"\nBenchmark completed in {time.time() - start_time:.2f} seconds.")

    def _execute_single_run(self, df_source: pd.DataFrame, target_col: str, 
                           sensitive_attrs: List[str], privileged_meta: Dict[str, Any], 
                           config_tuple: Tuple, run_num: int, dataset_name: str) -> Dict[str, Any]:
        """Executes one specific configuration in a sandboxed output environment."""
        with contextlib.redirect_stdout(None):
            (miss_strat, out_strat, enc_strat, model_name, 
             mit_info, unpriv_strat, bin_trans) = config_tuple
            
            mit_name, mit_param = mit_info
            
            # 1. Configuration Setup
            prep_config = PreprocessingConfig(
                missing_strategy=miss_strat,
                outlier_strategy=out_strat,
                encoding_strategy=enc_strat
            )
            
            if len(sensitive_attrs) == 1:
                privileged_val = privileged_meta[sensitive_attrs[0]]
            else:
                privileged_val = "_".join([str(privileged_meta[attr]) for attr in sensitive_attrs])

            # 2. Unified Preprocessing
            df_work, final_features, final_sens_col = self.preprocessing_service.run_pipeline(
                df_source, target_col, prep_config,
                sensitive_attrs=sensitive_attrs,
                privileged_val=privileged_val,
                binarize_sensitive=bin_trans
            )
            
            # 3. Fairness Setup for the runner
            eval_priv_val = 1 if bin_trans else privileged_val
            eval_unpriv_vals = [0] if bin_trans else [x for x in df_work[final_sens_col].unique() if str(x) != str(eval_priv_val)]

            # 4. Train/Test Split
            from sklearn.model_selection import train_test_split
            try:
                df_train, df_test = train_test_split(df_work, test_size=0.2, random_state=42 + run_num, stratify=df_work[target_col])
            except ValueError:
                df_train, df_test = train_test_split(df_work, test_size=0.2, random_state=42 + run_num)

            if df_train[target_col].nunique() < 2:
                return None

            # 5. Mitigation
            mitigator = None
            if mit_name.startswith('resampling'):
                from fairpath.fairness.mitigation import ResamplingMitigation
                mitigator = ResamplingMitigation(mit_param)
            elif mit_name == 'relabeling':
                mitigator = RelabelingMitigation()
            elif mit_name.startswith('synthetic'):
                mitigator = SyntheticMitigation(mit_param)
            
            # 6. Core Execution
            runner_res = self.experiment_runner.run(
                df_train, df_test, 
                target_col, final_sens_col, 
                eval_priv_val, eval_unpriv_vals,
                model_name,
                mitigation_strategy=mitigator,
                metric_choice='all'
            )

            # 7. Record Collection
            record = {
                'Dataset': dataset_name,
                'Model Type': model_name,
                'Mitigation Technique': mit_name,
                'Mitigation Detail': mit_param,
                'Run Number': run_num,
                'Sensitive Attributes': ", ".join(sensitive_attrs),
                'Privileged Value': privileged_val,
                'CV Mean Accuracy': runner_res['train_metrics'].get('CV Mean Accuracy', 0)
            }
            record.update(runner_res['test_metrics'])
            record.update(runner_res['fairness_metrics'])
            
            return record


    def _save_results_to_disk(self):
        """Persists benchmark results to CSV and Excel formats."""
        os.makedirs("outputs/reports", exist_ok=True)
        timestamp = int(time.time())
        filepath = os.path.join("outputs/reports", f"benchmark_results_{timestamp}.csv")
        
        df_final = pd.DataFrame(self.results)
        df_final.to_csv(filepath, index=False)
        
        try:
            excel_path = filepath.replace(".csv", ".xlsx")
            df_final.to_excel(excel_path, index=False)
            print(f"Benchmark results persisted: {filepath}")
        except Exception:
            print(f"Results saved to {filepath} (Excel fallback failed)")

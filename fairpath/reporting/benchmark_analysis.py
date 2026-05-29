import pandas as pd
from typing import Dict, Tuple

class BenchmarkAnalyzer:
    """Analyzes benchmark results to identify interesting patterns and outliers."""

    def __init__(self, df_results: pd.DataFrame):
        self.df = df_results

    def detect_high_performance_outliers(self, metric: str = "Test Accuracy", iqr_multiplier: float = 1.5, fallback_top_n: int = 5) -> Dict[str, Tuple[pd.DataFrame, bool]]:
        """
        Detects configurations with unusually high performance for each dataset using the IQR method.
        Falls back to top N performers if no statistical outliers are found.
        
        Args:
            metric: The column name of the metric to analyze (default: "Test Accuracy").
            iqr_multiplier: The multiplier for IQR to define the upper fence (default: 1.5).
            fallback_top_n: Number of top runs to return if no outliers are found.
            
        Returns:
            A dictionary where keys are dataset names and values are tuples:
            (DataFrame of runs, Boolean is_statistical_outlier)
        """
        outliers_by_dataset = {}
        
        # Handle case where metric might be named differently (e.g., "Accuracy" vs "Test Accuracy")
        target_metric = metric
        if metric not in self.df.columns:
            if "Accuracy" in self.df.columns: target_metric = "Accuracy"
            elif "Test Accuracy" in self.df.columns: target_metric = "Test Accuracy"
            else:
                print(f"Warning: Metric '{metric}' not found in results. Skipping outlier detection.")
                return {}

        # Analyze each dataset independently
        for dataset_name, group in self.df.groupby("Dataset"):
            # Calculate IQR stats
            q1 = group[target_metric].quantile(0.25)
            q3 = group[target_metric].quantile(0.75)
            iqr = q3 - q1
            upper_fence = q3 + (iqr_multiplier * iqr)
            
            # Identify outliers (high performance only)
            high_performers = group[group[target_metric] > upper_fence].copy()
            is_outlier = True
            
            # Fallback if no outliers found
            if high_performers.empty:
                high_performers = group.sort_values(by=target_metric, ascending=False).head(fallback_top_n).copy()
                is_outlier = False
            else:
                # Sort outliers by metric descending
                high_performers = high_performers.sort_values(by=target_metric, ascending=False)
                
            outliers_by_dataset[dataset_name] = (high_performers, is_outlier)
                
        return outliers_by_dataset

    def format_outlier_table(self, df_outliers: pd.DataFrame) -> pd.DataFrame:
        """
        Selects and reorders columns for cleaner reporting of outlier configurations.
        """
        # Define preferred column order
        config_cols = [
            "Model Type", "Mitigation Technique", "Mitigation Detail", 
            "Missing Values Strategy", "Outlier Strategy", "Encoding Strategy",
            "Sensitive Attributes"
        ]
        
        perf_cols = [c for c in df_outliers.columns if "Accuracy" in c or "F1" in c or "Balanced" in c]
        fair_cols = [c for c in df_outliers.columns if "Difference" in c or "Impact" in c or "Ratio" in c]
        
        # Combine and ensure columns exist
        final_cols = ["Dataset"] + [c for c in config_cols if c in df_outliers.columns] + \
                     [c for c in perf_cols if c in df_outliers.columns] + \
                     [c for c in fair_cols if c in df_outliers.columns]
                     
        return df_outliers[final_cols]

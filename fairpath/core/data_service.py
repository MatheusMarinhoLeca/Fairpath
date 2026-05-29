import pandas as pd
from typing import List, Tuple, Optional, Any, Dict
from fairpath.data import (
    load_dataset, handle_duplicate_columns, infer_numeric_types, 
    validate_dataset, create_composite_attribute, binarize_attribute
)
from fairpath.eda.statistics import get_basic_stats, get_comprehensive_stats
from fairpath.preprocessing.missing_values import impute_missing
from fairpath.preprocessing.outliers import OutlierRemover
from fairpath.preprocessing.encoding import one_hot_encode, label_encode
from fairpath.core.models import PreprocessingConfig

class DataService:
    """Service for dataset engineering and integrity operations.
    
    This service encapsulates all logic related to data loading, cleaning, 
    preprocessing, and statistical auditing.
    """

    def load_initial_data(self, path: str) -> Tuple[pd.DataFrame, List[str]]:
        """Loads a dataset and handles structural anomalies like duplicate columns.

        Args:
            path: Path to the .csv or .xlsx file.

        Returns:
            A tuple containing:
                - The loaded pandas DataFrame.
                - A list of column names that were renamed due to duplicates.
        """
        dataset = load_dataset(path)
        dataset, renamed_columns = handle_duplicate_columns(dataset)
        return dataset, renamed_columns

    def prepare_raw_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Performs initial type inference and structural cleanup.

        Args:
            df: The raw DataFrame.

        Returns:
            A tuple containing:
                - The cleaned DataFrame.
                - A list of columns that were auto-converted to numeric.
        """
        df_cleaned, converted_cols = infer_numeric_types(df)
        return df_cleaned, converted_cols

    def get_eda_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Computes basic exploratory statistics for the dashboard.

        Args:
            df: The DataFrame to analyze.

        Returns:
            A dictionary containing summary statistics.
        """
        return get_basic_stats(df)

    def apply_preprocessing(self, df: pd.DataFrame, target_col: str, 
                           config: PreprocessingConfig) -> Tuple[pd.DataFrame, List[str]]:
        """Applies a sequence of data cleaning steps defined in a config.

        Args:
            df: The DataFrame to preprocess.
            target_col: The outcome column name.
            config: An object containing preprocessing strategies.

        Returns:
            A tuple containing:
                - The preprocessed DataFrame.
                - The updated list of selected feature names.
        """
        df_processed = df.copy()
        current_features = config.selected_features.copy()

        # 1. Feature Selection
        if current_features:
            cols_to_keep = current_features + [target_col]
            df_processed = df_processed[[c for c in cols_to_keep if c in df_processed.columns]]

        # 2. Missing Value Imputation
        if config.missing_strategy and config.missing_strategy not in ['Skipped', 'None detected']:
            strategy_map = {'Mean/Median': 'mean', 'Mode': 'mode', 'Drop': 'drop'}
            strat = strategy_map.get(config.missing_strategy, config.missing_strategy)
            
            if strat == 'drop':
                cols_to_clean = current_features + [target_col] if current_features else None
                df_processed = impute_missing(df_processed, 'drop', columns=cols_to_clean)
                df_processed = impute_missing(df_processed, 'mode', columns=None)
            else:
                df_processed = impute_missing(df_processed, strat, columns=None)

        # 3. Outlier Removal
        if config.outlier_strategy == 'IQR-based removal':
            cols_to_clean = (current_features + [target_col]) if current_features else None
            remover = OutlierRemover(strategy='remove', columns=cols_to_clean)
            df_processed = remover.fit_transform(df_processed)
        
        return df_processed, current_features

    def encode_categorical(self, df: pd.DataFrame, target_col: str, strategy: str, 
                          columns: List[str], selected_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Encodes categorical features and tracks the resulting column names.

        Args:
            df: The DataFrame to encode.
            target_col: The outcome column name.
            strategy: '1' for One-hot, '2' for Label encoding.
            columns: The specific columns to encode.
            selected_features: The current list of model features to track.

        Returns:
            A tuple containing:
                - The encoded DataFrame.
                - The updated list of features (expanded if one-hot).
        """
        df_encoded = df.copy()
        new_features = selected_features.copy()

        if strategy == '1':
            df_encoded = one_hot_encode(df_encoded, target_col, columns=columns)
            if selected_features:
                expanded_features = []
                for feat in selected_features:
                    if feat in df_encoded.columns:
                        expanded_features.append(feat)
                    else:
                        dummies = [c for c in df_encoded.columns if c.startswith(f"{feat}_")]
                        expanded_features.extend(dummies)
                new_features = list(dict.fromkeys(expanded_features))
        else:
            df_encoded = label_encode(df_encoded, target_col, columns=columns)
            
        return df_encoded, new_features

    def handle_remaining_categorical(self, df: pd.DataFrame, target_col: str, 
                                   remaining_cols: List[str], strategy: str, 
                                   selected_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Handles columns missed in primary encoding to ensure model compatibility.

        Args:
            df: The DataFrame.
            target_col: The outcome column name.
            remaining_cols: List of unencoded categorical columns.
            strategy: '1' to encode automatically, '2' to drop from features.
            selected_features: The current list of model features.

        Returns:
            A tuple containing:
                - The updated DataFrame.
                - The updated list of features.
        """
        df_updated = df.copy()
        new_features = selected_features.copy()
        
        if strategy == '1':
            df_updated = label_encode(df_updated, target_col, columns=remaining_cols)
        elif strategy == '2':
            new_features = [f for f in selected_features if f not in remaining_cols]
            
        return df_updated, new_features

    def setup_composite_sensitive(self, df: pd.DataFrame, attributes: List[str]) -> Tuple[pd.DataFrame, str]:
        """Creates a composite sensitive attribute for intersectional fairness.

        Args:
            df: The dataset.
            attributes: Multiple protected attributes (e.g., ['Race', 'Gender']).

        Returns:
            A tuple containing:
                - The DataFrame with the new composite column.
                - The name of the created composite column.
        """
        return create_composite_attribute(df, attributes)

    def prepare_fairness_binary(self, df: pd.DataFrame, sensitive_col: str, privileged_group: Any) -> pd.DataFrame:
        """Transforms a multi-class sensitive attribute into a binary (1/0) format.

        Args:
            df: The dataset.
            sensitive_col: The sensitive attribute column.
            privileged_group: The value defined as privileged.

        Returns:
            The DataFrame with a binarized sensitive attribute.
        """
        return binarize_attribute(df, sensitive_col, privileged_group)

    def get_audit_stats(self, df: pd.DataFrame, target_col: str, sensitive_col: str, 
                        original_feat_count: int, selected_features: List[str],
                        display_mapping: Optional[Dict[Any, str]] = None) -> Dict[str, Any]:
        """Computes comprehensive statistics for the final fairness audit report.

        Args:
            df: The dataset.
            target_col: The outcome column name.
            sensitive_col: The sensitive attribute column name.
            original_feat_count: The number of features in the raw dataset.
            selected_features: The features used in the model.
            display_mapping: Optional mapping to rename groups in the output labels.

        Returns:
            A dictionary containing detailed statistics per group and class.
        """
        stats = get_comprehensive_stats(
            df, target_col, sensitive_col, 
            original_feat_count=original_feat_count, 
            selected_features=selected_features
        )
        
        if not display_mapping:
            return stats
            
        # Post-process to map numeric labels (0/1) to human-readable names
        mapped_stats = {}
        for k, v in stats.items():
            new_key = k
            # Handle keys like "Group '0' Size" or "Group '1' Pos. Rate"
            if k.startswith("Group '"):
                parts = k.split("'")
                if len(parts) >= 3:
                    group_val_str = parts[1]
                    suffix = parts[2]
                    
                    # Try to match the group value string to a mapped name
                    mapped_name = None
                    # Try direct string match
                    if group_val_str in display_mapping:
                        mapped_name = display_mapping[group_val_str]
                    # Try numeric match if it looks like a number
                    else:
                        try:
                            # Use float for comparison to handle 0.0 vs 0
                            f_val = float(group_val_str)
                            for m_key, m_name in display_mapping.items():
                                try:
                                    if float(m_key) == f_val:
                                        mapped_name = m_name
                                        break
                                except (ValueError, TypeError):
                                    continue
                        except ValueError:
                            pass
                    
                    if mapped_name:
                        new_key = f"Group '{mapped_name}'{suffix}"
            
            mapped_stats[new_key] = v
            
        return mapped_stats

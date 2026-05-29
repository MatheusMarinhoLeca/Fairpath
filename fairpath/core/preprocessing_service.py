import pandas as pd
from typing import List, Tuple, Any, Optional
from fairpath.preprocessing.missing_values import impute_missing
from fairpath.preprocessing.outliers import remove_outliers_iqr
from fairpath.preprocessing.encoding import one_hot_encode, label_encode
from fairpath.data import create_composite_attribute, binarize_attribute
from fairpath.core.models import PreprocessingConfig

class PreprocessingService:
    """Centralized service for consistent data preprocessing across the platform.
    
    This service unifies the cleaning, encoding, and attribute transformation 
    logic to ensure that both the interactive workflow and the benchmark engine 
    produce identical results for the same configuration.
    """

    def run_pipeline(self, 
                    df: pd.DataFrame, 
                    target_col: str, 
                    config: PreprocessingConfig,
                    sensitive_attrs: Optional[List[str]] = None,
                    privileged_val: Optional[Any] = None,
                    binarize_sensitive: bool = False) -> Tuple[pd.DataFrame, List[str], str]:
        """Executes the full preprocessing pipeline.

        Args:
            df: The raw input DataFrame.
            target_col: The outcome column name.
            config: Preprocessing configuration (missing values, outliers, encoding).
            sensitive_attrs: List of one or more sensitive attributes.
            privileged_val: Value of the privileged group.
            binarize_sensitive: Whether to transform the sensitive attribute to binary.

        Returns:
            A tuple containing:
                - The processed DataFrame.
                - The final list of model features.
                - The name of the final sensitive attribute column.
        """
        df_work = df.copy()
        # 1. Feature Selection Tracking
        # We don't drop columns here anymore to preserve all features for the final export.
        initial_selected_features = config.selected_features.copy() if config.selected_features else [c for c in df_work.columns if c not in [target_col]]

        # Columns to actually use for cleaning/encoding decisions
        work_features = initial_selected_features + [target_col]

        # 2. Cleaning: Missing Values
        if config.missing_strategy and config.missing_strategy not in ['none', 'Skip']:
            strategy_map = {
                'Mean/Median': 'mean', '1': 'mean',
                'Mode': 'mode', '2': 'mode',
                'Drop': 'drop', '3': 'drop'
            }
            strat = strategy_map.get(config.missing_strategy, config.missing_strategy)
            
            # Smart Imputation: Mean/Median only works on numbers. 
            # If the user chose it, we apply it to numbers and fallback to 'mode' for strings.
            if strat in ['mean', 'median']:
                num_cols = [c for c in work_features if c in df_work.columns and pd.api.types.is_numeric_dtype(df_work[c])]
                cat_cols = [c for c in work_features if c in df_work.columns and not pd.api.types.is_numeric_dtype(df_work[c])]
                
                if num_cols:
                    df_work = impute_missing(df_work, strat, columns=num_cols)
                if cat_cols:
                    df_work = impute_missing(df_work, 'mode', columns=cat_cols)
            else:
                # For 'mode' or 'drop', we can apply to all work features directly
                df_work = impute_missing(df_work, strat, columns=work_features)

        # 3. Cleaning: Outliers
        if config.outlier_strategy and config.outlier_strategy not in ['none', 'Skip']:
            # Benchmark uses 'iqr' as string, Workflow uses 'IQR-based removal'
            if config.outlier_strategy.lower() in ['iqr', 'iqr-based removal', '1']:
                df_work = remove_outliers_iqr(df_work, columns=work_features)

        # 4. Sensitive Attribute Setup
        final_sensitive_col = ""
        if sensitive_attrs:
            if len(sensitive_attrs) > 1:
                df_work, final_sensitive_col = create_composite_attribute(df_work, sensitive_attrs)
                # Recalculate privileged_val if it was intersectional
                if not isinstance(privileged_val, str) and isinstance(privileged_val, dict):
                    privileged_val = "_".join([str(privileged_val[attr]) for attr in sensitive_attrs])
            else:
                final_sensitive_col = sensitive_attrs[0]

            if binarize_sensitive:
                df_work = binarize_attribute(df_work, final_sensitive_col, privileged_val)

        # 5. Encoding
        # Determine features to encode (ONLY those explicitly selected, excluding target and sensitive)
        exclude = [target_col, final_sensitive_col]
        
        # We only encode columns that the user explicitly wants to "work with"
        cat_features = [c for c in initial_selected_features if c in df_work.columns and not pd.api.types.is_numeric_dtype(df_work[c])]
        cat_features = [c for c in cat_features if c not in exclude]
        
        # Final selected features tracking
        model_features = initial_selected_features.copy()

        if config.encoding_strategy:
            enc_strat = 'one-hot' if config.encoding_strategy in ['1', 'one-hot', 'One-Hot Encoding'] else 'label'
            
            if enc_strat == 'one-hot' and cat_features:
                df_work = one_hot_encode(df_work, target_col, columns=cat_features)
                # Update model features list if one-hot expanded the columns
                expanded_features = []
                for feat in model_features:
                    if feat in df_work.columns:
                        expanded_features.append(feat)
                    else:
                        dummies = [c for c in df_work.columns if c.startswith(f"{feat}_")]
                        expanded_features.extend(dummies)
                model_features = list(dict.fromkeys(expanded_features))
            elif enc_strat == 'label' and cat_features:
                df_work = label_encode(df_work, target_col, columns=cat_features)

        # Ensure target is numeric (LabelEncode if necessary)
        if df_work[target_col].dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            df_work[target_col] = LabelEncoder().fit_transform(df_work[target_col].astype(str))

        return df_work, model_features, final_sensitive_col

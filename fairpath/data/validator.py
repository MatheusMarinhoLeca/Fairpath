import os
import pandas as pd
from typing import List, Tuple, Any

def validate_file_path(file_path):
    if not os.path.exists(file_path):
        return False
    if not os.path.isfile(file_path):
        return False
    return True

def validate_dataset(df):
    if df.empty:
        return False, "Dataset is empty."
    if len(df.columns) < 2:
        return False, "Dataset has fewer than 2 columns."
    return True, "Dataset is valid."

def check_duplicates(df):
    """
    Checks for duplicate rows in the dataframe.
    Returns:
        (bool, int): (True if duplicates exist, count of duplicates)
    """
    dup_count = df.duplicated().sum()
    return dup_count > 0, dup_count

class SchemaValidator:
    """Performs explicit validation of dataset structure and semantic integrity."""
    
    @staticmethod
    def validate_schema(df: pd.DataFrame, expected_cols: List[str], target_col: str) -> Tuple[bool, str]:
        """Checks for column existence and basic type consistency.
        
        Returns:
            (bool, str): (True if valid, error message if not)
        """
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            return False, f"Missing required columns: {', '.join(missing)}"
            
        if target_col not in df.columns:
            return False, f"Target column '{target_col}' not found in dataset."
            
        return True, "Schema is valid."

    @staticmethod
    def validate_labels(df: pd.DataFrame, target_col: str, expected_labels: List[Any]) -> Tuple[bool, str]:
        """Ensures the target column contains expected classification labels.
        
        This prevents cases where the target exists but has semantically incorrect data.
        """
        unique_vals = df[target_col].unique().tolist()
        
        # Check if at least one expected label is present
        found = any(str(label) in [str(u) for u in unique_vals] for label in expected_labels)
        
        if not found:
            return False, f"Target column '{target_col}' does not contain expected labels {expected_labels}. Found: {unique_vals}"
            
        return True, "Labels are valid."

import os
import pandas as pd

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

def check_zero_variance(df):
    """
    Identifies columns with only one unique value (zero variance).
    Returns:
        list: List of column names with zero variance.
    """
    return [col for col in df.columns if df[col].nunique() <= 1]

import os

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

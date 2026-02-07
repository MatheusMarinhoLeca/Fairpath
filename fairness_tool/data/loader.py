import pandas as pd
import os

def load_dataset(file_path):
    _, ext = os.path.splitext(file_path)
    try:
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        raise IOError(f"Error loading file: {e}")

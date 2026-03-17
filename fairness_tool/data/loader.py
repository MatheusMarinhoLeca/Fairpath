import pandas as pd
import os
from .parsers import ParserFactory

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads a dataset from a file path using the appropriate parser.
    
    Supported formats:
    - CSV (.csv) with automatic encoding and separator detection.
    - Excel (.xlsx, .xls).
    
    Args:
        file_path (str): Path to the dataset file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported or file is empty/malformed.
        IOError: If there is an error reading the file.
    """
    try:
        parser = ParserFactory.get_parser(file_path)
        return parser.parse(file_path)
    except FileNotFoundError:
        raise
    except ValueError as e:
        # Re-raise ValueError as is, or wrap if specific message needed
        raise
    except Exception as e:
        # Wrap unexpected errors to maintain some backward compatibility with callers expecting generic errors,
        # though app.py catches Exception anyway.
        raise IOError(f"Error loading file: {e}")

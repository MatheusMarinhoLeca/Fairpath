import pandas as pd
import os
from abc import ABC, abstractmethod

class BaseParser(ABC):
    """Abstract base class for file parsers."""
    
    @abstractmethod
    def parse(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Parses the file into a pandas DataFrame."""
        pass

    def validate_file(self, file_path: str):
        """Basic validation for file existence and size."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"File is empty: {file_path}")

import csv

class CSVParser(BaseParser):
    """Parser for CSV files with deterministic separator detection."""
    
    def parse(self, file_path: str, **kwargs) -> pd.DataFrame:
        self.validate_file(file_path)
        
        # Try to detect separator using csv.Sniffer
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(2048)
                dialect = csv.Sniffer().sniff(sample)
                detected_sep = dialect.delimiter
        except Exception:
            # Fallback to comma if sniffer fails, or raise error if precision is required
            detected_sep = ','

        try:
            # We use a single pass with the detected separator.
            # If it fails, we fail loudly instead of guessing.
            df = pd.read_csv(
                file_path, 
                sep=detected_sep, 
                encoding='utf-8', 
                on_bad_lines='error', # Fail loudly on malformed lines
                index_col=False,
                **kwargs
            )
            
            # Standardize headers
            df.columns = df.columns.str.strip()
            
            # Final sanity check: if only 1 column, the separator might be wrong
            if df.shape[1] <= 1:
                # One last attempt with python engine auto-detection
                df_retry = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='warn')
                if df_retry.shape[1] > 1:
                    return df_retry
                raise ValueError(f"CSV parsing resulted in only one column using separator '{detected_sep}'. Please check the file format.")
                
            return df
                        
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file: {e}")

class ExcelParser(BaseParser):
    """Parser for Excel files (.xls, .xlsx)."""
    
    def parse(self, file_path: str, **kwargs) -> pd.DataFrame:
        self.validate_file(file_path)
        try:
            # engine='openpyxl' for xlsx, 'xlrd' for xls (if installed)
            # pandas usually auto-detects engine based on extension
            df = pd.read_excel(file_path, **kwargs)
            
            # Standardize headers
            df.columns = df.columns.astype(str).str.strip()
            return df
        except Exception as e:
            raise ValueError(f"Failed to parse Excel file: {e}")

class ParserFactory:
    """Factory to get the appropriate parser based on file extension."""
    
    _parsers = {
        '.csv': CSVParser,
        '.xlsx': ExcelParser,
        '.xls': ExcelParser
    }
    
    @classmethod
    def get_parser(cls, file_path: str) -> BaseParser:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        parser_class = cls._parsers.get(ext)
        if parser_class:
            return parser_class()
        
        raise ValueError(f"Unsupported file extension: {ext}")

    @classmethod
    def register_parser(cls, extension: str, parser_class: type):
        """Allows extending the factory with new parsers."""
        cls._parsers[extension.lower()] = parser_class

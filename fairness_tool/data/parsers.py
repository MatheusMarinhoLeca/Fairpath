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

class CSVParser(BaseParser):
    """Parser for CSV files with robust encoding and separator handling."""
    
    def parse(self, file_path: str, **kwargs) -> pd.DataFrame:
        self.validate_file(file_path)
        
        # Try common encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        separators = [',', ';', '\t', '|']
        
        last_exception = None

        # First try to detect separator with python engine if possible, otherwise iterate
        # Using engine='python' allows separator auto-detection but is slower.
        # Let's try explicit separators with C engine for performance first.
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(
                        file_path, 
                        sep=sep, 
                        encoding=encoding, 
                        on_bad_lines='warn', # Skip bad lines or warn
                        index_col=False, # Don't use first column as index by default
                        **kwargs
                    )
                    
                    # Basic heuristic: if we have only 1 column, maybe the separator was wrong
                    if df.shape[1] > 1:
                        # Standardize headers: strip whitespace
                        df.columns = df.columns.str.strip()
                        return df
                        
                except Exception as e:
                    last_exception = e
                    continue
        
        # Fallback: Try python engine for separator autodetection
        try:
            df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            pass

        raise ValueError(f"Failed to parse CSV file. Last error: {last_exception}")

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

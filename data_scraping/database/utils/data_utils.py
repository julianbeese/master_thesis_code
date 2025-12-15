"""
Utility-Funktionen für Datenverarbeitung
Unterstützt Polars, Pandas, PyArrow und DuckDB
"""

import polars as pl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import logging
from datetime import datetime
import json


class DataConverter:
    """Klasse für Konvertierung zwischen verschiedenen Datenformaten"""
    
    @staticmethod
    def polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
        """Konvertiert Polars DataFrame zu Pandas DataFrame"""
        return df.to_pandas()
    
    @staticmethod
    def pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
        """Konvertiert Pandas DataFrame zu Polars DataFrame"""
        return pl.from_pandas(df)
    
    @staticmethod
    def polars_to_arrow(df: pl.DataFrame) -> pa.Table:
        """Konvertiert Polars DataFrame zu PyArrow Table"""
        return df.to_arrow()
    
    @staticmethod
    def arrow_to_polars(table: pa.Table) -> pl.DataFrame:
        """Konvertiert PyArrow Table zu Polars DataFrame"""
        return pl.from_arrow(table)
    
    @staticmethod
    def pandas_to_arrow(df: pd.DataFrame) -> pa.Table:
        """Konvertiert Pandas DataFrame zu PyArrow Table"""
        return pa.Table.from_pandas(df)
    
    @staticmethod
    def arrow_to_pandas(table: pa.Table) -> pd.DataFrame:
        """Konvertiert PyArrow Table zu Pandas DataFrame"""
        return table.to_pandas()


class DataValidator:
    """Klasse für Datenvalidierung"""
    
    @staticmethod
    def validate_dataframe(df: Union[pl.DataFrame, pd.DataFrame], 
                          required_columns: Optional[List[str]] = None,
                          min_rows: int = 0) -> Dict[str, Any]:
        """
        Validiert DataFrame
        
        Args:
            df: DataFrame (Polars oder Pandas)
            required_columns: Liste der erforderlichen Spalten
            min_rows: Minimale Anzahl Zeilen
            
        Returns:
            Dictionary mit Validierungsergebnissen
        """
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # Konvertierung zu Pandas für einheitliche Validierung
        if isinstance(df, pl.DataFrame):
            df_pandas = df.to_pandas()
        else:
            df_pandas = df
        
        # Basis-Informationen
        result["info"]["shape"] = df_pandas.shape
        result["info"]["columns"] = list(df_pandas.columns)
        result["info"]["dtypes"] = df_pandas.dtypes.to_dict()
        
        # Zeilen-Check
        if len(df_pandas) < min_rows:
            result["is_valid"] = False
            result["errors"].append(f"Zu wenige Zeilen: {len(df_pandas)} < {min_rows}")
        
        # Spalten-Check
        if required_columns:
            missing_columns = set(required_columns) - set(df_pandas.columns)
            if missing_columns:
                result["is_valid"] = False
                result["errors"].append(f"Fehlende Spalten: {missing_columns}")
        
        # Null-Werte Check
        null_counts = df_pandas.isnull().sum()
        if null_counts.sum() > 0:
            result["warnings"].append(f"Null-Werte gefunden: {null_counts.to_dict()}")
        
        return result


class DataProcessor:
    """Klasse für Datenverarbeitung"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def clean_dataframe(self, df: Union[pl.DataFrame, pd.DataFrame],
                       remove_duplicates: bool = True,
                       fill_nulls: bool = True,
                       strip_strings: bool = True) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Bereinigt DataFrame
        
        Args:
            df: DataFrame
            remove_duplicates: Duplikate entfernen
            fill_nulls: Null-Werte füllen
            strip_strings: String-Whitespace entfernen
            
        Returns:
            Bereinigter DataFrame
        """
        original_type = type(df)
        
        # Konvertierung zu Pandas für Verarbeitung
        if isinstance(df, pl.DataFrame):
            df_work = df.to_pandas()
        else:
            df_work = df.copy()
        
        # String-Bereinigung
        if strip_strings:
            string_columns = df_work.select_dtypes(include=['object']).columns
            for col in string_columns:
                df_work[col] = df_work[col].astype(str).str.strip()
        
        # Duplikate entfernen
        if remove_duplicates:
            before_count = len(df_work)
            df_work = df_work.drop_duplicates()
            after_count = len(df_work)
            if before_count != after_count:
                self.logger.info(f"Duplikate entfernt: {before_count - after_count}")
        
        # Null-Werte füllen
        if fill_nulls:
            # Numerische Spalten: Median
            numeric_columns = df_work.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                df_work[col] = df_work[col].fillna(df_work[col].median())
            
            # String-Spalten: "Unknown"
            string_columns = df_work.select_dtypes(include=['object']).columns
            for col in string_columns:
                df_work[col] = df_work[col].fillna("Unknown")
        
        # Rückkonvertierung
        if original_type == pl.DataFrame:
            return pl.from_pandas(df_work)
        else:
            return df_work
    
    def batch_process(self, data: List[Any], batch_size: int = 1000,
                     processor_func: callable = None) -> List[Any]:
        """
        Verarbeitet Daten in Batches
        
        Args:
            data: Liste der zu verarbeitenden Daten
            batch_size: Größe der Batches
            processor_func: Funktion zur Verarbeitung
            
        Returns:
            Liste der verarbeiteten Daten
        """
        results = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            if processor_func:
                processed_batch = processor_func(batch)
            else:
                processed_batch = batch
            
            results.extend(processed_batch)
            
            if self.logger:
                self.logger.info(f"Batch {i//batch_size + 1} verarbeitet: {len(batch)} Elemente")
        
        return results


class FileManager:
    """Klasse für Datei-Management"""
    
    def __init__(self, base_path: Path, logger: Optional[logging.Logger] = None):
        self.base_path = Path(base_path)
        self.logger = logger or logging.getLogger(__name__)
    
    def save_dataframe(self, df: Union[pl.DataFrame, pd.DataFrame], 
                      file_path: Union[str, Path],
                      format: str = "parquet") -> None:
        """
        Speichert DataFrame in verschiedenen Formaten
        
        Args:
            df: DataFrame
            file_path: Pfad zur Datei
            format: Format ('parquet', 'csv', 'json')
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Konvertierung zu Pandas für Speicherung
        if isinstance(df, pl.DataFrame):
            df_save = df.to_pandas()
        else:
            df_save = df
        
        if format.lower() == "parquet":
            df_save.to_parquet(file_path, engine='pyarrow')
        elif format.lower() == "csv":
            df_save.to_csv(file_path, index=False)
        elif format.lower() == "json":
            df_save.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if self.logger:
            self.logger.info(f"DataFrame gespeichert: {file_path} ({format})")
    
    def load_dataframe(self, file_path: Union[str, Path],
                      format: str = "parquet") -> pd.DataFrame:
        """
        Lädt DataFrame aus verschiedenen Formaten
        
        Args:
            file_path: Pfad zur Datei
            format: Format ('parquet', 'csv', 'json')
            
        Returns:
            Pandas DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
        
        if format.lower() == "parquet":
            return pd.read_parquet(file_path, engine='pyarrow')
        elif format.lower() == "csv":
            return pd.read_csv(file_path)
        elif format.lower() == "json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def list_files(self, pattern: str = "*", 
                  recursive: bool = True) -> List[Path]:
        """
        Listet Dateien im Basis-Pfad auf
        
        Args:
            pattern: Glob-Pattern
            recursive: Rekursiv suchen
            
        Returns:
            Liste der gefundenen Dateien
        """
        if recursive:
            return list(self.base_path.rglob(pattern))
        else:
            return list(self.base_path.glob(pattern))


class PerformanceMonitor:
    """Klasse für Performance-Monitoring"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.memory_start = None
    
    def start_monitoring(self):
        """Startet Performance-Monitoring"""
        self.start_time = datetime.now()
        try:
            import psutil
            process = psutil.Process()
            self.memory_start = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self.memory_start = None
    
    def stop_monitoring(self, operation_name: str = "Operation"):
        """Stoppt Performance-Monitoring und loggt Ergebnisse"""
        if self.start_time is None:
            return
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        memory_info = ""
        if self.memory_start is not None:
            try:
                import psutil
                process = psutil.Process()
                memory_end = process.memory_info().rss / 1024 / 1024  # MB
                memory_diff = memory_end - self.memory_start
                memory_info = f", Memory: {memory_diff:+.2f}MB"
            except ImportError:
                pass
        
        if self.logger:
            self.logger.info(f"{operation_name} abgeschlossen: {duration:.2f}s{memory_info}")
        
        return {
            "operation": operation_name,
            "duration_seconds": duration,
            "memory_change_mb": memory_info
        }


# Convenience-Funktionen
def quick_convert(df: Union[pl.DataFrame, pd.DataFrame], 
                 target_type: str) -> Union[pl.DataFrame, pd.DataFrame, pa.Table]:
    """
    Schnelle Konvertierung zwischen Datenformaten
    
    Args:
        df: DataFrame
        target_type: Zieltyp ('pandas', 'polars', 'arrow')
        
    Returns:
        Konvertierter DataFrame/Table
    """
    converter = DataConverter()
    
    if target_type.lower() == "pandas":
        if isinstance(df, pl.DataFrame):
            return converter.polars_to_pandas(df)
        elif isinstance(df, pa.Table):
            return converter.arrow_to_pandas(df)
        else:
            return df
    
    elif target_type.lower() == "polars":
        if isinstance(df, pd.DataFrame):
            return converter.pandas_to_polars(df)
        elif isinstance(df, pa.Table):
            return converter.arrow_to_polars(df)
        else:
            return df
    
    elif target_type.lower() == "arrow":
        if isinstance(df, pl.DataFrame):
            return converter.polars_to_arrow(df)
        elif isinstance(df, pd.DataFrame):
            return converter.pandas_to_arrow(df)
        else:
            return df
    
    else:
        raise ValueError(f"Unsupported target type: {target_type}")


def validate_and_clean(df: Union[pl.DataFrame, pd.DataFrame],
                      required_columns: Optional[List[str]] = None,
                      min_rows: int = 0) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Kombiniert Validierung und Bereinigung
    
    Args:
        df: DataFrame
        required_columns: Erforderliche Spalten
        min_rows: Minimale Zeilen
        
    Returns:
        Bereinigter DataFrame
    """
    validator = DataValidator()
    processor = DataProcessor()
    
    # Validierung
    validation_result = validator.validate_dataframe(df, required_columns, min_rows)
    
    if not validation_result["is_valid"]:
        raise ValueError(f"DataFrame-Validierung fehlgeschlagen: {validation_result['errors']}")
    
    # Bereinigung
    return processor.clean_dataframe(df)

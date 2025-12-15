"""
Konfigurationsdatei für Datenbank-Management
Enthält alle wichtigen Einstellungen und Pfade
"""

from pathlib import Path
from typing import Dict, Any
import os


class DatabaseConfig:
    """Konfigurationsklasse für Datenbank-Einstellungen"""
    
    # Basis-Pfade
    BASE_DIR = Path(__file__).parent.parent.parent
    DATABASE_DIR = BASE_DIR / "database"
    DATA_DIR = DATABASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    LOGS_DIR = DATABASE_DIR / "logs"
    
    # Datenbank-Einstellungen
    DB_PATH = DATA_DIR / "debates_data.db"
    DB_BACKUP_DIR = DATA_DIR / "backups"
    
    # DuckDB-Einstellungen
    DUCKDB_CONFIG = {
        "memory_limit": "2GB",
        "threads": 4,
        "enable_progress_bar": True,
        "enable_http_metadata_cache": True,
        "enable_object_cache": True
    }
    
    # Polars-Einstellungen
    POLARS_CONFIG = {
        "streaming": True,
        "lazy": True,
        "collect_schema": True
    }
    
    # Pandas-Einstellungen
    PANDAS_CONFIG = {
        "display.max_columns": None,
        "display.max_rows": 100,
        "display.width": None,
        "display.max_colwidth": 50
    }
    
    # PyArrow-Einstellungen
    PYARROW_CONFIG = {
        "memory_pool": "system",
        "use_threads": True
    }
    
    # Logging-Einstellungen
    LOGGING_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "max_file_size": "10MB",
        "backup_count": 5
    }
    
    # XML-Parsing-Einstellungen
    XML_CONFIG = {
        "encoding": "utf-8",
        "parser": "lxml",
        "namespace_handling": True,
        "strip_whitespace": True
    }
    
    # Performance-Einstellungen
    PERFORMANCE_CONFIG = {
        "batch_size": 1000,
        "chunk_size": 10000,
        "max_workers": 4,
        "memory_threshold": "1GB"
    }
    
    @classmethod
    def get_db_path(cls) -> str:
        """Gibt den Datenbankpfad zurück"""
        return str(cls.DB_PATH)
    
    @classmethod
    def get_raw_data_dir(cls) -> Path:
        """Gibt das Raw-Data-Verzeichnis zurück"""
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return cls.RAW_DATA_DIR
    
    @classmethod
    def get_processed_data_dir(cls) -> Path:
        """Gibt das Processed-Data-Verzeichnis zurück"""
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return cls.PROCESSED_DATA_DIR
    
    @classmethod
    def get_logs_dir(cls) -> Path:
        """Gibt das Logs-Verzeichnis zurück"""
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.LOGS_DIR
    
    @classmethod
    def get_backup_dir(cls) -> Path:
        """Gibt das Backup-Verzeichnis zurück"""
        cls.DB_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        return cls.DB_BACKUP_DIR
    
    @classmethod
    def setup_directories(cls) -> None:
        """Erstellt alle notwendigen Verzeichnisse"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.LOGS_DIR,
            cls.DB_BACKUP_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Gibt alle Konfigurationen als Dictionary zurück"""
        return {
            "database": {
                "path": cls.get_db_path(),
                "duckdb_config": cls.DUCKDB_CONFIG
            },
            "data_directories": {
                "raw": str(cls.get_raw_data_dir()),
                "processed": str(cls.get_processed_data_dir()),
                "backups": str(cls.get_backup_dir())
            },
            "libraries": {
                "polars": cls.POLARS_CONFIG,
                "pandas": cls.PANDAS_CONFIG,
                "pyarrow": cls.PYARROW_CONFIG
            },
            "xml_parsing": cls.XML_CONFIG,
            "performance": cls.PERFORMANCE_CONFIG,
            "logging": cls.LOGGING_CONFIG
        }


# Umgebungsvariablen
class EnvironmentConfig:
    """Konfiguration basierend auf Umgebungsvariablen"""
    
    @staticmethod
    def get_database_url() -> str:
        """Gibt die Datenbank-URL aus Umgebungsvariablen zurück"""
        return os.getenv("DATABASE_URL", DatabaseConfig.get_db_path())
    
    @staticmethod
    def get_log_level() -> str:
        """Gibt das Log-Level aus Umgebungsvariablen zurück"""
        return os.getenv("LOG_LEVEL", "INFO")
    
    @staticmethod
    def get_max_workers() -> int:
        """Gibt die maximale Anzahl Worker aus Umgebungsvariablen zurück"""
        return int(os.getenv("MAX_WORKERS", "4"))
    
    @staticmethod
    def get_memory_limit() -> str:
        """Gibt das Memory-Limit aus Umgebungsvariablen zurück"""
        return os.getenv("MEMORY_LIMIT", "2GB")


# Globale Konfigurationsinstanz
config = DatabaseConfig()
env_config = EnvironmentConfig()

"""
Konfigurationsdatei für XML-Scraper
Enthält alle wichtigen Einstellungen für das Scraping von theyworkforyou.com
"""

from pathlib import Path
from typing import Dict, Any, List
import os


class ScraperConfig:
    """Konfigurationsklasse für XML-Scraper"""
    
    # Basis-Pfade
    BASE_DIR = Path(__file__).parent.parent.parent
    SCRAPER_DIR = BASE_DIR / "scraper"
    DATA_DIR = SCRAPER_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    LOGS_DIR = SCRAPER_DIR / "logs"
    
    # Scraping-Einstellungen
    BASE_URL = "https://www.theyworkforyou.com/pwdata/scrapedxml/westminhall/"
    START_YEAR = 2012
    END_YEAR = 2021
    
    # Download-Einstellungen
    MAX_WORKERS = 4
    DELAY_BETWEEN_REQUESTS = 1.0  # Sekunden
    REQUEST_TIMEOUT = 60  # Sekunden
    MAX_RETRIES = 3
    
    # User-Agent für Requests
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    
    # Datei-Einstellungen
    SUPPORTED_EXTENSIONS = ['.xml']
    MIN_FILE_SIZE = 100  # Bytes
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    # Logging-Einstellungen
    LOGGING_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "max_file_size": "10MB",
        "backup_count": 5
    }
    
    # Performance-Einstellungen
    PERFORMANCE_CONFIG = {
        "chunk_size": 8192,  # Bytes für Datei-Downloads
        "memory_threshold": "500MB",
        "enable_progress_bar": True,
        "enable_file_verification": True
    }
    
    # Roboter.txt und Ethik-Einstellungen
    ETHICS_CONFIG = {
        "respect_robots_txt": True,
        "max_requests_per_minute": 60,
        "respect_server_load": True,
        "user_agent_identification": True
    }
    
    # Datei-Patterns für Jahres-Extraktion
    YEAR_PATTERNS = [
        r'(\d{4})',  # Einfaches 4-stelliges Jahr
        r'(\d{4})-\d{2}-\d{2}',  # Datum-Format YYYY-MM-DD
        r'(\d{4})_\d{2}_\d{2}',  # Datum mit Unterstrichen YYYY_MM_DD
        r'(\d{4})\d{2}\d{2}',  # Kompaktes Datum YYYYMMDD
    ]
    
    # Ausschluss-Patterns (Dateien die nicht gescraped werden sollen)
    EXCLUDE_PATTERNS = [
        r'.*\.tmp$',
        r'.*\.temp$',
        r'.*\.bak$',
        r'.*\.old$',
    ]
    
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
    def setup_directories(cls) -> None:
        """Erstellt alle notwendigen Verzeichnisse"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Gibt alle Konfigurationen als Dictionary zurück"""
        return {
            "scraping": {
                "base_url": cls.BASE_URL,
                "start_year": cls.START_YEAR,
                "end_year": cls.END_YEAR,
                "max_workers": cls.MAX_WORKERS,
                "delay_between_requests": cls.DELAY_BETWEEN_REQUESTS,
                "request_timeout": cls.REQUEST_TIMEOUT,
                "max_retries": cls.MAX_RETRIES
            },
            "directories": {
                "raw_data": str(cls.get_raw_data_dir()),
                "processed_data": str(cls.get_processed_data_dir()),
                "logs": str(cls.get_logs_dir())
            },
            "file_settings": {
                "supported_extensions": cls.SUPPORTED_EXTENSIONS,
                "min_file_size": cls.MIN_FILE_SIZE,
                "max_file_size": cls.MAX_FILE_SIZE,
                "year_patterns": cls.YEAR_PATTERNS,
                "exclude_patterns": cls.EXCLUDE_PATTERNS
            },
            "performance": cls.PERFORMANCE_CONFIG,
            "ethics": cls.ETHICS_CONFIG,
            "logging": cls.LOGGING_CONFIG
        }
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """
        Validiert die Konfiguration
        
        Returns:
            Liste von Validierungsfehlern (leer wenn alles OK)
        """
        errors = []
        
        # Jahr-Validierung
        if cls.START_YEAR > cls.END_YEAR:
            errors.append("Start-Jahr muss kleiner oder gleich End-Jahr sein")
        
        if cls.START_YEAR < 2000 or cls.END_YEAR > 2030:
            errors.append("Jahre müssen zwischen 2000 und 2030 liegen")
        
        # Worker-Validierung
        if cls.MAX_WORKERS < 1 or cls.MAX_WORKERS > 20:
            errors.append("Max-Workers muss zwischen 1 und 20 liegen")
        
        # Delay-Validierung
        if cls.DELAY_BETWEEN_REQUESTS < 0:
            errors.append("Delay zwischen Requests muss >= 0 sein")
        
        # Timeout-Validierung
        if cls.REQUEST_TIMEOUT < 1:
            errors.append("Request-Timeout muss >= 1 Sekunde sein")
        
        return errors


class EnvironmentScraperConfig:
    """Konfiguration basierend auf Umgebungsvariablen"""
    
    @staticmethod
    def get_base_url() -> str:
        """Gibt die Basis-URL aus Umgebungsvariablen zurück"""
        return os.getenv("SCRAPER_BASE_URL", ScraperConfig.BASE_URL)
    
    @staticmethod
    def get_start_year() -> int:
        """Gibt das Start-Jahr aus Umgebungsvariablen zurück"""
        return int(os.getenv("SCRAPER_START_YEAR", str(ScraperConfig.START_YEAR)))
    
    @staticmethod
    def get_end_year() -> int:
        """Gibt das End-Jahr aus Umgebungsvariablen zurück"""
        return int(os.getenv("SCRAPER_END_YEAR", str(ScraperConfig.END_YEAR)))
    
    @staticmethod
    def get_max_workers() -> int:
        """Gibt die maximale Anzahl Worker aus Umgebungsvariablen zurück"""
        return int(os.getenv("SCRAPER_MAX_WORKERS", str(ScraperConfig.MAX_WORKERS)))
    
    @staticmethod
    def get_delay() -> float:
        """Gibt die Verzögerung aus Umgebungsvariablen zurück"""
        return float(os.getenv("SCRAPER_DELAY", str(ScraperConfig.DELAY_BETWEEN_REQUESTS)))
    
    @staticmethod
    def get_timeout() -> int:
        """Gibt das Timeout aus Umgebungsvariablen zurück"""
        return int(os.getenv("SCRAPER_TIMEOUT", str(ScraperConfig.REQUEST_TIMEOUT)))
    
    @staticmethod
    def get_log_level() -> str:
        """Gibt das Log-Level aus Umgebungsvariablen zurück"""
        return os.getenv("SCRAPER_LOG_LEVEL", "INFO")
    
    @staticmethod
    def is_parallel_enabled() -> bool:
        """Prüft ob parallele Downloads aktiviert sind"""
        return os.getenv("SCRAPER_PARALLEL", "true").lower() == "true"
    
    @staticmethod
    def get_user_agent() -> str:
        """Gibt den User-Agent aus Umgebungsvariablen zurück"""
        return os.getenv("SCRAPER_USER_AGENT", ScraperConfig.USER_AGENT)


class ScrapingTargets:
    """Definiert verschiedene Scraping-Ziele"""
    
    # Verschiedene Zeiträume
    TIME_PERIODS = {
        "full_range": (2012, 2021),
        "recent": (2018, 2021),
        "historical": (2012, 2017),
        "single_year_2020": (2020, 2020),
        "single_year_2021": (2021, 2021),
    }
    
    # Verschiedene URL-Endpunkte (falls verfügbar)
    URL_ENDPOINTS = {
        "westminhall": "https://www.theyworkforyou.com/pwdata/scrapedxml/westminhall/",
        "scotland": "https://www.theyworkforyou.com/pwdata/scrapedxml/scotland/",
        "wales": "https://www.theyworkforyou.com/pwdata/scrapedxml/wales/",
        "ni": "https://www.theyworkforyou.com/pwdata/scrapedxml/ni/",
    }
    
    @classmethod
    def get_target_config(cls, target_name: str) -> Dict[str, Any]:
        """
        Gibt Konfiguration für ein bestimmtes Ziel zurück
        
        Args:
            target_name: Name des Ziels
            
        Returns:
            Konfiguration für das Ziel
        """
        if target_name not in cls.TIME_PERIODS:
            raise ValueError(f"Unbekanntes Ziel: {target_name}")
        
        start_year, end_year = cls.TIME_PERIODS[target_name]
        
        return {
            "start_year": start_year,
            "end_year": end_year,
            "base_url": cls.URL_ENDPOINTS.get("westminhall", ScraperConfig.BASE_URL),
            "description": f"Scraping für {target_name} ({start_year}-{end_year})"
        }


# Globale Konfigurationsinstanzen
scraper_config = ScraperConfig()
env_scraper_config = EnvironmentScraperConfig()
scraping_targets = ScrapingTargets()

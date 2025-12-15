"""
Datenbank-Management System für XML-Scraping und -Parsing
"""

from .managers import DatabaseManager, create_database_manager, quick_query
from .config import DatabaseConfig, EnvironmentConfig, config, env_config
from .utils import (
    DataConverter,
    DataValidator,
    DataProcessor,
    FileManager,
    PerformanceMonitor,
    quick_convert,
    validate_and_clean
)

__version__ = "1.0.0"
__author__ = "Parliament Data Processing System"

__all__ = [
    # Managers
    'DatabaseManager',
    'create_database_manager', 
    'quick_query',
    
    # Config
    'DatabaseConfig',
    'EnvironmentConfig',
    'config',
    'env_config',
    
    # Utils
    'DataConverter',
    'DataValidator',
    'DataProcessor', 
    'FileManager',
    'PerformanceMonitor',
    'quick_convert',
    'validate_and_clean'
]

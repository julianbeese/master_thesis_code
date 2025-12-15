"""
XML-Scraper Modul für theyworkforyou.com
Integriert mit dem Datenbank-Management-System
"""

from .xml_scraper import XMLScraper
from .config import (
    ScraperConfig,
    EnvironmentScraperConfig,
    ScrapingTargets,
    scraper_config,
    env_scraper_config,
    scraping_targets
)
from .utils import (
    FileValidator,
    ProgressTracker,
    URLManager,
    DataIntegrityChecker,
    BatchProcessor,
    quick_file_validation,
    get_directory_summary
)

__version__ = "1.0.0"
__author__ = "Parliament Data Processing System"

__all__ = [
    # Main Scraper
    'XMLScraper',
    
    # Config
    'ScraperConfig',
    'EnvironmentScraperConfig',
    'ScrapingTargets',
    'scraper_config',
    'env_scraper_config',
    'scraping_targets',
    
    # Utils
    'FileValidator',
    'ProgressTracker',
    'URLManager',
    'DataIntegrityChecker',
    'BatchProcessor',
    'quick_file_validation',
    'get_directory_summary'
]

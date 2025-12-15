"""
Konfigurationsmodul für XML-Scraper
"""

from .scraper_config import (
    ScraperConfig, 
    EnvironmentScraperConfig, 
    ScrapingTargets,
    scraper_config,
    env_scraper_config,
    scraping_targets
)

__all__ = [
    'ScraperConfig',
    'EnvironmentScraperConfig', 
    'ScrapingTargets',
    'scraper_config',
    'env_scraper_config',
    'scraping_targets'
]

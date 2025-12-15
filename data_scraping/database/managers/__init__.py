"""
Manager-Module für Datenbank-Management
"""

from .database_manager import DatabaseManager, create_database_manager, quick_query

__all__ = ['DatabaseManager', 'create_database_manager', 'quick_query']

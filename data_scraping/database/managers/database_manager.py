"""
Datenbank-Manager für XML-Scraping und -Parsing Projekt
Verwendet DuckDB als Hauptdatenbank mit Polars, PyArrow und Pandas für Datenverarbeitung
"""

import duckdb
import polars as pl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import logging
from datetime import datetime


class DatabaseManager:
    """
    Hauptklasse für Datenbank-Management mit DuckDB
    Unterstützt Polars, Pandas und PyArrow für Datenverarbeitung
    """
    
    def __init__(self, db_path: str = "database/data/debates_data.db"):
        """
        Initialisiert den DatabaseManager
        
        Args:
            db_path: Pfad zur DuckDB-Datei
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # DuckDB-Verbindung
        self.connection = None
        self._connect()
        
        # Logging setup
        self._setup_logging()
        
    def _connect(self):
        """Stellt Verbindung zur DuckDB her"""
        try:
            self.connection = duckdb.connect(str(self.db_path))
            if hasattr(self, 'logger'):
                self.logger.info(f"Verbindung zu DuckDB hergestellt: {self.db_path}")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Fehler beim Verbinden zur Datenbank: {e}")
            raise
    
    def _setup_logging(self):
        """Setup für Logging"""
        log_dir = Path("database/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"db_manager_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> Any:
        """
        Führt SQL-Query aus
        
        Args:
            query: SQL-Query String
            parameters: Optional Parameter für prepared statements
            
        Returns:
            Query-Ergebnis
        """
        try:
            if parameters:
                result = self.connection.execute(query, parameters).fetchall()
            else:
                result = self.connection.execute(query).fetchall()
            
            self.logger.info(f"Query erfolgreich ausgeführt: {query[:100]}...")
            return result
        except Exception as e:
            self.logger.error(f"Fehler bei Query-Ausführung: {e}")
            raise
    
    def execute_query_as_dataframe(self, query: str, parameters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Führt Query aus und gibt Ergebnis als Pandas DataFrame zurück
        
        Args:
            query: SQL-Query
            parameters: Optional: Query-Parameter
            
        Returns:
            Pandas DataFrame
        """
        try:
            if parameters:
                result = self.connection.execute(query, parameters).df()
            else:
                result = self.connection.execute(query).df()
            
            self.logger.info(f"Query als DataFrame ausgeführt: {query[:100]}...")
            return result
        except Exception as e:
            self.logger.error(f"Fehler bei DataFrame-Query: {e}")
            raise
    
    def create_table_from_polars(self, df: pl.DataFrame, table_name: str, 
                                if_exists: str = "replace") -> None:
        """
        Erstellt Tabelle aus Polars DataFrame
        
        Args:
            df: Polars DataFrame
            table_name: Name der Tabelle
            if_exists: Verhalten wenn Tabelle existiert ('replace', 'append', 'fail')
        """
        try:
            # Polars zu DuckDB
            self.connection.register('temp_df', df.to_pandas())
            
            if if_exists == "replace":
                self.connection.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            self.connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df")
            self.logger.info(f"Tabelle '{table_name}' aus Polars DataFrame erstellt")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen der Tabelle {table_name}: {e}")
            raise
    
    def create_table_from_pandas(self, df: pd.DataFrame, table_name: str, 
                                if_exists: str = "replace") -> None:
        """
        Erstellt Tabelle aus Pandas DataFrame
        
        Args:
            df: Pandas DataFrame
            table_name: Name der Tabelle
            if_exists: Verhalten wenn Tabelle existiert
        """
        try:
            self.connection.register('temp_df', df)
            
            if if_exists == "replace":
                self.connection.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            self.connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df")
            self.logger.info(f"Tabelle '{table_name}' aus Pandas DataFrame erstellt")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen der Tabelle {table_name}: {e}")
            raise
    
    def read_table_to_polars(self, table_name: str) -> pl.DataFrame:
        """
        Liest Tabelle als Polars DataFrame
        
        Args:
            table_name: Name der Tabelle
            
        Returns:
            Polars DataFrame
        """
        try:
            result = self.connection.execute(f"SELECT * FROM {table_name}").df()
            return pl.from_pandas(result)
        except Exception as e:
            self.logger.error(f"Fehler beim Lesen der Tabelle {table_name}: {e}")
            raise
    
    def read_table_to_pandas(self, table_name: str) -> pd.DataFrame:
        """
        Liest Tabelle als Pandas DataFrame
        
        Args:
            table_name: Name der Tabelle
            
        Returns:
            Pandas DataFrame
        """
        try:
            return self.connection.execute(f"SELECT * FROM {table_name}").df()
        except Exception as e:
            self.logger.error(f"Fehler beim Lesen der Tabelle {table_name}: {e}")
            raise
    
    def save_to_parquet(self, table_name: str, file_path: str) -> None:
        """
        Speichert Tabelle als Parquet-Datei
        
        Args:
            table_name: Name der Tabelle
            file_path: Pfad für Parquet-Datei
        """
        try:
            df = self.read_table_to_pandas(table_name)
            df.to_parquet(file_path, engine='pyarrow')
            self.logger.info(f"Tabelle '{table_name}' als Parquet gespeichert: {file_path}")
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern als Parquet: {e}")
            raise
    
    def load_from_parquet(self, file_path: str, table_name: str) -> None:
        """
        Lädt Parquet-Datei in Tabelle
        
        Args:
            file_path: Pfad zur Parquet-Datei
            table_name: Name der Zieltabelle
        """
        try:
            df = pd.read_parquet(file_path, engine='pyarrow')
            self.create_table_from_pandas(df, table_name)
            self.logger.info(f"Parquet-Datei geladen: {file_path} -> {table_name}")
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Parquet-Datei: {e}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Gibt Informationen über eine Tabelle zurück
        
        Args:
            table_name: Name der Tabelle
            
        Returns:
            Dictionary mit Tabellen-Informationen
        """
        try:
            # Tabellen-Schema
            schema = self.connection.execute(f"DESCRIBE {table_name}").fetchall()
            
            # Anzahl Zeilen
            count = self.connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
            return {
                'table_name': table_name,
                'schema': schema,
                'row_count': count,
                'columns': [col[0] for col in schema]
            }
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Tabellen-Info: {e}")
            raise
    
    def list_tables(self) -> List[str]:
        """
        Listet alle Tabellen in der Datenbank auf
        
        Returns:
            Liste der Tabellennamen
        """
        try:
            result = self.connection.execute("SHOW TABLES").fetchall()
            return [table[0] for table in result]
        except Exception as e:
            self.logger.error(f"Fehler beim Auflisten der Tabellen: {e}")
            raise
    
    def close(self):
        """Schließt die Datenbankverbindung"""
        if self.connection:
            self.connection.close()
            self.logger.info("Datenbankverbindung geschlossen")
    
    def __enter__(self):
        """Context Manager Entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Exit"""
        self.close()


# Convenience-Funktionen für häufige Operationen
def create_database_manager(db_path: str = "database/data/parliament_data.db") -> DatabaseManager:
    """
    Factory-Funktion für DatabaseManager
    
    Args:
        db_path: Pfad zur Datenbank
        
    Returns:
        DatabaseManager Instanz
    """
    return DatabaseManager(db_path)


def quick_query(query: str, db_path: str = "database/data/parliament_data.db") -> Any:
    """
    Führt schnelle Query aus ohne permanente Verbindung
    
    Args:
        query: SQL-Query
        db_path: Pfad zur Datenbank
        
    Returns:
        Query-Ergebnis
    """
    with DatabaseManager(db_path) as db:
        return db.execute_query(query)

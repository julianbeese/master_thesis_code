"""
Debates Database Manager für DuckDB
Spezialisierter Manager für debates-Daten mit ML/NLP-Optimierungen
"""

try:
    import duckdb
    import polars as pl
    import pandas as pd
except ImportError:
    print("DuckDB, Polars oder Pandas nicht verfügbar. Installiere mit: pip install duckdb polars pandas")
    raise
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
import logging
from datetime import datetime
import sys

# Füge das parser-Verzeichnis zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent.parent / "parser"))

try:
    from debates_parser import DebatesXMLParser, DebatesDataProcessor
    from database_manager import DatabaseManager
except ImportError:
    # Fallback für direkte Ausführung
    sys.path.append(str(Path(__file__).parent))
    from debates_parser import DebatesXMLParser, DebatesDataProcessor
    from database_manager import DatabaseManager


class DebatesDatabaseManager(DatabaseManager):
    """
    Spezialisierter DatabaseManager für debates-Daten
    Erweitert den Basis-DatabaseManager um debates-spezifische Funktionen
    """
    
    def __init__(self, db_path: str = "database/data/debates_data.db"):
        """
        Initialisiert den DebatesDatabaseManager
        
        Args:
            db_path: Pfad zur DuckDB-Datei für debates
        """
        super().__init__(db_path)
        
        # Parser und Processor
        self.parser = DebatesXMLParser(self.logger)
        self.processor = DebatesDataProcessor(self.logger)
        
        # Tabellen-Schema (nur bei Bedarf erstellen)
        # self._create_tables()
    
    def _create_tables(self):
        """Erstellt die debates-spezifischen Tabellen"""
        try:
            # Sessions-Tabelle
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id VARCHAR PRIMARY KEY,
                    date DATE,
                    session_type VARCHAR,
                    total_debates INTEGER,
                    total_speeches INTEGER,
                    total_words INTEGER,
                    unique_speakers INTEGER,
                    duration_hours DOUBLE,
                    processing_timestamp TIMESTAMP
                )
            """)
            
            # Debatten-Tabelle
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS debates (
                    debate_id VARCHAR PRIMARY KEY,
                    session_id VARCHAR,
                    title TEXT,
                    heading_id VARCHAR,
                    time VARCHAR,
                    url VARCHAR,
                    column_number INTEGER,
                    total_speeches INTEGER,
                    total_words INTEGER,
                    unique_speakers INTEGER,
                    duration_minutes DOUBLE
                )
            """)
            
            # Sprecher-Tabelle
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS speakers (
                    speaker_name VARCHAR PRIMARY KEY,
                    person_id VARCHAR,
                    title VARCHAR,
                    party VARCHAR,
                    constituency VARCHAR,
                    first_appearance DATE,
                    total_speeches INTEGER DEFAULT 0,
                    total_words INTEGER DEFAULT 0
                )
            """)
            
            # Reden-Tabelle
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS speeches (
                    speech_id VARCHAR PRIMARY KEY,
                    debate_id VARCHAR,
                    session_id VARCHAR,
                    speaker_name VARCHAR,
                    speaker_person_id VARCHAR,
                    speaker_title VARCHAR,
                    speaker_party VARCHAR,
                    speaker_constituency VARCHAR,
                    time VARCHAR,
                    column_number INTEGER,
                    url VARCHAR,
                    word_count INTEGER,
                    char_count INTEGER,
                    is_intervention BOOLEAN,
                    paragraph_count INTEGER,
                    full_text TEXT
                )
            """)
            
            # Absätze-Tabelle
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS paragraphs (
                    paragraph_id VARCHAR PRIMARY KEY,
                    speech_id VARCHAR,
                    text TEXT,
                    class_name VARCHAR,
                    motion_text BOOLEAN,
                    word_count INTEGER
                )
            """)
            
            # ML-Features-Tabelle
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS ml_features (
                    speech_id VARCHAR PRIMARY KEY,
                    speaker_name VARCHAR,
                    speaker_person_id VARCHAR,
                    word_count INTEGER,
                    char_count INTEGER,
                    is_intervention BOOLEAN,
                    paragraph_count INTEGER,
                    avg_chars_per_word DOUBLE,
                    avg_words_per_paragraph DOUBLE,
                    speech_length_category VARCHAR,
                    anonymous_speaker BOOLEAN,
                    unknown_person_id BOOLEAN
                )
            """)
            
            # Indizes für bessere Performance
            self._create_indexes()
            
            self.logger.info("Debates-Tabellen erfolgreich erstellt")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen der Tabellen: {e}")
            raise
    
    def _create_indexes(self):
        """Erstellt Indizes für bessere Query-Performance"""
        try:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(date)",
                "CREATE INDEX IF NOT EXISTS idx_debates_session_id ON debates(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_speeches_debate_id ON speeches(debate_id)",
                "CREATE INDEX IF NOT EXISTS idx_speeches_speaker ON speeches(speaker_name)",
                "CREATE INDEX IF NOT EXISTS idx_speeches_date ON speeches(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_paragraphs_speech_id ON paragraphs(speech_id)",
                "CREATE INDEX IF NOT EXISTS idx_ml_features_speaker ON ml_features(speaker_name)",
                "CREATE INDEX IF NOT EXISTS idx_ml_features_length ON ml_features(speech_length_category)"
            ]
            
            for index_sql in indexes:
                self.connection.execute(index_sql)
            
            self.logger.info("Indizes erfolgreich erstellt")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen der Indizes: {e}")
            raise
    
    def import_xml_directory(self, directory_path: Union[str, Path], 
                           max_files: Optional[int] = None,
                           batch_size: int = 100) -> Dict[str, int]:
        """
        Importiert alle XML-Dateien aus einem Verzeichnis
        
        Args:
            directory_path: Pfad zum Verzeichnis mit XML-Dateien
            max_files: Maximale Anzahl zu verarbeitender Dateien
            batch_size: Batch-Größe für Datenbank-Inserts
            
        Returns:
            Dictionary mit Import-Statistiken
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Verzeichnis nicht gefunden: {directory_path}")
        
        self.logger.info(f"Starte Import aus {directory_path}")
        
        # XML-Dateien parsen
        sessions = self.parser.parse_directory(directory_path, max_files)
        
        if not sessions:
            self.logger.warning("Keine Sitzungen zum Importieren gefunden")
            return {'sessions': 0, 'debates': 0, 'speeches': 0, 'speakers': 0, 'paragraphs': 0}
        
        # Zu DataFrames konvertieren
        dataframes = self.processor.sessions_to_dataframes(sessions)
        
        # In Datenbank importieren
        import_stats = self._import_dataframes(dataframes, batch_size)
        
        # ML-Features erstellen und importieren
        if 'speeches' in dataframes:
            ml_features = self.processor.create_ml_features(dataframes['speeches'])
            self.create_table_from_polars(ml_features, 'ml_features', 'replace')
            import_stats['ml_features'] = len(ml_features)
        
        # Sprecher-Statistiken aktualisieren
        self._update_speaker_statistics()
        
        self.logger.info(f"Import abgeschlossen: {import_stats}")
        return import_stats
    
    def _import_dataframes(self, dataframes: Dict[str, pl.DataFrame], 
                          batch_size: int) -> Dict[str, int]:
        """Importiert DataFrames in die Datenbank"""
        import_stats = {}
        
        # Reihenfolge der Tabellen (wegen Foreign Keys)
        table_order = ['sessions', 'speakers', 'debates', 'speeches', 'paragraphs']
        all_tables = ['sessions', 'speakers', 'debates', 'speeches', 'paragraphs', 'ml_features']
        
        # Alle Tabellen löschen (in umgekehrter Reihenfolge wegen Foreign Keys)
        for table_name in reversed(all_tables):
            try:
                self.connection.execute(f"DROP TABLE IF EXISTS {table_name}")
            except Exception as e:
                self.logger.warning(f"Konnte Tabelle {table_name} nicht löschen: {e}")
        
        # Tabellen neu erstellen
        self._create_tables()
        
        # Foreign Key Constraints sind bereits entfernt
        
        for table_name in table_order:
            if table_name in dataframes:
                df = dataframes[table_name]
                
                # Duplikate entfernen (außer bei speakers, da bereits dedupliziert)
                if table_name != 'speakers':
                    df = df.unique()
                
                # In Datenbank importieren
                # Registriere DataFrame temporär
                self.connection.register('temp_df', df.to_pandas())
                
                # Füge Daten zur bestehenden Tabelle hinzu
                if table_name == 'speakers':
                    # Speakers-Tabelle hat zusätzliche Spalten
                    self.connection.execute(f"""
                        INSERT INTO {table_name} (speaker_name, person_id, title, party, constituency)
                        SELECT speaker_name, person_id, title, party, constituency FROM temp_df
                    """)
                else:
                    self.connection.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
                import_stats[table_name] = len(df)
                
                self.logger.info(f"Tabelle '{table_name}' importiert: {len(df)} Zeilen")
        
        # Foreign Key Constraints sind bereits entfernt
        
        return import_stats
    
    def _update_speaker_statistics(self):
        """Aktualisiert Sprecher-Statistiken"""
        try:
            # Erste Erscheinung und Statistiken berechnen
            self.connection.execute("""
                UPDATE speakers 
                SET 
                    first_appearance = (
                        SELECT MIN(s.date) 
                        FROM sessions s 
                        JOIN speeches sp ON s.session_id = sp.session_id 
                        WHERE sp.speaker_name = speakers.speaker_name
                    ),
                    total_speeches = (
                        SELECT COUNT(*) 
                        FROM speeches 
                        WHERE speaker_name = speakers.speaker_name
                    ),
                    total_words = (
                        SELECT SUM(word_count) 
                        FROM speeches 
                        WHERE speaker_name = speakers.speaker_name
                    )
            """)
            
            self.logger.info("Sprecher-Statistiken aktualisiert")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Aktualisieren der Sprecher-Statistiken: {e}")
    
    def get_debates_by_date_range(self, start_date: str, end_date: str) -> pl.DataFrame:
        """
        Gibt Debatten in einem Datumsbereich zurück
        
        Args:
            start_date: Startdatum (YYYY-MM-DD)
            end_date: Enddatum (YYYY-MM-DD)
            
        Returns:
            Polars DataFrame mit Debatten
        """
        query = """
            SELECT d.*, s.date, s.session_type
            FROM debates d
            JOIN sessions s ON d.session_id = s.session_id
            WHERE s.date BETWEEN ? AND ?
            ORDER BY s.date, d.column_number
        """
        
        result = self.connection.execute(query, [start_date, end_date]).df()
        return pl.from_pandas(result)
    
    def get_speaker_statistics(self, speaker_name: Optional[str] = None) -> pl.DataFrame:
        """
        Gibt Sprecher-Statistiken zurück
        
        Args:
            speaker_name: Optional: Name eines bestimmten Sprechers
            
        Returns:
            Polars DataFrame mit Sprecher-Statistiken
        """
        if speaker_name:
            query = """
                SELECT s.*, 
                       COUNT(sp.speech_id) as total_speeches,
                       SUM(sp.word_count) as total_words,
                       AVG(sp.word_count) as avg_words_per_speech
                FROM speakers s
                LEFT JOIN speeches sp ON s.speaker_name = sp.speaker_name
                WHERE s.speaker_name = ?
                GROUP BY s.speaker_name, s.person_id, s.title, s.party, s.constituency
            """
            result = self.connection.execute(query, [speaker_name]).df()
        else:
            query = """
                SELECT s.*, 
                       COUNT(sp.speech_id) as total_speeches,
                       SUM(sp.word_count) as total_words,
                       AVG(sp.word_count) as avg_words_per_speech
                FROM speakers s
                LEFT JOIN speeches sp ON s.speaker_name = sp.speaker_name
                GROUP BY s.speaker_name, s.person_id, s.title, s.party, s.constituency
                ORDER BY total_speeches DESC
            """
            result = self.connection.execute(query).df()
        
        return pl.from_pandas(result)
    
    def get_ml_dataset(self, min_word_count: int = 10, 
                      max_word_count: int = 10000) -> pl.DataFrame:
        """
        Gibt ML-Dataset zurück
        
        Args:
            min_word_count: Minimale Wortanzahl pro Rede
            max_word_count: Maximale Wortanzahl pro Rede
            
        Returns:
            Polars DataFrame mit ML-Features
        """
        query = """
            SELECT mf.*, s.date, s.session_type, d.title as debate_title
            FROM ml_features mf
            JOIN speeches sp ON mf.speech_id = sp.speech_id
            JOIN sessions s ON sp.session_id = s.session_id
            JOIN debates d ON sp.debate_id = d.debate_id
            WHERE mf.word_count BETWEEN ? AND ?
            AND mf.speaker_name IS NOT NULL
            AND mf.speaker_name != ''
        """
        
        result = self.connection.execute(query, [min_word_count, max_word_count]).df()
        return pl.from_pandas(result)
    
    def get_debate_topics(self, limit: int = 100) -> pl.DataFrame:
        """
        Gibt die häufigsten Debatten-Themen zurück
        
        Args:
            limit: Maximale Anzahl Ergebnisse
            
        Returns:
            Polars DataFrame mit Debatten-Themen
        """
        query = """
            SELECT 
                title,
                COUNT(*) as debate_count,
                SUM(total_speeches) as total_speeches,
                SUM(total_words) as total_words,
                AVG(duration_minutes) as avg_duration_minutes
            FROM debates
            WHERE title IS NOT NULL AND title != ''
            GROUP BY title
            ORDER BY debate_count DESC
            LIMIT ?
        """
        
        result = self.connection.execute(query, [limit]).df()
        return pl.from_pandas(result)
    
    def export_to_parquet(self, output_dir: Union[str, Path]):
        """
        Exportiert alle Tabellen als Parquet-Dateien
        
        Args:
            output_dir: Ausgabeverzeichnis
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tables = ['sessions', 'debates', 'speakers', 'speeches', 'paragraphs', 'ml_features']
        
        for table in tables:
            try:
                df = self.read_table_to_pandas(table)
                output_file = output_dir / f"{table}.parquet"
                df.to_parquet(output_file, engine='pyarrow')
                self.logger.info(f"Tabelle '{table}' als Parquet exportiert: {output_file}")
            except Exception as e:
                self.logger.error(f"Fehler beim Exportieren der Tabelle '{table}': {e}")
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Gibt umfassende Datenbank-Statistiken zurück"""
        stats = {}
        
        # Tabellen-Statistiken
        tables = ['sessions', 'debates', 'speakers', 'speeches', 'paragraphs', 'ml_features']
        for table in tables:
            try:
                count = self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[f"{table}_count"] = count
            except:
                stats[f"{table}_count"] = 0
        
        # Zeitraum
        try:
            date_range = self.connection.execute("""
                SELECT MIN(date) as start_date, MAX(date) as end_date 
                FROM sessions
            """).fetchone()
            stats['date_range'] = date_range
        except:
            stats['date_range'] = None
        
        # Top-Sprecher
        try:
            top_speakers = self.connection.execute("""
                SELECT speaker_name, COUNT(*) as speech_count
                FROM speeches
                WHERE speaker_name IS NOT NULL AND speaker_name != ''
                GROUP BY speaker_name
                ORDER BY speech_count DESC
                LIMIT 10
            """).fetchall()
            stats['top_speakers'] = top_speakers
        except:
            stats['top_speakers'] = []
        
        return stats


def main():
    """Hauptfunktion für Tests"""
    # Setup Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Database Manager initialisieren
    db_manager = DebatesDatabaseManager()
    
    try:
        # Test-Import (nur wenige Dateien)
        debates_dir = Path("scraper/data/debates")
        if debates_dir.exists():
            stats = db_manager.import_xml_directory(debates_dir, max_files=5)
            print(f"Import-Statistiken: {stats}")
            
            # Datenbank-Statistiken
            db_stats = db_manager.get_database_statistics()
            print(f"Datenbank-Statistiken: {db_stats}")
            
            # Test-Queries
            ml_dataset = db_manager.get_ml_dataset()
            print(f"ML-Dataset: {ml_dataset.shape}")
            
            top_speakers = db_manager.get_speaker_statistics()
            print(f"Top-Sprecher: {top_speakers.head()}")
        
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()

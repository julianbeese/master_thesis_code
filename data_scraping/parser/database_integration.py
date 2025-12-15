"""
Datenbank-Integration für geparste Parlamentsdaten
Speichert strukturierte Daten in DuckDB für ML-Modelle
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from database import DatabaseManager, config as db_config
from parser.xml_parser import ParliamentXMLParser, extract_text_for_ml
from parser.models.parliament_models import ParliamentSession


class ParliamentDatabaseIntegration:
    """
    Integration zwischen XML-Parser und Datenbank
    Speichert strukturierte Parlamentsdaten für ML-Modelle
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialisiert die Integration
        
        Args:
            db_manager: Optional: Bestehender DatabaseManager
        """
        self.db_manager = db_manager or DatabaseManager()
        self.logger = logging.getLogger(__name__)
        
        # Setup
        db_config.setup_directories()
    
    def create_parliament_tables(self) -> None:
        """Erstellt alle Tabellen für Parlamentsdaten"""
        
        # 1. Sessions-Tabelle
        sessions_schema = """
        CREATE TABLE IF NOT EXISTS parliament_sessions (
            session_id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            session_type TEXT NOT NULL,
            total_debates INTEGER,
            total_speeches INTEGER,
            total_words INTEGER,
            unique_speakers INTEGER,
            duration_hours REAL,
            processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # 2. Debates-Tabelle
        debates_schema = """
        CREATE TABLE IF NOT EXISTS parliament_debates (
            debate_id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            title TEXT NOT NULL,
            heading_id TEXT,
            time TEXT,
            url TEXT,
            column_number INTEGER,
            total_speeches INTEGER,
            total_words INTEGER,
            unique_speakers INTEGER,
            duration_minutes REAL,
            FOREIGN KEY (session_id) REFERENCES parliament_sessions(session_id)
        )
        """
        
        # 3. Speakers-Tabelle
        speakers_schema = """
        CREATE TABLE IF NOT EXISTS parliament_speakers (
            speaker_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            person_id TEXT,
            party TEXT,
            constituency TEXT,
            title TEXT,
            total_speeches INTEGER DEFAULT 0,
            total_words INTEGER DEFAULT 0,
            first_speech_date TEXT,
            last_speech_date TEXT
        )
        """
        
        # 4. Speeches-Tabelle (Haupttabelle für ML)
        speeches_schema = """
        CREATE TABLE IF NOT EXISTS parliament_speeches (
            speech_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            debate_id INTEGER NOT NULL,
            speaker_id INTEGER NOT NULL,
            speaker_name TEXT NOT NULL,
            speaker_person_id TEXT,
            speaker_title TEXT,
            time TEXT,
            column_number INTEGER,
            url TEXT,
            text TEXT NOT NULL,
            word_count INTEGER,
            char_count INTEGER,
            is_intervention BOOLEAN,
            paragraph_count INTEGER,
            processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES parliament_sessions(session_id),
            FOREIGN KEY (debate_id) REFERENCES parliament_debates(debate_id),
            FOREIGN KEY (speaker_id) REFERENCES parliament_speakers(speaker_id)
        )
        """
        
        # 5. Paragraphs-Tabelle
        paragraphs_schema = """
        CREATE TABLE IF NOT EXISTS parliament_paragraphs (
            paragraph_id INTEGER PRIMARY KEY,
            speech_id TEXT NOT NULL,
            text TEXT NOT NULL,
            paragraph_pid TEXT,
            class_name TEXT,
            motion_text BOOLEAN,
            word_count INTEGER,
            position_in_speech INTEGER,
            FOREIGN KEY (speech_id) REFERENCES parliament_speeches(speech_id)
        )
        """
        
        # Tabellen erstellen
        schemas = [
            ("parliament_sessions", sessions_schema),
            ("parliament_debates", debates_schema),
            ("parliament_speakers", speakers_schema),
            ("parliament_speeches", speeches_schema),
            ("parliament_paragraphs", paragraphs_schema)
        ]
        
        for table_name, schema in schemas:
            self.db_manager.execute_query(schema)
            self.logger.info(f"Tabelle '{table_name}' erstellt")
    
    def store_parliament_session(self, session: ParliamentSession) -> Dict[str, Any]:
        """
        Speichert eine ParliamentSession in der Datenbank
        
        Args:
            session: ParliamentSession-Objekt
            
        Returns:
            Dictionary mit Speicher-Ergebnissen
        """
        try:
            # 1. Session speichern
            session_data = (
                session.session_id,
                session.date,
                session.session_type,
                session.total_debates,
                session.total_speeches,
                session.total_words,
                session.unique_speakers,
                session.duration_hours
            )
            
            session_query = """
            INSERT OR REPLACE INTO parliament_sessions 
            (session_id, date, session_type, total_debates, total_speeches, 
             total_words, unique_speakers, duration_hours)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.db_manager.execute_query(session_query, session_data)
            
            # 2. Sprecher sammeln und speichern
            speaker_mapping = self._store_speakers(session)
            
            # 3. Debatten speichern
            debate_mapping = self._store_debates(session)
            
            # 4. Reden speichern
            speech_count = self._store_speeches(session, speaker_mapping, debate_mapping)
            
            # 5. Absätze speichern
            paragraph_count = self._store_paragraphs(session)
            
            self.logger.info(
                f"Session '{session.session_id}' gespeichert: "
                f"{len(speaker_mapping)} Sprecher, {len(debate_mapping)} Debatten, "
                f"{speech_count} Reden, {paragraph_count} Absätze"
            )
            
            return {
                'success': True,
                'session_id': session.session_id,
                'speakers': len(speaker_mapping),
                'debates': len(debate_mapping),
                'speeches': speech_count,
                'paragraphs': paragraph_count
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Session {session.session_id}: {e}")
            return {
                'success': False,
                'session_id': session.session_id,
                'error': str(e)
            }
    
    def _store_speakers(self, session: ParliamentSession) -> Dict[str, int]:
        """Speichert Sprecher und gibt Mapping zurück"""
        speaker_mapping = {}
        
        # Alle Sprecher sammeln
        all_speakers = set()
        for debate in session.debates:
            for speech in debate.speeches:
                all_speakers.add(speech.speaker)
        
        # Hole die nächste verfügbare Speaker-ID
        max_id_result = self.db_manager.execute_query("SELECT COALESCE(MAX(speaker_id), 0) FROM parliament_speakers")
        next_speaker_id = max_id_result[0][0] + 1 if max_id_result else 1
        
        for speaker in all_speakers:
            # Prüfe ob Sprecher bereits existiert
            existing_query = "SELECT speaker_id FROM parliament_speakers WHERE name = ?"
            existing_result = self.db_manager.execute_query(existing_query, (speaker.name,))
            
            if existing_result:
                speaker_id = existing_result[0][0]
            else:
                # Neuen Sprecher erstellen
                speaker_data = (
                    next_speaker_id,
                    speaker.name,
                    speaker.person_id,
                    speaker.party,
                    speaker.constituency,
                    speaker.title
                )
                
                insert_query = """
                INSERT INTO parliament_speakers 
                (speaker_id, name, person_id, party, constituency, title)
                VALUES (?, ?, ?, ?, ?, ?)
                """
                
                self.db_manager.execute_query(insert_query, speaker_data)
                speaker_id = next_speaker_id
                next_speaker_id += 1
            
            speaker_mapping[speaker.name] = speaker_id
        
        return speaker_mapping
    
    def _store_debates(self, session: ParliamentSession) -> Dict[str, int]:
        """Speichert Debatten und gibt Mapping zurück"""
        debate_mapping = {}
        
        # Hole die nächste verfügbare Debate-ID
        max_id_result = self.db_manager.execute_query("SELECT COALESCE(MAX(debate_id), 0) FROM parliament_debates")
        next_debate_id = max_id_result[0][0] + 1 if max_id_result else 1
        
        for debate in session.debates:
            debate_data = (
                next_debate_id,
                session.session_id,
                debate.title,
                debate.heading_id,
                debate.time,
                debate.url,
                debate.column_number,
                debate.total_speeches,
                debate.total_words,
                debate.unique_speakers,
                debate.duration_minutes
            )
            
            insert_query = """
            INSERT INTO parliament_debates 
            (debate_id, session_id, title, heading_id, time, url, column_number,
             total_speeches, total_words, unique_speakers, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.db_manager.execute_query(insert_query, debate_data)
            debate_mapping[debate.title] = next_debate_id
            next_debate_id += 1
        
        return debate_mapping
    
    def _store_speeches(self, session: ParliamentSession, 
                       speaker_mapping: Dict[str, int], 
                       debate_mapping: Dict[str, int]) -> int:
        """Speichert Reden und gibt Anzahl zurück"""
        speech_count = 0
        
        for debate in session.debates:
            debate_id = debate_mapping[debate.title]
            
            for speech in debate.speeches:
                speaker_id = speaker_mapping[speech.speaker.name]
                
                speech_data = (
                    speech.speech_id,
                    session.session_id,
                    debate_id,
                    speaker_id,
                    speech.speaker.name,
                    speech.speaker.person_id,
                    speech.speaker.title,
                    speech.time,
                    speech.column_number,
                    speech.url,
                    speech.full_text,
                    speech.word_count,
                    speech.char_count,
                    speech.is_intervention,
                    len(speech.paragraphs)
                )
                
                insert_query = """
                INSERT OR REPLACE INTO parliament_speeches 
                (speech_id, session_id, debate_id, speaker_id, speaker_name, 
                 speaker_person_id, speaker_title, time, column_number, url,
                 text, word_count, char_count, is_intervention, paragraph_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                self.db_manager.execute_query(insert_query, speech_data)
                speech_count += 1
        
        return speech_count
    
    def _store_paragraphs(self, session: ParliamentSession) -> int:
        """Speichert Absätze und gibt Anzahl zurück"""
        paragraph_count = 0
        
        # Hole die nächste verfügbare Paragraph-ID
        max_id_result = self.db_manager.execute_query("SELECT COALESCE(MAX(paragraph_id), 0) FROM parliament_paragraphs")
        next_paragraph_id = max_id_result[0][0] + 1 if max_id_result else 1
        
        for debate in session.debates:
            for speech in debate.speeches:
                for position, paragraph in enumerate(speech.paragraphs):
                    paragraph_data = (
                        next_paragraph_id,
                        speech.speech_id,
                        paragraph.text,
                        paragraph.paragraph_id,
                        paragraph.class_name,
                        paragraph.motion_text,
                        paragraph.word_count,
                        position + 1
                    )
                    
                    insert_query = """
                    INSERT INTO parliament_paragraphs 
                    (paragraph_id, speech_id, text, paragraph_pid, class_name, motion_text, 
                     word_count, position_in_speech)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    self.db_manager.execute_query(insert_query, paragraph_data)
                    paragraph_count += 1
                    next_paragraph_id += 1
        
        return paragraph_count
    
    def get_parliament_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken über gespeicherte Parlamentsdaten zurück"""
        try:
            # Basis-Statistiken
            stats = {}
            
            # Sessions
            session_count = self.db_manager.execute_query("SELECT COUNT(*) FROM parliament_sessions")[0][0]
            stats['total_sessions'] = session_count
            
            # Debatten
            debate_count = self.db_manager.execute_query("SELECT COUNT(*) FROM parliament_debates")[0][0]
            stats['total_debates'] = debate_count
            
            # Reden
            speech_count = self.db_manager.execute_query("SELECT COUNT(*) FROM parliament_speeches")[0][0]
            stats['total_speeches'] = speech_count
            
            # Sprecher
            speaker_count = self.db_manager.execute_query("SELECT COUNT(*) FROM parliament_speakers")[0][0]
            stats['total_speakers'] = speaker_count
            
            # Absätze
            paragraph_count = self.db_manager.execute_query("SELECT COUNT(*) FROM parliament_paragraphs")[0][0]
            stats['total_paragraphs'] = paragraph_count
            
            # Wörter
            word_result = self.db_manager.execute_query("SELECT SUM(word_count) FROM parliament_speeches")
            stats['total_words'] = word_result[0][0] or 0
            
            # Datum-Bereich
            date_result = self.db_manager.execute_query(
                "SELECT MIN(date), MAX(date) FROM parliament_sessions"
            )
            if date_result and date_result[0][0]:
                stats['date_range'] = f"{date_result[0][0]} - {date_result[0][1]}"
            else:
                stats['date_range'] = "Keine Daten"
            
            # Top-Sprecher
            top_speakers = self.db_manager.execute_query("""
                SELECT speaker_name, COUNT(*) as speech_count, SUM(word_count) as total_words
                FROM parliament_speeches 
                GROUP BY speaker_name 
                ORDER BY speech_count DESC 
                LIMIT 10
            """)
            stats['top_speakers'] = [
                {'name': row[0], 'speeches': row[1], 'words': row[2]} 
                for row in top_speakers
            ]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Statistiken: {e}")
            return {'error': str(e)}
    
    def export_ml_data(self, output_format: str = "parquet") -> Dict[str, Any]:
        """
        Exportiert Daten für ML-Modelle
        
        Args:
            output_format: Export-Format ('parquet', 'csv', 'json')
            
        Returns:
            Dictionary mit Export-Ergebnissen
        """
        try:
            # ML-Daten abrufen
            ml_query = """
            SELECT 
                s.session_id,
                s.date,
                s.session_type,
                d.title as debate_title,
                sp.speech_id,
                sp.speaker_name,
                sp.speaker_person_id,
                sp.speaker_title,
                sp.time,
                sp.text,
                sp.word_count,
                sp.char_count,
                sp.is_intervention,
                sp.paragraph_count,
                sp.processing_timestamp
            FROM parliament_speeches sp
            JOIN parliament_sessions s ON sp.session_id = s.session_id
            JOIN parliament_debates d ON sp.debate_id = d.debate_id
            ORDER BY s.date, sp.time
            """
            
            ml_data = self.db_manager.execute_query(ml_query)
            
            # Als DataFrame konvertieren
            import pandas as pd
            
            df = pd.DataFrame(ml_data, columns=[
                'session_id', 'date', 'session_type', 'debate_title',
                'speech_id', 'speaker_name', 'speaker_person_id', 'speaker_title',
                'time', 'text', 'word_count', 'char_count', 'is_intervention',
                'paragraph_count', 'processing_timestamp'
            ])
            
            # Export
            output_dir = Path("parser/data/exported")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if output_format.lower() == "parquet":
                output_file = output_dir / f"parliament_ml_data_{timestamp}.parquet"
                df.to_parquet(output_file, engine='pyarrow')
            elif output_format.lower() == "csv":
                output_file = output_dir / f"parliament_ml_data_{timestamp}.csv"
                df.to_csv(output_file, index=False)
            elif output_format.lower() == "json":
                output_file = output_dir / f"parliament_ml_data_{timestamp}.json"
                df.to_json(output_file, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {output_format}")
            
            self.logger.info(f"ML-Daten exportiert: {output_file} ({len(df)} Zeilen)")
            
            return {
                'success': True,
                'output_file': str(output_file),
                'row_count': len(df),
                'format': output_format
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Export: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def close(self):
        """Schließt die Datenbankverbindung"""
        if self.db_manager:
            self.db_manager.close()
    
    def __enter__(self):
        """Context Manager Entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Exit"""
        self.close()


# Convenience-Funktionen
def parse_and_store_xml_files(xml_files: List[Path], 
                             db_path: str = "database/data/parliament_data.db") -> Dict[str, Any]:
    """
    Parst XML-Dateien und speichert sie in der Datenbank
    
    Args:
        xml_files: Liste von XML-Dateipfaden
        db_path: Pfad zur Datenbank
        
    Returns:
        Dictionary mit Verarbeitungsergebnissen
    """
    with ParliamentDatabaseIntegration(DatabaseManager(db_path)) as integration:
        # Tabellen erstellen
        integration.create_parliament_tables()
        
        # XML-Dateien parsen
        parser = ParliamentXMLParser()
        sessions = parser.parse_multiple_files(xml_files)
        
        # Sessions speichern
        results = []
        for session in sessions:
            result = integration.store_parliament_session(session)
            results.append(result)
        
        # Statistiken
        stats = integration.get_parliament_statistics()
        
        return {
            'sessions_processed': len(sessions),
            'successful_stores': sum(1 for r in results if r['success']),
            'failed_stores': sum(1 for r in results if not r['success']),
            'database_statistics': stats,
            'results': results
        }


def export_parliament_data_for_ml(output_format: str = "parquet",
                                 db_path: str = "database/data/parliament_data.db") -> Dict[str, Any]:
    """
    Exportiert Parlamentsdaten für ML-Modelle
    
    Args:
        output_format: Export-Format
        db_path: Pfad zur Datenbank
        
    Returns:
        Dictionary mit Export-Ergebnissen
    """
    with ParliamentDatabaseIntegration(DatabaseManager(db_path)) as integration:
        return integration.export_ml_data(output_format)

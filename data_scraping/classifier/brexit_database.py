"""
Brexit-Datenbank für gefilterte Parliament-Debatten
Speichert nur Brexit-bezogene Debatten und Reden mit Klassifikationsmetadaten
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from database.managers.database_manager import DatabaseManager
from parser.models.parliament_models import Debate, Speech, ParliamentSession
from .brexit_classifier import BrexitClassificationResult


class BrexitDatabase:
    """
    Spezialisierte Datenbank für Brexit-bezogene Parliament-Daten
    Erweitert die Standard-Datenbank um Brexit-spezifische Tabellen und Funktionen
    """
    
    def __init__(self, db_path: str = "database/data/brexit_data.db"):
        """
        Initialisiert die Brexit-Datenbank
        
        Args:
            db_path: Pfad zur Brexit-Datenbank
        """
        self.db_path = db_path
        self.db_manager = DatabaseManager(db_path)
        
        # Setup Logging
        self._setup_logging()
        
        # Erstelle Tabellen
        self._create_brexit_tables()
        
        self.logger.info(f"Brexit-Datenbank initialisiert: {db_path}")
    
    def is_debate_processed(self, debate_id: str) -> bool:
        """
        Prüft ob eine Debatte bereits verarbeitet wurde
        
        Args:
            debate_id: ID der Debatte
            
        Returns:
            True wenn bereits verarbeitet, False sonst
        """
        try:
            query = "SELECT COUNT(*) FROM debates WHERE debate_id = ? AND classification_timestamp IS NOT NULL"
            result = self.db_manager.execute_query(query, [debate_id])
            return result[0][0] > 0 if result else False
        except Exception as e:
            self.logger.error(f"Fehler beim Prüfen der Debatte {debate_id}: {e}")
            return False
    
    def store_classification_result(self, debate_id: str, classification, debate_data):
        """
        Speichert ein Klassifikationsergebnis in der Datenbank
        
        Args:
            debate_id: ID der Debatte
            classification: Klassifikationsergebnis
            debate_data: Debatten-Daten
        """
        try:
            # Extrahiere Debatten-Informationen
            title = debate_data.title if hasattr(debate_data, 'title') else f"Debate {debate_id}"
            session_id = debate_data.session_id if hasattr(debate_data, 'session_id') else "unknown"
            
            # Erstelle Session falls sie nicht existiert (mit echten Session-Daten)
            session_data = getattr(debate_data, 'session', None)
            self._ensure_session_exists(session_id, session_data)
            
            # Speichere in debates Tabelle (mit allen ursprünglichen Daten + Klassifikations-Spalten)
            query = """
            INSERT OR REPLACE INTO debates 
            (debate_id, session_id, title, heading_id, time, url, column_number,
             total_speeches, total_words, unique_speakers, duration_minutes,
             is_brexit_related, confidence, classification_method, reasoning, 
             keywords_found, classification_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            keywords_json = json.dumps(classification.keywords_found) if classification.keywords_found else "[]"
            
            # Extrahiere alle Debatten-Informationen
            heading_id = getattr(debate_data, 'heading_id', '') or ''
            debate_time = getattr(debate_data, 'time', '') or ''
            debate_url = getattr(debate_data, 'url', '') or ''
            column_number = getattr(debate_data, 'column_number', 0) or 0
            total_speeches = getattr(debate_data, 'total_speeches', 0) or 0
            total_words = getattr(debate_data, 'total_words', 0) or 0
            unique_speakers = getattr(debate_data, 'unique_speakers', 0) or 0
            duration_minutes = 0.0  # Nicht verfügbar im Debate-Modell
            
            self.db_manager.execute_query(query, [
                debate_id,
                session_id,
                title,
                heading_id,
                debate_time,
                debate_url,
                column_number,
                total_speeches,
                total_words,
                unique_speakers,
                duration_minutes,
                classification.is_brexit_related,
                classification.confidence,
                classification.classification_method,
                getattr(classification, 'reasoning', ''),
                keywords_json,
                datetime.now().isoformat()
            ])
            
            # Nur wenn es eine Brexit-Debatte ist, kopiere alle zugehörigen Daten
            if classification.is_brexit_related:
                self._copy_brexit_debate_data(debate_id, session_id, debate_data)
            
            self.logger.debug(f"Klassifikationsergebnis für Debatte {debate_id} gespeichert")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Klassifikation für Debatte {debate_id}: {e}")
    
    def _ensure_session_exists(self, session_id: str, session_data=None):
        """Stellt sicher, dass eine Session existiert"""
        try:
            # Prüfe ob Session bereits existiert
            check_query = "SELECT COUNT(*) FROM sessions WHERE session_id = ?"
            result = self.db_manager.execute_query(check_query, [session_id])
            
            if result and result[0][0] == 0:
                # Session existiert nicht, erstelle sie
                insert_query = """
                INSERT INTO sessions 
                (session_id, date, session_type, total_debates, total_speeches, 
                 total_words, unique_speakers, duration_hours, processing_timestamp,
                 brexit_debates_count, classification_method, average_confidence, processing_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                # Verwende echte Session-Daten falls verfügbar
                if session_data:
                    date = getattr(session_data, 'date', datetime.now().strftime("%Y-%m-%d"))
                    session_type = getattr(session_data, 'session_type', 'unknown')
                    total_debates = getattr(session_data, 'total_debates', 0)
                    total_speeches = getattr(session_data, 'total_speeches', 0)
                    total_words = getattr(session_data, 'total_words', 0)
                    unique_speakers = getattr(session_data, 'unique_speakers', 0)
                    duration_hours = getattr(session_data, 'duration_hours', 0.0)
                else:
                    date = datetime.now().strftime("%Y-%m-%d")
                    session_type = "unknown"
                    total_debates = 0
                    total_speeches = 0
                    total_words = 0
                    unique_speakers = 0
                    duration_hours = 0.0
                
                self.db_manager.execute_query(insert_query, [
                    session_id,
                    date,
                    session_type,
                    total_debates,
                    total_speeches,
                    total_words,
                    unique_speakers,
                    duration_hours,
                    datetime.now().isoformat(),
                    0,  # Brexit debates count
                    "manual",  # Classification method
                    0.0,  # Average confidence
                    f"Auto-created for debate classification"
                ])
                
                self.logger.debug(f"Session {session_id} erstellt")
                
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen der Session {session_id}: {e}")
    
    def _copy_brexit_debate_data(self, debate_id: str, session_id: str, debate_data):
        """
        Kopiert alle Daten einer Brexit-Debatte aus der Quell-Datenbank
        
        Args:
            debate_id: ID der Debatte
            session_id: ID der Session
            debate_data: Debatten-Daten
        """
        try:
            # Kopiere alle Reden der Debatte
            for speech in debate_data.speeches:
                self._copy_speech_data(speech, debate_id, session_id)
                
                # Kopiere alle Absätze der Rede
                for paragraph in speech.paragraphs:
                    self._copy_paragraph_data(paragraph, speech.speech_id)
            
            # Kopiere Sprecher-Daten
            speakers = set()
            for speech in debate_data.speeches:
                if speech.speaker.name not in speakers:
                    self._copy_speaker_data(speech.speaker)
                    speakers.add(speech.speaker.name)
            
            self.logger.debug(f"Brexit-Debatte {debate_id} vollständig kopiert")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Kopieren der Brexit-Debatte {debate_id}: {e}")
    
    def _copy_speech_data(self, speech, debate_id: str, session_id: str):
        """Kopiert Rede-Daten"""
        try:
            query = """
            INSERT OR REPLACE INTO speeches 
            (speech_id, session_id, debate_id, speaker_name, speaker_person_id, 
             speaker_title, time, column_number, url, word_count, char_count, 
             is_intervention, paragraph_count, full_text, processing_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.db_manager.execute_query(query, [
                speech.speech_id,
                session_id,
                debate_id,
                speech.speaker.name,
                speech.speaker.person_id,
                speech.speaker.title,
                speech.time,
                speech.column_number,
                speech.url,
                speech.word_count,
                speech.char_count,
                speech.is_intervention,
                len(speech.paragraphs),
                speech.full_text,
                datetime.now().isoformat()
            ])
            
        except Exception as e:
            self.logger.error(f"Fehler beim Kopieren der Rede {speech.speech_id}: {e}")
    
    def _copy_paragraph_data(self, paragraph, speech_id: str):
        """Kopiert Absatz-Daten"""
        try:
            query = """
            INSERT OR REPLACE INTO paragraphs 
            (paragraph_id, speech_id, text, paragraph_pid, class_name, 
             motion_text, word_count, position_in_speech)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.db_manager.execute_query(query, [
                paragraph.paragraph_id,
                speech_id,
                paragraph.text,
                '',  # paragraph_pid nicht verfügbar
                paragraph.class_name,
                paragraph.motion_text,
                paragraph.word_count,
                0  # position_in_speech nicht verfügbar
            ])
            
        except Exception as e:
            self.logger.error(f"Fehler beim Kopieren des Absatzes {paragraph.paragraph_id}: {e}")
    
    def _copy_speaker_data(self, speaker):
        """Kopiert Sprecher-Daten"""
        try:
            query = """
            INSERT OR REPLACE INTO speakers 
            (speaker_name, person_id, title, party, constituency, 
             first_appearance, total_speeches, total_words)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.db_manager.execute_query(query, [
                speaker.name,
                speaker.person_id,
                speaker.title,
                speaker.party,
                speaker.constituency,
                None,  # first_appearance nicht verfügbar
                0,     # total_speeches nicht verfügbar
                0      # total_words nicht verfügbar
            ])
            
        except Exception as e:
            self.logger.error(f"Fehler beim Kopieren des Sprechers {speaker.name}: {e}")
    
    def copy_original_data_from_source(self, source_db_path: str):
        """
        Initialisiert die Brexit-Datenbank - keine Daten werden kopiert,
        da nur klassifizierte Brexit-Debatten gespeichert werden sollen.
        
        Args:
            source_db_path: Pfad zur Quell-Datenbank (wird nicht verwendet)
        """
        self.logger.info("Brexit-Datenbank bereit - nur klassifizierte Daten werden gespeichert")
    
    
    def _setup_logging(self):
        """Setup für Logging"""
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"brexit_database_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_brexit_tables(self):
        """Erstellt alle Brexit-Tabellen - exakt wie die ursprüngliche Datenbank, aber mit Klassifikations-Spalten"""
        
        # 1. Sessions-Tabelle (exakt wie ursprünglich + Klassifikations-Spalten)
        sessions_schema = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id VARCHAR PRIMARY KEY,
            date DATE,
            session_type VARCHAR,
            total_debates INTEGER,
            total_speeches INTEGER,
            total_words INTEGER,
            unique_speakers INTEGER,
            duration_hours DOUBLE,
            processing_timestamp TIMESTAMP,
            
            -- Brexit-Klassifikations-Spalten
            brexit_debates_count INTEGER DEFAULT 0,
            classification_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            classification_method VARCHAR,
            average_confidence DOUBLE DEFAULT 0.0,
            processing_notes TEXT
        )
        """
        
        # 2. Debates-Tabelle (exakt wie ursprünglich + Klassifikations-Spalten)
        debates_schema = """
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
            duration_minutes DOUBLE,
            
            -- Brexit-Klassifikations-Spalten
            is_brexit_related BOOLEAN DEFAULT FALSE,
            confidence DOUBLE DEFAULT 0.0,
            classification_method VARCHAR,
            reasoning TEXT,
            keywords_found TEXT, -- JSON-Array als String
            classification_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # 3. Speakers-Tabelle (exakt wie ursprünglich)
        speakers_schema = """
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
        """
        
        # 4. Speeches-Tabelle (exakt wie ursprünglich + Klassifikations-Spalten)
        speeches_schema = """
        CREATE TABLE IF NOT EXISTS speeches (
            speech_id VARCHAR PRIMARY KEY,
            session_id VARCHAR,
            debate_id VARCHAR,
            speaker_name VARCHAR,
            speaker_person_id VARCHAR,
            speaker_title VARCHAR,
            time VARCHAR,
            column_number INTEGER,
            url VARCHAR,
            word_count INTEGER,
            char_count INTEGER,
            is_intervention BOOLEAN,
            paragraph_count INTEGER,
            full_text TEXT,
            
            -- Brexit-Klassifikations-Spalten
            brexit_relevance_score DOUBLE DEFAULT 0.0,
            brexit_keywords_in_speech TEXT, -- JSON-Array
            processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # 5. Paragraphs-Tabelle (exakt wie ursprünglich)
        paragraphs_schema = """
        CREATE TABLE IF NOT EXISTS paragraphs (
            paragraph_id VARCHAR PRIMARY KEY,
            speech_id VARCHAR,
            text TEXT,
            paragraph_pid VARCHAR,
            class_name VARCHAR,
            motion_text BOOLEAN,
            word_count INTEGER,
            position_in_speech INTEGER
        )
        """
        
        # 6. Brexit-Keywords-Tabelle (für Analyse)
        keywords_schema = """
        CREATE TABLE IF NOT EXISTS brexit_keywords (
            keyword_id INTEGER PRIMARY KEY,
            keyword TEXT UNIQUE NOT NULL,
            category TEXT, -- 'direct', 'eu_related', 'economic', 'political'
            frequency INTEGER DEFAULT 0,
            first_occurrence TEXT,
            last_occurrence TEXT
        )
        """
        
        # 7. Klassifikations-Log-Tabelle
        classification_log_schema = """
        CREATE TABLE IF NOT EXISTS classification_log (
            log_id INTEGER PRIMARY KEY,
            session_id VARCHAR,
            debate_id VARCHAR,
            classification_method VARCHAR,
            processing_time_ms INTEGER,
            success BOOLEAN,
            error_message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Erstelle alle Tabellen
        schemas = [
            ("sessions", sessions_schema),
            ("debates", debates_schema),
            ("speakers", speakers_schema),
            ("speeches", speeches_schema),
            ("paragraphs", paragraphs_schema),
            ("brexit_keywords", keywords_schema),
            ("classification_log", classification_log_schema)
        ]
        
        for table_name, schema in schemas:
            try:
                self.db_manager.execute_query(schema)
                self.logger.info(f"Tabelle '{table_name}' erstellt/überprüft")
            except Exception as e:
                self.logger.error(f"Fehler beim Erstellen der Tabelle '{table_name}': {e}")
                raise
    
    def store_brexit_session(self, session: ParliamentSession, 
                           classification_results: Dict[str, Any]) -> bool:
        """
        Speichert eine Session mit Brexit-Klassifikationsergebnissen
        
        Args:
            session: ParliamentSession-Objekt
            classification_results: Ergebnisse der Brexit-Klassifikation
            
        Returns:
            True wenn erfolgreich gespeichert
        """
        try:
            # 1. Speichere Session-Metadaten
            session_data = {
                'session_id': session.session_id,
                'date': session.date,
                'session_type': session.session_type,
                'total_debates': session.total_debates,
                'brexit_debates_count': classification_results['classification_summary']['total_brexit_debates'],
                'classification_method': 'hybrid',  # Wird später dynamisch gesetzt
                'average_confidence': classification_results['classification_summary'].get('average_confidence', 0.0),
                'processing_notes': f"Processed {len(classification_results['brexit_debates'])} Brexit debates"
            }
            
            self._insert_session(session_data)
            
            # 2. Speichere Brexit-Debatten
            for debate_data in classification_results['brexit_debates']:
                debate = self._find_debate_by_title(session, debate_data['title'])
                if debate:
                    self._store_brexit_debate(debate, debate_data['classification'], session.session_id)
            
            self.logger.info(f"Session '{session.session_id}' erfolgreich gespeichert")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Session '{session.session_id}': {e}")
            return False
    
    def _insert_session(self, session_data: Dict[str, Any]):
        """Fügt Session-Daten in die Datenbank ein"""
        query = """
        INSERT OR REPLACE INTO brexit_sessions 
        (session_id, date, session_type, total_debates, brexit_debates_count, 
         classification_method, average_confidence, processing_notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        self.db_manager.execute_query(query, [
            session_data['session_id'],
            session_data['date'],
            session_data['session_type'],
            session_data['total_debates'],
            session_data['brexit_debates_count'],
            session_data['classification_method'],
            session_data['average_confidence'],
            session_data['processing_notes']
        ])
    
    def _find_debate_by_title(self, session: ParliamentSession, title: str) -> Optional[Debate]:
        """Findet Debate-Objekt anhand des Titels"""
        for debate in session.debates:
            if debate.title == title:
                return debate
        return None
    
    def _store_brexit_debate(self, debate: Debate, classification: BrexitClassificationResult, 
                           session_id: str):
        """Speichert eine Brexit-Debatte in die Datenbank"""
        
        # 1. Speichere Debatte - verwende die ursprüngliche debate_id
        debate_id = debate.debate_id if hasattr(debate, 'debate_id') else f"debate_{int(time.time())}"
        
        debate_query = """
        INSERT OR REPLACE INTO brexit_debates 
        (debate_id, session_id, title, heading_id, time, url, column_number,
         is_brexit_related, confidence, classification_method, reasoning, keywords_found,
         total_speeches, total_words, unique_speakers, duration_minutes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        self.db_manager.execute_query(debate_query, [
            debate_id,
            session_id,
            debate.title,
            debate.heading_id,
            debate.time,
            debate.url,
            debate.column_number,
            classification.is_brexit_related,
            classification.confidence,
            classification.classification_method,
            classification.reasoning,
            json.dumps(classification.keywords_found),
            debate.total_speeches,
            debate.total_words,
            debate.unique_speakers,
            debate.duration_minutes
        ])
        
        # 2. Speichere alle Reden der Debatte
        for speech in debate.speeches:
            self._store_brexit_speech(speech, debate_id, session_id, classification.keywords_found)
        
        # 3. Update Keyword-Frequenzen
        self._update_keyword_frequencies(classification.keywords_found, session_id)
    
    def _store_brexit_speech(self, speech: Speech, debate_id: str, session_id: str, 
                           debate_keywords: List[str]):
        """Speichert eine Rede in die Brexit-Datenbank"""
        
        # Finde Brexit-Keywords in dieser spezifischen Rede
        speech_text_lower = speech.full_text.lower()
        speech_keywords = [kw for kw in debate_keywords if kw in speech_text_lower]
        
        # Berechne Brexit-Relevanz-Score für diese Rede
        relevance_score = len(speech_keywords) / max(len(debate_keywords), 1) if debate_keywords else 0.0
        
        query = """
        INSERT OR REPLACE INTO brexit_speeches 
        (speech_id, debate_id, session_id, speaker_name, speaker_person_id, 
         speaker_party, speaker_constituency, speaker_title, speech_text, 
         word_count, char_count, time, column_number, url, is_intervention,
         brexit_relevance_score, brexit_keywords_in_speech)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        self.db_manager.execute_query(query, [
            speech.speech_id,
            debate_id,
            session_id,
            speech.speaker.name,
            speech.speaker.person_id,
            speech.speaker.party,
            speech.speaker.constituency,
            speech.speaker.title,
            speech.full_text,
            speech.word_count,
            speech.char_count,
            speech.time,
            speech.column_number,
            speech.url,
            speech.is_intervention,
            relevance_score,
            json.dumps(speech_keywords)
        ])
    
    def _update_keyword_frequencies(self, keywords: List[str], session_id: str):
        """Aktualisiert Keyword-Frequenzen in der Datenbank"""
        for keyword in keywords:
            # Prüfe ob Keyword bereits existiert
            existing = self.db_manager.execute_query(
                "SELECT keyword_id, frequency FROM brexit_keywords WHERE keyword = ?",
                [keyword]
            )
            
            if existing:
                # Update Frequenz
                keyword_id = existing[0][0]
                new_frequency = existing[0][1] + 1
                
                self.db_manager.execute_query(
                    "UPDATE brexit_keywords SET frequency = ?, last_occurrence = ? WHERE keyword_id = ?",
                    [new_frequency, session_id, keyword_id]
                )
            else:
                # Neues Keyword einfügen
                keyword_id = self.db_manager.execute_query(
                    "SELECT COALESCE(MAX(keyword_id), 0) + 1 FROM brexit_keywords"
                )[0][0]
                
                self.db_manager.execute_query(
                    "INSERT INTO brexit_keywords (keyword_id, keyword, category, frequency, first_occurrence, last_occurrence) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        keyword_id,
                        keyword,
                        self._categorize_keyword(keyword),
                        1,
                        session_id,
                        session_id
                    ]
                )
    
    def _categorize_keyword(self, keyword: str) -> str:
        """Kategorisiert ein Keyword"""
        keyword_lower = keyword.lower()
        
        if keyword_lower in {'brexit', 'eu exit', 'european union exit', 'leave the eu'}:
            return 'direct'
        elif any(term in keyword_lower for term in {'eu', 'european', 'union'}):
            return 'eu_related'
        elif any(term in keyword_lower for term in {'trade', 'market', 'economy', 'business'}):
            return 'economic'
        elif any(term in keyword_lower for term in {'sovereignty', 'control', 'independence'}):
            return 'political'
        else:
            return 'other'
    
    def get_brexit_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken über die Brexit-Datenbank zurück"""
        try:
            # Gesamtstatistiken
            total_sessions = self.db_manager.execute_query("SELECT COUNT(*) FROM sessions WHERE brexit_debates_count > 0")[0][0]
            total_debates = self.db_manager.execute_query("SELECT COUNT(*) FROM debates WHERE is_brexit_related = TRUE")[0][0]
            total_speeches = self.db_manager.execute_query("SELECT COUNT(*) FROM speeches WHERE debate_id IN (SELECT debate_id FROM debates WHERE is_brexit_related = TRUE)")[0][0]
            
            # Durchschnitts-Confidence
            avg_confidence = self.db_manager.execute_query(
                "SELECT AVG(confidence) FROM debates WHERE is_brexit_related = TRUE"
            )[0][0] or 0.0
            
            # Top Keywords
            top_keywords = self.db_manager.execute_query(
                "SELECT keyword, frequency FROM brexit_keywords ORDER BY frequency DESC LIMIT 10"
            )
            
            # Zeitraum
            date_range = self.db_manager.execute_query(
                "SELECT MIN(date), MAX(date) FROM sessions WHERE brexit_debates_count > 0"
            )[0]
            
            return {
                'total_sessions': total_sessions,
                'total_debates': total_debates,
                'total_speeches': total_speeches,
                'average_confidence': round(avg_confidence, 3),
                'top_keywords': [{'keyword': kw[0], 'frequency': kw[1]} for kw in top_keywords],
                'date_range': {
                    'earliest': date_range[0],
                    'latest': date_range[1]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Statistiken: {e}")
            return {}
    
    def search_brexit_debates(self, keyword: str = None, min_confidence: float = 0.0, 
                            date_from: str = None, date_to: str = None) -> List[Dict[str, Any]]:
        """
        Sucht Brexit-Debatten mit verschiedenen Filtern
        
        Args:
            keyword: Suchbegriff im Titel oder Text
            min_confidence: Mindest-Confidence
            date_from: Startdatum (YYYY-MM-DD)
            date_to: Enddatum (YYYY-MM-DD)
            
        Returns:
            Liste der gefundenen Debatten
        """
        try:
            query = """
            SELECT d.*, s.date, s.session_type
            FROM debates d
            JOIN sessions s ON d.session_id = s.session_id
            WHERE d.is_brexit_related = TRUE
            """
            
            params = {}
            
            if keyword:
                query += " AND (d.title LIKE ? OR d.reasoning LIKE ?)"
                params['keyword'] = f"%{keyword}%"
                params['keyword2'] = f"%{keyword}%"
            
            if min_confidence > 0:
                query += " AND d.confidence >= ?"
                params['min_confidence'] = min_confidence
            
            if date_from:
                query += " AND s.date >= ?"
                params['date_from'] = date_from
            
            if date_to:
                query += " AND s.date <= ?"
                params['date_to'] = date_to
            
            query += " ORDER BY d.confidence DESC, s.date DESC"
            
            results = self.db_manager.execute_query(query, params)
            
            # Konvertiere zu Dictionary-Format
            debates = []
            for row in results:
                debate = {
                    'debate_id': row[0],
                    'session_id': row[1],
                    'title': row[2],
                    'confidence': row[8],
                    'classification_method': row[9],
                    'reasoning': row[10],
                    'keywords_found': json.loads(row[11]) if row[11] else [],
                    'date': row[16],
                    'session_type': row[17]
                }
                debates.append(debate)
            
            return debates
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Suche: {e}")
            return []
    
    def close(self):
        """Schließt die Datenbankverbindung"""
        self.db_manager.close()
    
    def __enter__(self):
        """Context Manager Entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Exit"""
        self.close()


# Convenience-Funktionen
def create_brexit_database(db_path: str = "database/data/brexit_data.db") -> BrexitDatabase:
    """
    Factory-Funktion für BrexitDatabase
    
    Args:
        db_path: Pfad zur Brexit-Datenbank
        
    Returns:
        BrexitDatabase Instanz
    """
    return BrexitDatabase(db_path)


if __name__ == "__main__":
    # Test der Brexit-Datenbank
    print("Brexit-Datenbank Test")
    
    with create_brexit_database() as db:
        stats = db.get_brexit_statistics()
        print(f"Brexit-Datenbank Statistiken: {stats}")
        
        # Test-Suche
        debates = db.search_brexit_debates(min_confidence=0.7)
        print(f"Debatten mit hoher Confidence: {len(debates)}")

"""
Brexit-Filtering Pipeline
Hauptpipeline zum Klassifizieren und Filtern von Parliament-Debatten auf Brexit-Bezug
"""

import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import time
from tqdm import tqdm

# Lade .env Datei falls vorhanden
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv ist optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from database.managers.database_manager import DatabaseManager
from parser.xml_parser import ParliamentXMLParser
from parser.models.parliament_models import ParliamentSession
from .brexit_classifier import BrexitClassifier, create_brexit_classifier
from .brexit_database import BrexitDatabase, create_brexit_database


class BrexitPipeline:
    """
    Hauptpipeline für Brexit-Debatten-Filterung
    Verarbeitet XML-Dateien, klassifiziert Debatten und speichert Brexit-relevante Daten
    """
    
    def __init__(self, 
                 source_db_path: str = "database/data/parliament_data.db",
                 brexit_db_path: str = "database/data/brexit_data.db",
                 google_api_key: Optional[str] = None):
        """
        Initialisiert die Brexit-Pipeline
        
        Args:
            source_db_path: Pfad zur Hauptdatenbank mit allen Parliament-Daten
            brexit_db_path: Pfad zur Brexit-spezifischen Datenbank
            google_api_key: Google AI API Key für Gemini
        """
        self.source_db_path = source_db_path
        self.brexit_db_path = brexit_db_path
        
        # Setup Logging
        self._setup_logging()
        
        # Initialisiere Komponenten
        self.source_db_manager = DatabaseManager(source_db_path)
        self.brexit_db = create_brexit_database(brexit_db_path)
        self.classifier = create_brexit_classifier(google_api_key)
        self.xml_parser = ParliamentXMLParser()
        
        self.logger.info("Brexit-Pipeline initialisiert")
    
    def _setup_logging(self):
        """Setup für Logging"""
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"brexit_pipeline_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def process_xml_files(self, xml_directory: str, 
                         max_files: Optional[int] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Verarbeitet XML-Dateien und filtert Brexit-Debatten
        
        Args:
            xml_directory: Verzeichnis mit XML-Dateien
            max_files: Maximale Anzahl zu verarbeitender Dateien
            start_date: Startdatum (YYYY-MM-DD)
            end_date: Enddatum (YYYY-MM-DD)
            
        Returns:
            Verarbeitungsstatistiken
        """
        xml_path = Path(xml_directory)
        if not xml_path.exists():
            raise ValueError(f"XML-Verzeichnis existiert nicht: {xml_directory}")
        
        # Finde XML-Dateien
        xml_files = list(xml_path.glob("*.xml"))
        
        # Filtere nach Datum falls angegeben
        if start_date or end_date:
            xml_files = self._filter_files_by_date(xml_files, start_date, end_date)
        
        # Begrenze Anzahl falls angegeben
        if max_files:
            xml_files = xml_files[:max_files]
        
        self.logger.info(f"Verarbeite {len(xml_files)} XML-Dateien")
        
        # Verarbeitungsstatistiken
        stats = {
            'total_files': len(xml_files),
            'processed_files': 0,
            'failed_files': 0,
            'total_sessions': 0,
            'total_debates': 0,
            'brexit_debates': 0,
            'total_speeches': 0,
            'brexit_speeches': 0,
            'processing_time_seconds': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        # Verarbeite Dateien mit Progress Bar
        for xml_file in tqdm(xml_files, desc="Verarbeite XML-Dateien"):
            try:
                file_stats = self._process_single_xml_file(xml_file)
                
                # Update Gesamtstatistiken
                stats['processed_files'] += 1
                stats['total_sessions'] += file_stats.get('sessions', 0)
                stats['total_debates'] += file_stats.get('debates', 0)
                stats['brexit_debates'] += file_stats.get('brexit_debates', 0)
                stats['total_speeches'] += file_stats.get('speeches', 0)
                stats['brexit_speeches'] += file_stats.get('brexit_speeches', 0)
                
                self.logger.info(f"Verarbeitet: {xml_file.name} - {file_stats}")
                
            except Exception as e:
                stats['failed_files'] += 1
                error_msg = f"Fehler bei {xml_file.name}: {e}"
                stats['errors'].append(error_msg)
                self.logger.error(error_msg)
        
        stats['processing_time_seconds'] = time.time() - start_time
        
        self.logger.info(f"Verarbeitung abgeschlossen: {stats}")
        return stats
    
    def _filter_files_by_date(self, xml_files: List[Path], 
                            start_date: Optional[str], 
                            end_date: Optional[str]) -> List[Path]:
        """Filtert XML-Dateien nach Datum"""
        filtered_files = []
        
        for xml_file in xml_files:
            # Extrahiere Datum aus Dateinamen
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', xml_file.name)
            if not date_match:
                continue
            
            file_date = date_match.group(1)
            
            # Prüfe Datumsfilter
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            
            filtered_files.append(xml_file)
        
        return filtered_files
    
    def _process_single_xml_file(self, xml_file: Path) -> Dict[str, int]:
        """
        Verarbeitet eine einzelne XML-Datei
        
        Args:
            xml_file: Pfad zur XML-Datei
            
        Returns:
            Statistiken für diese Datei
        """
        # Parse XML-Datei
        session = self.xml_parser.parse_xml_file(xml_file)
        if not session:
            return {'sessions': 0, 'debates': 0, 'brexit_debates': 0, 'speeches': 0, 'brexit_speeches': 0}
        
        # Klassifiziere Session
        classification_results = self.classifier.classify_session(session)
        
        # Speichere in Brexit-Datenbank
        success = self.brexit_db.store_brexit_session(session, classification_results)
        
        if not success:
            raise Exception("Fehler beim Speichern in Brexit-Datenbank")
        
        # Berechne Statistiken
        brexit_speeches = sum(
            debate_data['debate_stats']['total_speeches'] 
            for debate_data in classification_results['brexit_debates']
        )
        
        return {
            'sessions': 1,
            'debates': len(session.debates),
            'brexit_debates': len(classification_results['brexit_debates']),
            'speeches': session.total_speeches,
            'brexit_speeches': brexit_speeches
        }
    
    def process_existing_database(self, 
                                session_ids: Optional[List[str]] = None,
                                date_from: Optional[str] = None,
                                date_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Verarbeitet bereits in der Datenbank gespeicherte Sessions
        
        Args:
            session_ids: Spezifische Session-IDs zu verarbeiten
            date_from: Startdatum (YYYY-MM-DD)
            date_to: Enddatum (YYYY-MM-DD)
            
        Returns:
            Verarbeitungsstatistiken
        """
        # Finde Sessions in der Hauptdatenbank
        query = "SELECT session_id, date FROM parliament_sessions WHERE 1=1"
        params = {}
        
        if session_ids:
            placeholders = ','.join(['?' for _ in session_ids])
            query += f" AND session_id IN ({placeholders})"
            params.update({f'session_{i}': sid for i, sid in enumerate(session_ids)})
        
        if date_from:
            query += " AND date >= ?"
            params['date_from'] = date_from
        
        if date_to:
            query += " AND date <= ?"
            params['date_to'] = date_to
        
        sessions = self.source_db.execute_query(query, params)
        
        self.logger.info(f"Verarbeite {len(sessions)} Sessions aus der Datenbank")
        
        stats = {
            'total_sessions': len(sessions),
            'processed_sessions': 0,
            'failed_sessions': 0,
            'total_debates': 0,
            'brexit_debates': 0,
            'processing_time_seconds': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        for session_id, date in tqdm(sessions, desc="Verarbeite Sessions"):
            try:
                session_stats = self._process_database_session(session_id)
                
                stats['processed_sessions'] += 1
                stats['total_debates'] += session_stats.get('debates', 0)
                stats['brexit_debates'] += session_stats.get('brexit_debates', 0)
                
            except Exception as e:
                stats['failed_sessions'] += 1
                error_msg = f"Fehler bei Session {session_id}: {e}"
                stats['errors'].append(error_msg)
                self.logger.error(error_msg)
        
        stats['processing_time_seconds'] = time.time() - start_time
        
        return stats
    
    def _process_database_session(self, session_id: str) -> Dict[str, int]:
        """
        Verarbeitet eine Session aus der Datenbank
        
        Args:
            session_id: Session-ID
            
        Returns:
            Statistiken für diese Session
        """
        # Lade Session-Daten aus der Datenbank
        session_data = self._load_session_from_database(session_id)
        if not session_data:
            return {'debates': 0, 'brexit_debates': 0}
        
        # Konvertiere zu ParliamentSession-Objekt
        session = self._convert_to_parliament_session(session_data)
        
        # Klassifiziere Session
        classification_results = self.classifier.classify_session(session)
        
        # Speichere in Brexit-Datenbank
        success = self.brexit_db.store_brexit_session(session, classification_results)
        
        if not success:
            raise Exception("Fehler beim Speichern in Brexit-Datenbank")
        
        return {
            'debates': len(session.debates),
            'brexit_debates': len(classification_results['brexit_debates'])
        }
    
    def _load_session_from_database(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Lädt Session-Daten aus der Hauptdatenbank"""
        try:
            # Lade Session-Info
            session_query = """
            SELECT session_id, date, session_type, total_debates, total_speeches, 
                   total_words, unique_speakers, duration_hours
            FROM parliament_sessions 
            WHERE session_id = ?
            """
            session_result = self.source_db_manager.execute_query(session_query, [session_id])
            
            if not session_result:
                self.logger.warning(f"Session {session_id} nicht in der Datenbank gefunden")
                return None
            
            session_data = {
                'session_id': session_result[0][0],
                'date': session_result[0][1],
                'session_type': session_result[0][2],
                'total_debates': session_result[0][3],
                'total_speeches': session_result[0][4],
                'total_words': session_result[0][5],
                'unique_speakers': session_result[0][6],
                'duration_hours': session_result[0][7]
            }
            
            # Lade alle Debatten dieser Session
            debates_query = """
            SELECT debate_id, title, heading_id, time, url, column_number,
                   total_speeches, total_words, unique_speakers
            FROM parliament_debates 
            WHERE session_id = ?
            ORDER BY debate_id
            """
            debates_result = self.source_db_manager.execute_query(debates_query, [session_id])
            
            debates = []
            for debate_row in debates_result:
                debate_id = debate_row[0]
                
                # Lade alle Reden dieser Debatte
                speeches_query = """
                SELECT ps.speech_id, ps.speaker_id, ps.speaker_name, ps.speaker_person_id,
                       ps.speaker_title, ps.time, ps.column_number, ps.url, ps.text,
                       ps.word_count, ps.char_count, ps.is_intervention,
                       sp.party, sp.constituency
                FROM parliament_speeches ps
                LEFT JOIN parliament_speakers sp ON ps.speaker_id = sp.speaker_id
                WHERE ps.debate_id = ?
                ORDER BY ps.speech_id
                """
                speeches_result = self.source_db_manager.execute_query(speeches_query, [debate_id])
                
                speeches = []
                for speech_row in speeches_result:
                    speech_data = {
                        'speech_id': speech_row[0],
                        'speaker_id': speech_row[1],
                        'speaker_name': speech_row[2],
                        'speaker_person_id': speech_row[3],
                        'speaker_title': speech_row[4],
                        'time': speech_row[5],
                        'column_number': speech_row[6],
                        'url': speech_row[7],
                        'text': speech_row[8],
                        'word_count': speech_row[9],
                        'char_count': speech_row[10],
                        'is_intervention': speech_row[11],
                        'speaker_party': speech_row[12],
                        'speaker_constituency': speech_row[13]
                    }
                    speeches.append(speech_data)
                
                debate_data = {
                    'debate_id': debate_id,
                    'title': debate_row[1],
                    'heading_id': debate_row[2],
                    'time': debate_row[3],
                    'url': debate_row[4],
                    'column_number': debate_row[5],
                    'total_speeches': debate_row[6],
                    'total_words': debate_row[7],
                    'unique_speakers': debate_row[8],
                    'speeches': speeches
                }
                debates.append(debate_data)
            
            session_data['debates'] = debates
            return session_data
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Session {session_id}: {e}")
            return None
    
    def _convert_to_parliament_session(self, session_data: Dict[str, Any]) -> ParliamentSession:
        """Konvertiert Datenbank-Daten zu ParliamentSession-Objekt"""
        from parser.models.parliament_models import ParliamentSession, Debate, Speech, Speaker
        
        # Erstelle Session-Objekt
        session = ParliamentSession(
            session_id=session_data['session_id'],
            date=session_data['date'],
            session_type=session_data.get('session_type', 'westminster')
        )
        
        # Konvertiere Debatten
        for debate_data in session_data.get('debates', []):
            debate = Debate(
                title=debate_data['title'],
                heading_id=debate_data.get('heading_id', ''),
                time=debate_data.get('time', ''),
                url=debate_data.get('url', ''),
                column_number=debate_data.get('column_number', 0)
            )
            
            # Konvertiere Reden
            for speech_data in debate_data.get('speeches', []):
                speaker = Speaker(
                    name=speech_data['speaker_name'],
                    person_id=speech_data.get('speaker_person_id', ''),
                    party=speech_data.get('speaker_party', ''),
                    constituency=speech_data.get('speaker_constituency', ''),
                    title=speech_data.get('speaker_title', '')
                )
                
                # Erstelle Paragraph-Objekt für den Text
                from parser.models.parliament_models import Paragraph
                paragraph = Paragraph(
                    text=speech_data.get('text', ''),
                    paragraph_id=speech_data['speech_id'],
                    class_name='speech',
                    motion_text=False
                )
                
                speech = Speech(
                    speech_id=speech_data['speech_id'],
                    speaker=speaker,
                    paragraphs=[paragraph],
                    time=speech_data.get('time', ''),
                    column_number=speech_data.get('column_number', 0),
                    url=speech_data.get('url', '')
                )
                
                debate.speeches.append(speech)
            
            session.debates.append(debate)
        
        return session
    
    def _classify_session_debates(self, session: ParliamentSession) -> Dict[str, Any]:
        """
        Klassifiziert alle Debatten einer Session auf Brexit-Bezug
        
        Args:
            session: ParliamentSession-Objekt
            
        Returns:
            Dictionary mit Klassifikationsergebnissen
        """
        brexit_debates = []
        all_confidences = []
        
        for debate in session.debates:
            try:
                # Klassifiziere Debatte
                classification = self.classifier.classify_debate(debate)
                
                if classification.is_brexit_related:
                    brexit_debates.append(debate)
                
                all_confidences.append(classification.confidence)
                
            except Exception as e:
                self.logger.error(f"Fehler bei Klassifikation von Debatte '{debate.title}': {e}")
                continue
        
        # Berechne durchschnittliche Confidence
        average_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
        return {
            'brexit_debates': brexit_debates,
            'classification_method': 'gemini_ai',
            'average_confidence': average_confidence
        }
    
    def process_all_sessions_from_database(self, max_sessions: Optional[int] = None) -> Dict[str, Any]:
        """
        Verarbeitet alle Sessions aus der bestehenden Parliament-Datenbank
        
        Args:
            max_sessions: Maximale Anzahl zu verarbeitender Sessions
            
        Returns:
            Verarbeitungsstatistiken
        """
        try:
            # Lade alle Session-IDs aus der Datenbank
            sessions_query = """
            SELECT session_id, date 
            FROM parliament_sessions 
            ORDER BY date DESC
            """
            if max_sessions:
                sessions_query += f" LIMIT {max_sessions}"
            
            sessions_result = self.source_db_manager.execute_query(sessions_query)
            
            if not sessions_result:
                self.logger.warning("Keine Sessions in der Datenbank gefunden")
                return {'total_files': 0, 'processed_files': 0, 'failed_files': 0}
            
            self.logger.info(f"Verarbeite {len(sessions_result)} Sessions aus der Datenbank")
            
            total_sessions = 0
            total_debates = 0
            brexit_debates = 0
            total_speeches = 0
            brexit_speeches = 0
            errors = []
            
            for session_row in sessions_result:
                session_id = session_row[0]
                session_date = session_row[1]
                
                try:
                    self.logger.info(f"Verarbeite Session: {session_id} ({session_date})")
                    
                    # Lade Session aus Datenbank
                    session_data = self._load_session_from_database(session_id)
                    if not session_data:
                        errors.append(f"Session {session_id} konnte nicht geladen werden")
                        continue
                    
                    # Konvertiere zu ParliamentSession-Objekt
                    session = self._convert_to_parliament_session(session_data)
                    
                    # Klassifiziere Debatten
                    classification_results = self._classify_session_debates(session)
                    
                    # Speichere in Brexit-Datenbank
                    if classification_results['brexit_debates']:
                        # Konvertiere Debate-Objekte zu Dictionary-Format
                        brexit_debates_data = []
                        for debate in classification_results['brexit_debates']:
                            # Klassifiziere die Debatte erneut, um die Klassifikationsdaten zu erhalten
                            classification = self.classifier.classify_debate(debate)
                            brexit_debates_data.append({
                                'title': debate.title,
                                'classification': classification
                            })
                        
                        # Erstelle classification_results Dictionary im erwarteten Format
                        classification_data = {
                            'brexit_debates': brexit_debates_data,
                            'classification_summary': {
                                'total_brexit_debates': len(classification_results['brexit_debates']),
                                'average_confidence': classification_results['average_confidence']
                            }
                        }
                        
                        success = self.brexit_db.store_brexit_session(
                            session, 
                            classification_data
                        )
                        
                        if success:
                            total_sessions += 1
                            total_debates += len(session.debates)
                            brexit_debates += len(classification_results['brexit_debates'])
                            total_speeches += sum(len(debate.speeches) for debate in session.debates)
                            brexit_speeches += sum(len(debate.speeches) for debate in classification_results['brexit_debates'])
                            
                            self.logger.info(f"Session {session_id} erfolgreich verarbeitet: "
                                           f"{len(classification_results['brexit_debates'])}/{len(session.debates)} Brexit-Debatten")
                        else:
                            errors.append(f"Fehler beim Speichern der Session {session_id}")
                    else:
                        self.logger.info(f"Session {session_id}: Keine Brexit-Debatten gefunden")
                        
                except Exception as e:
                    error_msg = f"Fehler bei Session {session_id}: {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
            
            processing_stats = {
                'total_files': len(sessions_result),
                'processed_files': total_sessions,
                'failed_files': len(errors),
                'total_sessions': total_sessions,
                'total_debates': total_debates,
                'brexit_debates': brexit_debates,
                'total_speeches': total_speeches,
                'brexit_speeches': brexit_speeches,
                'errors': errors
            }
            
            self.logger.info(f"Verarbeitung abgeschlossen: {processing_stats}")
            return processing_stats
            
        except Exception as e:
            self.logger.error(f"Fehler beim Verarbeiten der Sessions: {e}")
            return {'total_files': 0, 'processed_files': 0, 'failed_files': 1, 'errors': [str(e)]}
    
    def get_brexit_analytics(self) -> Dict[str, Any]:
        """
        Gibt umfassende Analytics über die Brexit-Daten zurück
        
        Returns:
            Analytics-Dictionary
        """
        # Basis-Statistiken
        stats = self.brexit_db.get_brexit_statistics()
        
        # Erweiterte Analytics
        analytics = {
            'overview': stats,
            'confidence_distribution': self._get_confidence_distribution(),
            'keyword_analysis': self._get_keyword_analysis(),
            'temporal_analysis': self._get_temporal_analysis(),
            'speaker_analysis': self._get_speaker_analysis()
        }
        
        return analytics
    
    def _get_confidence_distribution(self) -> Dict[str, Any]:
        """Analysiert Confidence-Verteilung"""
        try:
            query = """
            SELECT 
                CASE 
                    WHEN confidence >= 0.9 THEN 'Very High (0.9-1.0)'
                    WHEN confidence >= 0.7 THEN 'High (0.7-0.9)'
                    WHEN confidence >= 0.5 THEN 'Medium (0.5-0.7)'
                    ELSE 'Low (0.0-0.5)'
                END as confidence_range,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM brexit_debates
            GROUP BY confidence_range
            ORDER BY avg_confidence DESC
            """
            
            results = self.brexit_db.db_manager.execute_query(query)
            
            return {
                'distribution': [
                    {
                        'range': row[0],
                        'count': row[1],
                        'average_confidence': round(row[2], 3)
                    }
                    for row in results
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei Confidence-Analyse: {e}")
            return {}
    
    def _get_keyword_analysis(self) -> Dict[str, Any]:
        """Analysiert Keyword-Verwendung"""
        try:
            # Top Keywords
            top_keywords = self.brexit_db.db_manager.execute_query(
                "SELECT keyword, frequency, category FROM brexit_keywords ORDER BY frequency DESC LIMIT 20"
            )
            
            # Keywords nach Kategorie
            category_stats = self.brexit_db.db_manager.execute_query(
                "SELECT category, COUNT(*) as count, SUM(frequency) as total_frequency FROM brexit_keywords GROUP BY category"
            )
            
            return {
                'top_keywords': [
                    {
                        'keyword': row[0],
                        'frequency': row[1],
                        'category': row[2]
                    }
                    for row in top_keywords
                ],
                'category_breakdown': [
                    {
                        'category': row[0],
                        'unique_keywords': row[1],
                        'total_occurrences': row[2]
                    }
                    for row in category_stats
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei Keyword-Analyse: {e}")
            return {}
    
    def _get_temporal_analysis(self) -> Dict[str, Any]:
        """Analysiert zeitliche Verteilung der Brexit-Debatten"""
        try:
            # Debates pro Monat
            monthly_stats = self.brexit_db.db_manager.execute_query(
                """
                SELECT 
                    SUBSTR(date, 1, 7) as month,
                    COUNT(*) as debate_count,
                    AVG(confidence) as avg_confidence
                FROM brexit_debates bd
                JOIN brexit_sessions bs ON bd.session_id = bs.session_id
                GROUP BY month
                ORDER BY month
                """
            )
            
            return {
                'monthly_breakdown': [
                    {
                        'month': row[0],
                        'debate_count': row[1],
                        'average_confidence': round(row[2], 3)
                    }
                    for row in monthly_stats
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei temporaler Analyse: {e}")
            return {}
    
    def _get_speaker_analysis(self) -> Dict[str, Any]:
        """Analysiert Sprecher in Brexit-Debatten"""
        try:
            # Top Sprecher
            top_speakers = self.brexit_db.db_manager.execute_query(
                """
                SELECT 
                    speaker_name,
                    speaker_party,
                    COUNT(*) as speech_count,
                    SUM(word_count) as total_words,
                    AVG(brexit_relevance_score) as avg_relevance
                FROM brexit_speeches
                GROUP BY speaker_name, speaker_party
                ORDER BY speech_count DESC
                LIMIT 20
                """
            )
            
            return {
                'top_speakers': [
                    {
                        'name': row[0],
                        'party': row[1],
                        'speech_count': row[2],
                        'total_words': row[3],
                        'average_relevance': round(row[4], 3)
                    }
                    for row in top_speakers
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei Sprecher-Analyse: {e}")
            return {}
    
    def export_brexit_data(self, output_format: str = "parquet", 
                          output_path: Optional[str] = None) -> str:
        """
        Exportiert Brexit-Daten in verschiedenen Formaten
        
        Args:
            output_format: Export-Format ('parquet', 'csv', 'json')
            output_path: Ausgabepfad (optional)
            
        Returns:
            Pfad zur exportierten Datei
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"parser/data/exported/brexit_data_{timestamp}.{output_format}"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if output_format == "parquet":
                # Exportiere alle Brexit-Debatten als Parquet
                query = """
                SELECT 
                    bd.*,
                    bs.date,
                    bs.session_type
                FROM brexit_debates bd
                JOIN brexit_sessions bs ON bd.session_id = bs.session_id
                """
                
                df = self.brexit_db.db_manager.execute_query_as_dataframe(query)
                df.to_parquet(output_path, engine='pyarrow')
                
            elif output_format == "csv":
                # Exportiere als CSV
                query = """
                SELECT 
                    bd.*,
                    bs.date,
                    bs.session_type
                FROM brexit_debates bd
                JOIN brexit_sessions bs ON bd.session_id = bs.session_id
                """
                
                df = self.brexit_db.db_manager.execute_query_as_dataframe(query)
                df.to_csv(output_path, index=False)
                
            elif output_format == "json":
                # Exportiere als JSON
                debates = self.brexit_db.search_brexit_debates()
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(debates, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Brexit-Daten exportiert: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Fehler beim Export: {e}")
            raise
    
    def close(self):
        """Schließt alle Datenbankverbindungen"""
        self.source_db_manager.close()
        self.brexit_db.close()
    
    def __enter__(self):
        """Context Manager Entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Exit"""
        self.close()


# Convenience-Funktionen
def create_brexit_pipeline(google_api_key: Optional[str] = None) -> BrexitPipeline:
    """
    Factory-Funktion für BrexitPipeline
    
    Args:
        google_api_key: Google AI API Key
        
    Returns:
        BrexitPipeline Instanz
    """
    return BrexitPipeline(google_api_key=google_api_key)


if __name__ == "__main__":
    # Test der Brexit-Pipeline
    print("Brexit-Pipeline Test")
    print("Stelle sicher, dass GOOGLE_API_KEY Umgebungsvariable gesetzt ist")
    
    # Test mit kleinen XML-Dateien
    with create_brexit_pipeline() as pipeline:
        # Test mit Test-Dateien
        test_dir = "scraper/data/raw_test"
        if Path(test_dir).exists():
            stats = pipeline.process_xml_files(test_dir, max_files=2)
            print(f"Test-Ergebnisse: {stats}")
            
            # Analytics
            analytics = pipeline.get_brexit_analytics()
            print(f"Analytics: {analytics}")
        else:
            print(f"Test-Verzeichnis {test_dir} nicht gefunden")

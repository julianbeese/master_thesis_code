"""
Debates XML-Parser für theyworkforyou.com
Spezialisierter Parser für debates XML-Dateien mit DuckDB-Integration
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import logging
from datetime import datetime
import re
try:
    import polars as pl
    import pandas as pd
except ImportError:
    print("Polars oder Pandas nicht verfügbar. Installiere mit: pip install polars pandas")
    raise

try:
    from .models.parliament_models import (
        Speaker,
        Paragraph,
        Speech,
        Debate,
        ParliamentSession,
        clean_text,
        extract_person_id_from_url
    )
except ImportError:
    from models.parliament_models import (
        Speaker,
        Paragraph,
        Speech,
        Debate,
        ParliamentSession,
        clean_text,
        extract_person_id_from_url
    )


class DebatesXMLParser:
    """
    Spezialisierter Parser für debates XML-Dateien
    Extrahiert strukturierte Daten für ML/NLP-Anwendungen
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialisiert den Debates XML-Parser
        
        Args:
            logger: Optional Logger-Instanz
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # XML-Namespace (falls vorhanden)
        self.namespace = None
        
        # Statistiken
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_speeches': 0,
            'total_debates': 0,
            'total_words': 0
        }
    
    def parse_xml_file(self, file_path: Path) -> Optional[ParliamentSession]:
        """
        Parst eine debates XML-Datei und extrahiert strukturierte Daten
        
        Args:
            file_path: Pfad zur XML-Datei
            
        Returns:
            ParliamentSession-Objekt oder None bei Fehlern
        """
        try:
            self.logger.info(f"Parse debates XML-Datei: {file_path.name}")
            
            # XML-Datei laden
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Session-Informationen extrahieren
            session_id = self._extract_session_id(file_path.name)
            date = self._extract_date_from_filename(file_path.name)
            session_type = "debates"
            
            if not date:
                self.logger.warning(f"Konnte Datum nicht aus Dateinamen extrahieren: {file_path.name}")
                return None
            
            # Debatten extrahieren
            debates = self._extract_debates(root)
            
            # ParliamentSession erstellen
            session = ParliamentSession(
                session_id=session_id,
                date=date,
                session_type=session_type,
                debates=debates
            )
            
            # Statistiken aktualisieren
            self.stats['files_processed'] += 1
            self.stats['total_debates'] += session.total_debates
            self.stats['total_speeches'] += session.total_speeches
            self.stats['total_words'] += session.total_words
            
            self.logger.info(
                f"Debates XML erfolgreich geparst: {session.total_debates} Debatten, "
                f"{session.total_speeches} Reden, {session.total_words} Wörter"
            )
            
            return session
            
        except ET.ParseError as e:
            self.logger.error(f"XML-Parse-Fehler in {file_path.name}: {e}")
            self.stats['files_failed'] += 1
            return None
        except Exception as e:
            self.logger.error(f"Unerwarteter Fehler beim Parsen von {file_path.name}: {e}")
            self.stats['files_failed'] += 1
            return None
    
    def _extract_session_id(self, filename: str) -> str:
        """Extrahiert Session-ID aus Dateinamen"""
        # Entferne .xml Extension
        session_id = filename.replace('.xml', '')
        return session_id
    
    def _extract_date_from_filename(self, filename: str) -> Optional[str]:
        """Extrahiert Datum aus debates Dateinamen"""
        # Pattern: debates2021-12-16c.xml
        date_match = re.search(r'debates(\d{4}-\d{2}-\d{2})', filename)
        if date_match:
            return date_match.group(1)
        return None
    
    def _extract_debates(self, root: ET.Element) -> List[Debate]:
        """Extrahiert alle Debatten aus dem XML-Root"""
        debates = []
        current_debate = None
        
        for element in root:
            if element.tag == 'oral-heading':
                # Neue Debatte starten
                if current_debate:
                    debates.append(current_debate)
                
                current_debate = Debate(
                    title=clean_text(element.text or ""),
                    heading_id=element.get('id'),
                    time=element.get('time'),
                    url=element.get('url'),
                    column_number=self._safe_int(element.get('colnum'))
                )
            
            elif element.tag == 'major-heading':
                # Hauptüberschrift - aktualisiere aktuelle Debatte
                if current_debate:
                    current_debate.title += f" - {clean_text(element.text or '')}"
                    current_debate.heading_id = element.get('id')
            
            elif element.tag == 'minor-heading':
                # Unterüberschrift - neue Debatte oder Unterthema
                if current_debate:
                    debates.append(current_debate)
                
                current_debate = Debate(
                    title=clean_text(element.text or ""),
                    heading_id=element.get('id'),
                    time=element.get('time'),
                    url=element.get('url'),
                    column_number=self._safe_int(element.get('colnum'))
                )
            
            elif element.tag == 'speech' and current_debate:
                # Rede zur aktuellen Debatte hinzufügen
                speech = self._extract_speech(element)
                if speech:
                    current_debate.speeches.append(speech)
                    # Statistiken aktualisieren
                    current_debate._update_statistics()
        
        # Letzte Debatte hinzufügen
        if current_debate:
            debates.append(current_debate)
        
        return debates
    
    def _extract_speech(self, speech_element: ET.Element) -> Optional[Speech]:
        """Extrahiert eine Rede aus einem speech-Element"""
        try:
            speech_id = speech_element.get('id', '')
            speaker_name = speech_element.get('speakername', '')
            person_id = speech_element.get('person_id', '')
            speech_type = speech_element.get('type', '')
            time = speech_element.get('time', '')
            url = speech_element.get('url', '')
            column_number = self._safe_int(speech_element.get('colnum'))
            
            # Speaker erstellen
            speaker = Speaker.create(
                name=speaker_name,
                person_id=person_id
            )
            
            # Absätze extrahieren
            paragraphs = []
            for p_element in speech_element.findall('p'):
                paragraph = self._extract_paragraph(p_element)
                if paragraph:
                    paragraphs.append(paragraph)
            
            # Speech erstellen
            speech = Speech(
                speech_id=speech_id,
                speaker=speaker,
                paragraphs=paragraphs,
                time=time,
                column_number=column_number,
                url=url
            )
            
            return speech
            
        except Exception as e:
            self.logger.error(f"Fehler beim Extrahieren der Rede: {e}")
            return None
    
    def _extract_paragraph(self, p_element: ET.Element) -> Optional[Paragraph]:
        """Extrahiert einen Absatz aus einem p-Element"""
        try:
            text = clean_text(p_element.text or "")
            paragraph_id = p_element.get('pid', '')
            class_name = p_element.get('class', '')
            
            # Prüfe auf Motion-Text
            motion_text = 'motion' in class_name.lower()
            
            paragraph = Paragraph(
                text=text,
                paragraph_id=paragraph_id,
                class_name=class_name,
                motion_text=motion_text
            )
            
            return paragraph
            
        except Exception as e:
            self.logger.error(f"Fehler beim Extrahieren des Absatzes: {e}")
            return None
    
    def _safe_int(self, value: Optional[str]) -> Optional[int]:
        """Sichere Konvertierung zu Integer"""
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None
    
    def parse_directory(self, directory_path: Path, max_files: Optional[int] = None) -> List[ParliamentSession]:
        """
        Parst alle XML-Dateien in einem Verzeichnis
        
        Args:
            directory_path: Pfad zum Verzeichnis
            max_files: Maximale Anzahl zu verarbeitender Dateien
            
        Returns:
            Liste von ParliamentSession-Objekten
        """
        sessions = []
        xml_files = list(directory_path.glob("*.xml"))
        
        if max_files:
            xml_files = xml_files[:max_files]
        
        self.logger.info(f"Verarbeite {len(xml_files)} XML-Dateien aus {directory_path}")
        
        for i, xml_file in enumerate(xml_files, 1):
            self.logger.info(f"Verarbeite Datei {i}/{len(xml_files)}: {xml_file.name}")
            
            session = self.parse_xml_file(xml_file)
            if session:
                sessions.append(session)
        
        self.logger.info(f"Verarbeitung abgeschlossen: {len(sessions)} Sitzungen erfolgreich geparst")
        return sessions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Parser-Statistiken zurück"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Setzt Statistiken zurück"""
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_speeches': 0,
            'total_debates': 0,
            'total_words': 0
        }


class DebatesDataProcessor:
    """
    Datenprozessor für debates-Daten
    Konvertiert ParliamentSession-Objekte zu DataFrames für DuckDB
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialisiert den DebatesDataProcessor
        
        Args:
            logger: Optional Logger-Instanz
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def sessions_to_dataframes(self, sessions: List[ParliamentSession]) -> Dict[str, pl.DataFrame]:
        """
        Konvertiert ParliamentSession-Objekte zu Polars DataFrames
        
        Args:
            sessions: Liste von ParliamentSession-Objekten
            
        Returns:
            Dictionary mit DataFrames für verschiedene Tabellen
        """
        self.logger.info(f"Konvertiere {len(sessions)} Sitzungen zu DataFrames")
        
        # Listen für verschiedene Entitäten
        sessions_data = []
        debates_data = []
        speeches_data = []
        speakers_data = []
        paragraphs_data = []
        
        for session in sessions:
            # Session-Daten
            sessions_data.append({
                'session_id': session.session_id,
                'date': session.date,
                'session_type': session.session_type,
                'total_debates': session.total_debates,
                'total_speeches': session.total_speeches,
                'total_words': session.total_words,
                'unique_speakers': session.unique_speakers,
                'duration_hours': session.duration_hours,
                'processing_timestamp': session.processing_timestamp
            })
            
            for debate in session.debates:
                # Debatten-Daten
                debates_data.append({
                    'debate_id': f"{session.session_id}_{debate.heading_id or 'unknown'}",
                    'session_id': session.session_id,
                    'title': debate.title,
                    'heading_id': debate.heading_id,
                    'time': debate.time,
                    'url': debate.url,
                    'column_number': debate.column_number,
                    'total_speeches': debate.total_speeches,
                    'total_words': debate.total_words,
                    'unique_speakers': debate.unique_speakers,
                    'duration_minutes': debate.duration_minutes
                })
                
                for speech in debate.speeches:
                    # Rede-Daten
                    speeches_data.append({
                        'speech_id': speech.speech_id,
                        'debate_id': f"{session.session_id}_{debate.heading_id or 'unknown'}",
                        'session_id': session.session_id,
                        'speaker_name': speech.speaker.name,
                        'speaker_person_id': speech.speaker.person_id,
                        'speaker_title': speech.speaker.title,
                        'speaker_party': speech.speaker.party,
                        'speaker_constituency': speech.speaker.constituency,
                        'time': speech.time,
                        'column_number': speech.column_number,
                        'url': speech.url,
                        'word_count': speech.word_count,
                        'char_count': speech.char_count,
                        'is_intervention': speech.is_intervention,
                        'paragraph_count': len(speech.paragraphs),
                        'full_text': speech.full_text
                    })
                    
                    # Sprecher-Daten (einmalig pro Sprecher)
                    if speech.speaker.name:
                        speakers_data.append({
                            'speaker_name': speech.speaker.name,
                            'person_id': speech.speaker.person_id,
                            'title': speech.speaker.title,
                            'party': speech.speaker.party,
                            'constituency': speech.speaker.constituency
                        })
                    
                    for paragraph in speech.paragraphs:
                        # Absatz-Daten
                        paragraphs_data.append({
                            'paragraph_id': paragraph.paragraph_id,
                            'speech_id': speech.speech_id,
                            'text': paragraph.text,
                            'class_name': paragraph.class_name,
                            'motion_text': paragraph.motion_text,
                            'word_count': paragraph.word_count
                        })
        
        # DataFrames erstellen mit expliziten Datentypen
        dataframes = {}
        
        if sessions_data:
            dataframes['sessions'] = pl.DataFrame(sessions_data, schema={
                'session_id': pl.Utf8,
                'date': pl.Utf8,
                'session_type': pl.Utf8,
                'total_debates': pl.Int64,
                'total_speeches': pl.Int64,
                'total_words': pl.Int64,
                'unique_speakers': pl.Int64,
                'duration_hours': pl.Float64,
                'processing_timestamp': pl.Datetime
            })
        
        if debates_data:
            dataframes['debates'] = pl.DataFrame(debates_data, schema={
                'debate_id': pl.Utf8,
                'session_id': pl.Utf8,
                'title': pl.Utf8,
                'heading_id': pl.Utf8,
                'time': pl.Utf8,
                'url': pl.Utf8,
                'column_number': pl.Int64,
                'total_speeches': pl.Int64,
                'total_words': pl.Int64,
                'unique_speakers': pl.Int64,
                'duration_minutes': pl.Float64
            })
        
        if speeches_data:
            dataframes['speeches'] = pl.DataFrame(speeches_data, schema={
                'speech_id': pl.Utf8,
                'debate_id': pl.Utf8,
                'session_id': pl.Utf8,
                'speaker_name': pl.Utf8,
                'speaker_person_id': pl.Utf8,
                'speaker_title': pl.Utf8,
                'speaker_party': pl.Utf8,
                'speaker_constituency': pl.Utf8,
                'time': pl.Utf8,
                'column_number': pl.Int64,
                'url': pl.Utf8,
                'word_count': pl.Int64,
                'char_count': pl.Int64,
                'is_intervention': pl.Boolean,
                'paragraph_count': pl.Int64,
                'full_text': pl.Utf8
            })
        
        if speakers_data:
            # Duplikate entfernen
            speakers_df = pl.DataFrame(speakers_data, schema={
                'speaker_name': pl.Utf8,
                'person_id': pl.Utf8,
                'title': pl.Utf8,
                'party': pl.Utf8,
                'constituency': pl.Utf8
            })
            dataframes['speakers'] = speakers_df.unique(subset=['speaker_name'])
        
        if paragraphs_data:
            dataframes['paragraphs'] = pl.DataFrame(paragraphs_data, schema={
                'paragraph_id': pl.Utf8,
                'speech_id': pl.Utf8,
                'text': pl.Utf8,
                'class_name': pl.Utf8,
                'motion_text': pl.Boolean,
                'word_count': pl.Int64
            })
        
        self.logger.info(f"DataFrames erstellt: {list(dataframes.keys())}")
        return dataframes
    
    def create_ml_features(self, speeches_df: pl.DataFrame) -> pl.DataFrame:
        """
        Erstellt ML-Features aus den Rede-Daten
        
        Args:
            speeches_df: DataFrame mit Rede-Daten
            
        Returns:
            DataFrame mit ML-Features
        """
        self.logger.info("Erstelle ML-Features aus Rede-Daten")
        
        # Basis-Features
        ml_features = speeches_df.select([
            'speech_id',
            'speaker_name',
            'speaker_person_id',
            'word_count',
            'char_count',
            'is_intervention',
            'paragraph_count'
        ]).with_columns([
            # Text-Länge Features
            (pl.col('char_count') / pl.col('word_count')).alias('avg_chars_per_word'),
            (pl.col('word_count') / pl.col('paragraph_count')).alias('avg_words_per_paragraph'),
            
            # Kategorische Features
            pl.when(pl.col('word_count') < 50).then(pl.lit('short'))
            .when(pl.col('word_count') < 200).then(pl.lit('medium'))
            .otherwise(pl.lit('long')).alias('speech_length_category'),
            
            # Sprecher-Features
            pl.col('speaker_name').is_null().alias('anonymous_speaker'),
            pl.col('speaker_person_id').is_null().alias('unknown_person_id')
        ])
        
        self.logger.info(f"ML-Features erstellt: {ml_features.shape}")
        return ml_features


def main():
    """Hauptfunktion für Tests"""
    # Setup Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Parser initialisieren
    parser = DebatesXMLParser(logger)
    processor = DebatesDataProcessor(logger)
    
    # Test mit einer Datei
    test_file = Path("scraper/data/debates/debates2021-12-16c.xml")
    if test_file.exists():
        session = parser.parse_xml_file(test_file)
        if session:
            print(f"Session geparst: {session.session_id}")
            print(f"Debatten: {session.total_debates}")
            print(f"Reden: {session.total_speeches}")
            print(f"Wörter: {session.total_words}")
            
            # Zu DataFrames konvertieren
            dataframes = processor.sessions_to_dataframes([session])
            print(f"DataFrames erstellt: {list(dataframes.keys())}")
            
            # ML-Features erstellen
            if 'speeches' in dataframes:
                ml_features = processor.create_ml_features(dataframes['speeches'])
                print(f"ML-Features: {ml_features.shape}")
    else:
        print(f"Test-Datei nicht gefunden: {test_file}")


if __name__ == "__main__":
    main()

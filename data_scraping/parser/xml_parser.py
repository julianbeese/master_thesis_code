"""
XML-Parser für Parlamentsdaten
Extrahiert strukturierte Daten aus XML-Dateien von theyworkforyou.com
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import re

from .models.parliament_models import (
    ParliamentSession,
    Speech,
    Speaker,
    Paragraph,
    Debate,
    extract_date_from_filename,
    extract_session_type_from_filename,
    clean_text,
    extract_person_id_from_url
)


class ParliamentXMLParser:
    """
    Parser für Parlaments-XML-Dateien
    Extrahiert strukturierte Daten für ML-Modelle
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialisiert den XML-Parser
        
        Args:
            logger: Optional Logger-Instanz
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # XML-Namespace (falls vorhanden)
        self.namespace = None
    
    def parse_xml_file(self, file_path: Path) -> Optional[ParliamentSession]:
        """
        Parst eine XML-Datei und extrahiert strukturierte Daten
        
        Args:
            file_path: Pfad zur XML-Datei
            
        Returns:
            ParliamentSession-Objekt oder None bei Fehlern
        """
        try:
            self.logger.info(f"Parse XML-Datei: {file_path.name}")
            
            # XML-Datei laden
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Session-Informationen extrahieren
            session_id = self._extract_session_id(file_path.name)
            date = extract_date_from_filename(file_path.name)
            session_type = extract_session_type_from_filename(file_path.name)
            
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
            
            self.logger.info(
                f"XML erfolgreich geparst: {session.total_debates} Debatten, "
                f"{session.total_speeches} Reden, {session.total_words} Wörter"
            )
            
            return session
            
        except ET.ParseError as e:
            self.logger.error(f"XML-Parse-Fehler in {file_path.name}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unerwarteter Fehler beim Parsen von {file_path.name}: {e}")
            return None
    
    def _extract_session_id(self, filename: str) -> str:
        """Extrahiert Session-ID aus Dateinamen"""
        # Entferne .xml Extension
        session_id = filename.replace('.xml', '')
        return session_id
    
    def _extract_debates(self, root: ET.Element) -> List[Debate]:
        """Extrahiert alle Debatten aus dem XML-Root"""
        debates = []
        current_debate = None
        
        for element in root:
            if element.tag == 'minor-heading':
                # Neue Debatte starten
                if current_debate:
                    debates.append(current_debate)
                
                current_debate = self._create_debate_from_heading(element)
                
            elif element.tag == 'speech':
                # Rede extrahieren
                speech = self._extract_speech(element)
                if speech:
                    # Wenn keine aktuelle Debatte, eine Standard-Debatte erstellen
                    if not current_debate:
                        current_debate = Debate(title="General Debate")
                    
                    current_debate.speeches.append(speech)
                    current_debate._update_statistics()
        
        # Letzte Debatte hinzufügen
        if current_debate:
            debates.append(current_debate)
        
        return debates
    
    def _create_debate_from_heading(self, heading_element: ET.Element) -> Debate:
        """Erstellt Debate-Objekt aus minor-heading Element"""
        title = clean_text(heading_element.text or "")
        
        return Debate(
            title=title,
            heading_id=heading_element.get('id'),
            time=heading_element.get('time'),
            url=heading_element.get('url'),
            column_number=self._parse_int(heading_element.get('colnum'))
        )
    
    def _extract_speech(self, speech_element: ET.Element) -> Optional[Speech]:
        """Extrahiert Speech-Objekt aus speech Element"""
        try:
            # Sprecher-Informationen
            speaker_name = speech_element.get('speakername', '') or speech_element.get('speaker_name', '')
            person_id = speech_element.get('person_id', '')
            
            if not speaker_name:
                self.logger.warning("Rede ohne Sprecher-Name gefunden")
                return None
            
            # Speaker-Objekt erstellen
            speaker = Speaker.create(
                name=speaker_name,
                person_id=extract_person_id_from_url(person_id)
            )
            
            # Absätze extrahieren
            paragraphs = []
            for p_element in speech_element.findall('p'):
                paragraph = self._extract_paragraph(p_element)
                if paragraph:
                    paragraphs.append(paragraph)
            
            # Speech-Objekt erstellen
            speech = Speech(
                speech_id=speech_element.get('id', ''),
                speaker=speaker,
                paragraphs=paragraphs,
                time=speech_element.get('time'),
                column_number=self._parse_int(speech_element.get('colnum')),
                url=speech_element.get('url')
            )
            
            return speech
            
        except Exception as e:
            self.logger.error(f"Fehler beim Extrahieren der Rede: {e}")
            return None
    
    def _extract_paragraph(self, p_element: ET.Element) -> Optional[Paragraph]:
        """Extrahiert Paragraph-Objekt aus p Element"""
        try:
            text = clean_text(p_element.text or "")
            
            if not text.strip():
                return None
            
            # Zusätzliche Attribute
            paragraph_id = p_element.get('pid')
            class_name = p_element.get('class')
            motion_text = p_element.get('pwmotiontext') == 'yes'
            
            return Paragraph(
                text=text,
                paragraph_id=paragraph_id,
                class_name=class_name,
                motion_text=motion_text
            )
            
        except Exception as e:
            self.logger.error(f"Fehler beim Extrahieren des Absatzes: {e}")
            return None
    
    def _parse_int(self, value: Optional[str]) -> Optional[int]:
        """Konvertiert String zu Integer, gibt None bei Fehlern zurück"""
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None
    
    def parse_multiple_files(self, file_paths: List[Path]) -> List[ParliamentSession]:
        """
        Parst mehrere XML-Dateien
        
        Args:
            file_paths: Liste von XML-Dateipfaden
            
        Returns:
            Liste von ParliamentSession-Objekten
        """
        sessions = []
        
        for file_path in file_paths:
            session = self.parse_xml_file(file_path)
            if session:
                sessions.append(session)
        
        self.logger.info(f"Erfolgreich {len(sessions)} von {len(file_paths)} Dateien geparst")
        return sessions
    
    def get_parsing_statistics(self, sessions: List[ParliamentSession]) -> Dict[str, Any]:
        """
        Gibt Statistiken über geparste Sitzungen zurück
        
        Args:
            sessions: Liste von ParliamentSession-Objekten
            
        Returns:
            Dictionary mit Statistiken
        """
        if not sessions:
            return {}
        
        total_debates = sum(s.total_debates for s in sessions)
        total_speeches = sum(s.total_speeches for s in sessions)
        total_words = sum(s.total_words for s in sessions)
        
        # Sprecher-Statistiken
        all_speakers = set()
        for session in sessions:
            for debate in session.debates:
                for speech in debate.speeches:
                    all_speakers.add(speech.speaker.name)
        
        # Datum-Bereich
        dates = [s.date for s in sessions if s.date]
        date_range = f"{min(dates)} - {max(dates)}" if dates else "Unbekannt"
        
        return {
            'total_sessions': len(sessions),
            'total_debates': total_debates,
            'total_speeches': total_speeches,
            'total_words': total_words,
            'unique_speakers': len(all_speakers),
            'date_range': date_range,
            'average_words_per_speech': total_words / total_speeches if total_speeches > 0 else 0,
            'average_speeches_per_session': total_speeches / len(sessions) if sessions else 0
        }


# Convenience-Funktionen
def parse_single_xml_file(file_path: Path) -> Optional[ParliamentSession]:
    """
    Parst eine einzelne XML-Datei
    
    Args:
        file_path: Pfad zur XML-Datei
        
    Returns:
        ParliamentSession-Objekt oder None
    """
    parser = ParliamentXMLParser()
    return parser.parse_xml_file(file_path)


def parse_xml_directory(directory_path: Path, pattern: str = "*.xml") -> List[ParliamentSession]:
    """
    Parst alle XML-Dateien in einem Verzeichnis
    
    Args:
        directory_path: Verzeichnis-Pfad
        pattern: Glob-Pattern für Dateien
        
    Returns:
        Liste von ParliamentSession-Objekten
    """
    parser = ParliamentXMLParser()
    xml_files = list(directory_path.glob(pattern))
    return parser.parse_multiple_files(xml_files)


def extract_text_for_ml(session: ParliamentSession) -> List[Dict[str, Any]]:
    """
    Extrahiert Text-Daten für ML-Modelle
    
    Args:
        session: ParliamentSession-Objekt
        
    Returns:
        Liste von Dictionaries mit Text-Daten
    """
    ml_data = []
    
    for debate in session.debates:
        for speech in debate.speeches:
            ml_data.append({
                'session_id': session.session_id,
                'date': session.date,
                'session_type': session.session_type,
                'debate_title': debate.title,
                'speech_id': speech.speech_id,
                'speaker_name': speech.speaker.name,
                'speaker_person_id': speech.speaker.person_id,
                'speaker_title': speech.speaker.title,
                'time': speech.time,
                'text': speech.full_text,
                'word_count': speech.word_count,
                'char_count': speech.char_count,
                'is_intervention': speech.is_intervention,
                'paragraph_count': len(speech.paragraphs)
            })
    
    return ml_data

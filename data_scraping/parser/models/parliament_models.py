"""
Datenmodelle für Parlamentsdaten
Definiert die Struktur der extrahierten Daten
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import re


@dataclass(frozen=True)
class Speaker:
    """Modell für einen Sprecher"""
    name: str
    person_id: Optional[str] = None
    party: Optional[str] = None
    constituency: Optional[str] = None
    title: Optional[str] = None  # z.B. "Mr", "Mrs", "Dr"
    
    @classmethod
    def create(cls, name: str, person_id: Optional[str] = None, 
               party: Optional[str] = None, constituency: Optional[str] = None):
        """Erstellt einen Speaker mit bereinigten Daten"""
        if not name:
            return cls(name="", person_id=person_id, party=party, constituency=constituency)
        
        # Extrahiere Titel aus dem Namen
        title = None
        clean_name = name
        title_match = re.match(r'^(Mr|Mrs|Ms|Dr|Sir|Dame|Lord|Lady)\s+', name)
        if title_match:
            title = title_match.group(1)
            clean_name = name[len(title_match.group(0)):].strip()
        
        # Entferne Klammern und zusätzliche Informationen
        clean_name = re.sub(r'\s*\([^)]*\)', '', clean_name).strip()
        
        return cls(name=clean_name, person_id=person_id, party=party, 
                  constituency=constituency, title=title)


@dataclass
class Paragraph:
    """Modell für einen Absatz in einer Rede"""
    text: str
    paragraph_id: Optional[str] = None
    class_name: Optional[str] = None  # z.B. "indent"
    motion_text: bool = False
    word_count: int = field(init=False)
    
    def __post_init__(self):
        """Berechnet Wortanzahl"""
        self.word_count = len(self.text.split()) if self.text else 0


@dataclass
class Speech:
    """Modell für eine Rede"""
    speech_id: str
    speaker: Speaker
    paragraphs: List[Paragraph] = field(default_factory=list)
    time: Optional[str] = None
    column_number: Optional[int] = None
    url: Optional[str] = None
    word_count: int = field(init=False)
    char_count: int = field(init=False)
    
    def __post_init__(self):
        """Berechnet Statistiken"""
        self.word_count = sum(p.word_count for p in self.paragraphs)
        self.char_count = sum(len(p.text) for p in self.paragraphs)
    
    @property
    def full_text(self) -> str:
        """Gibt den vollständigen Text der Rede zurück"""
        return ' '.join(p.text for p in self.paragraphs)
    
    @property
    def is_intervention(self) -> bool:
        """Prüft ob es sich um eine Intervention handelt (kurze Rede)"""
        return self.word_count < 50


@dataclass
class Debate:
    """Modell für eine Debatte/Thema"""
    title: str
    heading_id: Optional[str] = None
    time: Optional[str] = None
    url: Optional[str] = None
    column_number: Optional[int] = None
    speeches: List[Speech] = field(default_factory=list)
    total_speeches: int = field(init=False)
    total_words: int = field(init=False)
    unique_speakers: int = field(init=False)
    
    def __post_init__(self):
        """Berechnet Statistiken"""
        self._update_statistics()
    
    def _update_statistics(self):
        """Aktualisiert die Statistiken"""
        self.total_speeches = len(self.speeches)
        self.total_words = sum(speech.word_count for speech in self.speeches)
        self.unique_speakers = len(set(speech.speaker.name for speech in self.speeches))
    
    @property
    def duration_minutes(self) -> Optional[float]:
        """Schätzt die Dauer der Debatte in Minuten basierend auf Wortanzahl"""
        if self.total_words == 0:
            return 0
        # Schätzung: 150 Wörter pro Minute (durchschnittliche Sprechgeschwindigkeit)
        return self.total_words / 150


@dataclass
class ParliamentSession:
    """Modell für eine Parlamentssitzung"""
    session_id: str
    date: str
    session_type: str = "westminster"  # westminster, scotland, wales, ni
    debates: List[Debate] = field(default_factory=list)
    total_debates: int = field(init=False)
    total_speeches: int = field(init=False)
    total_words: int = field(init=False)
    unique_speakers: int = field(init=False)
    processing_timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Berechnet Statistiken"""
        self.total_debates = len(self.debates)
        self.total_speeches = sum(debate.total_speeches for debate in self.debates)
        self.total_words = sum(debate.total_words for debate in self.debates)
        
        # Sammle alle Sprecher
        all_speakers = set()
        for debate in self.debates:
            for speech in debate.speeches:
                all_speakers.add(speech.speaker.name)
        self.unique_speakers = len(all_speakers)
    
    @property
    def duration_hours(self) -> float:
        """Schätzt die Gesamtdauer der Sitzung in Stunden"""
        total_minutes = sum(debate.duration_minutes or 0 for debate in self.debates)
        return total_minutes / 60
    
    def get_speaker_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken über Sprecher zurück"""
        speaker_stats = {}
        
        for debate in self.debates:
            for speech in debate.speeches:
                speaker_name = speech.speaker.name
                if speaker_name not in speaker_stats:
                    speaker_stats[speaker_name] = {
                        'speeches': 0,
                        'words': 0,
                        'debates': set()
                    }
                
                speaker_stats[speaker_name]['speeches'] += 1
                speaker_stats[speaker_name]['words'] += speech.word_count
                speaker_stats[speaker_name]['debates'].add(debate.title)
        
        # Konvertiere sets zu counts
        for speaker in speaker_stats:
            speaker_stats[speaker]['debate_count'] = len(speaker_stats[speaker]['debates'])
            del speaker_stats[speaker]['debates']
        
        return speaker_stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert die Sitzung zu einem Dictionary für JSON-Serialisierung"""
        return {
            'session_id': self.session_id,
            'date': self.date,
            'session_type': self.session_type,
            'total_debates': self.total_debates,
            'total_speeches': self.total_speeches,
            'total_words': self.total_words,
            'unique_speakers': self.unique_speakers,
            'duration_hours': self.duration_hours,
            'processing_timestamp': self.processing_timestamp.isoformat(),
            'debates': [
                {
                    'title': debate.title,
                    'heading_id': debate.heading_id,
                    'time': debate.time,
                    'total_speeches': debate.total_speeches,
                    'total_words': debate.total_words,
                    'unique_speakers': debate.unique_speakers,
                    'duration_minutes': debate.duration_minutes,
                    'speeches': [
                        {
                            'speech_id': speech.speech_id,
                            'speaker_name': speech.speaker.name,
                            'speaker_person_id': speech.speaker.person_id,
                            'speaker_title': speech.speaker.title,
                            'time': speech.time,
                            'word_count': speech.word_count,
                            'char_count': speech.char_count,
                            'is_intervention': speech.is_intervention,
                            'paragraphs': [
                                {
                                    'text': p.text,
                                    'paragraph_id': p.paragraph_id,
                                    'class_name': p.class_name,
                                    'motion_text': p.motion_text,
                                    'word_count': p.word_count
                                }
                                for p in speech.paragraphs
                            ]
                        }
                        for speech in debate.speeches
                    ]
                }
                for debate in self.debates
            ]
        }


# Hilfsfunktionen für Datenverarbeitung
def extract_date_from_filename(filename: str) -> Optional[str]:
    """Extrahiert Datum aus Dateinamen"""
    # Pattern: westminster2020-01-21a.xml
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if date_match:
        return date_match.group(1)
    return None


def extract_session_type_from_filename(filename: str) -> str:
    """Extrahiert Sitzungstyp aus Dateinamen"""
    if 'westminster' in filename.lower():
        return 'westminster'
    elif 'scotland' in filename.lower():
        return 'scotland'
    elif 'wales' in filename.lower():
        return 'wales'
    elif 'ni' in filename.lower():
        return 'ni'
    else:
        return 'unknown'


def clean_text(text: str) -> str:
    """Bereinigt Text von XML-Artifakten und normalisiert"""
    if not text:
        return ""
    
    # Entferne XML-Entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&apos;', "'")
    
    # Entferne überflüssige Whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Entferne führende/nachfolgende Whitespaces
    text = text.strip()
    
    return text


def extract_person_id_from_url(url: str) -> Optional[str]:
    """Extrahiert Person-ID aus URL"""
    if not url:
        return None
    
    # Pattern: uk.org.publicwhip/person/10499
    person_match = re.search(r'person/(\d+)', url)
    if person_match:
        return person_match.group(1)
    return None

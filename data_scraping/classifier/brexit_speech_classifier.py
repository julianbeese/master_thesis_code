"""
Brexit Speech Klassifikator für Parliament-Debatten
Verwendet Google Gemini für intelligente Erkennung von Brexit-bezogenen Reden
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import time
import re
import duckdb

# Lade .env Datei falls vorhanden
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv ist optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google Generative AI nicht installiert. Installiere mit: pip install google-generativeai")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class BrexitSpeechResult:
    """Ergebnis einer Brexit-Speech-Klassifikation"""
    speech_id: str
    is_brexit_related: bool
    confidence: float  # 0.0 bis 1.0
    reasoning: str
    keywords_found: List[str]
    classification_method: str  # "keyword", "gemini", "hybrid"
    processing_time: float  # Sekunden


class BrexitSpeechClassifier:
    """
    Klassifikator für Brexit-bezogene Parliament-Reden
    Verwendet Google Gemini für intelligente Erkennung
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """
        Initialisiert den Brexit-Speech-Klassifikator
        
        Args:
            api_key: Google AI API Key (falls nicht in Umgebungsvariablen)
            model_name: Gemini-Modell (gemini-1.5-flash ist günstig und schnell)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        # Setup Logging
        self._setup_logging()
        
        # Setup Gemini
        if GEMINI_AVAILABLE and self.api_key:
            self._setup_gemini()
            self.gemini_available = True
        else:
            self.logger.warning("Gemini nicht verfügbar - verwende nur Keyword-Matching")
            self.gemini_available = False
        
        # Brexit-Keywords für erste Filterung
        self.brexit_keywords = {
            # Direkte Brexit-Begriffe
            'brexit', 'brexit-related', 'brexit-related issues',
            'eu exit', 'european union exit', 'leave the eu', 'leaving the eu',
            'article 50', 'article 50 process',
            
            # EU-Bezogene Begriffe
            'european union', 'eu membership', 'eu referendum',
            'single market', 'customs union', 'european economic area',
            'free movement', 'freedom of movement',
            
            # Verhandlungsbegriffe
            'withdrawal agreement', 'trade deal', 'trade agreement',
            'future relationship', 'partnership agreement',
            'northern ireland protocol', 'irish border', 'backstop',
            
            # Institutionelle Begriffe
            'european court of justice', 'ecj', 'european commission',
            'european parliament', 'european council',
            
            # Wirtschaftliche Begriffe
            'european single market', 'customs union', 'tariffs',
            'trade barriers', 'economic partnership',
            
            # Politische Begriffe
            'sovereignty', 'take back control', 'independence',
            'european integration', 'ever closer union'
        }
        
        # Erweiterte Keywords für Kontext
        self.context_keywords = {
            'referendum', 'vote', 'campaign', 'remain', 'leave',
            'migration', 'immigration', 'borders', 'control',
            'economy', 'trade', 'business', 'investment',
            'fishing', 'agriculture', 'manufacturing'
        }
    
    def _setup_logging(self):
        """Setup für Logging"""
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"brexit_speech_classifier_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_gemini(self):
        """Setup für Google Gemini"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.logger.info(f"Gemini-Modell '{self.model_name}' erfolgreich initialisiert")
        except Exception as e:
            self.logger.error(f"Fehler beim Setup von Gemini: {e}")
            self.gemini_available = False
    
    def classify_speech(self, speech_text: str, speech_id: str, speaker_name: str = "") -> BrexitSpeechResult:
        """
        Klassifiziert eine einzelne Rede auf Brexit-Bezug
        
        Args:
            speech_text: Vollständiger Text der Rede
            speech_id: Eindeutige ID der Rede
            speaker_name: Name des Sprechers (optional)
            
        Returns:
            BrexitSpeechResult
        """
        start_time = time.time()
        
        # 1. Keyword-basierte Vorfilterung
        keyword_result = self._classify_by_keywords(speech_text)
        
        # 2. Wenn Keywords gefunden oder unsicher, verwende Gemini
        if keyword_result['is_brexit_related'] or keyword_result['confidence'] < 0.8:
            if self.gemini_available:
                gemini_result = self._classify_by_gemini(speech_text, speaker_name)
                result = self._combine_results(keyword_result, gemini_result)
            else:
                result = keyword_result
        else:
            result = keyword_result
        
        processing_time = time.time() - start_time
        
        return BrexitSpeechResult(
            speech_id=speech_id,
            is_brexit_related=result['is_brexit_related'],
            confidence=result['confidence'],
            reasoning=result['reasoning'],
            keywords_found=result['keywords_found'],
            classification_method=result['classification_method'],
            processing_time=processing_time
        )
    
    def _classify_by_keywords(self, speech_text: str) -> Dict[str, Any]:
        """
        Klassifiziert Rede basierend auf Keywords
        
        Args:
            speech_text: Text der Rede
            
        Returns:
            Dictionary mit Klassifikationsergebnissen
        """
        text_lower = speech_text.lower()
        
        # Suche nach Keywords
        found_keywords = []
        for keyword in self.brexit_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        # Berechne Confidence basierend auf Anzahl und Art der Keywords
        confidence = 0.0
        is_brexit = False
        
        if found_keywords:
            # Direkte Brexit-Begriffe haben höhere Gewichtung
            direct_brexit_terms = {'brexit', 'eu exit', 'european union exit', 'leave the eu'}
            direct_matches = sum(1 for kw in found_keywords if kw in direct_brexit_terms)
            
            confidence = min(0.9, 0.3 + (len(found_keywords) * 0.1) + (direct_matches * 0.2))
            is_brexit = confidence > 0.4
        
        reasoning = f"Keyword-basierte Analyse: {len(found_keywords)} relevante Begriffe gefunden"
        if found_keywords:
            reasoning += f": {', '.join(found_keywords[:5])}"
        
        return {
            'is_brexit_related': is_brexit,
            'confidence': confidence,
            'reasoning': reasoning,
            'keywords_found': found_keywords,
            'classification_method': "keyword"
        }
    
    def _classify_by_gemini(self, speech_text: str, speaker_name: str = "") -> Dict[str, Any]:
        """
        Klassifiziert Rede mit Google Gemini
        
        Args:
            speech_text: Text der Rede
            speaker_name: Name des Sprechers
            
        Returns:
            Dictionary mit Klassifikationsergebnissen
        """
        try:
            # Erstelle Prompt für Gemini
            prompt = self._create_gemini_prompt(speech_text, speaker_name)
            
            # API-Call mit Rate Limiting
            time.sleep(0.1)  # 100ms Pause zwischen Calls
            response = self.model.generate_content(prompt)
            
            # Parse Response
            result = self._parse_gemini_response(response.text)
            
            self.logger.info(f"Gemini-Klassifikation für Speaker '{speaker_name}': {result['is_brexit_related']} (Confidence: {result['confidence']})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei Gemini-Klassifikation: {e}")
            # Fallback zu Keyword-Ergebnis
            return self._classify_by_keywords(speech_text)
    
    def _create_gemini_prompt(self, speech_text: str, speaker_name: str = "") -> str:
        """
        Erstellt Prompt für Gemini-Klassifikation
        
        Args:
            speech_text: Text der Rede
            speaker_name: Name des Sprechers
            
        Returns:
            Prompt-String
        """
        # Begrenze Text für Token-Limit
        max_chars = 3000  # Gemini Token-Limit
        if len(speech_text) > max_chars:
            speech_text = speech_text[:max_chars] + "..."
        
        prompt = f"""
Du bist ein Experte für britische Politik und Brexit. Analysiere die folgende Parliament-Rede und bestimme, ob sie sich auf Brexit oder verwandte EU-Themen bezieht.

SPRECHER: {speaker_name if speaker_name else "Unbekannt"}

REDE-TEXT:
{speech_text}

Bitte antworte im folgenden JSON-Format:
{{
    "is_brexit_related": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Kurze Begründung deiner Entscheidung",
    "keywords_found": ["liste", "der", "relevanten", "begriffe"]
}}

Kriterien für Brexit-Bezug:
- Direkte Erwähnung von Brexit, EU-Austritt, Referendum
- Diskussion über EU-Mitgliedschaft, Single Market, Customs Union
- Verhandlungen über Withdrawal Agreement, Trade Deals
- Northern Ireland Protocol, Irish Border
- EU-Institutionen (ECJ, European Commission, etc.)
- Wirtschaftliche Aspekte der EU-Mitgliedschaft
- Migration, Grenzkontrollen im EU-Kontext
- Souveränität, "Take Back Control"
- Brexit-bezogene Gesetze, Verordnungen, Abkommen

Antworte NUR mit dem JSON, keine zusätzlichen Erklärungen.
"""
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parst Gemini-Response zu Dictionary
        
        Args:
            response_text: Rohe Response von Gemini
            
        Returns:
            Dictionary mit Klassifikationsergebnissen
        """
        try:
            # Extrahiere JSON aus Response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                return {
                    'is_brexit_related': bool(data.get('is_brexit_related', False)),
                    'confidence': float(data.get('confidence', 0.0)),
                    'reasoning': str(data.get('reasoning', '')),
                    'keywords_found': data.get('keywords_found', []),
                    'classification_method': "gemini"
                }
            else:
                raise ValueError("Kein JSON in Response gefunden")
                
        except Exception as e:
            self.logger.error(f"Fehler beim Parsen der Gemini-Response: {e}")
            # Fallback
            return {
                'is_brexit_related': False,
                'confidence': 0.0,
                'reasoning': f"Fehler beim Parsen: {e}",
                'keywords_found': [],
                'classification_method': "gemini_error"
            }
    
    def _combine_results(self, keyword_result: Dict[str, Any], 
                        gemini_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Kombiniert Keyword- und Gemini-Ergebnisse
        
        Args:
            keyword_result: Keyword-basiertes Ergebnis
            gemini_result: Gemini-basiertes Ergebnis
            
        Returns:
            Kombiniertes Ergebnis
        """
        # Gewichtung: 30% Keywords, 70% Gemini
        combined_confidence = (keyword_result['confidence'] * 0.3) + (gemini_result['confidence'] * 0.7)
        
        # Entscheidung basierend auf höherer Confidence
        is_brexit = combined_confidence > 0.5
        
        # Kombiniere Keywords
        all_keywords = list(set(keyword_result['keywords_found'] + gemini_result['keywords_found']))
        
        # Kombiniere Reasoning
        reasoning = f"Hybrid-Analyse: Keywords ({keyword_result['confidence']:.2f}) + Gemini ({gemini_result['confidence']:.2f})"
        
        return {
            'is_brexit_related': is_brexit,
            'confidence': combined_confidence,
            'reasoning': reasoning,
            'keywords_found': all_keywords,
            'classification_method': "hybrid"
        }
    
    def classify_speeches_from_db(self, db_path: str, output_db_path: str, 
                                 batch_size: int = 100, max_speeches: Optional[int] = None) -> Dict[str, Any]:
        """
        Klassifiziert alle Reden aus der Datenbank und speichert Ergebnisse
        
        Args:
            db_path: Pfad zur Quell-Datenbank
            output_db_path: Pfad zur Ausgabe-Datenbank
            batch_size: Anzahl Reden pro Batch
            max_speeches: Maximale Anzahl zu verarbeitender Reden (für Tests)
            
        Returns:
            Dictionary mit Klassifikationsstatistiken
        """
        self.logger.info(f"Starte Klassifikation von Reden aus {db_path}")
        
        # Verbinde zur Quell-Datenbank
        conn = duckdb.connect(db_path, read_only=True)
        
        # Hole alle Reden mit Text
        query = """
            SELECT speech_id, speaker_name, full_text, word_count, char_count
            FROM speeches 
            WHERE full_text IS NOT NULL AND full_text != ''
            ORDER BY word_count DESC
        """
        
        if max_speeches:
            query += f" LIMIT {max_speeches}"
        
        speeches = conn.execute(query).fetchall()
        conn.close()
        
        self.logger.info(f"Gefunden: {len(speeches)} Reden zur Klassifikation")
        
        # Erstelle Ausgabe-Datenbank
        self._create_output_database(output_db_path)
        
        # Klassifiziere in Batches
        results = []
        brexit_speeches = []
        stats = {
            'total_processed': 0,
            'brexit_related': 0,
            'total_confidence': 0.0,
            'processing_times': [],
            'classification_methods': {}
        }
        
        for i in range(0, len(speeches), batch_size):
            batch = speeches[i:i + batch_size]
            self.logger.info(f"Verarbeite Batch {i//batch_size + 1}/{(len(speeches) + batch_size - 1)//batch_size}")
            
            for speech_id, speaker_name, full_text, word_count, char_count in batch:
                # Klassifiziere Rede
                result = self.classify_speech(full_text, speech_id, speaker_name or "")
                
                # Speichere Ergebnis
                results.append({
                    'speech_id': result.speech_id,
                    'is_brexit_related': result.is_brexit_related,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning,
                    'keywords_found': json.dumps(result.keywords_found),
                    'classification_method': result.classification_method,
                    'processing_time': result.processing_time,
                    'speaker_name': speaker_name,
                    'word_count': word_count,
                    'char_count': char_count
                })
                
                # Wenn Brexit-bezogen, speichere auch in separater Tabelle
                if result.is_brexit_related:
                    brexit_speeches.append({
                        'speech_id': result.speech_id,
                        'speaker_name': speaker_name,
                        'full_text': full_text,
                        'word_count': word_count,
                        'char_count': char_count,
                        'confidence': result.confidence,
                        'reasoning': result.reasoning,
                        'keywords_found': json.dumps(result.keywords_found),
                        'classification_method': result.classification_method
                    })
                
                # Update Statistiken
                stats['total_processed'] += 1
                if result.is_brexit_related:
                    stats['brexit_related'] += 1
                stats['total_confidence'] += result.confidence
                stats['processing_times'].append(result.processing_time)
                
                method = result.classification_method
                stats['classification_methods'][method] = stats['classification_methods'].get(method, 0) + 1
                
                # Log Fortschritt
                if stats['total_processed'] % 100 == 0:
                    self.logger.info(f"Verarbeitet: {stats['total_processed']} Reden, {stats['brexit_related']} Brexit-bezogen")
            
            # Speichere Batch in Datenbank
            batch_results = results[-len(batch):]
            batch_brexit_speeches = [s for s in brexit_speeches if s['speech_id'] in [r['speech_id'] for r in batch_results]]
            self._save_batch_to_db(output_db_path, batch_results, batch_brexit_speeches)
        
        # Finale Statistiken
        stats['average_confidence'] = stats['total_confidence'] / stats['total_processed'] if stats['total_processed'] > 0 else 0
        stats['average_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times']) if stats['processing_times'] else 0
        stats['brexit_percentage'] = (stats['brexit_related'] / stats['total_processed'] * 100) if stats['total_processed'] > 0 else 0
        
        self.logger.info(f"Klassifikation abgeschlossen: {stats['total_processed']} Reden, {stats['brexit_related']} Brexit-bezogen ({stats['brexit_percentage']:.1f}%)")
        
        return stats
    
    def _create_output_database(self, output_db_path: str):
        """Erstellt Ausgabe-Datenbank mit Tabellen"""
        conn = duckdb.connect(output_db_path)
        
        # Klassifikationsergebnisse
        conn.execute("""
            CREATE TABLE IF NOT EXISTS classification_results (
                speech_id VARCHAR PRIMARY KEY,
                is_brexit_related BOOLEAN,
                confidence FLOAT,
                reasoning TEXT,
                keywords_found TEXT,
                classification_method VARCHAR,
                processing_time FLOAT,
                speaker_name VARCHAR,
                word_count INTEGER,
                char_count INTEGER
            )
        """)
        
        # Brexit-bezogene Reden
        conn.execute("""
            CREATE TABLE IF NOT EXISTS brexit_speeches (
                speech_id VARCHAR PRIMARY KEY,
                speaker_name VARCHAR,
                full_text TEXT,
                word_count INTEGER,
                char_count INTEGER,
                confidence FLOAT,
                reasoning TEXT,
                keywords_found TEXT,
                classification_method VARCHAR
            )
        """)
        
        # Statistiken
        conn.execute("""
            CREATE TABLE IF NOT EXISTS classification_stats (
                id INTEGER PRIMARY KEY,
                total_processed INTEGER,
                brexit_related INTEGER,
                brexit_percentage FLOAT,
                average_confidence FLOAT,
                average_processing_time FLOAT,
                classification_methods TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.close()
    
    def _save_batch_to_db(self, output_db_path: str, results: List[Dict], brexit_speeches: List[Dict]):
        """Speichert Batch-Ergebnisse in Datenbank"""
        conn = duckdb.connect(output_db_path)
        
        # Speichere Klassifikationsergebnisse
        for result in results:
            conn.execute("""
                INSERT OR REPLACE INTO classification_results 
                (speech_id, is_brexit_related, confidence, reasoning, keywords_found, 
                 classification_method, processing_time, speaker_name, word_count, char_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result['speech_id'], result['is_brexit_related'], result['confidence'],
                result['reasoning'], result['keywords_found'], result['classification_method'],
                result['processing_time'], result['speaker_name'], result['word_count'], result['char_count']
            ))
        
        # Speichere Brexit-Reden
        for speech in brexit_speeches:
            conn.execute("""
                INSERT OR REPLACE INTO brexit_speeches 
                (speech_id, speaker_name, full_text, word_count, char_count, 
                 confidence, reasoning, keywords_found, classification_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                speech['speech_id'], speech['speaker_name'], speech['full_text'],
                speech['word_count'], speech['char_count'], speech['confidence'],
                speech['reasoning'], speech['keywords_found'], speech['classification_method']
            ))
        
        conn.close()


# Convenience-Funktionen
def create_brexit_speech_classifier(api_key: Optional[str] = None) -> BrexitSpeechClassifier:
    """
    Factory-Funktion für BrexitSpeechClassifier
    
    Args:
        api_key: Google AI API Key
        
    Returns:
        BrexitSpeechClassifier Instanz
    """
    return BrexitSpeechClassifier(api_key=api_key)


def classify_speech_quick(speech_text: str, speech_id: str, api_key: Optional[str] = None) -> BrexitSpeechResult:
    """
    Schnelle Klassifikation einer einzelnen Rede
    
    Args:
        speech_text: Text der Rede
        speech_id: ID der Rede
        api_key: Google AI API Key
        
    Returns:
        BrexitSpeechResult
    """
    classifier = create_brexit_speech_classifier(api_key)
    return classifier.classify_speech(speech_text, speech_id)


if __name__ == "__main__":
    # Test des Klassifikators
    print("Brexit-Speech-Klassifikator Test")
    print("Stelle sicher, dass GOOGLE_API_KEY Umgebungsvariable gesetzt ist")
    
    # Test mit Beispiel-Text
    test_text = """
    Mr Speaker, I rise to speak about the implications of Brexit for our fishing industry. 
    The withdrawal from the European Union has fundamentally changed our relationship with 
    the single market and customs union. We must ensure that our fishing communities 
    benefit from taking back control of our waters.
    """
    
    classifier = create_brexit_speech_classifier()
    result = classifier.classify_speech(test_text, "test_speech_1", "Test Speaker")
    
    print(f"Test-Ergebnis: {result}")

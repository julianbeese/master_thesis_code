"""
Brexit-Klassifikator für Parliament-Debatten
Verwendet Google Gemini für intelligente Erkennung von Brexit-bezogenen Debatten
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

from database.managers.database_manager import DatabaseManager
from parser.models.parliament_models import Debate, Speech, ParliamentSession


@dataclass
class BrexitClassificationResult:
    """Ergebnis einer Brexit-Klassifikation"""
    is_brexit_related: bool
    confidence: float  # 0.0 bis 1.0
    reasoning: str
    keywords_found: List[str]
    classification_method: str  # "keyword", "gemini", "hybrid"


class BrexitClassifier:
    """
    Klassifikator für Brexit-bezogene Parliament-Debatten
    Verwendet Google Gemini für intelligente Erkennung
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """
        Initialisiert den Brexit-Klassifikator
        
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

        self.brexit_keywords = {
            # Kern-Brexit-Begriffe
            'brexit', 'brexit-related', 'brexit-related issues',
            'british exit', 'eu exit', 'european union exit',
            'leave the eu', 'leaving the eu', 'vote to leave',
            'in or out', 'article 50', 'article 50 process',

            # EU-Institutionen und Mitgliedschaft
            'european union', 'eu', 'eu membership', 'member state',
            'european commission', 'european parliament', 'european council',
            'european court of justice', 'ecj',

            # EU-Verträge und -Abkommen
            'withdrawal agreement', 'trade deal', 'trade agreement',
            'partnership agreement', 'future relationship',
            'european treaty', 'maastricht treaty',

            # Wirtschaft und Handel
            'single market', 'european single market', 'customs union',
            'european economic area', 'tariffs', 'trade barriers',
            'economic partnership', 'eurozone',

            # Bewegungsfreiheit und Grenzen
            'free movement', 'freedom of movement', 'schengen',
            'northern ireland protocol', 'irish border', 'backstop',

            # Politische Schlagworte
            'sovereignty', 'take back control', 'independence',
            'european integration', 'ever closer union',
            'eu referendum'
        }

    def _setup_logging(self):
        """Setup für Logging"""
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"brexit_classifier_{datetime.now().strftime('%Y%m%d')}.log"),
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
    
    def classify_debate(self, debate: Debate) -> BrexitClassificationResult:
        """
        Klassifiziert eine einzelne Debatte auf Brexit-Bezug
        
        Args:
            debate: Debate-Objekt
            
        Returns:
            BrexitClassificationResult
        """
        # 1. Keyword-basierte Vorfilterung
        keyword_result = self._classify_by_keywords(debate)
        
        # 2. Wenn Keywords gefunden oder unsicher, verwende Gemini
        if keyword_result.is_brexit_related or keyword_result.confidence < 0.8:
            if self.gemini_available:
                gemini_result = self._classify_by_gemini(debate)
                return self._combine_results(keyword_result, gemini_result)
            else:
                return keyword_result
        
        return keyword_result
    
    def _classify_by_keywords(self, debate: Debate) -> BrexitClassificationResult:
        """
        Klassifiziert Debatte basierend auf Keywords
        
        Args:
            debate: Debate-Objekt
            
        Returns:
            BrexitClassificationResult
        """
        # Sammle Text aus Titel und Reden
        text_to_analyze = debate.title.lower()
        
        # Füge ersten Teil der Reden hinzu (für Kontext)
        speech_texts = []
        for speech in debate.speeches[:5]:  # Erste 5 Reden
            speech_texts.append(speech.full_text[:8000].lower())  # Erste 8000 Zeichen pro Rede
        
        full_text = text_to_analyze + " " + " ".join(speech_texts)
        
        # Suche nach Keywords
        found_keywords = []
        for keyword in self.brexit_keywords:
            if keyword in full_text:
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
        
        return BrexitClassificationResult(
            is_brexit_related=is_brexit,
            confidence=confidence,
            reasoning=reasoning,
            keywords_found=found_keywords,
            classification_method="keyword"
        )
    
    def _classify_by_gemini(self, debate: Debate) -> BrexitClassificationResult:
        """
        Klassifiziert Debatte mit Google Gemini
        
        Args:
            debate: Debate-Objekt
            
        Returns:
            BrexitClassificationResult
        """
        try:
            # Erstelle Prompt für Gemini
            prompt = self._create_gemini_prompt(debate)
            
            # API-Call mit Rate Limiting
            time.sleep(0.1)  # 100ms Pause zwischen Calls
            response = self.model.generate_content(prompt)
            
            # Parse Response
            result = self._parse_gemini_response(response.text)
            
            self.logger.info(f"Gemini-Klassifikation für '{debate.title}': {result.is_brexit_related} (Confidence: {result.confidence})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei Gemini-Klassifikation: {e}")
            # Fallback zu Keyword-Ergebnis
            return self._classify_by_keywords(debate)
    
    def _create_gemini_prompt(self, debate: Debate) -> str:
        """
        Erstellt Prompt für Gemini-Klassifikation
        
        Args:
            debate: Debate-Objekt
            
        Returns:
            Prompt-String
        """
        # Sammle Text für Analyse
        debate_text = debate.title
        
        # Füge ersten Teil der Reden hinzu (begrenzt für Token-Limit)
        speech_texts = []
        total_chars = 0
        max_chars = 8000  # Begrenze für Gemini
        
        for speech in debate.speeches[:5]:  # Nur erste 5 Reden
            if total_chars + len(speech.full_text) > max_chars:
                break
            speech_texts.append(speech.full_text[:8000])  # Erste 8000 Zeichen pro Rede
            total_chars += len(speech.full_text)
        
        full_text = debate_text + "\n\n" + "\n\n".join(speech_texts)
        
        prompt = f"""
Du bist ein Experte für britische Politik und Brexit. Analysiere die folgende Parliament-Debatte und bestimme, ob sie sich auf Brexit oder verwandte EU-Themen bezieht.

DEBATTEN-TITEL: {debate.title}

DEBATTEN-INHALT:
{full_text}

Bitte antworte im folgenden JSON-Format:
{{
    "is_brexit_related": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Kurze Begründung deiner Entscheidung"
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

Antworte NUR mit dem JSON, keine zusätzlichen Erklärungen.
"""
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> BrexitClassificationResult:
        """
        Parst Gemini-Response zu BrexitClassificationResult
        
        Args:
            response_text: Rohe Response von Gemini
            
        Returns:
            BrexitClassificationResult
        """
        try:
            # Extrahiere JSON aus Response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                return BrexitClassificationResult(
                    is_brexit_related=bool(data.get('is_brexit_related', False)),
                    confidence=float(data.get('confidence', 0.0)),
                    reasoning=str(data.get('reasoning', '')),
                    keywords_found=[],  # Keine Keywords von AI
                    classification_method="gemini"
                )
            else:
                raise ValueError("Kein JSON in Response gefunden")
                
        except Exception as e:
            self.logger.error(f"Fehler beim Parsen der Gemini-Response: {e}")
            # Fallback
            return BrexitClassificationResult(
                is_brexit_related=False,
                confidence=0.0,
                reasoning=f"Fehler beim Parsen: {e}",
                keywords_found=[],
                classification_method="gemini_error"
            )
    
    def _combine_results(self, keyword_result: BrexitClassificationResult, 
                        gemini_result: BrexitClassificationResult) -> BrexitClassificationResult:
        """
        Kombiniert Keyword- und Gemini-Ergebnisse
        
        Args:
            keyword_result: Keyword-basiertes Ergebnis
            gemini_result: Gemini-basiertes Ergebnis
            
        Returns:
            Kombiniertes Ergebnis
        """
        # Gewichtung: 30% Keywords, 70% Gemini
        combined_confidence = (keyword_result.confidence * 0.3) + (gemini_result.confidence * 0.7)
        
        # Entscheidung basierend auf höherer Confidence
        is_brexit = combined_confidence > 0.5
        
        # Verwende nur Keyword-basierte Keywords (keine AI-generierten)
        all_keywords = keyword_result.keywords_found
        
        # Kombiniere Reasoning
        reasoning = f"Hybrid-Analyse: Keywords ({keyword_result.confidence:.2f}) + Gemini ({gemini_result.confidence:.2f})"
        
        return BrexitClassificationResult(
            is_brexit_related=is_brexit,
            confidence=combined_confidence,
            reasoning=reasoning,
            keywords_found=all_keywords,
            classification_method="hybrid"
        )
    
    def classify_session(self, session: ParliamentSession) -> Dict[str, Any]:
        """
        Klassifiziert alle Debatten einer Session
        
        Args:
            session: ParliamentSession-Objekt
            
        Returns:
            Dictionary mit Klassifikationsergebnissen
        """
        results = {
            'session_id': session.session_id,
            'date': session.date,
            'total_debates': len(session.debates),
            'brexit_debates': [],
            'classification_summary': {
                'total_brexit_debates': 0,
                'total_confidence': 0.0,
                'classification_methods': {}
            }
        }
        
        for debate in session.debates:
            classification = self.classify_debate(debate)
            
            if classification.is_brexit_related:
                results['brexit_debates'].append({
                    'title': debate.title,
                    'classification': classification,
                    'debate_stats': {
                        'total_speeches': debate.total_speeches,
                        'total_words': debate.total_words,
                        'duration_minutes': debate.duration_minutes
                    }
                })
                
                results['classification_summary']['total_brexit_debates'] += 1
                results['classification_summary']['total_confidence'] += classification.confidence
        
        # Berechne Durchschnitts-Confidence
        if results['classification_summary']['total_brexit_debates'] > 0:
            results['classification_summary']['average_confidence'] = (
                results['classification_summary']['total_confidence'] / 
                results['classification_summary']['total_brexit_debates']
            )
        else:
            results['classification_summary']['average_confidence'] = 0.0
        
        return results


# Convenience-Funktionen
def create_brexit_classifier(api_key: Optional[str] = None) -> BrexitClassifier:
    """
    Factory-Funktion für BrexitClassifier
    
    Args:
        api_key: Google AI API Key
        
    Returns:
        BrexitClassifier Instanz
    """
    return BrexitClassifier(api_key=api_key)


def classify_debate_quick(debate: Debate, api_key: Optional[str] = None) -> BrexitClassificationResult:
    """
    Schnelle Klassifikation einer einzelnen Debatte
    
    Args:
        debate: Debate-Objekt
        api_key: Google AI API Key
        
    Returns:
        BrexitClassificationResult
    """
    classifier = create_brexit_classifier(api_key)
    return classifier.classify_debate(debate)


if __name__ == "__main__":
    # Test des Klassifikators
    print("Brexit-Klassifikator Test")
    print("Stelle sicher, dass GOOGLE_API_KEY Umgebungsvariable gesetzt ist")
    
    # Beispiel-Test
    from parser.models.parliament_models import Debate, Speech, Speaker, Paragraph
    
    # Erstelle Test-Debatte
    test_speaker = Speaker(name="Test Speaker", party="Test Party")
    test_paragraph = Paragraph(text="This debate is about Brexit and the future relationship with the European Union.")
    test_speech = Speech(
        speech_id="test_1",
        speaker=test_speaker,
        paragraphs=[test_paragraph]
    )
    test_debate = Debate(
        title="Brexit and EU Future Relationship",
        speeches=[test_speech]
    )
    
    # Teste Klassifikator
    classifier = create_brexit_classifier()
    result = classifier.classify_debate(test_debate)
    
    print(f"Test-Ergebnis: {result}")

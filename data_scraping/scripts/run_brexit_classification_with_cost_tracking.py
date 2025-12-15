#!/usr/bin/env python3
"""
Brexit-Klassifikation mit Kosten-Tracking
Klassifiziert Debatten mit Google Gemini API und verfolgt die Kosten
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import argparse
from dataclasses import dataclass, asdict
import csv

# Lade .env Datei
from dotenv import load_dotenv
load_dotenv()

# Füge Projekt-Root zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from database.managers.database_manager import DatabaseManager
from classifier.brexit_classifier import BrexitClassifier, create_brexit_classifier
from classifier.brexit_database import BrexitDatabase, create_brexit_database


@dataclass
class CostTracking:
    """Datenklasse für Kosten-Tracking"""
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_characters: int = 0
    total_cost_usd: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    
    def get_duration_seconds(self) -> float:
        return self.end_time - self.start_time if self.end_time > 0 else 0.0
    
    def get_requests_per_minute(self) -> float:
        duration = self.get_duration_seconds()
        return (self.total_requests / duration * 60) if duration > 0 else 0.0


class CostTracker:
    """Kosten-Tracker für Google Gemini API"""
    
    # Gemini 1.5 Flash Preise (Stand: 2024)
    # Input: $0.075 per 1M tokens
    # Output: $0.30 per 1M tokens
    INPUT_COST_PER_1M_TOKENS = 0.075
    OUTPUT_COST_PER_1M_TOKENS = 0.30
    
    # Geschätzte Token-zu-Character-Ratio (ca. 4 Zeichen pro Token)
    CHARS_PER_TOKEN = 4
    
    def __init__(self):
        self.cost_data = CostTracking()
        self.cost_data.start_time = time.time()
        
    def add_request(self, input_text: str, output_text: str = ""):
        """Fügt eine API-Anfrage zum Kosten-Tracking hinzu"""
        input_chars = len(input_text)
        output_chars = len(output_text)
        
        # Geschätzte Token-Anzahl
        input_tokens = input_chars / self.CHARS_PER_TOKEN
        output_tokens = output_chars / self.CHARS_PER_TOKEN
        
        # Kosten berechnen
        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_1M_TOKENS
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M_TOKENS
        total_cost = input_cost + output_cost
        
        # Tracking-Daten aktualisieren
        self.cost_data.total_requests += 1
        self.cost_data.total_input_tokens += int(input_tokens)
        self.cost_data.total_output_tokens += int(output_tokens)
        self.cost_data.total_characters += input_chars + output_chars
        self.cost_data.total_cost_usd += total_cost
        
    def finalize(self):
        """Finalisiert das Kosten-Tracking"""
        self.cost_data.end_time = time.time()
        
    def get_summary(self) -> Dict[str, Any]:
        """Gibt eine Zusammenfassung der Kosten zurück"""
        return {
            'total_requests': self.cost_data.total_requests,
            'total_input_tokens': self.cost_data.total_input_tokens,
            'total_output_tokens': self.cost_data.total_output_tokens,
            'total_characters': self.cost_data.total_characters,
            'total_cost_usd': round(self.cost_data.total_cost_usd, 4),
            'duration_seconds': round(self.cost_data.get_duration_seconds(), 2),
            'requests_per_minute': round(self.cost_data.get_requests_per_minute(), 2),
            'avg_cost_per_request': round(self.cost_data.total_cost_usd / max(self.cost_data.total_requests, 1), 6),
            'estimated_cost_for_1000_debates': round(self.cost_data.total_cost_usd * (1000 / max(self.cost_data.total_requests, 1)), 2)
        }
    
    def save_to_csv(self, filename: str):
        """Speichert Kosten-Daten in CSV-Datei"""
        summary = self.get_summary()
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow(['Metric', 'Value', 'Unit'])
            
            # Daten
            writer.writerow(['Total Requests', summary['total_requests'], 'count'])
            writer.writerow(['Total Input Tokens', summary['total_input_tokens'], 'tokens'])
            writer.writerow(['Total Output Tokens', summary['total_output_tokens'], 'tokens'])
            writer.writerow(['Total Characters', summary['total_characters'], 'characters'])
            writer.writerow(['Total Cost', summary['total_cost_usd'], 'USD'])
            writer.writerow(['Duration', summary['duration_seconds'], 'seconds'])
            writer.writerow(['Requests per Minute', summary['requests_per_minute'], 'req/min'])
            writer.writerow(['Avg Cost per Request', summary['avg_cost_per_request'], 'USD'])
            writer.writerow(['Estimated Cost for 1000 Debates', summary['estimated_cost_for_1000_debates'], 'USD'])


class CostTrackingBrexitClassifier(BrexitClassifier):
    """Erweiterte Brexit-Klassifikator mit Kosten-Tracking"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        super().__init__(api_key, model_name)
        self.cost_tracker = CostTracker()
        
    def _classify_by_gemini(self, debate) -> Any:
        """Überschreibt die Gemini-Klassifikation mit Kosten-Tracking"""
        if not self.gemini_available:
            self.logger.warning("Gemini nicht verfügbar, verwende Keyword-Klassifikation")
            return super()._classify_by_keywords(debate)
        
        # Erstelle Prompt für die Klassifikation
        prompt = self._create_classification_prompt(debate)
        
        try:
            # API-Aufruf mit Kosten-Tracking
            response = self.model.generate_content(prompt)
            
            # Kosten-Tracking
            self.cost_tracker.add_request(prompt, response.text)
            
            # Parse Response
            return self._parse_gemini_response(response.text)
            
        except Exception as e:
            self.logger.error(f"Fehler bei Gemini-Klassifikation: {e}")
            self.logger.info("Fallback zu Keyword-basierter Klassifikation")
            # Fallback zu Keyword-basierter Klassifikation
            return super()._classify_by_keywords(debate)
    
    def _create_classification_prompt(self, debate) -> str:
        """Erstellt einen Prompt für die Gemini-Klassifikation"""
        # Kürze den Text falls zu lang (Gemini hat Token-Limits)
        max_chars = 8000  # Sicherheitspuffer
        debate_text = debate.title + " " + " ".join([speech.full_text for speech in debate.speeches[:5]])
        
        if len(debate_text) > max_chars:
            debate_text = debate_text[:max_chars] + "..."
        
        prompt = f"""
Analysiere die folgende Parlamentsdebatte und bestimme, ob sie Brexit-bezogen ist.

Debatte: "{debate.title}"

Text: {debate_text}

Antworte im folgenden JSON-Format:
{{
    "is_brexit_related": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Kurze Begründung",
    "keywords_found": ["keyword1", "keyword2"]
}}

Brexit-bezogene Themen umfassen:
- EU-Austritt, Brexit-Verhandlungen
- EU-Handel, Binnenmarkt
- EU-Immigration, Freizügigkeit
- EU-Souveränität, Kontrolle
- EU-Regulierung, Gesetze
"""
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Any:
        """Parst die Gemini-Response"""
        try:
            # Versuche JSON zu extrahieren
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                from classifier.brexit_classifier import BrexitClassificationResult
                return BrexitClassificationResult(
                    is_brexit_related=data.get('is_brexit_related', False),
                    confidence=data.get('confidence', 0.5),
                    keywords_found=data.get('keywords_found', []),
                    classification_method='gemini',
                    reasoning=data.get('reasoning', '')
                )
        except Exception as e:
            self.logger.warning(f"Fehler beim Parsen der Gemini-Response: {e}")
        
        # Fallback
        from classifier.brexit_classifier import BrexitClassificationResult
        return BrexitClassificationResult(
            is_brexit_related=False,
            confidence=0.0,
            keywords_found=[],
            classification_method='gemini',
            reasoning='Fehler beim Parsen der Response'
        )
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Gibt Kosten-Zusammenfassung zurück"""
        return self.cost_tracker.get_summary()
    
    def save_cost_report(self, filename: str):
        """Speichert Kosten-Report"""
        self.cost_tracker.finalize()
        self.cost_tracker.save_to_csv(filename)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup für Logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"brexit_cost_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def parse_arguments():
    """Kommandozeilen-Argumente parsen"""
    parser = argparse.ArgumentParser(description="Brexit-Klassifikation mit Kosten-Tracking")
    
    parser.add_argument(
        "--max-debates",
        type=int,
        default=1000,
        help="Maximale Anzahl zu klassifizierender Debatten (Standard: 1000)"
    )
    
    parser.add_argument(
        "--source-db",
        type=str,
        default="database/data/debates_data.db",
        help="Pfad zur Hauptdatenbank (Standard: database/data/debates_data.db)"
    )
    
    parser.add_argument(
        "--brexit-db",
        type=str,
        default="database/data/brexit_data.db",
        help="Pfad zur Brexit-Datenbank (Standard: database/data/brexit_data.db)"
    )
    
    parser.add_argument(
        "--google-api-key",
        type=str,
        help="Google API Key (falls nicht in Umgebungsvariablen)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini-Modell (Standard: gemini-2.0-flash)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Ausgabeverzeichnis für Reports (Standard: results)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log-Level (Standard: INFO)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Zeige nur Kostenschätzung ohne Ausführung"
    )
    
    return parser.parse_args()


def estimate_costs(max_debates: int, avg_debate_length: int = 5000) -> Dict[str, Any]:
    """Schätzt die Kosten für die Klassifikation"""
    # Geschätzte Token-Anzahl pro Debatte
    chars_per_debate = avg_debate_length
    tokens_per_debate = chars_per_debate / 4  # 4 Zeichen pro Token
    
    # Input + Output Token (Output ist meist kürzer)
    total_tokens_per_debate = tokens_per_debate + (tokens_per_debate * 0.1)  # 10% für Output
    
    # Kosten berechnen
    input_cost = (tokens_per_debate * max_debates / 1_000_000) * 0.075
    output_cost = (tokens_per_debate * 0.1 * max_debates / 1_000_000) * 0.30
    total_cost = input_cost + output_cost
    
    return {
        'max_debates': max_debates,
        'avg_debate_length_chars': avg_debate_length,
        'estimated_tokens_per_debate': int(tokens_per_debate),
        'estimated_total_tokens': int(total_tokens_per_debate * max_debates),
        'estimated_input_cost': round(input_cost, 4),
        'estimated_output_cost': round(output_cost, 4),
        'estimated_total_cost': round(total_cost, 4),
        'estimated_cost_per_debate': round(total_cost / max_debates, 6)
    }


def main():
    """Hauptfunktion"""
    args = parse_arguments()
    
    # Logging setup
    logger = setup_logging(args.log_level)
    
    logger.info("=== Brexit-Klassifikation mit Kosten-Tracking ===")
    logger.info(f"Max Debatten: {args.max_debates}")
    logger.info(f"Quell-Datenbank: {args.source_db}")
    logger.info(f"Brexit-Datenbank: {args.brexit_db}")
    logger.info(f"Gemini-Modell: {args.model}")
    
    # Kostenschätzung
    cost_estimate = estimate_costs(args.max_debates)
    logger.info("=== KOSTENSCHÄTZUNG ===")
    for key, value in cost_estimate.items():
        logger.info(f"{key}: {value}")
    
    if args.dry_run:
        logger.info("Dry-Run Modus - keine Ausführung")
        return 0
    
    # Bestätigung
    print(f"\n💰 Geschätzte Kosten: ${cost_estimate['estimated_total_cost']:.4f}")
    print(f"📊 Geschätzte Token: {cost_estimate['estimated_total_tokens']:,}")
    print("Möchtest du fortfahren? (y/N): y")
    response = "y"  # Automatisch mit y antworten
    if response.lower() != 'y':
        logger.info("Abgebrochen durch Benutzer")
        return 0
    
    # Ausgabeverzeichnis erstellen
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Initialisiere Komponenten
        logger.info("Initialisiere Komponenten...")
        
        # Datenbank-Manager
        source_db = DatabaseManager(args.source_db)
        
        # Brexit-Klassifikator mit Kosten-Tracking
        classifier = CostTrackingBrexitClassifier(
            api_key=args.google_api_key,
            model_name=args.model
        )
        
        # Prüfe Gemini-Verfügbarkeit
        if not classifier.gemini_available:
            logger.warning("⚠️  Gemini API nicht verfügbar - verwende nur Keyword-Klassifikation")
            logger.warning("   Für Gemini-Klassifikation: Google API Key setzen")
        else:
            logger.info("✅ Gemini API verfügbar")
        
        # Brexit-Datenbank
        brexit_db = create_brexit_database(args.brexit_db)
        
        # Kopiere ursprüngliche Daten aus der Quell-Datenbank
        logger.info("Kopiere ursprüngliche Parliament-Daten...")
        brexit_db.copy_original_data_from_source(args.source_db)
        
        # Lade Debatten aus der Datenbank (angepasst für parliament_data.db Schema)
        logger.info("Lade Debatten aus der Datenbank...")
        debates_query = """
        SELECT d.debate_id, d.title, d.session_id, s.date
        FROM debates d
        JOIN sessions s ON d.session_id = s.session_id
        ORDER BY s.date DESC
        LIMIT ?
        """
        
        debates_result = source_db.execute_query(debates_query, [args.max_debates])
        
        if not debates_result:
            logger.error("Keine Debatten in der Datenbank gefunden")
            return 1
        
        logger.info(f"Gefunden: {len(debates_result)} Debatten")
        
        # Klassifikation durchführen
        logger.info("Starte Klassifikation...")
        start_time = time.time()
        
        processed_debates = 0
        brexit_debates = 0
        errors = []
        
        for i, (debate_id, title, session_id, date) in enumerate(debates_result, 1):
            try:
                # Prüfe ob Debatte bereits verarbeitet wurde
                if brexit_db.is_debate_processed(debate_id):
                    logger.info(f"⏭️  Debatte {i}/{len(debates_result)} bereits verarbeitet: {title[:50]}...")
                    processed_debates += 1
                    continue
                
                logger.info(f"Verarbeite Debatte {i}/{len(debates_result)}: {title[:50]}...")
                
                # Lade vollständige Debatten-Daten
                debate_data = load_debate_data(source_db, debate_id)
                if not debate_data:
                    logger.warning(f"Debatte {debate_id} konnte nicht geladen werden")
                    continue
                
                # Klassifiziere Debatte
                classification = classifier.classify_debate(debate_data)
                
                # Speichere Klassifikationsergebnis in Brexit-Datenbank
                brexit_db.store_classification_result(debate_id, classification, debate_data)
                
                processed_debates += 1
                if classification.is_brexit_related:
                    brexit_debates += 1
                    logger.info(f"  ✅ Brexit-Debatte gefunden (Confidence: {classification.confidence:.2f})")
                else:
                    logger.info(f"  ❌ Keine Brexit-Debatte (Confidence: {classification.confidence:.2f})")
                
                # Fortschritt anzeigen
                if i % 10 == 0:
                    cost_summary = classifier.get_cost_summary()
                    logger.info(f"Fortschritt: {i}/{len(debates_result)} | "
                              f"Kosten bisher: ${cost_summary['total_cost_usd']:.4f} | "
                              f"Brexit-Debatten: {brexit_debates}")
                
            except Exception as e:
                error_msg = f"Fehler bei Debatte {debate_id}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Finalisiere Kosten-Tracking
        classifier.cost_tracker.finalize()
        cost_summary = classifier.get_cost_summary()
        
        # Speichere Kosten-Report
        cost_report_file = output_dir / f"cost_report_{timestamp}.csv"
        classifier.save_cost_report(str(cost_report_file))
        
        # Speichere JSON-Report
        json_report = {
            'execution_summary': {
                'total_debates_processed': processed_debates,
                'brexit_debates_found': brexit_debates,
                'brexit_percentage': round((brexit_debates / max(processed_debates, 1)) * 100, 2),
                'duration_seconds': round(duration, 2),
                'errors': len(errors)
            },
            'cost_analysis': cost_summary,
            'cost_estimate': cost_estimate,
            'errors': errors
        }
        
        json_report_file = output_dir / f"classification_report_{timestamp}.json"
        with open(json_report_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        
        # Zusammenfassung
        logger.info("=== KLASSIFIKATION ABGESCHLOSSEN ===")
        logger.info(f"Verarbeitete Debatten: {processed_debates}")
        logger.info(f"Brexit-Debatten gefunden: {brexit_debates}")
        logger.info(f"Brexit-Anteil: {(brexit_debates / max(processed_debates, 1)) * 100:.1f}%")
        logger.info(f"Dauer: {duration:.1f} Sekunden")
        logger.info(f"Fehler: {len(errors)}")
        
        logger.info("=== KOSTEN-ANALYSE ===")
        logger.info(f"Gesamtkosten: ${cost_summary['total_cost_usd']:.4f}")
        logger.info(f"Kosten pro Debatte: ${cost_summary['avg_cost_per_request']:.6f}")
        logger.info(f"Geschätzte Kosten für 1000 Debatten: ${cost_summary['estimated_cost_for_1000_debates']:.2f}")
        logger.info(f"API-Anfragen: {cost_summary['total_requests']}")
        logger.info(f"Geschwindigkeit: {cost_summary['requests_per_minute']:.1f} Anfragen/Minute")
        
        logger.info("=== REPORTS ===")
        logger.info(f"Kosten-Report: {cost_report_file}")
        logger.info(f"JSON-Report: {json_report_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if 'source_db' in locals():
            source_db.close()
        if 'brexit_db' in locals():
            brexit_db.close()


def load_debate_data(db_manager: DatabaseManager, debate_id: str):
    """Lädt vollständige Debatten-Daten aus der Datenbank (angepasst für parliament_data.db Schema)"""
    try:
        # Lade Debatten-Info (mit allen Metadaten)
        debate_query = """
        SELECT d.debate_id, d.title, d.session_id, d.heading_id, d.time, d.url, d.column_number,
               d.total_speeches, d.total_words, d.unique_speakers, d.duration_minutes
        FROM debates d
        WHERE d.debate_id = ?
        """
        
        debate_result = db_manager.execute_query(debate_query, [debate_id])
        if not debate_result:
            return None
        
        debate_row = debate_result[0]
        
        # Lade alle Reden für diese Debatte
        speeches_query = """
        SELECT speech_id, speaker_name, speaker_person_id, speaker_title, 
               time, url, is_intervention, column_number, full_text, word_count, char_count
        FROM speeches
        WHERE debate_id = ?
        ORDER BY speech_id
        """
        
        speeches_result = db_manager.execute_query(speeches_query, [debate_id])
        
        # Erstelle Debate-Objekt
        from parser.models.parliament_models import Debate, Speech, Speaker, Paragraph
        
        debate = Debate(
            title=debate_row[1],  # title
            heading_id=debate_row[3] or '',  # heading_id
            time=debate_row[4] or '',  # time
            url=debate_row[5] or '',  # url
            column_number=debate_row[6] or 0  # column_number
        )
        
        # Setze zusätzliche Metadaten (nur verfügbare Attribute)
        debate.debate_id = debate_row[0]  # debate_id
        debate.session_id = debate_row[2]  # session_id
        # total_speeches, total_words, unique_speakers werden automatisch berechnet
        # duration_minutes ist nicht im Debate-Modell verfügbar
        
        # Füge alle Reden hinzu
        for speech_row in speeches_result:
            speaker = Speaker(
                name=speech_row[1] or 'Unknown',      # speaker_name
                person_id=speech_row[2] or '',        # speaker_person_id
                party='',                             # nicht verfügbar
                constituency='',                      # nicht verfügbar
                title=speech_row[3] or ''             # speaker_title
            )
            
            # Verwende den echten Text aus der Datenbank
            paragraph = Paragraph(
                text=speech_row[8] or '',             # text (echter Redeinhalt)
                paragraph_id=speech_row[0],           # speech_id
                class_name='speech',
                motion_text=False
            )
            
            speech = Speech(
                speech_id=speech_row[0],              # speech_id
                speaker=speaker,
                paragraphs=[paragraph],
                time=speech_row[4] or '',             # time
                column_number=speech_row[7] or 0,     # column_number
                url=speech_row[5] or ''               # url
            )
            
            debate.speeches.append(speech)
        
        return debate
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Fehler beim Laden der Debatte {debate_id}: {e}")
        return None


if __name__ == "__main__":
    sys.exit(main())

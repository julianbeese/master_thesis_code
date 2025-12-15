#!/usr/bin/env python3
"""
Schneller Test der Brexit-Klassifikation mit Kosten-Tracking
Testet mit einer kleinen Anzahl von Debatten um Kosten zu schätzen
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import argparse

# Füge Projekt-Root zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from database.managers.database_manager import DatabaseManager
from classifier.brexit_classifier import BrexitClassifier, create_brexit_classifier


def estimate_costs_for_debates(num_debates: int, avg_debate_length: int = 5000) -> Dict[str, Any]:
    """Schätzt die Kosten für eine bestimmte Anzahl von Debatten"""
    
    # Gemini 1.5 Flash Preise (Stand: 2024)
    INPUT_COST_PER_1M_TOKENS = 0.075  # $0.075 per 1M input tokens
    OUTPUT_COST_PER_1M_TOKENS = 0.30  # $0.30 per 1M output tokens
    CHARS_PER_TOKEN = 4  # Geschätzte Token-zu-Character-Ratio
    
    # Berechnungen
    chars_per_debate = avg_debate_length
    input_tokens_per_debate = chars_per_debate / CHARS_PER_TOKEN
    output_tokens_per_debate = input_tokens_per_debate * 0.1  # Output ist meist kürzer
    
    total_input_tokens = input_tokens_per_debate * num_debates
    total_output_tokens = output_tokens_per_debate * num_debates
    
    input_cost = (total_input_tokens / 1_000_000) * INPUT_COST_PER_1M_TOKENS
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_1M_TOKENS
    total_cost = input_cost + output_cost
    
    return {
        'num_debates': num_debates,
        'avg_debate_length_chars': avg_debate_length,
        'input_tokens_per_debate': int(input_tokens_per_debate),
        'output_tokens_per_debate': int(output_tokens_per_debate),
        'total_input_tokens': int(total_input_tokens),
        'total_output_tokens': int(total_output_tokens),
        'input_cost_usd': round(input_cost, 4),
        'output_cost_usd': round(output_cost, 4),
        'total_cost_usd': round(total_cost, 4),
        'cost_per_debate_usd': round(total_cost / num_debates, 6)
    }


def test_classification_with_costs(num_debates: int = 10, db_path: str = "database/data/parliament_data.db"):
    """Testet die Klassifikation mit einer kleinen Anzahl von Debatten"""
    
    print(f"🧪 Test der Brexit-Klassifikation mit {num_debates} Debatten")
    print("=" * 60)
    
    # Kostenschätzung
    cost_estimate = estimate_costs_for_debates(num_debates)
    print("💰 KOSTENSCHÄTZUNG:")
    for key, value in cost_estimate.items():
        print(f"   {key}: {value}")
    print()
    
    # Bestätigung
    response = input(f"Möchtest du {num_debates} Debatten klassifizieren? Geschätzte Kosten: ${cost_estimate['total_cost_usd']:.4f} (y/N): ")
    if response.lower() != 'y':
        print("Abgebrochen.")
        return
    
    try:
        # Initialisiere Komponenten
        print("🔧 Initialisiere Komponenten...")
        
        # Prüfe Google API Key
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ GOOGLE_API_KEY Umgebungsvariable nicht gesetzt!")
            print("   Setze sie mit: export GOOGLE_API_KEY='dein-api-key'")
            return
        
        # Datenbank
        db = DatabaseManager(db_path)
        
        # Klassifikator
        classifier = create_brexit_classifier(api_key)
        
        # Lade Test-Debatten (angepasst für debates_data.db Schema)
        print(f"📊 Lade {num_debates} Debatten aus der Datenbank...")
        debates_query = """
        SELECT d.debate_id, d.title, d.session_id, s.date
        FROM debates d
        JOIN sessions s ON d.session_id = s.session_id
        ORDER BY s.date DESC
        LIMIT ?
        """
        
        debates_result = db.execute_query(debates_query, [num_debates])
        
        if not debates_result:
            print("❌ Keine Debatten in der Datenbank gefunden")
            return
        
        print(f"✅ {len(debates_result)} Debatten geladen")
        print()
        
        # Klassifikation durchführen
        print("🤖 Starte Klassifikation...")
        start_time = time.time()
        
        results = []
        brexit_count = 0
        
        for i, (debate_id, title, session_id, date) in enumerate(debates_result, 1):
            print(f"📝 Debatte {i}/{len(debates_result)}: {title[:60]}...")
            
            try:
                # Lade Debatten-Daten
                debate_data = load_debate_for_test(db, debate_id)
                if not debate_data:
                    print(f"   ⚠️  Debatte {debate_id} konnte nicht geladen werden")
                    continue
                
                # Klassifiziere
                classification = classifier.classify_debate(debate_data)
                
                result = {
                    'debate_id': debate_id,
                    'title': title,
                    'date': str(date) if date else None,  # Konvertiere Datum zu String
                    'is_brexit_related': classification.is_brexit_related,
                    'confidence': classification.confidence,
                    'method': classification.classification_method,
                    'keywords': classification.keywords_found,
                    'reasoning': getattr(classification, 'reasoning', ''),
                    'full_debate_text': get_full_debate_text(debate_data),
                    'speeches': get_speeches_data(debate_data)
                }
                
                results.append(result)
                
                if classification.is_brexit_related:
                    brexit_count += 1
                    print(f"   ✅ Brexit-Debatte (Confidence: {classification.confidence:.2f})")
                else:
                    print(f"   ❌ Keine Brexit-Debatte (Confidence: {classification.confidence:.2f})")
                
            except Exception as e:
                print(f"   ❌ Fehler: {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Ergebnisse
        print()
        print("📊 ERGEBNISSE:")
        print(f"   Verarbeitete Debatten: {len(results)}")
        print(f"   Brexit-Debatten: {brexit_count}")
        print(f"   Brexit-Anteil: {(brexit_count / max(len(results), 1)) * 100:.1f}%")
        print(f"   Dauer: {duration:.1f} Sekunden")
        print(f"   Geschwindigkeit: {len(results) / duration * 60:.1f} Debatten/Minute")
        
        # Kosten-Extrapolation
        print()
        print("💰 KOSTEN-EXTRAPOLATION:")
        cost_per_debate = cost_estimate['cost_per_debate_usd']
        print(f"   Kosten pro Debatte: ${cost_per_debate:.6f}")
        print(f"   Geschätzte Kosten für 100 Debatten: ${cost_per_debate * 100:.4f}")
        print(f"   Geschätzte Kosten für 1000 Debatten: ${cost_per_debate * 1000:.2f}")
        print(f"   Geschätzte Kosten für 10000 Debatten: ${cost_per_debate * 10000:.2f}")
        
        # Speichere Ergebnisse
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_summary': {
                    'num_debates_tested': len(results),
                    'brexit_debates_found': brexit_count,
                    'brexit_percentage': (brexit_count / max(len(results), 1)) * 100,
                    'duration_seconds': duration,
                    'debates_per_minute': len(results) / duration * 60
                },
                'cost_estimate': cost_estimate,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Ergebnisse gespeichert: {results_file}")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'db' in locals():
            db.close()


def get_full_debate_text(debate_data):
    """Extrahiert den vollständigen Text einer Debatte"""
    if not debate_data or not hasattr(debate_data, 'speeches'):
        return ""
    
    full_text = f"Titel: {debate_data.title}\n\n"
    
    for speech in debate_data.speeches:
        if hasattr(speech, 'speaker') and speech.speaker:
            full_text += f"{speech.speaker.name}: "
        full_text += speech.full_text + "\n\n"
    
    return full_text.strip()


def get_speeches_data(debate_data):
    """Extrahiert strukturierte Reden-Daten"""
    if not debate_data or not hasattr(debate_data, 'speeches'):
        return []
    
    speeches = []
    for speech in debate_data.speeches:
        speech_data = {
            'speech_id': speech.speech_id,
            'speaker_name': speech.speaker.name if speech.speaker else 'Unknown',
            'speaker_party': speech.speaker.party if speech.speaker else '',
            'speaker_constituency': speech.speaker.constituency if speech.speaker else '',
            'speech_text': speech.full_text,
            'word_count': speech.word_count,
            'char_count': speech.char_count,
            'time': speech.time,
            'url': speech.url
        }
        speeches.append(speech_data)
    
    return speeches


def load_debate_for_test(db_manager: DatabaseManager, debate_id: str):
    """Lädt eine Debatte für den Test (angepasst für debates_data.db Schema)"""
    try:
        # Lade Debatten-Info (angepasst für debates_data.db Schema)
        debate_query = """
        SELECT debate_id, session_id, title, url
        FROM debates
        WHERE debate_id = ?
        """
        
        debate_result = db_manager.execute_query(debate_query, [debate_id])
        if not debate_result:
            return None
        
        debate_row = debate_result[0]
        
        # Lade alle Reden für diese Debatte (Schema: speeches hat bereits alle Sprecher-Info)
        speeches_query = """
        SELECT speech_id, speaker_name, speaker_person_id, speaker_title, speaker_party, speaker_constituency,
               time, url, word_count, char_count, full_text
        FROM speeches
        WHERE debate_id = ?
        ORDER BY speech_id
        """
        
        speeches_result = db_manager.execute_query(speeches_query, [debate_id])
        
        # Erstelle Debate-Objekt
        from parser.models.parliament_models import Debate, Speech, Speaker, Paragraph
        
        debate = Debate(
            title=debate_row[2],  # title
            heading_id='',
            time='',
            url=debate_row[3] or '',  # url
            column_number=0
        )
        
        # Füge alle Reden hinzu
        for speech_row in speeches_result:
            speaker = Speaker(
                name=speech_row[1] or 'Unknown',      # speaker_name
                person_id=speech_row[2] or '',        # speaker_person_id
                party=speech_row[4] or '',            # speaker_party
                constituency=speech_row[5] or '',     # speaker_constituency
                title=speech_row[3] or ''             # speaker_title
            )
            
            paragraph = Paragraph(
                text=speech_row[10] or '',            # full_text
                paragraph_id=speech_row[0],           # speech_id
                class_name='speech',
                motion_text=False
            )
            
            speech = Speech(
                speech_id=speech_row[0],              # speech_id
                speaker=speaker,
                paragraphs=[paragraph],
                time=speech_row[6] or '',             # time
                column_number=0,
                url=speech_row[7] or ''               # url
            )
            
            debate.speeches.append(speech)
        
        return debate
        
    except Exception as e:
        print(f"Fehler beim Laden der Debatte {debate_id}: {e}")
        return None


def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description="Test der Brexit-Klassifikation mit Kosten-Tracking")
    
    parser.add_argument(
        "--num-debates",
        type=int,
        default=10,
        help="Anzahl der Test-Debatten (Standard: 10)"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="database/data/parliament_data.db",
        help="Pfad zur Datenbank (Standard: database/data/parliament_data.db)"
    )
    
    args = parser.parse_args()
    
    test_classification_with_costs(args.num_debates, args.db_path)


if __name__ == "__main__":
    main()

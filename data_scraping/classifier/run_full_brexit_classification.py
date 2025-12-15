"""
Vollständige Brexit-Speech-Klassifikation
Klassifiziert alle 650.000+ Reden aus der optimierten Datenbank
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import time

# Füge Projekt-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from classifier.brexit_speech_classifier import create_brexit_speech_classifier


def main():
    """Hauptfunktion für vollständige Brexit-Speech-Klassifikation"""
    
    parser = argparse.ArgumentParser(description='Vollständige Brexit-Speech-Klassifikation mit Google Gemini')
    parser.add_argument('--input-db', 
                       default='database/data/debates_data_optimized.db',
                       help='Pfad zur Eingabe-Datenbank')
    parser.add_argument('--output-db', 
                       default='database/data/brexit_speeches_full.db',
                       help='Pfad zur Ausgabe-Datenbank')
    parser.add_argument('--batch-size', 
                       type=int, 
                       default=25,
                       help='Batch-Größe für Verarbeitung (kleiner = stabiler)')
    parser.add_argument('--api-key', 
                       help='Google API Key (falls nicht in Umgebungsvariablen)')
    parser.add_argument('--resume', 
                       action='store_true',
                       help='Fortsetzen der Klassifikation (falls unterbrochen)')
    
    args = parser.parse_args()
    
    # Überprüfe API Key
    api_key = args.api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ FEHLER: GOOGLE_API_KEY nicht gefunden!")
        print("Setze die Umgebungsvariable oder verwende --api-key")
        return 1
    
    # Überprüfe Eingabe-Datenbank
    if not Path(args.input_db).exists():
        print(f"❌ FEHLER: Eingabe-Datenbank nicht gefunden: {args.input_db}")
        return 1
    
    # Erstelle Ausgabe-Verzeichnis
    output_path = Path(args.output_db)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("🚀 VOLLSTÄNDIGE Brexit-Speech-Klassifikation gestartet")
    print(f"📁 Eingabe-Datenbank: {args.input_db}")
    print(f"📁 Ausgabe-Datenbank: {args.output_db}")
    print(f"📦 Batch-Größe: {args.batch_size}")
    print(f"⏰ Startzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Schätze Verarbeitungszeit
    print("⏱️  GESCHÄTZTE VERARBEITUNGSZEIT:")
    print("   - 650.000 Reden")
    print("   - ~1.2 Sekunden pro Rede (inkl. API-Calls)")
    print("   - Geschätzte Gesamtzeit: ~22 Stunden")
    print("   - Empfehlung: Über Nacht laufen lassen")
    print()
    
    # Bestätigung
    response = input("Möchten Sie fortfahren? (y/N): ")
    if response.lower() != 'y':
        print("Abgebrochen.")
        return 0
    
    # Erstelle Klassifikator
    try:
        classifier = create_brexit_speech_classifier(api_key=api_key)
        print("✅ Klassifikator erfolgreich initialisiert")
    except Exception as e:
        print(f"❌ Fehler beim Initialisieren des Klassifikators: {e}")
        return 1
    
    # Starte Klassifikation
    start_time = time.time()
    
    try:
        stats = classifier.classify_speeches_from_db(
            db_path=args.input_db,
            output_db_path=args.output_db,
            batch_size=args.batch_size,
            max_speeches=None  # Alle Reden
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Zeige Ergebnisse
        print("\n" + "="*60)
        print("📊 VOLLSTÄNDIGE KLASSIFIKATIONS-ERGEBNISSE")
        print("="*60)
        print(f"⏱️  Gesamtdauer: {duration:.1f} Sekunden ({duration/3600:.1f} Stunden)")
        print(f"📝 Verarbeitete Reden: {stats['total_processed']:,}")
        print(f"🇪🇺 Brexit-bezogene Reden: {stats['brexit_related']:,}")
        print(f"📈 Brexit-Anteil: {stats['brexit_percentage']:.1f}%")
        print(f"🎯 Durchschnittliche Confidence: {stats['average_confidence']:.3f}")
        print(f"⚡ Durchschnittliche Verarbeitungszeit: {stats['average_processing_time']:.3f}s pro Rede")
        
        print(f"\n🔧 Klassifikations-Methoden:")
        for method, count in stats['classification_methods'].items():
            percentage = (count / stats['total_processed'] * 100) if stats['total_processed'] > 0 else 0
            print(f"   {method}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n💾 Ergebnisse gespeichert in: {args.output_db}")
        
        # Zeige Top Brexit-Sprecher
        print(f"\n🏆 TOP BREXIT-SPRECHER:")
        try:
            import duckdb
            conn = duckdb.connect(args.output_db)
            top_speakers = conn.execute("""
                SELECT speaker_name, COUNT(*) as brexit_speeches, AVG(confidence) as avg_confidence
                FROM brexit_speeches 
                WHERE speaker_name IS NOT NULL AND speaker_name != ''
                GROUP BY speaker_name 
                ORDER BY brexit_speeches DESC 
                LIMIT 10
            """).fetchall()
            
            for speaker, count, confidence in top_speakers:
                print(f"   {speaker}: {count} Brexit-Reden (Ø {confidence:.3f} Confidence)")
            
            conn.close()
        except Exception as e:
            print(f"   Fehler beim Laden der Top-Sprecher: {e}")
        
        print("\n✅ Vollständige Klassifikation erfolgreich abgeschlossen!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Klassifikation durch Benutzer abgebrochen")
        print("💡 Tipp: Verwende --resume um fortzufahren")
        return 1
    except Exception as e:
        print(f"\n❌ Fehler bei der Klassifikation: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

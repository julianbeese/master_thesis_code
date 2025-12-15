#!/usr/bin/env python3
"""
Brexit-Filtering Runner
Hauptskript zum Ausführen der Brexit-Debatten-Filterung
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Lade .env Datei falls vorhanden
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv ist optional

# Füge Projekt-Root zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent))

from .brexit_pipeline import create_brexit_pipeline
from .brexit_database import create_brexit_database


def main():
    """Hauptfunktion für Brexit-Filtering"""
    parser = argparse.ArgumentParser(description="Brexit-Debatten-Filterung Pipeline")
    
    # Input-Optionen
    parser.add_argument("--xml-dir", type=str, 
                       help="Verzeichnis mit XML-Dateien (z.B. scraper/data/raw) - Veraltet: verwende --from-database")
    parser.add_argument("--max-files", type=int, 
                       help="Maximale Anzahl zu verarbeitender XML-Dateien")
    parser.add_argument("--from-database", action="store_true",
                       help="Verarbeite Sessions aus der bestehenden Parliament-Datenbank")
    parser.add_argument("--max-sessions", type=int, default=None,
                       help="Maximale Anzahl Sessions aus der Datenbank zu verarbeiten")
    parser.add_argument("--start-date", type=str, 
                       help="Startdatum (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, 
                       help="Enddatum (YYYY-MM-DD)")
    
    # API-Key
    parser.add_argument("--api-key", type=str, 
                       help="Google AI API Key (falls nicht in Umgebungsvariable)")
    
    # Output-Optionen
    parser.add_argument("--export-format", type=str, choices=["parquet", "csv", "json"],
                       default="parquet", help="Export-Format für Ergebnisse")
    parser.add_argument("--export-path", type=str,
                       help="Pfad für Export-Datei")
    
    # Analytics
    parser.add_argument("--analytics", action="store_true",
                       help="Zeige detaillierte Analytics")
    parser.add_argument("--search", type=str,
                       help="Suche nach spezifischem Begriff")
    
    # Test-Modus
    parser.add_argument("--test", action="store_true",
                       help="Test-Modus mit kleinen Dateien")
    
    args = parser.parse_args()
    
    # API-Key prüfen
    api_key = args.api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ Fehler: Google AI API Key nicht gefunden!")
        print("Setze GOOGLE_API_KEY Umgebungsvariable oder verwende --api-key")
        print("API Key bekommst du hier: https://makersuite.google.com/app/apikey")
        return 1
    
    print("🚀 Brexit-Filtering Pipeline gestartet")
    print(f"📅 Zeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialisiere Pipeline
        with create_brexit_pipeline(google_api_key=api_key) as pipeline:
            
            if args.test:
                # Test-Modus
                print("🧪 Test-Modus aktiviert")
                test_dir = "scraper/data/raw_test"
                if Path(test_dir).exists():
                    stats = pipeline.process_xml_files(test_dir, max_files=2)
                    print(f"✅ Test abgeschlossen: {stats}")
                else:
                    print(f"❌ Test-Verzeichnis nicht gefunden: {test_dir}")
                    return 1
            
            elif args.from_database:
                # Datenbank-Verarbeitung (NEU!)
                print("🗄️ Verarbeite Sessions aus der Parliament-Datenbank")
                
                stats = pipeline.process_all_sessions_from_database(
                    max_sessions=args.max_sessions
                )
                
                print("📊 Verarbeitungsstatistiken:")
                print(f"  • Verarbeitete Sessions: {stats['processed_files']}/{stats['total_files']}")
                print(f"  • Fehlgeschlagene Sessions: {stats['failed_files']}")
                print(f"  • Gesamte Debatten: {stats['total_debates']}")
                print(f"  • Brexit-Debatten: {stats['brexit_debates']}")
                print(f"  • Gesamte Reden: {stats['total_speeches']}")
                print(f"  • Brexit-Reden: {stats['brexit_speeches']}")
                
                if stats['errors']:
                    print(f"⚠️  Fehler aufgetreten: {len(stats['errors'])}")
                    for error in stats['errors'][:5]:  # Zeige erste 5 Fehler
                        print(f"    - {error}")
            
            elif args.xml_dir:
                # XML-Verarbeitung (Veraltet)
                print(f"📁 Verarbeite XML-Dateien aus: {args.xml_dir}")
                print("⚠️  Warnung: XML-Verarbeitung ist veraltet. Verwende --from-database für bessere Performance.")
                
                if not Path(args.xml_dir).exists():
                    print(f"❌ Verzeichnis nicht gefunden: {args.xml_dir}")
                    return 1
                
                stats = pipeline.process_xml_files(
                    xml_directory=args.xml_dir,
                    max_files=args.max_files,
                    start_date=args.start_date,
                    end_date=args.end_date
                )
                
                print("📊 Verarbeitungsstatistiken:")
                print(f"  • Verarbeitete Dateien: {stats['processed_files']}/{stats['total_files']}")
                print(f"  • Fehlgeschlagene Dateien: {stats['failed_files']}")
                print(f"  • Gesamte Debatten: {stats['total_debates']}")
                print(f"  • Brexit-Debatten: {stats['brexit_debates']}")
                print(f"  • Verarbeitungszeit: {stats['processing_time_seconds']:.1f}s")
                
                if stats['errors']:
                    print(f"⚠️  Fehler aufgetreten: {len(stats['errors'])}")
                    for error in stats['errors'][:5]:  # Zeige erste 5 Fehler
                        print(f"    - {error}")
            
            elif args.search:
                # Suche
                print(f"🔍 Suche nach: {args.search}")
                with create_brexit_database() as db:
                    debates = db.search_brexit_debates(keyword=args.search)
                    print(f"📋 Gefundene Debatten: {len(debates)}")
                    
                    for debate in debates[:10]:  # Zeige erste 10
                        print(f"  • {debate['title']} (Confidence: {debate['confidence']:.2f})")
            
            else:
                # Zeige Hilfe
                print("ℹ️  Verwende --help für verfügbare Optionen")
                print("\nBeispiele:")
                print("  python run_brexit_filtering.py --test")
                print("  python run_brexit_filtering.py --from-database --max-sessions 10")
                print("  python run_brexit_filtering.py --search 'trade deal'")
                print("  python run_brexit_filtering.py --analytics")
                return 0
            
            # Analytics anzeigen
            if args.analytics or args.from_database or args.xml_dir or args.test:
                print("\n📈 Brexit-Analytics:")
                analytics = pipeline.get_brexit_analytics()
                
                overview = analytics.get('overview', {})
                print(f"  • Gesamte Sessions: {overview.get('total_sessions', 0)}")
                print(f"  • Gesamte Debatten: {overview.get('total_debates', 0)}")
                print(f"  • Gesamte Reden: {overview.get('total_speeches', 0)}")
                print(f"  • Durchschnitts-Confidence: {overview.get('average_confidence', 0):.3f}")
                
                # Top Keywords
                keyword_analysis = analytics.get('keyword_analysis', {})
                top_keywords = keyword_analysis.get('top_keywords', [])[:5]
                if top_keywords:
                    print("  • Top Keywords:")
                    for kw in top_keywords:
                        print(f"    - {kw['keyword']}: {kw['frequency']}x")
            
            # Export
            if args.export_format and (args.xml_dir or args.test):
                print(f"\n💾 Exportiere Daten als {args.export_format.upper()}...")
                export_path = pipeline.export_brexit_data(
                    output_format=args.export_format,
                    output_path=args.export_path
                )
                print(f"✅ Exportiert: {export_path}")
        
        print("\n🎉 Brexit-Filtering erfolgreich abgeschlossen!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Verarbeitung durch Benutzer abgebrochen")
        return 1
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

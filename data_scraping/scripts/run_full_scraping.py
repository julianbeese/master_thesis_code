#!/usr/bin/env python3
"""
Vollständiger Scraping-Prozess für Parlamentsdaten
Startet den kompletten Workflow: Scraping → Parsing → Datenbank → Export
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import time

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from scraper.run_scraper import main as scraper_main
from parser.database_integration import export_parliament_data_for_ml


def run_full_scraping(start_year: int, end_year: int, max_workers: int = 2, delay: float = 2.0):
    """
    Führt den kompletten Scraping-Prozess durch
    
    Args:
        start_year: Startjahr für Scraping
        end_year: Endjahr für Scraping
        max_workers: Anzahl paralleler Worker
        delay: Verzögerung zwischen Requests
    """
    print("🚀 Parliament Data Processing System - Vollständiger Scraping-Prozess")
    print("=" * 70)
    print(f"Startzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Jahre: {start_year} - {end_year}")
    print(f"Worker: {max_workers}, Verzögerung: {delay}s")
    print()
    
    start_time = time.time()
    
    try:
        # Schritt 1: Scraping mit Datenbank-Integration
        print("📥 Schritt 1: XML-Scraping mit Datenbank-Integration")
        print("-" * 50)
        
        # Simuliere scraper_main Aufruf
        scraper_args = [
            '--with-database',
            f'--start-year', str(start_year),
            f'--end-year', str(end_year),
            f'--max-workers', str(max_workers),
            f'--delay', str(delay)
        ]
        
        # Temporär sys.argv überschreiben
        original_argv = sys.argv
        sys.argv = ['run_full_scraping.py'] + scraper_args
        
        try:
            scraper_main()
        finally:
            sys.argv = original_argv
        
        scraping_time = time.time() - start_time
        print(f"✅ Scraping abgeschlossen in {scraping_time/60:.1f} Minuten")
        
        # Schritt 2: ML-Daten exportieren
        print(f"\n📤 Schritt 2: ML-Daten exportieren")
        print("-" * 50)
        
        export_start = time.time()
        
        # Export in verschiedenen Formaten
        formats = ['parquet', 'csv', 'json']
        export_results = {}
        
        for format_type in formats:
            print(f"Exportiere als {format_type.upper()}...")
            result = export_parliament_data_for_ml(format_type)
            
            if result['success']:
                export_results[format_type] = result
                print(f"✅ {format_type.upper()}: {result['output_file']} ({result['row_count']} Zeilen)")
            else:
                print(f"❌ {format_type.upper()}: {result.get('error', 'Unbekannter Fehler')}")
        
        export_time = time.time() - export_start
        print(f"✅ Export abgeschlossen in {export_time:.1f} Sekunden")
        
        # Schritt 3: Zusammenfassung
        total_time = time.time() - start_time
        
        print(f"\n📊 ZUSAMMENFASSUNG")
        print("=" * 70)
        print(f"Gesamtzeit: {total_time/60:.1f} Minuten")
        print(f"Scraping-Zeit: {scraping_time/60:.1f} Minuten")
        print(f"Export-Zeit: {export_time:.1f} Sekunden")
        print(f"Jahre verarbeitet: {end_year - start_year + 1}")
        
        if export_results:
            print(f"\n📁 Exportierte Dateien:")
            for format_type, result in export_results.items():
                print(f"   {format_type.upper()}: {result['output_file']}")
        
        print(f"\n🎉 Vollständiger Scraping-Prozess erfolgreich abgeschlossen!")
        print(f"   Die Daten sind bereit für ML-Modelle!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Fehler beim Scraping-Prozess: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Hauptfunktion mit Argument-Parsing"""
    parser = argparse.ArgumentParser(
        description="Vollständiger Scraping-Prozess für Parlamentsdaten",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Standard (2012-2021)
  python run_full_scraping.py
  
  # Nur 2020
  python run_full_scraping.py --start-year 2020 --end-year 2020
  
  # 2019-2021 mit 4 Workern
  python run_full_scraping.py --start-year 2019 --end-year 2021 --max-workers 4
  
  # Mit höherer Verzögerung
  python run_full_scraping.py --delay 3.0
        """
    )
    
    parser.add_argument(
        '--start-year', 
        type=int, 
        default=2012,
        help='Startjahr für Scraping (Standard: 2012)'
    )
    
    parser.add_argument(
        '--end-year', 
        type=int, 
        default=2021,
        help='Endjahr für Scraping (Standard: 2021)'
    )
    
    parser.add_argument(
        '--max-workers', 
        type=int, 
        default=2,
        help='Anzahl paralleler Worker (Standard: 2)'
    )
    
    parser.add_argument(
        '--delay', 
        type=float, 
        default=2.0,
        help='Verzögerung zwischen Requests in Sekunden (Standard: 2.0)'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Zeige nur die Konfiguration ohne Ausführung'
    )
    
    args = parser.parse_args()
    
    # Validierung
    if args.start_year > args.end_year:
        print("❌ Fehler: Startjahr muss kleiner oder gleich Endjahr sein")
        return 1
    
    if args.start_year < 2000 or args.end_year > 2030:
        print("❌ Fehler: Jahre müssen zwischen 2000 und 2030 liegen")
        return 1
    
    if args.max_workers < 1 or args.max_workers > 10:
        print("❌ Fehler: Worker-Anzahl muss zwischen 1 und 10 liegen")
        return 1
    
    if args.delay < 0.5 or args.delay > 10.0:
        print("❌ Fehler: Verzögerung muss zwischen 0.5 und 10.0 Sekunden liegen")
        return 1
    
    # Dry-run
    if args.dry_run:
        print("🔍 DRY-RUN - Konfiguration:")
        print(f"   Jahre: {args.start_year} - {args.end_year}")
        print(f"   Worker: {args.max_workers}")
        print(f"   Verzögerung: {args.delay}s")
        print(f"   Geschätzte Zeit: {(args.end_year - args.start_year + 1) * 30} Minuten")
        return 0
    
    # Bestätigung für große Datensätze
    years = args.end_year - args.start_year + 1
    if years > 5:
        print(f"⚠️  WARNUNG: Du möchtest {years} Jahre scrapen!")
        print(f"   Das kann {years * 30} Minuten oder länger dauern.")
        response = input("Möchtest du fortfahren? (y/N): ")
        if response.lower() != 'y':
            print("Abgebrochen.")
            return 0
    
    # Scraping starten
    success = run_full_scraping(
        start_year=args.start_year,
        end_year=args.end_year,
        max_workers=args.max_workers,
        delay=args.delay
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

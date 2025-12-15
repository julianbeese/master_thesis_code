#!/usr/bin/env python3
"""
Kommandozeilen-Skript für XML-Scraper
Einfache Nutzung des Scrapers über die Kommandozeile
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from scraper import XMLScraper, scraper_config
from scraper.database_integration import ScraperDatabaseIntegration
from scraper.utils import DataIntegrityChecker, get_directory_summary


def main():
    """Hauptfunktion für Kommandozeilen-Interface"""
    parser = argparse.ArgumentParser(
        description="XML-Scraper für theyworkforyou.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python run_scraper.py --scrape-only
  python run_scraper.py --with-database --parallel
  python run_scraper.py --validate-only
  python run_scraper.py --stats
  python run_scraper.py --export-parquet
        """
    )
    
    # Hauptoptionen
    parser.add_argument('--scrape-only', action='store_true',
                       help='Nur Scraping durchführen (ohne Datenbank)')
    parser.add_argument('--with-database', action='store_true',
                       help='Scraping mit Datenbank-Integration')
    parser.add_argument('--validate-only', action='store_true',
                       help='Nur bestehende Dateien validieren')
    parser.add_argument('--stats', action='store_true',
                       help='Statistiken anzeigen')
    parser.add_argument('--export-parquet', action='store_true',
                       help='Daten als Parquet exportieren')
    
    # Scraping-Optionen
    parser.add_argument('--start-year', type=int, default=2012,
                       help='Start-Jahr für Scraping (Standard: 2012)')
    parser.add_argument('--end-year', type=int, default=2021,
                       help='End-Jahr für Scraping (Standard: 2021)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximale Anzahl paralleler Downloads (Standard: 4)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Verzögerung zwischen Requests in Sekunden (Standard: 1.0)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Sequenzielle Downloads verwenden')
    
    # Ausgabe-Optionen
    parser.add_argument('--output-format', choices=['text', 'json'], default='text',
                       help='Ausgabeformat (Standard: text)')
    parser.add_argument('--output-file', type=str,
                       help='Ausgabedatei für Ergebnisse')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Ausführliche Ausgabe')
    
    args = parser.parse_args()
    
    # Logging-Level setzen
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Ergebnisse sammeln
    results = {}
    
    try:
        if args.scrape_only:
            print("=== XML-Scraping (ohne Datenbank) ===")
            results['scraping'] = run_scraping_only(args)
            
        elif args.with_database:
            print("=== XML-Scraping mit Datenbank-Integration ===")
            results['pipeline'] = run_scraping_with_database(args)
            
        elif args.validate_only:
            print("=== Datei-Validierung ===")
            results['validation'] = run_validation_only(args)
            
        elif args.stats:
            print("=== Statistiken ===")
            results['statistics'] = show_statistics(args)
            
        elif args.export_parquet:
            print("=== Parquet-Export ===")
            results['export'] = run_parquet_export(args)
            
        else:
            # Standard: Scraping mit Datenbank
            print("=== Standard: XML-Scraping mit Datenbank-Integration ===")
            results['pipeline'] = run_scraping_with_database(args)
        
        # Ergebnisse ausgeben
        output_results(results, args)
        
    except KeyboardInterrupt:
        print("\n\nScraping durch Benutzer abgebrochen")
        sys.exit(1)
    except Exception as e:
        print(f"\nFehler: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_scraping_only(args):
    """Führt nur Scraping durch (ohne Datenbank)"""
    scraper = XMLScraper(
        start_year=args.start_year,
        end_year=args.end_year,
        max_workers=args.max_workers,
        delay_between_requests=args.delay
    )
    
    use_parallel = not args.no_parallel
    return scraper.run_scraping(use_parallel=use_parallel)


def run_scraping_with_database(args):
    """Führt Scraping mit Datenbank-Integration durch"""
    with ScraperDatabaseIntegration() as integration:
        use_parallel = not args.no_parallel
        return integration.run_complete_scraping_pipeline(use_parallel=use_parallel)


def run_validation_only(args):
    """Validiert nur bestehende Dateien"""
    raw_data_dir = scraper_config.get_raw_data_dir()
    
    if not raw_data_dir.exists():
        return {"error": "Raw-Data-Verzeichnis existiert nicht"}
    
    checker = DataIntegrityChecker()
    return checker.check_directory_integrity(raw_data_dir)


def show_statistics(args):
    """Zeigt Statistiken an"""
    with ScraperDatabaseIntegration() as integration:
        db_stats = integration.get_scraping_statistics()
    
    # Verzeichnis-Statistiken
    raw_data_dir = scraper_config.get_raw_data_dir()
    dir_summary = get_directory_summary(raw_data_dir)
    
    return {
        "database_statistics": db_stats,
        "directory_summary": dir_summary
    }


def run_parquet_export(args):
    """Exportiert Daten als Parquet"""
    with ScraperDatabaseIntegration() as integration:
        return integration.export_scraping_data_to_parquet()


def output_results(results, args):
    """Gibt Ergebnisse im gewünschten Format aus"""
    if args.output_format == 'json':
        output = json.dumps(results, indent=2, default=str)
    else:
        output = format_text_output(results)
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Ergebnisse gespeichert in: {args.output_file}")
    else:
        print(output)


def format_text_output(results):
    """Formatiert Ergebnisse als Text"""
    output_lines = []
    
    for key, value in results.items():
        output_lines.append(f"\n=== {key.upper()} ===")
        
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    output_lines.append(f"\n{sub_key}:")
                    for k, v in sub_value.items():
                        output_lines.append(f"  {k}: {v}")
                else:
                    output_lines.append(f"{sub_key}: {sub_value}")
        else:
            output_lines.append(str(value))
    
    return "\n".join(output_lines)


if __name__ == "__main__":
    main()

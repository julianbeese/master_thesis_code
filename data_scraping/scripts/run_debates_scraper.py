#!/usr/bin/env python3
"""
Ausführungsskript für den Debates XML-Scraper
Lädt XML-Dateien von theyworkforyou.com/pwdata/scrapedxml/debates/ herunter
"""

import sys
import os
from pathlib import Path

# Füge das scraper-Verzeichnis zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent / "scraper"))

from debates_scraper import DebatesXMLScraper


def main():
    """Hauptfunktion für das Debates-Scraping"""
    print("=== Debates XML-Scraper für TheyWorkForYou ===")
    print("Lädt XML-Dateien von der debates-Seite herunter...")
    print("Zeitraum: 2012-2021")
    print("Zielverzeichnis: scraper/data/debates/")
    print()
    
    # Scraper initialisieren
    scraper = DebatesXMLScraper(
        base_url="https://www.theyworkforyou.com/pwdata/scrapedxml/debates/",
        save_dir="scraper/data/debates",
        start_year=2012,
        end_year=2021,
        max_workers=4,
        delay_between_requests=1.0
    )
    
    try:
        # Scraping durchführen
        results = scraper.run_scraping(use_parallel=True)
        
        # Ergebnisse anzeigen
        print("\n" + "="*50)
        print("SCRAPING-ERGEBNISSE")
        print("="*50)
        print(f"Gesamt gefundene Dateien: {results.get('total_files', 0)}")
        print(f"Erfolgreich heruntergeladen: {results.get('downloaded', 0)}")
        print(f"Übersprungen (bereits vorhanden): {results.get('skipped', 0)}")
        print(f"Fehler: {results.get('errors', 0)}")
        
        if 'duration_seconds' in results:
            print(f"Gesamtdauer: {results['duration_seconds']:.2f} Sekunden")
            print(f"Dateien pro Minute: {results.get('files_per_minute', 0):.2f}")
        
        if 'valid_files' in results:
            print(f"Gültige Dateien: {results['valid_files']}")
            print(f"Leere Dateien: {results.get('empty_files', 0)}")
            print(f"Korrupte Dateien: {results.get('corrupted_files', 0)}")
        
        print(f"\nDateien gespeichert in: {scraper.save_dir.absolute()}")
        
        # Liste der heruntergeladenen Dateien anzeigen
        downloaded_files = scraper.list_downloaded_files()
        if downloaded_files:
            print(f"\nErste 10 heruntergeladene Dateien:")
            for i, file_path in enumerate(downloaded_files[:10], 1):
                print(f"  {i}. {file_path.name}")
            if len(downloaded_files) > 10:
                print(f"  ... und {len(downloaded_files) - 10} weitere")
        
    except KeyboardInterrupt:
        print("\n\nScraping durch Benutzer abgebrochen (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFehler beim Scraping: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

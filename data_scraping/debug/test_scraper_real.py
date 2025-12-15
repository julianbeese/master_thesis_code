#!/usr/bin/env python3
"""
Echter Test des XML-Scrapers mit begrenzter Anzahl von Dateien
"""

import sys
from pathlib import Path
import time
from datetime import datetime

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from scraper import XMLScraper
from scraper.database_integration import ScraperDatabaseIntegration
from scraper.utils import DataIntegrityChecker, get_directory_summary


def test_scraper_with_limited_files():
    """Testet den Scraper mit begrenzter Anzahl von Dateien"""
    print("🚀 Echter Test des XML-Scrapers")
    print("=" * 50)
    
    # Konfiguration für Test (nur wenige Dateien)
    test_config = {
        "start_year": 2020,
        "end_year": 2020,  # Nur ein Jahr für Test
        "max_workers": 1,  # Nur ein Worker für Test
        "delay_between_requests": 2.0,  # Höhere Verzögerung
        "save_dir": "scraper/data/raw_test"
    }
    
    print(f"Test-Konfiguration:")
    for key, value in test_config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # 1. Scraper initialisieren
        print("1. Initialisiere Scraper...")
        scraper = XMLScraper(**test_config)
        print("✅ Scraper initialisiert")
        
        # 2. Links sammeln
        print("\n2. Sammle XML-Links...")
        start_time = time.time()
        
        links = scraper.get_xml_links()
        link_time = time.time() - start_time
        
        print(f"✅ {len(links)} XML-Links gefunden (Zeit: {link_time:.2f}s)")
        
        if links:
            print(f"   Erste 5 Links:")
            for i, link in enumerate(links[:5]):
                print(f"     {i+1}. {link['filename']} (Jahr: {link['year']})")
            
            if len(links) > 5:
                print(f"     ... und {len(links) - 5} weitere")
        
        # 3. Teste Download von nur 2 Dateien
        if links:
            print(f"\n3. Teste Download von 2 Dateien...")
            
            test_links = links[:2]  # Nur 2 Dateien für Test
            download_start = time.time()
            
            for i, link in enumerate(test_links, 1):
                print(f"   Download {i}/{len(test_links)}: {link['filename']}")
                
                success, message = scraper.download_file(link['url'], link['filename'])
                
                if success:
                    print(f"     ✅ Erfolgreich: {message}")
                else:
                    print(f"     ❌ Fehler: {message}")
            
            download_time = time.time() - download_start
            print(f"   Download-Zeit: {download_time:.2f}s")
        
        # 4. Validiere heruntergeladene Dateien
        print(f"\n4. Validiere heruntergeladene Dateien...")
        raw_dir = Path(test_config['save_dir'])
        
        if raw_dir.exists():
            checker = DataIntegrityChecker()
            integrity_results = checker.check_directory_integrity(raw_dir)
            
            print(f"   Gesamt: {integrity_results['total_files']} Dateien")
            print(f"   Gültig: {integrity_results['valid_files']} Dateien")
            print(f"   Ungültig: {integrity_results['invalid_files']} Dateien")
            print(f"   Leer: {integrity_results['empty_files']} Dateien")
            
            if integrity_results['total_files'] > 0:
                validity_rate = (integrity_results['valid_files'] / integrity_results['total_files']) * 100
                print(f"   Validitätsrate: {validity_rate:.1f}%")
        
        # 5. Verzeichnis-Zusammenfassung
        print(f"\n5. Verzeichnis-Zusammenfassung...")
        summary = get_directory_summary(raw_dir)
        
        if 'error' not in summary:
            print(f"   Verzeichnis: {summary['directory']}")
            print(f"   Dateien: {summary['total_files']}")
            print(f"   Größe: {summary['total_size_mb']:.2f} MB")
            print(f"   Jahre: {summary['years']}")
            print(f"   Erweiterungen: {summary['file_extensions']}")
        
        print(f"\n🎉 Scraper-Test erfolgreich abgeschlossen!")
        return True
        
    except Exception as e:
        print(f"\n❌ Fehler beim Scraper-Test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_integration():
    """Testet die Datenbank-Integration"""
    print("\n🗄️ Test der Datenbank-Integration")
    print("=" * 50)
    
    try:
        with ScraperDatabaseIntegration() as integration:
            print("1. Teste Datenbank-Verbindung...")
            
            # Erstelle Test-Tabelle
            integration.create_scraping_metadata_table()
            print("✅ Metadaten-Tabelle erstellt")
            
            # Teste Metadaten-Speicherung
            test_metadata = [
                {
                    'filename': 'test_2020.xml',
                    'url': 'https://test.example.com/test_2020.xml',
                    'year': 2020,
                    'file_size_bytes': 1024,
                    'file_hash_md5': 'test_hash_123',
                    'is_valid_xml': True,
                    'validation_error': '',
                    'processing_status': 'test'
                }
            ]
            
            integration.store_scraping_metadata(test_metadata)
            print("✅ Test-Metadaten gespeichert")
            
            # Hole Statistiken
            stats = integration.get_scraping_statistics()
            print(f"✅ Statistiken abgerufen: {stats['total_files']} Dateien")
            
            print("🎉 Datenbank-Integration erfolgreich getestet!")
            return True
            
    except Exception as e:
        print(f"❌ Fehler bei Datenbank-Integration: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Hauptfunktion"""
    print("🧪 Parliament Data Processing System - Echter Test")
    print("=" * 60)
    print(f"Startzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Warnung
    print("⚠️  WICHTIG: Dieser Test lädt echte Daten von theyworkforyou.com herunter!")
    print("   - Nur 2 Dateien werden heruntergeladen")
    print("   - Nur Jahr 2020 wird gescrapt")
    print("   - Höhere Verzögerung zwischen Requests")
    print()
    
    # Benutzer-Bestätigung
    response = input("Möchten Sie den echten Test fortsetzen? (y/N): ")
    if response.lower() != 'y':
        print("Test abgebrochen.")
        return
    
    print()
    
    # Führe Tests durch
    scraper_success = test_scraper_with_limited_files()
    db_success = test_database_integration()
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("📊 TEST-ZUSAMMENFASSUNG")
    print("=" * 60)
    print(f"Scraper-Test: {'✅ Erfolgreich' if scraper_success else '❌ Fehlgeschlagen'}")
    print(f"Datenbank-Test: {'✅ Erfolgreich' if db_success else '❌ Fehlgeschlagen'}")
    print(f"Gesamt: {'🎉 Alle Tests erfolgreich!' if scraper_success and db_success else '❌ Einige Tests fehlgeschlagen'}")
    print(f"Endzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
XML-Parsing-Prozess für alle gescrapten Parlamentsdaten
Parst alle XML-Dateien und speichert sie strukturiert in der Datenbank
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import time
from typing import List, Dict, Any

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from database import DatabaseManager, config as db_config
from parser.database_integration import ParliamentDatabaseIntegration, parse_and_store_xml_files
from parser.xml_parser import ParliamentXMLParser


def setup_logging() -> logging.Logger:
    """Setup für Logging"""
    log_dir = Path("parser/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"xml_parsing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_xml_files(raw_data_dir: Path) -> List[Path]:
    """Sammelt alle XML-Dateien aus dem Raw-Data-Verzeichnis"""
    xml_files = list(raw_data_dir.glob("*.xml"))
    return sorted(xml_files)


def run_parsing_process(xml_files: List[Path], batch_size: int = 50, db_path: str = None) -> Dict[str, Any]:
    """
    Führt den Parsing-Prozess für alle XML-Dateien durch
    
    Args:
        xml_files: Liste der XML-Dateien
        batch_size: Anzahl Dateien pro Batch
        
    Returns:
        Dictionary mit Verarbeitungsergebnissen
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starte XML-Parsing für {len(xml_files)} Dateien")
    logger.info(f"Batch-Größe: {batch_size}")
    
    start_time = time.time()
    total_processed = 0
    total_successful = 0
    total_failed = 0
    
    # Verarbeitung in Batches
    for i in range(0, len(xml_files), batch_size):
        batch_files = xml_files[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(xml_files) + batch_size - 1) // batch_size
        
        logger.info(f"Verarbeite Batch {batch_num}/{total_batches} ({len(batch_files)} Dateien)")
        
        try:
            # Batch verarbeiten
            result = parse_and_store_xml_files(batch_files, db_path or "database/data/parliament_data.db")
            
            # Statistiken aktualisieren
            total_processed += result['sessions_processed']
            total_successful += result['successful_stores']
            total_failed += result['failed_stores']
            
            logger.info(
                f"Batch {batch_num} abgeschlossen: "
                f"{result['sessions_processed']} verarbeitet, "
                f"{result['successful_stores']} erfolgreich, "
                f"{result['failed_stores']} fehlgeschlagen"
            )
            
            # Fortschritt anzeigen
            progress = (i + len(batch_files)) / len(xml_files) * 100
            logger.info(f"Fortschritt: {progress:.1f}% ({i + len(batch_files)}/{len(xml_files)})")
            
        except Exception as e:
            logger.error(f"Fehler in Batch {batch_num}: {e}")
            total_failed += len(batch_files)
    
    total_time = time.time() - start_time
    
    return {
        'total_files': len(xml_files),
        'total_processed': total_processed,
        'total_successful': total_successful,
        'total_failed': total_failed,
        'total_time_minutes': total_time / 60,
        'files_per_minute': len(xml_files) / (total_time / 60) if total_time > 0 else 0
    }


def get_database_statistics(db_path: str = None) -> Dict[str, Any]:
    """Gibt aktuelle Datenbank-Statistiken zurück"""
    try:
        with ParliamentDatabaseIntegration(DatabaseManager(db_path) if db_path else None) as integration:
            return integration.get_parliament_statistics()
    except Exception as e:
        logging.getLogger(__name__).error(f"Fehler beim Abrufen der Statistiken: {e}")
        return {'error': str(e)}


def export_ml_data(db_path: str = None) -> Dict[str, Any]:
    """Exportiert die verarbeiteten Daten für ML-Modelle"""
    try:
        with ParliamentDatabaseIntegration(DatabaseManager(db_path) if db_path else None) as integration:
            # Export in verschiedenen Formaten
            results = {}
            formats = ['parquet', 'csv', 'json']
            
            for format_type in formats:
                result = integration.export_ml_data(format_type)
                results[format_type] = result
            
            return results
    except Exception as e:
        logging.getLogger(__name__).error(f"Fehler beim Export: {e}")
        return {'error': str(e)}


def main():
    """Hauptfunktion"""
    logger = setup_logging()
    
    print("🚀 Parliament Data XML-Parsing System")
    print("=" * 50)
    print(f"Startzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 1. Verzeichnisse einrichten
        logger.info("Richte Verzeichnisse ein...")
        db_config.setup_directories()
        
        # 2. XML-Dateien sammeln
        raw_data_dir = Path("scraper/data/raw")
        if not raw_data_dir.exists():
            logger.error(f"Raw-Data-Verzeichnis nicht gefunden: {raw_data_dir}")
            return 1
        
        xml_files = get_xml_files(raw_data_dir)
        logger.info(f"Gefunden: {len(xml_files)} XML-Dateien")
        
        if not xml_files:
            logger.warning("Keine XML-Dateien gefunden!")
            return 1
        
        # 3. Datenbank-Tabellen erstellen
        logger.info("Erstelle Datenbank-Tabellen...")
        # Verwende eine neue Datenbank-Datei um Lock-Konflikte zu vermeiden
        db_path = f"database/data/parliament_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        with ParliamentDatabaseIntegration(DatabaseManager(db_path)) as integration:
            integration.create_parliament_tables()
        
        # 4. Parsing-Prozess starten
        logger.info("Starte XML-Parsing-Prozess...")
        parsing_results = run_parsing_process(xml_files, batch_size=50, db_path=db_path)
        
        # 5. Statistiken anzeigen
        logger.info("Sammle Datenbank-Statistiken...")
        db_stats = get_database_statistics(db_path)
        
        # 6. ML-Daten exportieren
        logger.info("Exportiere ML-Daten...")
        export_results = export_ml_data(db_path)
        
        # 7. Zusammenfassung
        print(f"\n📊 PARSING-ERGEBNISSE")
        print("=" * 50)
        print(f"Gesamtzeit: {parsing_results['total_time_minutes']:.1f} Minuten")
        print(f"Dateien verarbeitet: {parsing_results['total_processed']}/{parsing_results['total_files']}")
        print(f"Erfolgreich: {parsing_results['total_successful']}")
        print(f"Fehlgeschlagen: {parsing_results['total_failed']}")
        print(f"Geschwindigkeit: {parsing_results['files_per_minute']:.1f} Dateien/Minute")
        
        print(f"\n📈 DATENBANK-STATISTIKEN")
        print("=" * 50)
        if 'error' not in db_stats:
            print(f"Sessions: {db_stats.get('total_sessions', 0)}")
            print(f"Debatten: {db_stats.get('total_debates', 0)}")
            print(f"Reden: {db_stats.get('total_speeches', 0)}")
            print(f"Sprecher: {db_stats.get('total_speakers', 0)}")
            print(f"Absätze: {db_stats.get('total_paragraphs', 0)}")
            print(f"Wörter: {db_stats.get('total_words', 0):,}")
            print(f"Datum-Bereich: {db_stats.get('date_range', 'Unbekannt')}")
        else:
            print(f"Fehler: {db_stats['error']}")
        
        print(f"\n📁 EXPORT-ERGEBNISSE")
        print("=" * 50)
        if 'error' not in export_results:
            for format_type, result in export_results.items():
                if result.get('success'):
                    print(f"{format_type.upper()}: {result['output_file']} ({result['row_count']} Zeilen)")
                else:
                    print(f"{format_type.upper()}: Fehler - {result.get('error', 'Unbekannt')}")
        else:
            print(f"Export-Fehler: {export_results['error']}")
        
        print(f"\n🎉 XML-Parsing erfolgreich abgeschlossen!")
        print(f"   Die Daten sind bereit für ML-Modelle!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

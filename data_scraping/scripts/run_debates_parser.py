#!/usr/bin/env python3
"""
Hauptskript für das Parsing und Importieren von debates XML-Daten
Führt den kompletten Workflow aus: XML-Parsing -> DuckDB-Import -> ML-Features
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import argparse

# Füge die notwendigen Verzeichnisse zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent / "parser"))
sys.path.append(str(Path(__file__).parent.parent / "database" / "managers"))

from debates_parser import DebatesXMLParser, DebatesDataProcessor
from debates_database_manager import DebatesDatabaseManager


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup für Logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"debates_parser_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def parse_arguments():
    """Kommandozeilen-Argumente parsen"""
    parser = argparse.ArgumentParser(description="Debates XML-Parser und Datenbank-Import")
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="scraper/data/debates",
        help="Verzeichnis mit XML-Dateien (Standard: scraper/data/debates)"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="database/data/debates_data.db",
        help="Pfad zur DuckDB-Datei (Standard: database/data/debates_data.db)"
    )
    
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximale Anzahl zu verarbeitender Dateien (Standard: alle)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch-Größe für Datenbank-Inserts (Standard: 100)"
    )
    
    parser.add_argument(
        "--export-parquet",
        type=str,
        help="Exportiere Daten als Parquet-Dateien in das angegebene Verzeichnis"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log-Level (Standard: INFO)"
    )
    
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test-Modus: Verarbeite nur wenige Dateien"
    )
    
    return parser.parse_args()


def main():
    """Hauptfunktion"""
    args = parse_arguments()
    
    # Logging setup
    logger = setup_logging(args.log_level)
    
    logger.info("=== Debates XML-Parser und Datenbank-Import ===")
    logger.info(f"Input-Verzeichnis: {args.input_dir}")
    logger.info(f"Datenbank-Pfad: {args.db_path}")
    logger.info(f"Max-Dateien: {args.max_files or 'alle'}")
    logger.info(f"Batch-Größe: {args.batch_size}")
    logger.info(f"Test-Modus: {args.test_mode}")
    
    # Test-Modus: Begrenze auf wenige Dateien
    if args.test_mode and not args.max_files:
        args.max_files = 5
        logger.info(f"Test-Modus aktiviert: Max-Dateien auf {args.max_files} gesetzt")
    
    try:
        # Database Manager initialisieren
        logger.info("Initialisiere Database Manager...")
        db_manager = DebatesDatabaseManager(args.db_path)
        
        # Input-Verzeichnis prüfen
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            logger.error(f"Input-Verzeichnis nicht gefunden: {input_dir}")
            return 1
        
        xml_files = list(input_dir.glob("*.xml"))
        if not xml_files:
            logger.error(f"Keine XML-Dateien in {input_dir} gefunden")
            return 1
        
        logger.info(f"Gefunden: {len(xml_files)} XML-Dateien")
        
        # Import durchführen
        logger.info("Starte XML-Parsing und Datenbank-Import...")
        start_time = datetime.now()
        
        import_stats = db_manager.import_xml_directory(
            input_dir,
            max_files=args.max_files,
            batch_size=args.batch_size
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Ergebnisse anzeigen
        logger.info("=== Import-Ergebnisse ===")
        logger.info(f"Verarbeitungsdauer: {duration:.2f} Sekunden")
        logger.info(f"Import-Statistiken: {import_stats}")
        
        # Datenbank-Statistiken
        db_stats = db_manager.get_database_statistics()
        logger.info("=== Datenbank-Statistiken ===")
        for key, value in db_stats.items():
            logger.info(f"{key}: {value}")
        
        # ML-Dataset-Info
        try:
            ml_dataset = db_manager.get_ml_dataset()
            logger.info(f"ML-Dataset: {ml_dataset.shape[0]} Reden mit Features")
        except Exception as e:
            logger.warning(f"ML-Dataset konnte nicht erstellt werden: {e}")
        
        # Parquet-Export
        if args.export_parquet:
            export_dir = Path(args.export_parquet)
            logger.info(f"Exportiere Daten als Parquet nach {export_dir}")
            db_manager.export_to_parquet(export_dir)
            logger.info("Parquet-Export abgeschlossen")
        
        # Beispiel-Queries
        logger.info("=== Beispiel-Queries ===")
        
        # Top-Sprecher
        try:
            top_speakers = db_manager.get_speaker_statistics()
            logger.info("Top 5 Sprecher:")
            for i, row in enumerate(top_speakers.head(5).iter_rows(named=True)):
                logger.info(f"  {i+1}. {row['speaker_name']}: {row['total_speeches']} Reden")
        except Exception as e:
            logger.warning(f"Top-Sprecher-Query fehlgeschlagen: {e}")
        
        # Debatten-Themen
        try:
            topics = db_manager.get_debate_topics(limit=5)
            logger.info("Top 5 Debatten-Themen:")
            for i, row in enumerate(topics.head(5).iter_rows(named=True)):
                logger.info(f"  {i+1}. {row['title'][:50]}... ({row['debate_count']} Debatten)")
        except Exception as e:
            logger.warning(f"Debatten-Themen-Query fehlgeschlagen: {e}")
        
        logger.info("=== Verarbeitung erfolgreich abgeschlossen ===")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Verarbeitung durch Benutzer abgebrochen")
        return 1
    except Exception as e:
        logger.error(f"Fehler bei der Verarbeitung: {e}", exc_info=True)
        return 1
    finally:
        if 'db_manager' in locals():
            db_manager.close()


if __name__ == "__main__":
    sys.exit(main())

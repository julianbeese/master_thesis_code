"""
Integration zwischen XML-Scraper und Datenbank-Management-System
Ermöglicht nahtlose Übertragung von gescrapten Daten in die Datenbank
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from database import DatabaseManager, config as db_config
from scraper import XMLScraper, scraper_config, FileValidator, DataIntegrityChecker


class ScraperDatabaseIntegration:
    """
    Klasse für die Integration zwischen Scraper und Datenbank
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialisiert die Integration
        
        Args:
            db_manager: Optional: Bestehender DatabaseManager
        """
        self.db_manager = db_manager or DatabaseManager()
        self.logger = logging.getLogger(__name__)
        
        # Setup
        scraper_config.setup_directories()
        db_config.setup_directories()
    
    def create_scraping_metadata_table(self) -> None:
        """Erstellt Tabelle für Scraping-Metadaten"""
        metadata_schema = """
        CREATE TABLE IF NOT EXISTS scraping_metadata (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            url TEXT NOT NULL,
            year INTEGER,
            download_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size_bytes INTEGER,
            file_hash_md5 TEXT,
            is_valid_xml BOOLEAN,
            validation_error TEXT,
            processing_status TEXT DEFAULT 'downloaded',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        self.db_manager.execute_query(metadata_schema)
        self.logger.info("Scraping-Metadaten-Tabelle erstellt")
    
    def store_scraping_metadata(self, metadata_list: List[Dict[str, Any]]) -> None:
        """
        Speichert Scraping-Metadaten in der Datenbank
        
        Args:
            metadata_list: Liste von Metadaten-Dictionaries
        """
        if not metadata_list:
            return
        
        # Erstelle Tabelle falls nicht vorhanden
        self.create_scraping_metadata_table()
        
        # Hole die nächste verfügbare ID
        max_id_result = self.db_manager.execute_query("SELECT COALESCE(MAX(id), 0) FROM scraping_metadata")
        next_id = max_id_result[0][0] + 1 if max_id_result else 1
        
        # Bereite Daten für Insert vor
        insert_data = []
        for i, metadata in enumerate(metadata_list):
            insert_data.append((
                next_id + i,  # ID dynamisch generieren
                metadata.get('filename', ''),
                metadata.get('url', ''),
                metadata.get('year'),
                metadata.get('download_date', datetime.now()),
                metadata.get('file_size_bytes', 0),
                metadata.get('file_hash_md5', ''),
                metadata.get('is_valid_xml', False),
                metadata.get('validation_error', ''),
                metadata.get('processing_status', 'downloaded')
            ))
        
        # Batch-Insert
        insert_query = """
        INSERT INTO scraping_metadata 
        (id, filename, url, year, download_date, file_size_bytes, file_hash_md5, 
         is_valid_xml, validation_error, processing_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        for data in insert_data:
            self.db_manager.execute_query(insert_query, data)
        
        self.logger.info(f"{len(metadata_list)} Metadaten-Einträge gespeichert")
    
    def get_scraping_statistics(self) -> Dict[str, Any]:
        """
        Gibt Scraping-Statistiken aus der Datenbank zurück
        
        Returns:
            Dictionary mit Statistiken
        """
        try:
            # Gesamtstatistiken
            total_query = "SELECT COUNT(*) as total FROM scraping_metadata"
            total_result = self.db_manager.execute_query(total_query)
            total_files = total_result[0][0] if total_result else 0
            
            # Nach Jahr gruppiert
            year_query = """
            SELECT year, COUNT(*) as count, 
                   SUM(CASE WHEN is_valid_xml = 1 THEN 1 ELSE 0 END) as valid_count
            FROM scraping_metadata 
            WHERE year IS NOT NULL
            GROUP BY year 
            ORDER BY year
            """
            year_results = self.db_manager.execute_query(year_query)
            
            # Nach Status gruppiert
            status_query = """
            SELECT processing_status, COUNT(*) as count
            FROM scraping_metadata 
            GROUP BY processing_status
            """
            status_results = self.db_manager.execute_query(status_query)
            
            # Validierungsstatistiken
            validation_query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_valid_xml = 1 THEN 1 ELSE 0 END) as valid,
                SUM(CASE WHEN is_valid_xml = 0 THEN 1 ELSE 0 END) as invalid
            FROM scraping_metadata
            """
            validation_result = self.db_manager.execute_query(validation_query)
            validation_stats = validation_result[0] if validation_result else (0, 0, 0)
            
            return {
                "total_files": total_files,
                "by_year": {str(row[0]): {"total": row[1], "valid": row[2]} for row in year_results},
                "by_status": {row[0]: row[1] for row in status_results},
                "validation": {
                    "total": validation_stats[0],
                    "valid": validation_stats[1],
                    "invalid": validation_stats[2],
                    "validity_rate": (validation_stats[1] / validation_stats[0] * 100) if validation_stats[0] > 0 else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Statistiken: {e}")
            return {"error": str(e)}
    
    def validate_and_store_metadata(self, raw_data_dir: Path) -> Dict[str, Any]:
        """
        Validiert alle Dateien im Raw-Data-Verzeichnis und speichert Metadaten
        
        Args:
            raw_data_dir: Verzeichnis mit rohen XML-Dateien
            
        Returns:
            Dictionary mit Validierungsergebnissen
        """
        if not raw_data_dir.exists():
            return {"error": "Raw-Data-Verzeichnis existiert nicht"}
        
        xml_files = list(raw_data_dir.glob("*.xml"))
        metadata_list = []
        
        self.logger.info(f"Validiere {len(xml_files)} XML-Dateien")
        
        for file_path in xml_files:
            # Datei-Informationen sammeln
            file_info = FileValidator.get_file_info(file_path)
            is_valid, error_msg = FileValidator.is_valid_xml_file(file_path)
            
            metadata = {
                'filename': file_path.name,
                'url': '',  # URL nicht verfügbar für bereits heruntergeladene Dateien
                'year': file_info.get('year'),
                'download_date': file_info.get('created'),
                'file_size_bytes': file_info.get('size_bytes', 0),
                'file_hash_md5': file_info.get('hash_md5', ''),
                'is_valid_xml': is_valid,
                'validation_error': error_msg if not is_valid else '',
                'processing_status': 'validated' if is_valid else 'validation_failed'
            }
            
            metadata_list.append(metadata)
        
        # Metadaten in Datenbank speichern
        self.store_scraping_metadata(metadata_list)
        
        # Statistiken berechnen
        valid_count = sum(1 for m in metadata_list if m['is_valid_xml'])
        invalid_count = len(metadata_list) - valid_count
        
        return {
            "total_files": len(metadata_list),
            "valid_files": valid_count,
            "invalid_files": invalid_count,
            "validity_rate": (valid_count / len(metadata_list) * 100) if metadata_list else 0
        }
    
    def run_complete_scraping_pipeline(self, use_parallel: bool = True) -> Dict[str, Any]:
        """
        Führt die komplette Scraping-Pipeline aus
        
        Args:
            use_parallel: Verwende parallele Downloads
            
        Returns:
            Dictionary mit Pipeline-Ergebnissen
        """
        self.logger.info("=== Starte komplette Scraping-Pipeline ===")
        
        try:
            # 1. Scraping durchführen
            scraper = XMLScraper(
                start_year=scraper_config.START_YEAR,
                end_year=scraper_config.END_YEAR,
                max_workers=scraper_config.MAX_WORKERS,
                delay_between_requests=scraper_config.DELAY_BETWEEN_REQUESTS
            )
            
            scraping_results = scraper.run_scraping(use_parallel=use_parallel)
            
            # 2. Metadaten validieren und speichern
            raw_data_dir = scraper_config.get_raw_data_dir()
            validation_results = self.validate_and_store_metadata(raw_data_dir)
            
            # 3. Statistiken aus Datenbank abrufen
            db_statistics = self.get_scraping_statistics()
            
            # 4. Ergebnisse zusammenfassen
            pipeline_results = {
                "scraping": scraping_results,
                "validation": validation_results,
                "database_statistics": db_statistics,
                "pipeline_completed": True,
                "completion_time": datetime.now()
            }
            
            self.logger.info("=== Scraping-Pipeline erfolgreich abgeschlossen ===")
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Fehler in Scraping-Pipeline: {e}")
            return {
                "pipeline_completed": False,
                "error": str(e),
                "completion_time": datetime.now()
            }
    
    def export_scraping_data_to_parquet(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Exportiert Scraping-Daten als Parquet-Dateien
        
        Args:
            output_dir: Optional: Ausgabeverzeichnis
            
        Returns:
            Dictionary mit Export-Ergebnissen
        """
        if output_dir is None:
            output_dir = scraper_config.get_processed_data_dir()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Metadaten als Parquet exportieren
            metadata_df = self.db_manager.read_table_to_pandas("scraping_metadata")
            metadata_parquet_path = output_dir / "scraping_metadata.parquet"
            metadata_df.to_parquet(metadata_parquet_path, engine='pyarrow')
            
            # Statistiken als JSON exportieren
            statistics = self.get_scraping_statistics()
            statistics_json_path = output_dir / "scraping_statistics.json"
            
            import json
            with open(statistics_json_path, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, indent=2, default=str)
            
            self.logger.info(f"Scraping-Daten exportiert nach {output_dir}")
            
            return {
                "export_successful": True,
                "metadata_parquet": str(metadata_parquet_path),
                "statistics_json": str(statistics_json_path),
                "export_time": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Export: {e}")
            return {
                "export_successful": False,
                "error": str(e),
                "export_time": datetime.now()
            }
    
    def close(self):
        """Schließt die Datenbankverbindung"""
        if self.db_manager:
            self.db_manager.close()
    
    def __enter__(self):
        """Context Manager Entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Exit"""
        self.close()


# Convenience-Funktionen
def run_scraping_with_database_integration(use_parallel: bool = True) -> Dict[str, Any]:
    """
    Führt Scraping mit Datenbank-Integration aus
    
    Args:
        use_parallel: Verwende parallele Downloads
        
    Returns:
        Dictionary mit Ergebnissen
    """
    with ScraperDatabaseIntegration() as integration:
        return integration.run_complete_scraping_pipeline(use_parallel=use_parallel)


def get_scraping_dashboard_data() -> Dict[str, Any]:
    """
    Gibt Dashboard-Daten für Scraping-Status zurück
    
    Returns:
        Dictionary mit Dashboard-Daten
    """
    with ScraperDatabaseIntegration() as integration:
        return integration.get_scraping_statistics()


if __name__ == "__main__":
    # Beispiel-Nutzung
    print("=== XML-Scraper mit Datenbank-Integration ===")
    
    with ScraperDatabaseIntegration() as integration:
        # Komplette Pipeline ausführen
        results = integration.run_complete_scraping_pipeline(use_parallel=True)
        
        print("\n=== Pipeline-Ergebnisse ===")
        for key, value in results.items():
            print(f"{key}: {value}")
        
        # Daten exportieren
        export_results = integration.export_scraping_data_to_parquet()
        print(f"\nExport erfolgreich: {export_results.get('export_successful', False)}")

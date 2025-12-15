"""
Utility-Funktionen für XML-Scraper
Enthält Hilfsfunktionen für Dateiverarbeitung, Validierung und Monitoring
"""

import os
import re
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import json
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


class FileValidator:
    """Klasse für Datei-Validierung"""
    
    @staticmethod
    def is_valid_xml_file(file_path: Path) -> Tuple[bool, str]:
        """
        Validiert ob eine Datei gültiges XML enthält
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            Tuple (is_valid, error_message)
        """
        try:
            if not file_path.exists():
                return False, "Datei existiert nicht"
            
            if file_path.stat().st_size == 0:
                return False, "Datei ist leer"
            
            # Versuche XML zu parsen
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # Erste 1000 Zeichen
                
                if not content.strip().startswith('<?xml'):
                    return False, "Keine XML-Deklaration gefunden"
                
                # Versuche vollständiges XML zu parsen
                f.seek(0)
                full_content = f.read()
                ET.fromstring(full_content)
                
            return True, "XML ist gültig"
            
        except ET.ParseError as e:
            return False, f"XML-Parse-Fehler: {e}"
        except Exception as e:
            return False, f"Unerwarteter Fehler: {e}"
    
    @staticmethod
    def get_file_info(file_path: Path) -> Dict[str, Any]:
        """
        Sammelt Informationen über eine Datei
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            Dictionary mit Datei-Informationen
        """
        try:
            stat = file_path.stat()
            
            # Hash berechnen
            file_hash = FileValidator._calculate_file_hash(file_path)
            
            # Jahr extrahieren
            year = FileValidator._extract_year_from_filename(file_path.name)
            
            return {
                "filename": file_path.name,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "hash_md5": file_hash,
                "year": year,
                "extension": file_path.suffix
            }
        except Exception as e:
            return {
                "filename": file_path.name,
                "error": str(e)
            }
    
    @staticmethod
    def _calculate_file_hash(file_path: Path) -> str:
        """Berechnet MD5-Hash einer Datei"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    @staticmethod
    def _extract_year_from_filename(filename: str) -> Optional[int]:
        """Extrahiert Jahr aus Dateinamen"""
        patterns = [
            r'(\d{4})',
            r'(\d{4})-\d{2}-\d{2}',
            r'(\d{4})_\d{2}_\d{2}',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    year = int(match.group(1))
                    if 2000 <= year <= 2030:
                        return year
                except ValueError:
                    continue
        return None


class ProgressTracker:
    """Klasse für Progress-Tracking"""
    
    def __init__(self, total_items: int, logger: Optional[logging.Logger] = None):
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.start_time = datetime.now()
        self.logger = logger or logging.getLogger(__name__)
        self.last_update = time.time()
        self.update_interval = 5.0  # Sekunden
    
    def update(self, success: bool = True, item_name: str = ""):
        """
        Aktualisiert den Progress
        
        Args:
            success: Ob das Item erfolgreich verarbeitet wurde
            item_name: Name des Items (für Logging)
        """
        if success:
            self.completed_items += 1
        else:
            self.failed_items += 1
        
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self._log_progress(item_name)
            self.last_update = current_time
    
    def _log_progress(self, item_name: str):
        """Loggt den aktuellen Progress"""
        progress_percent = (self.completed_items / self.total_items) * 100
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        if self.completed_items > 0:
            avg_time_per_item = elapsed_time / self.completed_items
            remaining_items = self.total_items - self.completed_items
            estimated_remaining = remaining_items * avg_time_per_item
            eta = datetime.now() + timedelta(seconds=estimated_remaining)
        else:
            eta = "Unbekannt"
        
        self.logger.info(
            f"Progress: {self.completed_items}/{self.total_items} "
            f"({progress_percent:.1f}%) - "
            f"Fehler: {self.failed_items} - "
            f"ETA: {eta.strftime('%H:%M:%S') if isinstance(eta, datetime) else eta}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Gibt aktuelle Statistiken zurück"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "success_rate": (self.completed_items / self.total_items) * 100 if self.total_items > 0 else 0,
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_formatted": str(timedelta(seconds=int(elapsed_time))),
            "items_per_second": self.completed_items / elapsed_time if elapsed_time > 0 else 0
        }


class URLManager:
    """Klasse für URL-Management"""
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """
        Prüft ob eine URL gültig ist
        
        Args:
            url: URL zum Prüfen
            
        Returns:
            True wenn gültig, False sonst
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def normalize_url(url: str, base_url: str) -> str:
        """
        Normalisiert eine URL relativ zu einer Basis-URL
        
        Args:
            url: URL zum Normalisieren
            base_url: Basis-URL
            
        Returns:
            Normalisierte URL
        """
        return urljoin(base_url, url)
    
    @staticmethod
    def extract_filename_from_url(url: str) -> str:
        """
        Extrahiert Dateinamen aus URL
        
        Args:
            url: URL
            
        Returns:
            Dateiname
        """
        return os.path.basename(urlparse(url).path)
    
    @staticmethod
    def check_url_accessibility(url: str, timeout: int = 10) -> Tuple[bool, str]:
        """
        Prüft ob eine URL erreichbar ist
        
        Args:
            url: URL zum Prüfen
            timeout: Timeout in Sekunden
            
        Returns:
            Tuple (is_accessible, status_message)
        """
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            if response.status_code == 200:
                return True, f"OK ({response.status_code})"
            else:
                return False, f"HTTP {response.status_code}"
        except requests.RequestException as e:
            return False, f"Request-Fehler: {e}"
        except Exception as e:
            return False, f"Unerwarteter Fehler: {e}"


class DataIntegrityChecker:
    """Klasse für Datenintegritätsprüfung"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def check_directory_integrity(self, directory: Path) -> Dict[str, Any]:
        """
        Prüft die Integrität aller Dateien in einem Verzeichnis
        
        Args:
            directory: Verzeichnis zum Prüfen
            
        Returns:
            Dictionary mit Integritäts-Ergebnissen
        """
        if not directory.exists():
            return {"error": "Verzeichnis existiert nicht"}
        
        xml_files = list(directory.glob("*.xml"))
        results = {
            "total_files": len(xml_files),
            "valid_files": 0,
            "invalid_files": 0,
            "empty_files": 0,
            "corrupted_files": 0,
            "file_details": []
        }
        
        for file_path in xml_files:
            file_info = FileValidator.get_file_info(file_path)
            is_valid, error_msg = FileValidator.is_valid_xml_file(file_path)
            
            file_detail = {
                "filename": file_path.name,
                "is_valid": is_valid,
                "error": error_msg if not is_valid else None,
                "size_bytes": file_info.get("size_bytes", 0),
                "year": file_info.get("year")
            }
            
            results["file_details"].append(file_detail)
            
            if file_info.get("size_bytes", 0) == 0:
                results["empty_files"] += 1
            elif is_valid:
                results["valid_files"] += 1
            else:
                results["invalid_files"] += 1
                if "parse" in error_msg.lower():
                    results["corrupted_files"] += 1
        
        return results
    
    def generate_integrity_report(self, directory: Path, output_file: Optional[Path] = None) -> str:
        """
        Generiert einen detaillierten Integritätsbericht
        
        Args:
            directory: Verzeichnis zum Prüfen
            output_file: Optional: Datei zum Speichern des Berichts
            
        Returns:
            Bericht als String
        """
        results = self.check_directory_integrity(directory)
        
        report_lines = [
            f"=== Integritätsbericht für {directory} ===",
            f"Generiert am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Gesamtanzahl Dateien: {results['total_files']}",
            f"Gültige Dateien: {results['valid_files']}",
            f"Ungültige Dateien: {results['invalid_files']}",
            f"Leere Dateien: {results['empty_files']}",
            f"Korrupte Dateien: {results['corrupted_files']}",
            ""
        ]
        
        if results['invalid_files'] > 0:
            report_lines.append("=== Ungültige Dateien ===")
            for file_detail in results['file_details']:
                if not file_detail['is_valid']:
                    report_lines.append(
                        f"- {file_detail['filename']}: {file_detail['error']}"
                    )
            report_lines.append("")
        
        # Statistiken nach Jahr
        year_stats = {}
        for file_detail in results['file_details']:
            year = file_detail.get('year')
            if year:
                if year not in year_stats:
                    year_stats[year] = {'total': 0, 'valid': 0}
                year_stats[year]['total'] += 1
                if file_detail['is_valid']:
                    year_stats[year]['valid'] += 1
        
        if year_stats:
            report_lines.append("=== Statistiken nach Jahr ===")
            for year in sorted(year_stats.keys()):
                stats = year_stats[year]
                report_lines.append(
                    f"{year}: {stats['valid']}/{stats['total']} gültig "
                    f"({(stats['valid']/stats['total']*100):.1f}%)"
                )
        
        report = "\n".join(report_lines)
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Integritätsbericht gespeichert: {output_file}")
        
        return report


class BatchProcessor:
    """Klasse für Batch-Verarbeitung"""
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4,
                 logger: Optional[logging.Logger] = None):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.logger = logger or logging.getLogger(__name__)
    
    def process_files_in_batches(self, files: List[Path], 
                                processor_func: callable,
                                progress_tracker: Optional[ProgressTracker] = None) -> List[Any]:
        """
        Verarbeitet Dateien in Batches
        
        Args:
            files: Liste der zu verarbeitenden Dateien
            processor_func: Funktion zur Verarbeitung einer Datei
            progress_tracker: Optional: Progress-Tracker
            
        Returns:
            Liste der Verarbeitungsergebnisse
        """
        results = []
        
        # Dateien in Batches aufteilen
        batches = [files[i:i + self.batch_size] for i in range(0, len(files), self.batch_size)]
        
        self.logger.info(f"Verarbeite {len(files)} Dateien in {len(batches)} Batches")
        
        for batch_idx, batch in enumerate(batches, 1):
            self.logger.info(f"Verarbeite Batch {batch_idx}/{len(batches)} ({len(batch)} Dateien)")
            
            # Batch parallel verarbeiten
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(processor_func, file_path): file_path
                    for file_path in batch
                }
                
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if progress_tracker:
                            progress_tracker.update(success=True, item_name=file_path.name)
                            
                    except Exception as e:
                        self.logger.error(f"Fehler bei Verarbeitung von {file_path.name}: {e}")
                        results.append({"file": file_path.name, "error": str(e)})
                        
                        if progress_tracker:
                            progress_tracker.update(success=False, item_name=file_path.name)
        
        return results


# Convenience-Funktionen
def quick_file_validation(file_path: Union[str, Path]) -> bool:
    """
    Schnelle Datei-Validierung
    
    Args:
        file_path: Pfad zur Datei
        
    Returns:
        True wenn gültig, False sonst
    """
    is_valid, _ = FileValidator.is_valid_xml_file(Path(file_path))
    return is_valid


def get_directory_summary(directory: Union[str, Path]) -> Dict[str, Any]:
    """
    Gibt eine Zusammenfassung eines Verzeichnisses zurück
    
    Args:
        directory: Verzeichnis-Pfad
        
    Returns:
        Dictionary mit Zusammenfassung
    """
    directory = Path(directory)
    if not directory.exists():
        return {"error": "Verzeichnis existiert nicht"}
    
    xml_files = list(directory.glob("*.xml"))
    
    summary = {
        "directory": str(directory),
        "total_files": len(xml_files),
        "total_size_mb": sum(f.stat().st_size for f in xml_files) / (1024 * 1024),
        "years": set(),
        "file_extensions": set()
    }
    
    for file_path in xml_files:
        year = FileValidator._extract_year_from_filename(file_path.name)
        if year:
            summary["years"].add(year)
        summary["file_extensions"].add(file_path.suffix)
    
    summary["years"] = sorted(list(summary["years"]))
    summary["file_extensions"] = list(summary["file_extensions"])
    
    return summary

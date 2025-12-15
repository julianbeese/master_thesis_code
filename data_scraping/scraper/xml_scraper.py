"""
XML-Scraper für theyworkforyou.com
Lädt alle XML-Dateien aus dem Zeitraum 2012-2021 herunter
"""

import os
import requests
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


class XMLScraper:
    """
    Hauptklasse für das Scraping von XML-Dateien von theyworkforyou.com
    """
    
    def __init__(self, base_url: str = "https://www.theyworkforyou.com/pwdata/scrapedxml/westminhall/",
                 save_dir: str = "scraper/data/raw",
                 start_year: int = 2012,
                 end_year: int = 2021,
                 max_workers: int = 4,
                 delay_between_requests: float = 1.0):
        """
        Initialisiert den XML-Scraper
        
        Args:
            base_url: Basis-URL der XML-Dateien
            save_dir: Verzeichnis zum Speichern der Dateien
            start_year: Startjahr für das Scraping
            end_year: Endjahr für das Scraping
            max_workers: Maximale Anzahl paralleler Downloads
            delay_between_requests: Verzögerung zwischen Requests (Sekunden)
        """
        self.base_url = base_url
        self.save_dir = Path(save_dir)
        self.start_year = start_year
        self.end_year = end_year
        self.max_workers = max_workers
        self.delay_between_requests = delay_between_requests
        
        # Verzeichnisse erstellen
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging setup
        self._setup_logging()
        
        # Session für bessere Performance
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Statistiken
        self.stats = {
            'total_files': 0,
            'downloaded': 0,
            'skipped': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _setup_logging(self):
        """Setup für Logging"""
        log_dir = Path("scraper/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"xml_scraper_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _extract_year_from_filename(self, filename: str) -> Optional[int]:
        """
        Extrahiert das Jahr aus dem Dateinamen
        
        Args:
            filename: Dateiname
            
        Returns:
            Jahr oder None wenn nicht extrahierbar
        """
        # Verschiedene Patterns für Jahres-Extraktion
        patterns = [
            r'(\d{4})',  # Einfaches 4-stelliges Jahr
            r'(\d{4})-\d{2}-\d{2}',  # Datum-Format
            r'(\d{4})_\d{2}_\d{2}',  # Datum mit Unterstrichen
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    year = int(match.group(1))
                    if 2000 <= year <= 2030:  # Plausibilitätsprüfung
                        return year
                except ValueError:
                    continue
        
        return None
    
    def _is_file_in_year_range(self, filename: str) -> bool:
        """
        Prüft ob die Datei im gewünschten Zeitraum liegt
        
        Args:
            filename: Dateiname
            
        Returns:
            True wenn im Zeitraum, False sonst
        """
        year = self._extract_year_from_filename(filename)
        if year is None:
            return False
        
        return self.start_year <= year <= self.end_year
    
    def _get_file_hash(self, file_path: Path) -> str:
        """
        Berechnet Hash einer Datei für Integritätsprüfung
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            MD5-Hash der Datei
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def get_xml_links(self) -> List[Dict[str, str]]:
        """
        Ruft alle XML-Dateilinks von der Basis-URL ab
        
        Returns:
            Liste von Dictionaries mit URL und Dateiname
        """
        self.logger.info(f"Lade XML-Links von {self.base_url}")
        
        try:
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.xml'):
                    full_url = urljoin(self.base_url, href)
                    filename = os.path.basename(href)
                    
                    # Prüfe ob Datei im gewünschten Zeitraum liegt
                    if self._is_file_in_year_range(filename):
                        links.append({
                            'url': full_url,
                            'filename': filename,
                            'year': self._extract_year_from_filename(filename)
                        })
            
            self.logger.info(f"Gefunden: {len(links)} XML-Dateien im Zeitraum {self.start_year}-{self.end_year}")
            return links
            
        except requests.RequestException as e:
            self.logger.error(f"Fehler beim Laden der XML-Links: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unerwarteter Fehler beim Parsen der Links: {e}")
            raise
    
    def download_file(self, url: str, filename: str) -> Tuple[bool, str]:
        """
        Lädt eine einzelne XML-Datei herunter
        
        Args:
            url: URL der Datei
            filename: Dateiname
            
        Returns:
            Tuple (success, message)
        """
        file_path = self.save_dir / filename
        
        try:
            # Prüfe ob Datei bereits existiert
            if file_path.exists():
                self.logger.info(f"Datei bereits vorhanden: {filename}")
                return True, "Already exists"
            
            # Download
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            # Speichere Datei
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Verzögerung zwischen Requests
            time.sleep(self.delay_between_requests)
            
            self.logger.info(f"Erfolgreich heruntergeladen: {filename} ({len(response.content)} bytes)")
            return True, "Downloaded successfully"
            
        except requests.RequestException as e:
            error_msg = f"Download-Fehler für {filename}: {e}"
            self.logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unerwarteter Fehler für {filename}: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def download_files_parallel(self, links: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Lädt Dateien parallel herunter
        
        Args:
            links: Liste der zu downloadenden Links
            
        Returns:
            Dictionary mit Download-Statistiken
        """
        self.stats['start_time'] = datetime.now()
        self.stats['total_files'] = len(links)
        
        self.logger.info(f"Starte parallelen Download von {len(links)} Dateien mit {self.max_workers} Workern")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit alle Downloads
            future_to_link = {
                executor.submit(self.download_file, link['url'], link['filename']): link
                for link in links
            }
            
            # Verarbeite Ergebnisse
            for future in as_completed(future_to_link):
                link = future_to_link[future]
                try:
                    success, message = future.result()
                    if success:
                        self.stats['downloaded'] += 1
                    else:
                        self.stats['errors'] += 1
                        if "already exists" in message.lower():
                            self.stats['skipped'] += 1
                            self.stats['downloaded'] += 1  # Zähle als erfolgreich
                            self.stats['errors'] -= 1
                except Exception as e:
                    self.logger.error(f"Fehler beim Verarbeiten von {link['filename']}: {e}")
                    self.stats['errors'] += 1
        
        self.stats['end_time'] = datetime.now()
        return self.stats
    
    def download_files_sequential(self, links: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Lädt Dateien sequenziell herunter (für Debugging oder bei Problemen)
        
        Args:
            links: Liste der zu downloadenden Links
            
        Returns:
            Dictionary mit Download-Statistiken
        """
        self.stats['start_time'] = datetime.now()
        self.stats['total_files'] = len(links)
        
        self.logger.info(f"Starte sequenziellen Download von {len(links)} Dateien")
        
        for i, link in enumerate(links, 1):
            self.logger.info(f"Download {i}/{len(links)}: {link['filename']}")
            
            success, message = self.download_file(link['url'], link['filename'])
            
            if success:
                self.stats['downloaded'] += 1
            else:
                self.stats['errors'] += 1
                if "already exists" in message.lower():
                    self.stats['skipped'] += 1
                    self.stats['downloaded'] += 1
                    self.stats['errors'] -= 1
        
        self.stats['end_time'] = datetime.now()
        return self.stats
    
    def get_download_statistics(self) -> Dict[str, any]:
        """
        Gibt detaillierte Download-Statistiken zurück
        
        Returns:
            Dictionary mit Statistiken
        """
        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            self.stats['duration_seconds'] = duration
            self.stats['files_per_minute'] = (self.stats['downloaded'] / duration) * 60 if duration > 0 else 0
        
        return self.stats.copy()
    
    def list_downloaded_files(self) -> List[Path]:
        """
        Listet alle heruntergeladenen XML-Dateien auf
        
        Returns:
            Liste der Dateipfade
        """
        xml_files = list(self.save_dir.glob("*.xml"))
        self.logger.info(f"Gefunden: {len(xml_files)} XML-Dateien im Download-Verzeichnis")
        return xml_files
    
    def verify_downloads(self) -> Dict[str, int]:
        """
        Verifiziert heruntergeladene Dateien
        
        Returns:
            Dictionary mit Verifikations-Statistiken
        """
        xml_files = self.list_downloaded_files()
        verification_stats = {
            'total_files': len(xml_files),
            'valid_files': 0,
            'corrupted_files': 0,
            'empty_files': 0
        }
        
        for file_path in xml_files:
            try:
                if file_path.stat().st_size == 0:
                    verification_stats['empty_files'] += 1
                    self.logger.warning(f"Leere Datei gefunden: {file_path.name}")
                else:
                    # Einfache XML-Validierung
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(1000)  # Erste 1000 Zeichen lesen
                        if content.strip().startswith('<?xml') or '<' in content:
                            verification_stats['valid_files'] += 1
                        else:
                            verification_stats['corrupted_files'] += 1
                            self.logger.warning(f"Möglicherweise korrupte Datei: {file_path.name}")
            except Exception as e:
                verification_stats['corrupted_files'] += 1
                self.logger.error(f"Fehler beim Verifizieren von {file_path.name}: {e}")
        
        return verification_stats
    
    def run_scraping(self, use_parallel: bool = True) -> Dict[str, any]:
        """
        Führt das komplette Scraping durch
        
        Args:
            use_parallel: Verwende parallele Downloads
            
        Returns:
            Dictionary mit Scraping-Ergebnissen
        """
        self.logger.info("=== XML-Scraping gestartet ===")
        self.logger.info(f"Ziel-URL: {self.base_url}")
        self.logger.info(f"Zeitraum: {self.start_year}-{self.end_year}")
        self.logger.info(f"Speicherort: {self.save_dir}")
        
        try:
            # 1. Links sammeln
            links = self.get_xml_links()
            
            if not links:
                self.logger.warning("Keine XML-Dateien im gewünschten Zeitraum gefunden")
                return self.get_download_statistics()
            
            # 2. Downloads durchführen
            if use_parallel:
                self.download_files_parallel(links)
            else:
                self.download_files_sequential(links)
            
            # 3. Verifikation
            verification_stats = self.verify_downloads()
            
            # 4. Statistiken
            final_stats = self.get_download_statistics()
            final_stats.update(verification_stats)
            
            self.logger.info("=== XML-Scraping abgeschlossen ===")
            self.logger.info(f"Heruntergeladen: {final_stats['downloaded']}/{final_stats['total_files']}")
            self.logger.info(f"Fehler: {final_stats['errors']}")
            if 'duration_seconds' in final_stats:
                self.logger.info(f"Dauer: {final_stats['duration_seconds']:.2f} Sekunden")
            
            return final_stats
            
        except Exception as e:
            self.logger.error(f"Kritischer Fehler beim Scraping: {e}")
            raise


def main():
    """Hauptfunktion für Kommandozeilen-Nutzung"""
    scraper = XMLScraper(
        start_year=2012,
        end_year=2021,
        max_workers=4,
        delay_between_requests=1.0
    )
    
    try:
        results = scraper.run_scraping(use_parallel=True)
        print("\n=== Scraping-Ergebnisse ===")
        for key, value in results.items():
            print(f"{key}: {value}")
            
    except KeyboardInterrupt:
        print("\nScraping durch Benutzer abgebrochen")
    except Exception as e:
        print(f"\nFehler beim Scraping: {e}")


if __name__ == "__main__":
    main()

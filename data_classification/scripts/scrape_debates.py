#!/usr/bin/env python3
"""
Scraper für TheyWorkForYou XML-Dateien
Lädt pro Datum nur die Datei mit dem letzten Buchstaben im Alphabet herunter
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
from collections import defaultdict
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://www.theyworkforyou.com/pwdata/scrapedxml/debates/"
DATA_DIR = Path("../data/raw")
YEAR_START = 2012
YEAR_END = 2022
MAX_WORKERS = 10  # Anzahl paralleler Downloads


def get_file_list():
    """Holt die Liste aller verfügbaren XML-Dateien von der Website"""
    print(f"Lade Dateiliste von {BASE_URL}...")
    response = requests.get(BASE_URL)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    # Finde alle Links zu XML-Dateien
    xml_files = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith('.xml') and 'debates' in href:
            xml_files.append(href)

    return xml_files


def parse_filename(filename):
    """
    Extrahiert Datum und Suffix-Buchstaben aus dem Dateinamen
    Beispiel: debates2008-06-16c.xml -> ('2008-06-16', 'c')
    """
    match = re.match(r'debates(\d{4}-\d{2}-\d{2})([a-z])\.xml', filename)
    if match:
        date = match.group(1)
        letter = match.group(2)
        return date, letter
    return None, None


def filter_latest_per_date(file_list):
    """
    Filtert die Dateiliste, sodass pro Datum nur die Datei
    mit dem letzten Buchstaben im Alphabet zurückgegeben wird
    Berücksichtigt nur Jahre von YEAR_START bis YEAR_END (einschließlich)
    """
    # Gruppiere Dateien nach Datum
    files_by_date = defaultdict(list)

    for filename in file_list:
        date, letter = parse_filename(filename)
        if date and letter:
            # Prüfe Jahreszeitraum
            year = int(date.split('-')[0])
            if YEAR_START <= year <= YEAR_END:
                files_by_date[date].append((letter, filename))

    # Wähle pro Datum die Datei mit dem letzten Buchstaben
    filtered_files = []
    for date, files in files_by_date.items():
        # Sortiere nach Buchstabe und nimm den letzten
        files.sort(key=lambda x: x[0])
        last_file = files[-1][1]
        filtered_files.append(last_file)

    return sorted(filtered_files)


def download_file(filename, output_dir):
    """Lädt eine einzelne XML-Datei herunter"""
    url = BASE_URL + filename
    output_path = output_dir / filename

    # Überspringe, falls Datei bereits existiert
    if output_path.exists():
        return (filename, True, "bereits vorhanden")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        output_path.write_bytes(response.content)
        return (filename, True, "heruntergeladen")
    except Exception as e:
        return (filename, False, str(e))


def main():
    """Hauptfunktion des Scrapers"""
    print("TheyWorkForYou XML Debates Scraper")
    print(f"Zeitraum: {YEAR_START} - {YEAR_END} (einschließlich)")
    print("=" * 50)

    # Erstelle data-Verzeichnis falls nötig
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Ausgabeverzeichnis: {DATA_DIR.absolute()}\n")

    # Hole Dateiliste
    all_files = get_file_list()
    print(f"Gefunden: {len(all_files)} XML-Dateien\n")

    # Filtere: nur letzte Datei pro Datum
    files_to_download = filter_latest_per_date(all_files)
    print(f"Nach Filterung: {len(files_to_download)} Dateien zum Download\n")

    # Lade Dateien herunter (parallel)
    print(f"Starte Downloads ({MAX_WORKERS} parallele Threads)...\n")
    success_count = 0
    skipped_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Starte alle Downloads
        future_to_file = {
            executor.submit(download_file, filename, DATA_DIR): filename
            for filename in files_to_download
        }

        # Sammle Ergebnisse
        for i, future in enumerate(as_completed(future_to_file), 1):
            filename, success, message = future.result()

            if success:
                if message == "heruntergeladen":
                    success_count += 1
                    print(f"[{i}/{len(files_to_download)}] ✓ {filename}")
                else:
                    skipped_count += 1
                    print(f"[{i}/{len(files_to_download)}] ⊘ {filename} (bereits vorhanden)")
            else:
                error_count += 1
                print(f"[{i}/{len(files_to_download)}] ✗ {filename} - Fehler: {message}")

    # Zusammenfassung
    print("\n" + "=" * 50)
    print(f"Fertig!")
    print(f"  Heruntergeladen: {success_count}")
    print(f"  Übersprungen: {skipped_count}")
    print(f"  Fehler: {error_count}")
    print(f"  Gesamt: {len(files_to_download)}")
    print(f"Speicherort: {DATA_DIR.absolute()}")


if __name__ == "__main__":
    main()

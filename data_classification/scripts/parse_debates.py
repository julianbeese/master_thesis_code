#!/usr/bin/env python3
"""
Parser für TheyWorkForYou XML-Dateien
Liest XML-Dateien und speichert die Daten in einer DuckDB-Datenbank
"""

import duckdb
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import re


DATA_DIR = Path("../data/raw")
DB_FILE = "../data/processed/debates.duckdb"


def create_tables(conn):
    """Erstellt die Datenbanktabellen"""
    print("Erstelle Datenbanktabellen...")

    # Tabelle: debates
    conn.execute("""
        CREATE TABLE IF NOT EXISTS debates (
            debate_id VARCHAR PRIMARY KEY,
            date DATE NOT NULL,
            file_name VARCHAR NOT NULL,
            major_heading_text VARCHAR,
            colnum VARCHAR,
            time VARCHAR,
            url VARCHAR
        )
    """)

    # Tabelle: topics
    conn.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            topic_id VARCHAR PRIMARY KEY,
            debate_id VARCHAR NOT NULL,
            minor_heading_text VARCHAR,
            colnum VARCHAR,
            time VARCHAR,
            url VARCHAR,
            FOREIGN KEY (debate_id) REFERENCES debates(debate_id)
        )
    """)

    # Tabelle: speeches
    conn.execute("""
        CREATE TABLE IF NOT EXISTS speeches (
            speech_id VARCHAR PRIMARY KEY,
            topic_id VARCHAR,
            debate_id VARCHAR NOT NULL,
            speaker_name VARCHAR,
            person_id VARCHAR,
            speaker_office VARCHAR,
            speech_type VARCHAR,
            oral_qnum VARCHAR,
            colnum VARCHAR,
            time VARCHAR,
            url VARCHAR,
            speech_text TEXT NOT NULL,
            paragraph_count INTEGER,
            FOREIGN KEY (topic_id) REFERENCES topics(topic_id),
            FOREIGN KEY (debate_id) REFERENCES debates(debate_id)
        )
    """)

    print("  ✓ Tabellen erstellt")


def extract_date_from_filename(filename):
    """
    Extrahiert das Datum aus dem Dateinamen
    Beispiel: debates2012-01-10b.xml -> 2012-01-10
    """
    match = re.match(r'debates(\d{4}-\d{2}-\d{2})[a-z]\.xml', filename)
    if match:
        return match.group(1)
    return None


def get_element_text(element):
    """Extrahiert den kompletten Text aus einem XML-Element und seinen Kindelementen"""
    if element is None:
        return ""

    # Sammle Text aus allen <p> Tags
    paragraphs = []
    for p in element.findall('.//p'):
        # Hole den gesamten Text inkl. Text in Kindelementen
        text_parts = [p.text or '']
        for child in p:
            text_parts.append(child.text or '')
            text_parts.append(child.tail or '')
        text_parts.append(p.tail or '')

        paragraph_text = ''.join(text_parts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    return '\n\n'.join(paragraphs)


def parse_xml_file(file_path, conn):
    """Parst eine einzelne XML-Datei und fügt die Daten in die Datenbank ein"""
    file_name = file_path.name
    date_str = extract_date_from_filename(file_name)

    if not date_str:
        print(f"  ✗ Konnte Datum nicht aus {file_name} extrahieren")
        return 0, 0, 0

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"  ✗ Fehler beim Parsen von {file_name}: {e}")
        return 0, 0, 0

    # Tracking-Variablen
    # Fallback debate_id für Dateien ohne major-heading
    current_debate_id = f"{file_name}_default"
    current_topic_id = None

    debates_count = 0
    topics_count = 0
    speeches_count = 0

    for element in root:
        element_id = element.attrib.get('id', '')

        # MAJOR HEADING = Neue Debatte
        if element.tag == 'major-heading':
            current_debate_id = element_id
            current_topic_id = None

            major_heading_text = element.text.strip() if element.text else None

            conn.execute("""
                INSERT OR IGNORE INTO debates
                (debate_id, date, file_name, major_heading_text, colnum, time, url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                element_id,
                date_str,
                file_name,
                major_heading_text,
                element.attrib.get('colnum'),
                element.attrib.get('time'),
                element.attrib.get('url')
            ])
            debates_count += 1

        # MINOR HEADING = Neues Thema
        elif element.tag == 'minor-heading':
            current_topic_id = element_id

            minor_heading_text = element.text.strip() if element.text else None

            # Wenn es keine major-heading gab, erstelle eine Default-Debatte
            if current_debate_id == f"{file_name}_default":
                conn.execute("""
                    INSERT OR IGNORE INTO debates
                    (debate_id, date, file_name, major_heading_text, colnum, time, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [
                    current_debate_id,
                    date_str,
                    file_name,
                    None,
                    None,
                    None,
                    None
                ])

            conn.execute("""
                INSERT OR IGNORE INTO topics
                (topic_id, debate_id, minor_heading_text, colnum, time, url)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                element_id,
                current_debate_id,
                minor_heading_text,
                element.attrib.get('colnum'),
                element.attrib.get('time'),
                element.attrib.get('url')
            ])
            topics_count += 1

        # SPEECH = Redebeitrag
        elif element.tag == 'speech':
            # Extrahiere den kompletten Redetext
            speech_text = get_element_text(element)

            # Zähle Paragraphen
            paragraph_count = len(element.findall('.//p'))

            # Hole Attribute
            speaker_name = element.attrib.get('speakername')
            person_id = element.attrib.get('person_id') or element.attrib.get('speakerid')
            speaker_office = element.attrib.get('speakeroffice')
            speech_type = element.attrib.get('type')
            oral_qnum = element.attrib.get('oral-qnum')

            conn.execute("""
                INSERT OR IGNORE INTO speeches
                (speech_id, topic_id, debate_id, speaker_name, person_id,
                 speaker_office, speech_type, oral_qnum, colnum, time, url,
                 speech_text, paragraph_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                element_id,
                current_topic_id,
                current_debate_id,
                speaker_name,
                person_id,
                speaker_office,
                speech_type,
                oral_qnum,
                element.attrib.get('colnum'),
                element.attrib.get('time'),
                element.attrib.get('url'),
                speech_text,
                paragraph_count
            ])
            speeches_count += 1

    return debates_count, topics_count, speeches_count


def main():
    """Hauptfunktion"""
    print("TheyWorkForYou XML Parser → DuckDB")
    print("=" * 50)

    # Finde alle XML-Dateien
    xml_files = sorted(DATA_DIR.glob("*.xml"))

    if not xml_files:
        print(f"Keine XML-Dateien in {DATA_DIR} gefunden!")
        return

    print(f"Gefunden: {len(xml_files)} XML-Dateien\n")

    # Verbinde mit DuckDB
    print(f"Verbinde mit Datenbank: {DB_FILE}")
    conn = duckdb.connect(DB_FILE)

    # Erstelle Tabellen
    create_tables(conn)
    print()

    # Parse alle Dateien
    print("Starte Parsing...\n")

    total_debates = 0
    total_topics = 0
    total_speeches = 0

    for i, xml_file in enumerate(xml_files, 1):
        print(f"[{i}/{len(xml_files)}] {xml_file.name}")

        debates, topics, speeches = parse_xml_file(xml_file, conn)

        total_debates += debates
        total_topics += topics
        total_speeches += speeches

        print(f"  → {debates} Debatten, {topics} Themen, {speeches} Reden")

    # Commit und Schließen
    conn.commit()

    # Statistiken
    print("\n" + "=" * 50)
    print("Fertig!")
    print(f"\nEingefügt:")
    print(f"  Debatten: {total_debates}")
    print(f"  Themen:   {total_topics}")
    print(f"  Reden:    {total_speeches}")

    # Datenbank-Statistiken
    print(f"\nDatenbank-Inhalt:")
    debates_in_db = conn.execute("SELECT COUNT(*) FROM debates").fetchone()[0]
    topics_in_db = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    speeches_in_db = conn.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]

    print(f"  Debatten: {debates_in_db}")
    print(f"  Themen:   {topics_in_db}")
    print(f"  Reden:    {speeches_in_db}")

    # Beispiel-Abfragen
    print(f"\nBeispiel-Statistiken:")

    # Top 5 Redner
    top_speakers = conn.execute("""
        SELECT speaker_name, COUNT(*) as speech_count
        FROM speeches
        WHERE speaker_name IS NOT NULL
        GROUP BY speaker_name
        ORDER BY speech_count DESC
        LIMIT 5
    """).fetchall()

    print("\n  Top 5 Redner:")
    for name, count in top_speakers:
        print(f"    {name}: {count} Reden")

    # Datum-Range
    date_range = conn.execute("""
        SELECT MIN(date), MAX(date)
        FROM debates
    """).fetchone()

    print(f"\n  Zeitraum: {date_range[0]} bis {date_range[1]}")

    conn.close()
    print(f"\nDatenbank gespeichert: {DB_FILE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Debug-Skript für Parser-Logik
"""

import sys
from pathlib import Path
import xml.etree.ElementTree as ET

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from parser.xml_parser import ParliamentXMLParser

def debug_parser_logic():
    """Debuggt die Parser-Logik"""
    print("🔍 Debug Parser-Logik")
    print("=" * 50)
    
    xml_file = Path("scraper/data/raw_test/westminster2020-01-21a.xml")
    
    if not xml_file.exists():
        print("❌ XML-Datei nicht gefunden")
        return
    
    # Parser initialisieren
    parser = ParliamentXMLParser()
    
    # XML parsen
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    print("Teste _extract_debates...")
    debates = parser._extract_debates(root)
    
    print(f"Anzahl Debatten: {len(debates)}")
    
    for i, debate in enumerate(debates):
        print(f"Debatte {i+1}: '{debate.title}'")
        print(f"  - Reden: {len(debate.speeches)}")
        print(f"  - Wörter: {debate.total_words}")
        
        if debate.speeches:
            first_speech = debate.speeches[0]
            print(f"  - Erste Rede: {first_speech.speaker.name} ({first_speech.word_count} Wörter)")
    
    print()
    
    # Teste einzelne Rede-Extraktion
    print("Teste _extract_speech...")
    speeches = root.findall('speech')
    
    if speeches:
        first_speech_element = speeches[0]
        print(f"Erstes Speech-Element: {first_speech_element.get('speakername')}")
        
        speech = parser._extract_speech(first_speech_element)
        if speech:
            print(f"✅ Rede erfolgreich extrahiert:")
            print(f"  - Sprecher: {speech.speaker.name}")
            print(f"  - Wörter: {speech.word_count}")
            print(f"  - Absätze: {len(speech.paragraphs)}")
        else:
            print("❌ Rede-Extraktion fehlgeschlagen")


if __name__ == "__main__":
    debug_parser_logic()

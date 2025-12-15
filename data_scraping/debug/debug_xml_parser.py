#!/usr/bin/env python3
"""
Debug-Skript für XML-Parser
"""

import sys
from pathlib import Path
import xml.etree.ElementTree as ET

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

def debug_xml_structure():
    """Debuggt die XML-Struktur"""
    print("🔍 Debug XML-Struktur")
    print("=" * 50)
    
    xml_file = Path("scraper/data/raw_test/westminster2020-01-21a.xml")
    
    if not xml_file.exists():
        print("❌ XML-Datei nicht gefunden")
        return
    
    # XML parsen
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    print(f"Root-Tag: {root.tag}")
    print(f"Root-Attribute: {root.attrib}")
    print()
    
    # Alle direkten Kinder analysieren
    print("Direkte Kinder:")
    for i, child in enumerate(root):
        print(f"  {i}: {child.tag} - {child.attrib}")
        
        # Erste paar speech-Elemente genauer analysieren
        if child.tag == 'speech' and i < 3:
            print(f"    speakername: {child.get('speakername', 'NICHT GEFUNDEN')}")
            print(f"    person_id: {child.get('person_id', 'NICHT GEFUNDEN')}")
            
            # Absätze zählen
            paragraphs = child.findall('p')
            print(f"    Absätze: {len(paragraphs)}")
            
            if paragraphs:
                first_p = paragraphs[0]
                print(f"    Erster Absatz: '{first_p.text[:100] if first_p.text else 'LEER'}...'")
    
    print()
    
    # Statistiken
    speeches = root.findall('speech')
    headings = root.findall('minor-heading')
    
    print(f"Statistiken:")
    print(f"  Speech-Elemente: {len(speeches)}")
    print(f"  Minor-Heading-Elemente: {len(headings)}")
    
    # Sprecher-Namen sammeln
    speaker_names = []
    for speech in speeches:
        name = speech.get('speakername')
        if name:
            speaker_names.append(name)
    
    print(f"  Sprecher mit Namen: {len(speaker_names)}")
    print(f"  Eindeutige Sprecher: {len(set(speaker_names))}")
    
    if speaker_names:
        print(f"  Erste 5 Sprecher: {speaker_names[:5]}")


if __name__ == "__main__":
    debug_xml_structure()

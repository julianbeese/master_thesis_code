#!/usr/bin/env python3
"""
Test des XML-Parsers mit echten Parlamentsdaten
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from parser import ParliamentXMLParser
from parser.database_integration import ParliamentDatabaseIntegration
from parser.xml_parser import extract_text_for_ml


def test_xml_parser():
    """Testet den XML-Parser mit echten Daten"""
    print("🧪 Test des XML-Parsers")
    print("=" * 50)
    
    # Logging aktivieren
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 1. Parser initialisieren
        print("1. Initialisiere XML-Parser...")
        parser = ParliamentXMLParser()
        print("✅ Parser initialisiert")
        
        # 2. XML-Dateien finden
        print("\n2. Suche XML-Dateien...")
        xml_dir = Path("scraper/data/raw_test")
        xml_files = list(xml_dir.glob("*.xml"))
        
        if not xml_files:
            print("❌ Keine XML-Dateien gefunden")
            return False
        
        print(f"✅ {len(xml_files)} XML-Dateien gefunden:")
        for xml_file in xml_files:
            print(f"   - {xml_file.name}")
        
        # 3. Erste Datei parsen
        print(f"\n3. Parse erste Datei: {xml_files[0].name}")
        session = parser.parse_xml_file(xml_files[0])
        
        if not session:
            print("❌ Parsing fehlgeschlagen")
            return False
        
        print("✅ XML erfolgreich geparst!")
        print(f"   Session ID: {session.session_id}")
        print(f"   Datum: {session.date}")
        print(f"   Typ: {session.session_type}")
        print(f"   Debatten: {session.total_debates}")
        print(f"   Reden: {session.total_speeches}")
        print(f"   Wörter: {session.total_words}")
        print(f"   Sprecher: {session.unique_speakers}")
        print(f"   Dauer: {session.duration_hours:.2f} Stunden")
        
        # 4. Detaillierte Analyse
        print(f"\n4. Detaillierte Analyse...")
        
        if session.debates:
            first_debate = session.debates[0]
            print(f"   Erste Debatte: '{first_debate.title}'")
            print(f"   - Reden: {first_debate.total_speeches}")
            print(f"   - Wörter: {first_debate.total_words}")
            print(f"   - Sprecher: {first_debate.unique_speakers}")
            
            if first_debate.speeches:
                first_speech = first_debate.speeches[0]
                print(f"   Erste Rede von: {first_speech.speaker.name}")
                print(f"   - Wörter: {first_speech.word_count}")
                print(f"   - Zeichen: {first_speech.char_count}")
                print(f"   - Absätze: {len(first_speech.paragraphs)}")
                print(f"   - Intervention: {first_speech.is_intervention}")
                
                # Ersten Absatz anzeigen
                if first_speech.paragraphs:
                    first_paragraph = first_speech.paragraphs[0]
                    preview = first_paragraph.text[:200] + "..." if len(first_paragraph.text) > 200 else first_paragraph.text
                    print(f"   Erster Absatz: '{preview}'")
        
        # 5. ML-Daten extrahieren
        print(f"\n5. Extrahiere ML-Daten...")
        ml_data = extract_text_for_ml(session)
        print(f"✅ {len(ml_data)} ML-Datensätze extrahiert")
        
        if ml_data:
            first_ml = ml_data[0]
            print(f"   Erster ML-Datensatz:")
            print(f"   - Sprecher: {first_ml['speaker_name']}")
            print(f"   - Wörter: {first_ml['word_count']}")
            print(f"   - Text-Länge: {len(first_ml['text'])} Zeichen")
        
        # 6. Sprecher-Statistiken
        print(f"\n6. Sprecher-Statistiken...")
        speaker_stats = session.get_speaker_statistics()
        print(f"✅ Statistiken für {len(speaker_stats)} Sprecher")
        
        # Top 5 Sprecher
        top_speakers = sorted(speaker_stats.items(), key=lambda x: x[1]['words'], reverse=True)[:5]
        print("   Top 5 Sprecher (nach Wörtern):")
        for i, (speaker, stats) in enumerate(top_speakers, 1):
            print(f"   {i}. {speaker}: {stats['words']} Wörter, {stats['speeches']} Reden")
        
        print(f"\n🎉 XML-Parser-Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"\n❌ Fehler beim XML-Parser-Test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_integration():
    """Testet die Datenbank-Integration"""
    print("\n🗄️ Test der Datenbank-Integration")
    print("=" * 50)
    
    try:
        with ParliamentDatabaseIntegration() as integration:
            print("1. Erstelle Datenbank-Tabellen...")
            integration.create_parliament_tables()
            print("✅ Tabellen erstellt")
            
            # XML-Datei parsen
            xml_dir = Path("scraper/data/raw_test")
            xml_files = list(xml_dir.glob("*.xml"))
            
            if not xml_files:
                print("❌ Keine XML-Dateien für Test gefunden")
                return False
            
            print(f"\n2. Parse und speichere: {xml_files[0].name}")
            parser = ParliamentXMLParser()
            session = parser.parse_xml_file(xml_files[0])
            
            if not session:
                print("❌ XML-Parsing fehlgeschlagen")
                return False
            
            # In Datenbank speichern
            result = integration.store_parliament_session(session)
            
            if result['success']:
                print("✅ Session erfolgreich gespeichert!")
                print(f"   - Sprecher: {result['speakers']}")
                print(f"   - Debatten: {result['debates']}")
                print(f"   - Reden: {result['speeches']}")
                print(f"   - Absätze: {result['paragraphs']}")
            else:
                print(f"❌ Speichern fehlgeschlagen: {result.get('error', 'Unbekannter Fehler')}")
                return False
            
            # Statistiken abrufen
            print(f"\n3. Hole Datenbank-Statistiken...")
            stats = integration.get_parliament_statistics()
            
            print("✅ Statistiken abgerufen:")
            print(f"   - Sessions: {stats.get('total_sessions', 0)}")
            print(f"   - Debatten: {stats.get('total_debates', 0)}")
            print(f"   - Reden: {stats.get('total_speeches', 0)}")
            print(f"   - Sprecher: {stats.get('total_speakers', 0)}")
            print(f"   - Absätze: {stats.get('total_paragraphs', 0)}")
            print(f"   - Wörter: {stats.get('total_words', 0)}")
            
            # ML-Daten exportieren
            print(f"\n4. Exportiere ML-Daten...")
            export_result = integration.export_ml_data("parquet")
            
            if export_result['success']:
                print("✅ ML-Daten exportiert!")
                print(f"   - Datei: {export_result['output_file']}")
                print(f"   - Zeilen: {export_result['row_count']}")
            else:
                print(f"❌ Export fehlgeschlagen: {export_result.get('error', 'Unbekannter Fehler')}")
                return False
            
            print(f"\n🎉 Datenbank-Integration-Test erfolgreich!")
            return True
            
    except Exception as e:
        print(f"❌ Fehler bei Datenbank-Integration: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Hauptfunktion"""
    print("🧪 Parliament Data Processing System - XML-Parser Test")
    print("=" * 60)
    print(f"Startzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Führe Tests durch
    parser_success = test_xml_parser()
    db_success = test_database_integration()
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("📊 TEST-ZUSAMMENFASSUNG")
    print("=" * 60)
    print(f"XML-Parser-Test: {'✅ Erfolgreich' if parser_success else '❌ Fehlgeschlagen'}")
    print(f"Datenbank-Integration: {'✅ Erfolgreich' if db_success else '❌ Fehlgeschlagen'}")
    print(f"Gesamt: {'🎉 Alle Tests erfolgreich!' if parser_success and db_success else '❌ Einige Tests fehlgeschlagen'}")
    print(f"Endzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if parser_success and db_success:
        print("\n🚀 Das System ist bereit für ML-Modelle!")
        print("   - XML-Daten werden erfolgreich geparst")
        print("   - Strukturierte Daten werden in der Datenbank gespeichert")
        print("   - ML-Daten können exportiert werden")
    else:
        print("\n⚠️  Bitte beheben Sie die Fehler vor der Nutzung.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test-Skript für .env Konfiguration
Prüft ob alle notwendigen Umgebungsvariablen korrekt gesetzt sind
"""

import os
import sys
from pathlib import Path

# Lade .env Datei falls vorhanden
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env Datei erfolgreich geladen")
except ImportError:
    print("⚠️  python-dotenv nicht installiert - .env Datei wird nicht automatisch geladen")
    print("   Installiere mit: pip install python-dotenv")
except Exception as e:
    print(f"❌ Fehler beim Laden der .env Datei: {e}")

def test_environment():
    """Testet die Umgebungsvariablen"""
    print("\n🔍 Teste Umgebungsvariablen...")
    
    # Teste Google API Key
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        if len(api_key) > 20:  # Google API Keys sind normalerweise länger
            print("✅ GOOGLE_API_KEY ist gesetzt und sieht gültig aus")
        else:
            print("⚠️  GOOGLE_API_KEY ist gesetzt, aber sieht zu kurz aus")
    else:
        print("❌ GOOGLE_API_KEY ist nicht gesetzt")
        print("   Setze es in der .env Datei oder als Umgebungsvariable")
    
    # Teste optionale Variablen
    optional_vars = {
        'BREXIT_DB_PATH': 'database/data/brexit_data.db',
        'SOURCE_DB_PATH': 'database/data/parliament_data.db',
        'LOG_LEVEL': 'INFO',
        'GEMINI_MODEL': 'gemini-1.5-flash'
    }
    
    print("\n📋 Optionale Konfigurationen:")
    for var, default in optional_vars.items():
        value = os.getenv(var, default)
        if value == default:
            print(f"  {var}: {value} (Standard)")
        else:
            print(f"  {var}: {value} (benutzerdefiniert)")

def test_imports():
    """Testet ob alle notwendigen Module importiert werden können"""
    print("\n📦 Teste Python-Module...")
    
    modules_to_test = [
        ('google.generativeai', 'Google Generative AI'),
        ('duckdb', 'DuckDB'),
        ('polars', 'Polars'),
        ('pandas', 'Pandas'),
        ('lxml', 'lxml'),
        ('tqdm', 'tqdm')
    ]
    
    for module, name in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {name} ist verfügbar")
        except ImportError:
            print(f"❌ {name} ist nicht installiert")
            print(f"   Installiere mit: pip install {module}")

def test_project_structure():
    """Testet die Projektstruktur"""
    print("\n📁 Teste Projektstruktur...")
    
    required_files = [
        'brexit_classifier.py',
        'brexit_database.py', 
        'brexit_pipeline.py',
        'run_brexit_filtering.py',
        'requirements.txt'
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} gefunden")
        else:
            print(f"❌ {file} nicht gefunden")
    
    # Teste Verzeichnisse
    required_dirs = [
        'database',
        'parser',
        'scraper'
    ]
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/ Verzeichnis gefunden")
        else:
            print(f"❌ {dir_name}/ Verzeichnis nicht gefunden")

def main():
    """Hauptfunktion"""
    print("🧪 Brexit-System Konfigurationstest")
    print("=" * 50)
    
    test_environment()
    test_imports()
    test_project_structure()
    
    print("\n" + "=" * 50)
    print("🎯 Nächste Schritte:")
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("1. Erstelle eine .env Datei mit deinem GOOGLE_API_KEY")
        print("2. Oder setze die Umgebungsvariable: export GOOGLE_API_KEY='dein-key'")
    else:
        print("1. ✅ API Key ist konfiguriert")
    
    print("2. Teste das System: python run_brexit_filtering.py --test")
    print("3. Verarbeite XML-Dateien: python run_brexit_filtering.py --xml-dir scraper/data/raw --max-files 5")

if __name__ == "__main__":
    main()

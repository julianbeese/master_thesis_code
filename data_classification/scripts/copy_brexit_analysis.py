#!/usr/bin/env python3
"""
Einfaches Skript zum Kopieren der brexit_analysis.duckdb Datenbank
und Erstellen einer neuen Datenbank unter brexit_llm_filtered.duckdb
"""

import os
import shutil
import sys
from pathlib import Path

def copy_brexit_analysis_db():
    """
    Kopiert die brexit_analysis.duckdb Datenbank zu brexit_llm_filtered.duckdb
    """
    # Pfade definieren
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / "processed"
    
    source_db = data_dir / "brexit_analysis.duckdb"
    target_db = data_dir / "brexit_llm_filtered.duckdb"
    
    # Prüfen ob Quelldatenbank existiert
    if not source_db.exists():
        print(f"Fehler: Quelldatenbank {source_db} existiert nicht!")
        return False
    
    # Prüfen ob Zieldatenbank bereits existiert
    if target_db.exists():
        print(f"Warnung: Zieldatenbank {target_db} existiert bereits!")
        response = input("Möchten Sie sie überschreiben? (y/N): ")
        if response.lower() != 'y':
            print("Kopieren abgebrochen.")
            return False
        else:
            # Alte Zieldatenbank löschen
            target_db.unlink()
            print(f"Alte Zieldatenbank {target_db} wurde gelöscht.")
    
    try:
        # Datenbank kopieren
        print(f"Kopiere {source_db} nach {target_db}...")
        shutil.copy2(source_db, target_db)
        print(f"Erfolgreich kopiert: {target_db}")
        
        # Prüfen ob Kopie erfolgreich war
        if target_db.exists():
            source_size = source_db.stat().st_size
            target_size = target_db.stat().st_size
            print(f"Quelldatei Größe: {source_size:,} Bytes")
            print(f"Zieldatei Größe: {target_size:,} Bytes")
            
            if source_size == target_size:
                print("✅ Kopieren erfolgreich abgeschlossen!")
                return True
            else:
                print("⚠️  Warnung: Dateigrößen stimmen nicht überein!")
                return False
        else:
            print("❌ Fehler: Zieldatei wurde nicht erstellt!")
            return False
            
    except Exception as e:
        print(f"❌ Fehler beim Kopieren: {e}")
        return False

def main():
    """Hauptfunktion"""
    print("=== Brexit Analysis Datenbank Kopierer ===")
    print()
    
    success = copy_brexit_analysis_db()
    
    if success:
        print("\n🎉 Skript erfolgreich abgeschlossen!")
        sys.exit(0)
    else:
        print("\n💥 Skript fehlgeschlagen!")
        sys.exit(1)

if __name__ == "__main__":
    main()

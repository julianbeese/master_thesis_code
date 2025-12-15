#!/usr/bin/env python3
"""
Script zum Leeren aller LLM-Spalten in der brexit_analysis.duckdb
Ermöglicht einen Neustart der LLM-Klassifizierung
Löscht auch die is_brexit_relevant Spalte falls vorhanden
"""

import duckdb
import sys
from pathlib import Path
from datetime import datetime

# Konfiguration
DB_PATH = "/Users/julianbeese/Developer/Master/data_collection/data/processed/brexit_analysis.duckdb"

# Alle möglichen LLM-Spalten (basierend auf der Codebase-Analyse)
LLM_COLUMNS = [
    # Hauptklassifizierung
    "llm_processed",
    "llm_classified_brexit", 
    "llm_confidence_score",
    "llm_reasoning",
    "llm_key_indicators",
    
    # Provider-Informationen
    "llm_provider",
    "llm_model",
    
    # Kosten und Performance
    "llm_cost_usd",
    "llm_processing_time",
    "llm_input_tokens",
    "llm_output_tokens",
    
    # Fehlerbehandlung
    "llm_error",
    "llm_processed_at",
    
    # Legacy-Spalten (falls vorhanden)
    "brexit_llm_confidence",
    "brexit_llm_reasoning",
    "brexit_confidence",
    "brexit_related",
    
    # Spalte die gelöscht werden soll
    "is_brexit_relevant"
]

def check_database_exists():
    """Prüft ob die Datenbank existiert"""
    if not Path(DB_PATH).exists():
        print(f"❌ Datenbank nicht gefunden: {DB_PATH}")
        return False
    print(f"✅ Datenbank gefunden: {DB_PATH}")
    return True

def get_existing_columns(conn):
    """Holt alle existierenden Spalten der speeches-Tabelle"""
    try:
        columns_info = conn.execute("PRAGMA table_info('speeches')").fetchall()
        existing_columns = [col[1] for col in columns_info]
        return existing_columns
    except Exception as e:
        print(f"❌ Fehler beim Abrufen der Spalten: {e}")
        return []

def drop_column_if_exists(conn, column_name):
    """Löscht eine Spalte falls sie existiert"""
    try:
        existing_columns = get_existing_columns(conn)
        if column_name in existing_columns:
            print(f"🗑️  Lösche Spalte: {column_name}")
            conn.execute(f"ALTER TABLE speeches DROP COLUMN {column_name}")
            conn.commit()
            print(f"✅ Spalte {column_name} erfolgreich gelöscht")
            return True
        else:
            print(f"ℹ️  Spalte {column_name} existiert nicht - nichts zu löschen")
            return True
    except Exception as e:
        print(f"❌ Fehler beim Löschen der Spalte {column_name}: {e}")
        return False

def clear_llm_columns(conn):
    """Leert alle LLM-Spalten"""
    print("\n🔍 Prüfe existierende LLM-Spalten...")
    
    existing_columns = get_existing_columns(conn)
    if not existing_columns:
        print("❌ Keine Spalten gefunden!")
        return False
    
    # Zuerst: Lösche die is_brexit_relevant Spalte falls sie existiert
    if "is_brexit_relevant" in existing_columns:
        if not drop_column_if_exists(conn, "is_brexit_relevant"):
            return False
    
    # Finde andere LLM-Spalten die tatsächlich existieren (ohne is_brexit_relevant)
    found_llm_columns = []
    for col in LLM_COLUMNS:
        if col in existing_columns and col != "is_brexit_relevant":
            found_llm_columns.append(col)
            print(f"  ✓ Gefunden: {col}")
    
    if not found_llm_columns:
        print("ℹ️  Keine weiteren LLM-Spalten gefunden - nichts zu leeren")
        return True
    
    print(f"\n🧹 Leere {len(found_llm_columns)} LLM-Spalten...")
    
    try:
        # Erstelle UPDATE-Statement für alle LLM-Spalten
        update_parts = []
        for col in found_llm_columns:
            # Bestimme den Standardwert basierend auf dem Spaltentyp
            if col in ["llm_processed", "llm_classified_brexit", "brexit_related"]:
                update_parts.append(f"{col} = FALSE")
            elif col in ["llm_confidence_score", "llm_cost_usd", "llm_processing_time", 
                        "brexit_llm_confidence", "brexit_confidence"]:
                update_parts.append(f"{col} = NULL")
            elif col in ["llm_input_tokens", "llm_output_tokens"]:
                update_parts.append(f"{col} = NULL")
            else:
                update_parts.append(f"{col} = NULL")
        
        if update_parts:
            update_sql = f"UPDATE speeches SET {', '.join(update_parts)}"
            print(f"📝 Führe aus: {update_sql}")
            
            result = conn.execute(update_sql)
            affected_rows = result.rowcount if hasattr(result, 'rowcount') else "unbekannt"
            print(f"✅ {affected_rows} Zeilen aktualisiert")
            
            # Committe die Änderungen
            conn.commit()
            print("✅ Änderungen gespeichert")
            
    except Exception as e:
        print(f"❌ Fehler beim Leeren der Spalten: {e}")
        return False
    
    return True

def show_statistics(conn):
    """Zeigt Statistiken nach dem Leeren"""
    print("\n📊 Statistiken nach dem Leeren:")
    
    try:
        # Zähle leere vs. gefüllte LLM-Spalten
        total_speeches = conn.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]
        print(f"  Gesamt Speeches: {total_speeches:,}")
        
        # Prüfe leere Spalten
        for col in ["llm_processed", "llm_classified_brexit", "brexit_related"]:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM speeches WHERE {col} = TRUE").fetchone()[0]
                print(f"  {col} = TRUE: {count:,}")
            except:
                pass
        
        # Prüfe NULL-Werte in numerischen Spalten
        for col in ["llm_confidence_score", "brexit_confidence"]:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM speeches WHERE {col} IS NOT NULL").fetchone()[0]
                print(f"  {col} (nicht NULL): {count:,}")
            except:
                pass
                
    except Exception as e:
        print(f"⚠️  Fehler beim Anzeigen der Statistiken: {e}")

def main():
    """Hauptfunktion"""
    print("=" * 80)
    print("🧹 LLM-SPALTEN LEEREN & LÖSCHEN SCRIPT")
    print("=" * 80)
    print(f"Ziel: {DB_PATH}")
    print(f"Zeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Prüfe Datenbank
    if not check_database_exists():
        sys.exit(1)
    
    # Verbinde zur Datenbank
    try:
        conn = duckdb.connect(DB_PATH)
        print("✅ Datenbankverbindung erfolgreich")
    except Exception as e:
        print(f"❌ Fehler beim Verbinden zur Datenbank: {e}")
        print("\n💡 MÖGLICHE LÖSUNGEN:")
        print("   1. Schließen Sie alle anderen Programme, die die Datenbank verwenden")
        print("   2. Schließen Sie Cursor und öffnen Sie es erneut")
        print("   3. Warten Sie einen Moment und versuchen Sie es erneut")
        print("   4. Prüfen Sie, ob andere Python-Scripts laufen")
        sys.exit(1)
    
    try:
        # Leere LLM-Spalten
        if clear_llm_columns(conn):
            print("\n✅ LLM-Spalten erfolgreich geleert!")
            
            # Zeige Statistiken
            show_statistics(conn)
            
            print("\n🎉 Fertig! Sie können jetzt mit der LLM-Klassifizierung neu starten.")
            print("💡 Tipp: Führen Sie Ihr Klassifizierungs-Script erneut aus.")
            
        else:
            print("\n❌ Fehler beim Leeren der LLM-Spalten!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        sys.exit(1)
        
    finally:
        conn.close()
        print("🔌 Datenbankverbindung geschlossen")

if __name__ == "__main__":
    main()

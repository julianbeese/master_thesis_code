#!/usr/bin/env python3
"""
Skript zur Aggregation von Frames aus Chunks in die Speeches- und Debates-Tabellen.

Dieses Skript:
1. Erstellt eine neue Spalte 'aggregated_frames' in der speeches-Tabelle
2. Aggregiert die frame_label Werte aus den zugehörigen Chunks
3. Speichert die Frames kommagetrennt, wenn es > 1 Chunk pro Speech gibt
4. Erstellt eine neue Spalte 'aggregated_frames' in der debates-Tabelle
5. Aggregiert die aggregated_frames Werte aus den zugehörigen Speeches
6. Speichert die Frames kommagetrennt, wenn es > 1 Speech pro Debatte gibt
"""

import duckdb
import os
from pathlib import Path

# Datenbankpfad
DB_PATH = "thesis_final.duckdb"

def main():
    """Hauptfunktion zur Aggregation der Frames."""
    
    # Prüfe ob Datenbank existiert
    if not os.path.exists(DB_PATH):
        print(f"Fehler: Datenbank {DB_PATH} nicht gefunden!")
        return
    
    print(f"Verbinde mit Datenbank: {DB_PATH}")
    conn = duckdb.connect(DB_PATH)
    
    try:
        # Prüfe ob Spalte bereits existiert
        print("\nPrüfe ob Spalte 'aggregated_frames' bereits existiert...")
        result = conn.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'speeches' AND column_name = 'aggregated_frames'
        """).fetchall()
        
        if result:
            print("Spalte 'aggregated_frames' existiert bereits. Überspringe Erstellung.")
        else:
            print("Erstelle neue Spalte 'aggregated_frames' in speeches-Tabelle...")
            conn.execute("""
                ALTER TABLE speeches 
                ADD COLUMN aggregated_frames VARCHAR
            """)
            print("✓ Spalte erfolgreich erstellt.")
        
        # Aggregiere Frames aus Chunks
        print("\nAggregiere Frames aus Chunks...")
        
        # SQL-Query zur Aggregation:
        # - Gruppiere nach speech_id
        # - Zähle Anzahl Chunks
        # - Sammle eindeutige frame_label Werte (ohne NULL)
        # - Wenn > 1 Chunk: Kommagetrennte Liste
        # - Wenn 1 Chunk: Einzelner Frame-Wert
        # - Wenn 0 Chunks oder alle NULL: NULL
        update_query = """
        UPDATE speeches
        SET aggregated_frames = frame_agg.aggregated
        FROM (
            SELECT 
                c.speech_id,
                CASE 
                    WHEN COUNT(*) > 1 THEN 
                        -- Mehrere Chunks: Kommagetrennte Liste eindeutiger Frames
                        STRING_AGG(DISTINCT c.frame_label, ', ' ORDER BY c.frame_label)
                    WHEN COUNT(*) = 1 THEN 
                        -- Ein Chunk: Einzelner Frame-Wert
                        MAX(c.frame_label)
                    ELSE 
                        -- Keine Chunks: NULL
                        NULL
                END AS aggregated
            FROM chunks c
            WHERE c.frame_label IS NOT NULL
            GROUP BY c.speech_id
        ) AS frame_agg
        WHERE speeches.speech_id = frame_agg.speech_id
        """
        
        conn.execute(update_query)
        
        # Zähle aktualisierte Zeilen
        updated_rows = conn.execute("""
            SELECT COUNT(*) 
            FROM speeches 
            WHERE aggregated_frames IS NOT NULL
        """).fetchone()[0]
        print(f"✓ {updated_rows} Speeches mit aggregierten Frames aktualisiert.")
        
        # Statistiken ausgeben
        print("\nStatistiken:")
        stats = conn.execute("""
            SELECT 
                COUNT(*) as total_speeches,
                COUNT(aggregated_frames) as speeches_with_frames,
                COUNT(*) - COUNT(aggregated_frames) as speeches_without_frames
            FROM speeches
        """).fetchone()
        
        print(f"  Gesamtanzahl Speeches: {stats[0]}")
        print(f"  Speeches mit aggregierten Frames: {stats[1]}")
        print(f"  Speeches ohne Frames: {stats[2]}")
        
        # Beispiel-Ergebnisse anzeigen
        print("\nBeispiel-Ergebnisse (erste 10 Einträge mit Frames):")
        examples = conn.execute("""
            SELECT 
                speech_id,
                aggregated_frames,
                (SELECT COUNT(*) FROM chunks WHERE chunks.speech_id = speeches.speech_id) as chunk_count
            FROM speeches
            WHERE aggregated_frames IS NOT NULL
            LIMIT 10
        """).fetchall()
        
        for speech_id, frames, chunk_count in examples:
            print(f"  {speech_id[:50]}... | Chunks: {chunk_count} | Frames: {frames}")
        
        # ========================================================================
        # AGGREGATION FÜR DEBATES
        # ========================================================================
        print("\n" + "="*80)
        print("AGGREGATION FÜR DEBATES")
        print("="*80)
        
        # Prüfe ob Spalte bereits existiert
        print("\nPrüfe ob Spalte 'aggregated_frames' in debates bereits existiert...")
        result = conn.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'debates' AND column_name = 'aggregated_frames'
        """).fetchall()
        
        if result:
            print("Spalte 'aggregated_frames' existiert bereits. Überspringe Erstellung.")
        else:
            print("Erstelle neue Spalte 'aggregated_frames' in debates-Tabelle...")
            conn.execute("""
                ALTER TABLE debates 
                ADD COLUMN aggregated_frames VARCHAR
            """)
            print("✓ Spalte erfolgreich erstellt.")
        
        # Aggregiere Frames aus Speeches
        print("\nAggregiere Frames aus Speeches...")
        
        # SQL-Query zur Aggregation für Debates:
        # - Gruppiere nach debate_id
        # - Sammle alle aggregated_frames aus den zugehörigen Speeches
        # - Splitte kommagetrennte Werte auf, sammle eindeutige Frames
        # - Wenn > 1 Speech: Kommagetrennte Liste
        # - Wenn 1 Speech: Einzelner Frame-Wert (oder kommagetrennt wenn mehrere Frames)
        # - Wenn 0 Speeches oder alle NULL: NULL
        
        # DuckDB: string_split gibt ein Array zurück, unnest extrahiert die Werte
        update_debates_query = """
        UPDATE debates
        SET aggregated_frames = frame_agg.aggregated
        FROM (
            SELECT 
                s.debate_id,
                CASE 
                    WHEN COUNT(DISTINCT s.speech_id) > 1 THEN 
                        -- Mehrere Speeches: Kommagetrennte Liste eindeutiger Frames
                        -- Wir müssen die kommagetrennten Werte aus speeches.aggregated_frames aufsplitten
                        STRING_AGG(DISTINCT TRIM(frame_value), ', ' ORDER BY TRIM(frame_value))
                    WHEN COUNT(DISTINCT s.speech_id) = 1 THEN 
                        -- Ein Speech: Verwende den Wert direkt (kann bereits kommagetrennt sein)
                        MAX(s.aggregated_frames)
                    ELSE 
                        -- Keine Speeches: NULL
                        NULL
                END AS aggregated
            FROM speeches s
            CROSS JOIN unnest(string_split(s.aggregated_frames, ',')) AS t(frame_value)
            WHERE s.aggregated_frames IS NOT NULL
            GROUP BY s.debate_id
        ) AS frame_agg
        WHERE debates.debate_id = frame_agg.debate_id
        """
        
        conn.execute(update_debates_query)
        
        # Zähle aktualisierte Zeilen
        updated_debates = conn.execute("""
            SELECT COUNT(*) 
            FROM debates 
            WHERE aggregated_frames IS NOT NULL
        """).fetchone()[0]
        print(f"✓ {updated_debates} Debates mit aggregierten Frames aktualisiert.")
        
        # Statistiken ausgeben
        print("\nStatistiken für Debates:")
        debate_stats = conn.execute("""
            SELECT 
                COUNT(*) as total_debates,
                COUNT(aggregated_frames) as debates_with_frames,
                COUNT(*) - COUNT(aggregated_frames) as debates_without_frames
            FROM debates
        """).fetchone()
        
        print(f"  Gesamtanzahl Debates: {debate_stats[0]}")
        print(f"  Debates mit aggregierten Frames: {debate_stats[1]}")
        print(f"  Debates ohne Frames: {debate_stats[2]}")
        
        # Beispiel-Ergebnisse anzeigen
        print("\nBeispiel-Ergebnisse für Debates (erste 10 Einträge mit Frames):")
        debate_examples = conn.execute("""
            SELECT 
                d.debate_id,
                d.aggregated_frames,
                (SELECT COUNT(*) FROM speeches WHERE speeches.debate_id = d.debate_id) as speech_count
            FROM debates d
            WHERE d.aggregated_frames IS NOT NULL
            LIMIT 10
        """).fetchall()
        
        for debate_id, frames, speech_count in debate_examples:
            print(f"  {debate_id[:50]}... | Speeches: {speech_count} | Frames: {frames}")
        
        print("\n✓ Aggregation für Debates erfolgreich abgeschlossen!")
        print("\n" + "="*80)
        print("✓ Gesamte Aggregation erfolgreich abgeschlossen!")
        
    except Exception as e:
        print(f"\n✗ Fehler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
        print("\nDatenbankverbindung geschlossen.")

if __name__ == "__main__":
    main()


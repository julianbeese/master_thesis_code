#!/usr/bin/env python3
"""
SYNC RAILWAY ZU LOKALER DUCKDB
Synchronisiert annotierte Daten von Railway zurück in lokale DuckDB
"""

import duckdb
import psycopg2
from psycopg2.extras import RealDictCursor
import argparse
from pathlib import Path
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

class RailwayToLocalSync:
    def __init__(self, local_duckdb_path: str, postgres_url: str):
        self.local_duckdb_path = local_duckdb_path
        self.postgres_url = postgres_url
    
    def ensure_columns_exist(self, duck_conn):
        """Stellt sicher, dass alle benötigten Spalten in der DuckDB existieren"""
        print("🔧 Prüfe Spalten in lokaler DuckDB...")
        
        try:
            # Prüfe ob pre_brexit Spalte existiert
            try:
                duck_conn.execute("SELECT pre_brexit FROM chunks LIMIT 1")
            except:
                print("➕ Füge pre_brexit Spalte hinzu...")
                duck_conn.execute("ALTER TABLE chunks ADD COLUMN pre_brexit BOOLEAN")
                # Setze pre_brexit basierend auf debate_date
                duck_conn.execute("""
                    UPDATE chunks 
                    SET pre_brexit = (debate_date < '2016-06-23'::date)
                    WHERE pre_brexit IS NULL
                """)
                print("✅ pre_brexit Spalte hinzugefügt und gefüllt")
            
            # Prüfe ob brexit_position Spalte existiert
            try:
                duck_conn.execute("SELECT brexit_position FROM chunks LIMIT 1")
            except:
                print("➕ Füge brexit_position Spalte hinzu...")
                duck_conn.execute("ALTER TABLE chunks ADD COLUMN brexit_position VARCHAR(100)")
                print("✅ brexit_position Spalte hinzugefügt")
                
        except Exception as e:
            print(f"⚠️ Fehler beim Hinzufügen der Spalten: {e}")
            # Versuche trotzdem fortzufahren
        
    def sync_annotations_from_railway(self):
        """Synchronisiert Annotationen von Railway zurück in lokale DuckDB"""
        print("🔄 SYNC RAILWAY ZU LOKALER DUCKDB")
        print("=" * 50)
        
        # Prüfe ob finale DB existiert
        final_db_path = self.local_duckdb_path.replace("debates_brexit_chunked", "debates_brexit_chunked_synced")
        
        if not Path(final_db_path).exists():
            print(f"📋 Kopiere {self.local_duckdb_path} → {final_db_path}")
            shutil.copy2(self.local_duckdb_path, final_db_path)
            print("✅ Datenbank kopiert")
        else:
            print(f"✅ Finale Datenbank existiert bereits: {final_db_path}")
        
        # PostgreSQL Verbindung zu Railway
        pg_conn = psycopg2.connect(self.postgres_url)
        pg_cursor = pg_conn.cursor(cursor_factory=RealDictCursor)
        
        # DuckDB Verbindung zur finalen DB
        duck_conn = duckdb.connect(final_db_path)
        
        # Prüfe und füge neue Spalten hinzu falls nötig
        self.ensure_columns_exist(duck_conn)
        
        try:
            # Hole annotierte Chunks von Railway
            print("📥 Lade annotierte Chunks von Railway...")
            pg_cursor.execute("""
                SELECT 
                    chunk_id, speech_id, debate_id, speaker_name, speaker_party,
                    debate_title, debate_date, chunk_text, chunk_index, total_chunks,
                    word_count, char_count, chunking_method, assigned_user,
                    frame_label, annotation_confidence, annotation_notes,
                    pre_brexit, brexit_position,
                    created_at, updated_at
                FROM chunks 
                WHERE assigned_user IS NOT NULL
                ORDER BY chunk_id
            """)
            
            railway_chunks = pg_cursor.fetchall()
            print(f"✅ {len(railway_chunks):,} annotierte Chunks von Railway geladen")
            
            # Hole Agreement-Daten von Railway
            try:
                pg_cursor.execute("""
                    SELECT 
                        chunk_id, annotator1, annotator2, label1, label2,
                        agreement_score, agreement_perfect, created_at, updated_at
                    FROM agreement_chunks
                    ORDER BY chunk_id
                """)
                railway_agreements = pg_cursor.fetchall()
                print(f"✅ {len(railway_agreements):,} Agreement-Daten von Railway geladen")
            except:
                railway_agreements = []
                print("⚠️ Keine Agreement-Daten auf Railway gefunden")
            
            # Aktualisiere lokale Chunks mit Railway-Daten
            print("🔄 Aktualisiere lokale Chunks...")
            updated_chunks = 0
            
            for chunk in railway_chunks:
                # Aktualisiere Chunk mit Railway-Daten
                duck_conn.execute("""
                    UPDATE chunks SET
                        assigned_user = ?,
                        frame_label = ?,
                        annotation_confidence = ?,
                        annotation_notes = ?,
                        pre_brexit = ?,
                        brexit_position = ?,
                        updated_at = ?
                    WHERE chunk_id = ?
                """, (
                    chunk['assigned_user'],
                    chunk['frame_label'],
                    chunk['annotation_confidence'],
                    chunk['annotation_notes'],
                    chunk['pre_brexit'],
                    chunk['brexit_position'],
                    chunk['updated_at'],
                    chunk['chunk_id']
                ))
                updated_chunks += 1
            
            print(f"✅ {updated_chunks:,} Chunks aktualisiert")
            
            # Synchronisiere Agreement-Daten
            if railway_agreements:
                print("🔄 Synchronisiere Agreement-Daten...")
                
                # Lösche alte Agreement-Daten
                duck_conn.execute("DELETE FROM agreement_chunks")
                
                # Füge Railway Agreement-Daten ein
                for agreement in railway_agreements:
                    duck_conn.execute("""
                        INSERT INTO agreement_chunks (
                            chunk_id, annotator1, annotator2, label1, label2,
                            agreement_score, agreement_perfect, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        agreement['chunk_id'],
                        agreement['annotator1'],
                        agreement['annotator2'],
                        agreement['label1'],
                        agreement['label2'],
                        agreement['agreement_score'],
                        agreement['agreement_perfect'],
                        agreement['created_at'],
                        agreement['updated_at']
                    ))
                
                print(f"✅ {len(railway_agreements):,} Agreement-Daten synchronisiert")
            
            # Statistiken
            self.print_sync_statistics(duck_conn)
            
            print("✅ Synchronisation erfolgreich abgeschlossen!")
            print(f"📁 Finale Datenbank: {final_db_path}")
            
        except Exception as e:
            print(f"❌ Fehler bei Synchronisation: {e}")
            raise
        finally:
            pg_cursor.close()
            pg_conn.close()
            duck_conn.close()
    
    def print_sync_statistics(self, duck_conn):
        """Zeigt Synchronisations-Statistiken"""
        print("\n📊 SYNCHRONISATIONS-STATISTIKEN")
        print("=" * 40)
        
        # Gesamt-Chunks
        total_chunks = duck_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        
        # Annotierte Chunks
        annotated_chunks = duck_conn.execute("""
            SELECT COUNT(*) FROM chunks 
            WHERE assigned_user IS NOT NULL
        """).fetchone()[0]
        
        # Chunks mit Frame-Labels
        labeled_chunks = duck_conn.execute("""
            SELECT COUNT(*) FROM chunks 
            WHERE frame_label IS NOT NULL
        """).fetchone()[0]
        
        # Agreement-Chunks
        try:
            agreement_chunks = duck_conn.execute("SELECT COUNT(*) FROM agreement_chunks").fetchone()[0]
        except:
            agreement_chunks = 0
        
        # Annotatoren-Statistiken
        annotator_stats = duck_conn.execute("""
            SELECT assigned_user, COUNT(*) as count
            FROM chunks 
            WHERE assigned_user IS NOT NULL
            GROUP BY assigned_user
            ORDER BY count DESC
        """).fetchall()
        
        print(f"Gesamt Chunks:           {total_chunks:,}")
        print(f"Annotierte Chunks:        {annotated_chunks:,} ({annotated_chunks/total_chunks*100:.1f}%)")
        print(f"Chunks mit Frame-Labels:   {labeled_chunks:,} ({labeled_chunks/total_chunks*100:.1f}%)")
        print(f"Agreement-Chunks:         {agreement_chunks:,}")
        
        print(f"\nAnnotatoren-Statistiken:")
        for annotator, count in annotator_stats:
            print(f"  {annotator:15}: {count:6,} Chunks")
        
        # Frame-Label-Statistiken
        try:
            frame_stats = duck_conn.execute("""
                SELECT frame_label, COUNT(*) as count
                FROM chunks 
                WHERE frame_label IS NOT NULL
                GROUP BY frame_label
                ORDER BY count DESC
                LIMIT 10
            """).fetchall()
            
            if frame_stats:
                print(f"\nTop Frame-Labels:")
                for frame_label, count in frame_stats:
                    print(f"  {frame_label:20}: {count:6,} Chunks")
        except:
            pass
        
        # Brexit-Position Statistiken
        try:
            brexit_stats = duck_conn.execute("""
                SELECT brexit_position, COUNT(*) as count
                FROM chunks 
                WHERE brexit_position IS NOT NULL AND brexit_position != ''
                GROUP BY brexit_position
                ORDER BY count DESC
            """).fetchall()
            
            if brexit_stats:
                print(f"\nBrexit-Positionen:")
                for position, count in brexit_stats:
                    print(f"  {position:20}: {count:6,} Chunks")
        except:
            pass
        
        # Pre-Brexit vs Post-Brexit Statistiken
        try:
            timing_stats = duck_conn.execute("""
                SELECT 
                    COUNT(CASE WHEN pre_brexit = true THEN 1 END) as pre_brexit_count,
                    COUNT(CASE WHEN pre_brexit = false THEN 1 END) as post_brexit_count
                FROM chunks
            """).fetchone()
            
            if timing_stats:
                print(f"\nBrexit-Timing:")
                print(f"  Pre-Brexit:         {timing_stats[0]:6,} Chunks")
                print(f"  Post-Brexit:        {timing_stats[1]:6,} Chunks")
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description='Sync Railway zu lokaler DuckDB')
    parser.add_argument('--duckdb', 
                       default='/Users/julianbeese/Developer/Master/data_collection/data/processed/debates_brexit_chunked.duckdb',
                       help='Pfad zur lokalen DuckDB-Datei')
    parser.add_argument('--postgres-url', help='PostgreSQL URL (optional, nutzt DATABASE_URL)')
    
    args = parser.parse_args()
    
    if not Path(args.duckdb).exists():
        print(f"❌ DuckDB-Datei {args.duckdb} nicht gefunden!")
        return
    
    postgres_url = args.postgres_url or os.getenv('DATABASE_URL')
    if not postgres_url:
        print("❌ PostgreSQL URL benötigt! Setze DATABASE_URL oder nutze --postgres-url")
        return
    
    syncer = RailwayToLocalSync(args.duckdb, postgres_url)
    syncer.sync_annotations_from_railway()

if __name__ == "__main__":
    main()

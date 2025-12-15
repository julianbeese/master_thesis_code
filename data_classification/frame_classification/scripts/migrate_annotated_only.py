#!/usr/bin/env python3
"""
MIGRATION NUR FÜR ANNOTIERTE CHUNKS
Migriert nur die 2.000 annotierten Chunks (0.9% der Daten)
"""

import duckdb
import psycopg2
from psycopg2.extras import execute_values
import argparse
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class AnnotatedChunksMigration:
    def __init__(self, duckdb_path: str, postgres_url: str):
        self.duckdb_path = duckdb_path
        self.postgres_url = postgres_url
        
    def migrate_annotated_chunks(self):
        """Migriert nur annotierte Chunks"""
        print("🎯 MIGRATION NUR ANNOTIERTE CHUNKS")
        print("=" * 50)
        
        # PostgreSQL Verbindung
        pg_conn = psycopg2.connect(self.postgres_url)
        pg_conn.autocommit = False
        pg_cursor = pg_conn.cursor()
        
        # DuckDB Verbindung
        duck_conn = duckdb.connect(self.duckdb_path, read_only=True)
        
        try:
            # Erstelle Tabellen
            self.create_tables(pg_cursor)
            
            # Hole nur annotierte Chunks
            print("📊 Lade annotierte Chunks...")
            annotated_chunks = duck_conn.execute("""
                SELECT 
                    chunk_id, speech_id, debate_id, speaker_name, speaker_party,
                    debate_title, debate_date, chunk_text, chunk_index, total_chunks,
                    word_count, char_count, chunking_method, assigned_user,
                    frame_label, annotation_confidence, annotation_notes,
                    created_at, updated_at
                FROM chunks 
                WHERE assigned_user IS NOT NULL
                ORDER BY chunk_id
            """).fetchall()
            
            print(f"✅ {len(annotated_chunks):,} annotierte Chunks gefunden")
            
            # Migriere in Batches
            batch_size = 1000
            for i in range(0, len(annotated_chunks), batch_size):
                batch = annotated_chunks[i:i+batch_size]
                self.insert_chunks_batch(pg_cursor, batch)
                print(f"  📥 Migriert: {min(i+batch_size, len(annotated_chunks)):,}/{len(annotated_chunks):,}")
            
            # Agreement-Chunks
            try:
                agreement_chunks = duck_conn.execute("""
                    SELECT chunk_id, annotator1, annotator2, label1, label2,
                           agreement_score, agreement_perfect, created_at, updated_at
                    FROM agreement_chunks
                """).fetchall()
                
                if agreement_chunks:
                    print(f"📥 Migriere {len(agreement_chunks):,} Agreement-Chunks...")
                    self.insert_agreement_batch(pg_cursor, agreement_chunks)
            except:
                print("⚠️ Keine Agreement-Chunks gefunden")
            
            pg_conn.commit()
            print("✅ Migration erfolgreich abgeschlossen!")
            
        except Exception as e:
            pg_conn.rollback()
            print(f"❌ Fehler: {e}")
            raise
        finally:
            duck_conn.close()
            pg_cursor.close()
            pg_conn.close()
    
    def create_tables(self, cursor):
        """Erstellt Tabellen"""
        print("📦 Erstelle Tabellen...")
        
        # Chunks-Tabelle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id VARCHAR PRIMARY KEY,
                speech_id VARCHAR NOT NULL,
                debate_id VARCHAR,
                speaker_name VARCHAR,
                speaker_party VARCHAR,
                debate_title VARCHAR,
                debate_date DATE,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                word_count INTEGER NOT NULL,
                char_count INTEGER NOT NULL,
                chunking_method VARCHAR NOT NULL,
                assigned_user VARCHAR,
                frame_label VARCHAR,
                annotation_confidence INTEGER,
                annotation_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Agreement-Tabelle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agreement_chunks (
                chunk_id VARCHAR PRIMARY KEY,
                annotator1 VARCHAR NOT NULL,
                annotator2 VARCHAR NOT NULL,
                label1 VARCHAR,
                label2 VARCHAR,
                agreement_score FLOAT,
                agreement_perfect BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Indizes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_chunks_assigned_user ON chunks(assigned_user)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_frame_label ON chunks(frame_label)",
            "CREATE INDEX IF NOT EXISTS idx_agreement_annotator1 ON agreement_chunks(annotator1)",
            "CREATE INDEX IF NOT EXISTS idx_agreement_annotator2 ON agreement_chunks(annotator2)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        print("✅ Tabellen erstellt")
    
    def insert_chunks_batch(self, cursor, batch_data):
        """Batch-Insert für Chunks"""
        insert_sql = """
            INSERT INTO chunks (
                chunk_id, speech_id, debate_id, speaker_name, speaker_party,
                debate_title, debate_date, chunk_text, chunk_index, total_chunks,
                word_count, char_count, chunking_method, assigned_user,
                frame_label, annotation_confidence, annotation_notes,
                created_at, updated_at
            ) VALUES %s
        """
        
        execute_values(cursor, insert_sql, batch_data, page_size=1000)
    
    def insert_agreement_batch(self, cursor, batch_data):
        """Batch-Insert für Agreement"""
        insert_sql = """
            INSERT INTO agreement_chunks (
                chunk_id, annotator1, annotator2, label1, label2,
                agreement_score, agreement_perfect, created_at, updated_at
            ) VALUES %s
        """
        
        execute_values(cursor, insert_sql, batch_data, page_size=1000)

def main():
    parser = argparse.ArgumentParser(description='Migration nur annotierte Chunks')
    parser.add_argument('--duckdb', required=True, help='Pfad zur DuckDB-Datei')
    parser.add_argument('--postgres-url', help='PostgreSQL URL (optional, nutzt DATABASE_URL)')
    
    args = parser.parse_args()
    
    if not Path(args.duckdb).exists():
        print(f"❌ DuckDB-Datei {args.duckdb} nicht gefunden!")
        return
    
    postgres_url = args.postgres_url or os.getenv('DATABASE_URL')
    if not postgres_url:
        print("❌ PostgreSQL URL benötigt! Setze DATABASE_URL oder nutze --postgres-url")
        return
    
    migrator = AnnotatedChunksMigration(args.duckdb, postgres_url)
    migrator.migrate_annotated_chunks()

if __name__ == "__main__":
    main()

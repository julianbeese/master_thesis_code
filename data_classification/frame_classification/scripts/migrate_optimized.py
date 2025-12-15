#!/usr/bin/env python3
"""
OPTIMIERTE RAILWAY MIGRATION
Schneller Export mit DuckDB Native CSV + PostgreSQL Bulk Import
"""

import duckdb
import psycopg2
from psycopg2.extras import execute_batch, execute_values
import pandas as pd
import argparse
from pathlib import Path
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import numpy as np
from dotenv import load_dotenv

# Lade Umgebungsvariablen
load_dotenv()

class OptimizedRailwayMigration:
    def __init__(self, duckdb_path: str, postgres_url: Optional[str] = None):
        self.duckdb_path = duckdb_path
        self.postgres_url = postgres_url or os.getenv('DATABASE_URL')
        
        if not self.postgres_url:
            raise ValueError("PostgreSQL URL benötigt! Setze DATABASE_URL oder nutze --postgres-url")
        
        # OPTIMIERTE Performance-Einstellungen
        self.batch_size = 15000  # Noch größere Batches
        self.page_size = 5000    # Größere Seiten für execute_batch
        self.csv_chunk_size = 50000  # Für CSV-Export
        
    def copy_database_complete(self, input_db: str, output_db: str):
        """OPTIMIERTE Datenbankkopie - nur für Tests"""
        print("Kopiere KOMPLETTE Datenbank (optimiert)...")
        
        # Erstelle neue DB und attache die alte
        conn = duckdb.connect(output_db)
        conn.execute(f"ATTACH '{input_db}' AS source_db (READ_ONLY)")
        
        # Hole alle Tabellen aus der Quell-DB
        tables = conn.execute("SHOW TABLES FROM source_db").fetchall()
        
        print(f"Kopiere {len(tables)} Tabellen vollständig:")
        
        for (table_name,) in tables:
            # Direkte VOLLSTÄNDIGE Tabellenkopie mit CREATE TABLE AS
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM source_db.{table_name}")
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  ✓ {table_name}: {count:,} Zeilen")
        
        conn.close()
        return output_db
    
    def migrate_chunks_optimized(self):
        """OPTIMIERTE Chunk-Migration mit größeren Batches"""
        print("🚀 Starte optimierte Migration...")
        
        # PostgreSQL Verbindung
        pg_conn = psycopg2.connect(self.postgres_url)
        pg_conn.autocommit = False
        pg_cursor = pg_conn.cursor()
        
        # DuckDB Verbindung
        duck_conn = duckdb.connect(self.duckdb_path, read_only=True)
        
        try:
            # Erstelle Tabellen
            self.create_optimized_tables(pg_cursor)
            
            # OPTIMIERTE Chunk-Migration
            print("📊 Migriere Chunks (optimiert)...")
            
            # Hole Chunk-Count für Progress
            total_chunks = duck_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            print(f"  📤 Exportiere {total_chunks:,} Chunks aus DuckDB...")
            
            # OPTIMIERTE Batch-Verarbeitung
            offset = 0
            processed = 0
            
            while offset < total_chunks:
                # Größere Batches für bessere Performance
                batch_data = duck_conn.execute(f"""
                    SELECT 
                        chunk_id, speech_id, debate_id, speaker_name, speaker_party,
                        debate_title, debate_date, chunk_text, chunk_index, total_chunks,
                        word_count, char_count, chunking_method, assigned_user,
                        frame_label, annotation_confidence, annotation_notes,
                        created_at, updated_at
                    FROM chunks 
                    ORDER BY chunk_id
                    LIMIT {self.batch_size} OFFSET {offset}
                """).fetchall()
                
                if not batch_data:
                    break
                
                # OPTIMIERTE Batch-Insert
                self.insert_chunks_batch(pg_cursor, batch_data)
                
                processed += len(batch_data)
                offset += self.batch_size
                
                print(f"  📥 Importiert: {processed:,}/{total_chunks:,} Chunks ({processed/total_chunks*100:.1f}%)")
            
            # Agreement-Chunks (falls vorhanden)
            try:
                agreement_data = duck_conn.execute("SELECT * FROM agreement_chunks").fetchall()
                if agreement_data:
                    print(f"  📥 Importiere {len(agreement_data):,} Agreement-Chunks...")
                    self.insert_agreement_batch(pg_cursor, agreement_data)
            except:
                print("  ⚠️ Keine Agreement-Chunks gefunden")
            
            pg_conn.commit()
            print("✅ Migration erfolgreich abgeschlossen!")
            
        except Exception as e:
            pg_conn.rollback()
            print(f"❌ Fehler bei Migration: {e}")
            raise
        finally:
            duck_conn.close()
            pg_cursor.close()
            pg_conn.close()
    
    def create_optimized_tables(self, cursor):
        """Erstellt optimierte Tabellen"""
        print("📦 Erstelle optimierte Tabellen...")
        
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
        
        # OPTIMIERTE Indizes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_chunks_speech_id ON chunks(speech_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_assigned_user ON chunks(assigned_user)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_frame_label ON chunks(frame_label)",
            "CREATE INDEX IF NOT EXISTS idx_agreement_annotator1 ON agreement_chunks(annotator1)",
            "CREATE INDEX IF NOT EXISTS idx_agreement_annotator2 ON agreement_chunks(annotator2)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        print("✅ Tabellen erstellt")
    
    def insert_chunks_batch(self, cursor, batch_data):
        """OPTIMIERTE Batch-Insert für Chunks"""
        insert_sql = """
            INSERT INTO chunks (
                chunk_id, speech_id, debate_id, speaker_name, speaker_party,
                debate_title, debate_date, chunk_text, chunk_index, total_chunks,
                word_count, char_count, chunking_method, assigned_user,
                frame_label, annotation_confidence, annotation_notes,
                created_at, updated_at
            ) VALUES %s
        """
        
        # Nutze execute_values für maximale Performance
        execute_values(
            cursor, insert_sql, batch_data,
            template=None, page_size=self.page_size
        )
    
    def insert_agreement_batch(self, cursor, batch_data):
        """OPTIMIERTE Batch-Insert für Agreement"""
        insert_sql = """
            INSERT INTO agreement_chunks (
                chunk_id, annotator1, annotator2, label1, label2,
                agreement_score, agreement_perfect, created_at, updated_at
            ) VALUES %s
        """
        
        execute_values(
            cursor, insert_sql, batch_data,
            template=None, page_size=self.page_size
        )
    
    def export_to_csv_ultra_fast(self, output_dir: str):
        """ULTRA-SCHNELLER CSV-Export mit DuckDB Native"""
        print("📦 ULTRA-SCHNELLER CSV-Export...")
        Path(output_dir).mkdir(exist_ok=True)
        
        duck_conn = duckdb.connect(self.duckdb_path, read_only=True)
        
        # OPTIMIERTE CSV-Exports
        print("  📤 Exportiere Chunks (ultra-schnell)...")
        duck_conn.execute(f"""
            COPY chunks 
            TO '{output_dir}/chunks.csv' 
            WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',', QUOTE '"')
        """)
        
        try:
            duck_conn.execute(f"""
                COPY agreement_chunks 
                TO '{output_dir}/agreement_chunks.csv' 
                WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',', QUOTE '"')
            """)
            print("  📤 Exportiere Agreement (ultra-schnell)...")
        except:
            print("  ⚠️ Keine Agreement-Chunks gefunden")
        
        # Erstelle Import-SQL
        import_sql = f"""
-- ULTRA-SCHNELLER IMPORT für Railway
-- Nutze diese SQL-Datei für schnellen Import

-- 1. Lade CSV-Dateien zu Railway hoch
-- 2. Führe diese SQL-Befehle aus:

\\copy chunks FROM '{output_dir}/chunks.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',', QUOTE '"');

-- Falls Agreement-Chunks vorhanden:
\\copy agreement_chunks FROM '{output_dir}/agreement_chunks.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',', QUOTE '"');

-- Erstelle Indizes für Performance:
CREATE INDEX IF NOT EXISTS idx_chunks_speech_id ON chunks(speech_id);
CREATE INDEX IF NOT EXISTS idx_chunks_assigned_user ON chunks(assigned_user);
CREATE INDEX IF NOT EXISTS idx_chunks_frame_label ON chunks(frame_label);
CREATE INDEX IF NOT EXISTS idx_agreement_annotator1 ON agreement_chunks(annotator1);
CREATE INDEX IF NOT EXISTS idx_agreement_annotator2 ON agreement_chunks(annotator2);
"""
        
        with open(f"{output_dir}/import.sql", "w") as f:
            f.write(import_sql)
        
        duck_conn.close()
        print("✅ ULTRA-SCHNELLER CSV-Export abgeschlossen!")
        print(f"📁 Export-Verzeichnis: {output_dir}/")
        print("📝 Nutze import.sql für ULTRA-SCHNELLEN Import in Railway")

def migrate_with_fallback(duckdb_path: str, postgres_url: str):
    """Migration mit Fallback zu CSV-Export"""
    print("=" * 70)
    print("🚀 OPTIMIERTE RAILWAY MIGRATION")
    print("=" * 70)
    
    migrator = OptimizedRailwayMigration(duckdb_path, postgres_url)
    
    try:
        print("🚀 Starte direkte Migration...")
        migrator.migrate_chunks_optimized()
        print("✅ Direkte Migration erfolgreich!")
        
    except Exception as e:
        print(f"❌ Direkte Migration fehlgeschlagen: {e}")
        print("💡 Tipp: Stelle sicher, dass DATABASE_URL gesetzt ist oder nutze --postgres-url")
        
        print("\n📦 Fallback: ULTRA-SCHNELLER CSV-Export...")
        migrator.export_to_csv_ultra_fast("railway_export_optimized")
        
        print("\n📌 Nächste Schritte:")
        print("1. Stelle sicher, dass die Daten korrekt migriert wurden")
        print("2. Teste die Annotation-App mit der Railway-Datenbank")
        print("3. Überwache die Performance mit Railway Metrics")

def main():
    parser = argparse.ArgumentParser(description='OPTIMIERTE Railway Migration')
    parser.add_argument('--duckdb', required=True, help='Pfad zur DuckDB-Datei')
    parser.add_argument('--postgres-url', help='PostgreSQL URL (optional, nutzt DATABASE_URL)')
    parser.add_argument('--method', choices=['direct', 'csv'], default='direct', 
                       help='Migration-Methode: direct oder csv')
    
    args = parser.parse_args()
    
    if not Path(args.duckdb).exists():
        print(f"❌ DuckDB-Datei {args.duckdb} nicht gefunden!")
        return
    
    if args.method == 'csv':
        print("📦 CSV-Export-Modus...")
        migrator = OptimizedRailwayMigration(args.duckdb, args.postgres_url)
        migrator.export_to_csv_ultra_fast("railway_export_optimized")
    else:
        migrate_with_fallback(args.duckdb, args.postgres_url)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Optimiertes Datenbank-Chunking für bessere Performance
Erstellt eine neue Datenbank mit ALLEN Tabellen + neue Chunks-Tabelle
"""

import duckdb
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import time
import random

# Konfiguration (robuste Pfadauflösung relativ zum Projekt-Root)
# Dieses Skript liegt unter frame_classification/scripts/current/...
# Projekt-Root ist drei Ebenen höher
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DB = PROJECT_ROOT / "data/processed/brexit_analysis.duckdb"
DEFAULT_OUTPUT_DB = PROJECT_ROOT / "data/processed/debates_brexit_chunked.duckdb"

class OptimizedDatabaseChunker:
    def __init__(self):
        # Verzichte auf spaCy - nutze nur regex-basierte Satztrennung für Geschwindigkeit
        self.sentence_endings = re.compile(r'[.!?]+[\s\n]')
        
    def clean_text(self, text: str) -> str:
        """Optimierte Textbereinigung"""
        if not text:
            return ""
        
        # Kombinierte Regex für alle Bereinigungen in einem Durchgang
        # HTML-Tags entfernen
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalisiere Whitespaces in einem Durchgang
        text = re.sub(r'\r\n|\r', '\n', text)
        text = re.sub(r'[\t ]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def split_into_sentences_fast(self, text: str) -> List[str]:
        """Schnelle Satztrennung mit Regex"""
        if not text:
            return []
        
        # Einfache aber schnelle Regex-basierte Trennung
        sentences = self.sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_by_paragraph_units_fast(self, text: str, max_chars: int = 1300) -> List[str]:
        """Optimiertes Chunking mit minimalen Regex-Operationen"""
        if not text:
            return []

        # Einmalige Absatztrennung
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)
            
            # Wenn der Absatz allein zu groß ist
            if para_size > max_chars:
                # Speichere aktuellen Chunk wenn vorhanden
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Teile großen Absatz mit 800-Zeichen-Limit für Sentences
                sentence_limit = 800
                sentences = self.split_into_sentences_fast(para)
                temp_chunk = []
                temp_size = 0
                
                for sent in sentences:
                    sent_size = len(sent)
                    if temp_size + sent_size + (2 if temp_chunk else 0) <= sentence_limit:
                        temp_chunk.append(sent)
                        temp_size += sent_size + (2 if temp_chunk else 0)
                    else:
                        if temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                        temp_chunk = [sent]
                        temp_size = sent_size
                
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
            
            # Wenn Absatz in aktuellen Chunk passt
            elif current_size + para_size + (4 if current_chunk else 0) <= max_chars:
                current_chunk.append(para)
                current_size += para_size + (4 if current_chunk else 0)
            
            # Neuer Chunk nötig
            else:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return [c for c in chunks if c]

def process_speech_batch(speeches_batch, start_id):
    """Verarbeitet eine Batch von Speeches parallel"""
    chunker = OptimizedDatabaseChunker()
    batch_chunks = []
    chunk_id = start_id
    
    for speech in speeches_batch:
        speech_id, debate_id, speaker_name, speaker_party, speech_text, _, debate_title, debate_date = speech
        
        if not speech_text:
            continue
        
        clean_text = chunker.clean_text(speech_text)
        if not clean_text:
            continue
        
        chunks = chunker.chunk_by_paragraph_units_fast(clean_text, max_chars=1300)
        
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            
            chunk_data = (
                f"chunk_{chunk_id:06d}",
                speech_id,
                debate_id,
                speaker_name,
                speaker_party,
                debate_title,
                debate_date,
                chunk_text,
                i,
                len(chunks),
                len(chunk_text.split()),
                len(chunk_text),
                'semantic',
                None,  # assigned_user - wird später gesetzt
                None,  # frame_label
                None,  # annotation_confidence
                None   # annotation_notes
            )
            
            batch_chunks.append(chunk_data)
            chunk_id += 1
    
    return batch_chunks

def copy_database_complete(input_db: str, output_db: str):
    """Kopiert die KOMPLETTE Datenbank mit allen Tabellen und Daten"""
    print("Kopiere KOMPLETTE Datenbank (optimiert)...")
    start_time = time.time()
    
    # Lösche Output-DB falls vorhanden
    if Path(output_db).exists():
        Path(output_db).unlink()
    
    # Erstelle neue DB und attache die alte
    conn = duckdb.connect(output_db)
    conn.execute(f"ATTACH '{input_db}' AS source_db (READ_ONLY)")
    
    # Hole alle Tabellen aus der Quell-DB
    tables = conn.execute("SHOW TABLES FROM source_db").fetchall()
    
    print(f"Kopiere {len(tables)} Tabellen vollständig:")
    
    for (table_name,) in tables:
        # Direkte VOLLSTÄNDIGE Tabellenkopie mit CREATE TABLE AS
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM source_db.{table_name}")
        
        # Verifiziere, dass ALLE Zeilen kopiert wurden
        source_count = conn.execute(f"SELECT COUNT(*) FROM source_db.{table_name}").fetchone()[0]
        target_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        if source_count != target_count:
            print(f"  ⚠️ WARNUNG: {table_name} - Quelle: {source_count}, Ziel: {target_count}")
        else:
            print(f"  ✓ {table_name}: {target_count:,} Zeilen (100% kopiert)")
    
    conn.execute("DETACH source_db")
    
    elapsed = time.time() - start_time
    print(f"✓ KOMPLETTE Datenbankkopie abgeschlossen in {elapsed:.2f} Sekunden")
    
    return conn

def copy_database_brexit_subset(input_db: str, output_db: str, min_confidence: float):
    """Erstellt eine neue DB mit NUR Brexit-relevanten Speeches und referenzierten Debates"""
    print("Kopiere Brexit-Subset (Speeches + referenzierte Debates)...")
    start_time = time.time()
    
    # Lösche Output-DB falls vorhanden
    out_path = Path(output_db)
    if out_path.exists():
        out_path.unlink()
    
    conn = duckdb.connect(output_db)
    conn.execute(f"ATTACH '{input_db}' AS source_db (READ_ONLY)")
    
    # Speeches (gefiltert)
    print("  • Erstelle Tabelle 'speeches' (Brexit-gefiltert)...")
    conn.execute(f"""
        CREATE TABLE speeches AS
        SELECT *
        FROM source_db.speeches s
        WHERE s.llm_processed = TRUE
          AND s.llm_classified_brexit = TRUE
          AND (s.llm_confidence_score IS NULL OR s.llm_confidence_score >= {min_confidence})
    """)
    
    # Debates (nur referenzierte)
    print("  • Erstelle Tabelle 'debates' (nur referenzierte Einträge)...")
    conn.execute("""
        CREATE TABLE debates AS
        SELECT *
        FROM source_db.debates d
        WHERE d.debate_id IN (SELECT DISTINCT debate_id FROM speeches WHERE debate_id IS NOT NULL)
    """)
    
    # Optional: weitere Tabellen bei Bedarf hinzufügen
    # Aktuell nutzt das Skript nur speeches und debates
    
    conn.execute("DETACH source_db")
    
    # Verifikation
    speech_count = conn.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]
    debate_count = conn.execute("SELECT COUNT(*) FROM debates").fetchone()[0]
    print(f"  ✓ {speech_count:,} Brexit-Speeches, {debate_count:,} referenzierte Debates")
    
    elapsed = time.time() - start_time
    print(f"✓ Brexit-Subset erstellt in {elapsed:.2f} Sekunden")
    return conn

def assign_chunks_for_annotation(conn, total_chunks: int):
    """Weist genau 1.000 Chunks für Doppel-Annotation zu"""
    print("\n" + "=" * 70)
    print("CHUNK-ZUWEISUNG FÜR ANNOTATION")
    print("=" * 70)
    
    # Annotatoren
    annotators = ["Max", "Julian", "Lina", "Julius", "Rike"]
    
    # Feste Anzahl für Annotation
    annotation_count = min(1000, total_chunks)
    
    if total_chunks < 1000:
        print(f"⚠️ Nur {total_chunks} Chunks verfügbar (weniger als 1.000)")
    else:
        print(f"✓ Wähle 1.000 von {total_chunks:,} Chunks für Annotation")
    
    # Zufällige Auswahl von 1.000 Chunks
    random.seed(42)  # Reproduzierbare Ergebnisse
    
    # Hole alle Chunk-IDs
    all_chunk_ids = conn.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id").fetchall()
    all_chunk_ids = [row[0] for row in all_chunk_ids]
    
    # Wähle zufällig 1.000 Chunks aus
    selected_chunk_ids = random.sample(all_chunk_ids, annotation_count)
    
    print(f"✓ {len(selected_chunk_ids)} Chunks ausgewählt")
    
    # Erstelle Agreement-Tabelle für Doppel-Annotation
    print("Erstelle Agreement-Tabelle...")
    conn.execute("""
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
    
    # Zuweisung mit Doppel-Annotation
    assignment_stats = {annotator: 0 for annotator in annotators}
    duplicated_chunks = []
    
    for chunk_id in selected_chunk_ids:
        # Wähle 2 zufällige Annotatoren für diesen Chunk
        selected_annotators = random.sample(annotators, 2)
        
        # Primäre Zuweisung (Original-Chunk)
        conn.execute("""
            UPDATE chunks 
            SET assigned_user = ? 
            WHERE chunk_id = ?
        """, (selected_annotators[0], chunk_id))
        
        assignment_stats[selected_annotators[0]] += 1
        
        # Hole Chunk-Daten für Duplikation
        chunk_data = conn.execute("""
            SELECT speech_id, debate_id, speaker_name, speaker_party,
                   debate_title, debate_date, chunk_text, chunk_index, 
                   total_chunks, word_count, char_count, chunking_method
            FROM chunks 
            WHERE chunk_id = ?
        """, (chunk_id,)).fetchone()
        
        # Erstelle Duplikat für zweiten Annotator
        dup_chunk_id = f"{chunk_id}_dup"
        duplicated_chunks.append((
            dup_chunk_id,
            chunk_data[0],  # speech_id
            chunk_data[1],  # debate_id
            chunk_data[2],  # speaker_name
            chunk_data[3],  # speaker_party
            chunk_data[4],  # debate_title
            chunk_data[5],  # debate_date
            chunk_data[6],  # chunk_text
            chunk_data[7],  # chunk_index
            chunk_data[8],  # total_chunks
            chunk_data[9],  # word_count
            chunk_data[10], # char_count
            chunk_data[11], # chunking_method
            selected_annotators[1],  # assigned_user (zweiter Annotator)
            None,  # frame_label
            None,  # annotation_confidence
            None   # annotation_notes
        ))
        
        assignment_stats[selected_annotators[1]] += 1
        
        # Speichere Agreement-Paar
        conn.execute("""
            INSERT INTO agreement_chunks (chunk_id, annotator1, annotator2)
            VALUES (?, ?, ?)
        """, (chunk_id, selected_annotators[0], selected_annotators[1]))
    
    # Füge duplizierte Chunks ein (Batch-Insert für Performance)
    if duplicated_chunks:
        print(f"Füge {len(duplicated_chunks)} Duplikate für Doppel-Annotation ein...")
        
        insert_dup_sql = """
        INSERT INTO chunks (
            chunk_id, speech_id, debate_id, speaker_name, speaker_party,
            debate_title, debate_date, chunk_text, chunk_index, total_chunks,
            word_count, char_count, chunking_method, assigned_user,
            frame_label, annotation_confidence, annotation_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        conn.executemany(insert_dup_sql, duplicated_chunks)
    
    # Statistiken ausgeben
    print("\n✓ Zuweisung abgeschlossen:")
    print(f"  • {annotation_count} Original-Chunks ausgewählt")
    print(f"  • {len(duplicated_chunks)} Duplikate erstellt")
    print(f"  • {annotation_count * 2} Annotationen insgesamt")
    
    print("\nAnnotationen pro User:")
    for annotator, count in sorted(assignment_stats.items()):
        print(f"  {annotator:10}: {count:4} Chunks")
    
    # Zeige unzugewiesene Chunks
    unassigned = conn.execute("""
        SELECT COUNT(*) 
        FROM chunks 
        WHERE assigned_user IS NULL 
        AND chunk_id NOT LIKE '%_dup'
    """).fetchone()[0]
    
    print(f"\n  Unzugewiesen: {unassigned:,} Chunks (werden nicht annotiert)")
    
    return annotation_count

def main():
    parser = argparse.ArgumentParser(description='Optimiertes Datenbank-Chunking')
    parser.add_argument('--input-db', default=str(DEFAULT_INPUT_DB), help='Pfad zur Input-Datenbank')
    parser.add_argument('--output-db', default=str(DEFAULT_OUTPUT_DB), help='Pfad zur Output-Datenbank')
    parser.add_argument('--max-speeches', type=int, help='Maximale Anzahl Reden (für Tests)')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch-Größe für Verarbeitung')
    parser.add_argument('--workers', type=int, default=mp.cpu_count(), help='Anzahl paralleler Worker')
    parser.add_argument('--filter-brexit', action='store_true', help='Nur als Brexit-relevant klassifizierte Speeches verarbeiten')
    parser.add_argument('--min-confidence', type=float, default=0.0, help='Mindest-Confidence für Brexit-Klassifikation (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Prüfe Input-Datenbank
    if not Path(args.input_db).exists():
        print(f"✗ Input-Datenbank {args.input_db} nicht gefunden!")
        return
    
    print("=" * 70)
    print("OPTIMIERTES DATENBANK-CHUNKING")
    print("=" * 70)
    print(f"Input:   {args.input_db}")
    print(f"Output:  {args.output_db}")
    print(f"Workers: {args.workers} CPU-Kerne")
    print("=" * 70)
    
    total_start = time.time()
    
    # SCHRITT 1: Kopiere Datenbank (vollständig oder Brexit-Subset)
    if args.filter_brexit:
        conn = copy_database_brexit_subset(args.input_db, args.output_db, args.min_confidence)
    else:
        conn = copy_database_complete(args.input_db, args.output_db)
    
    # SCHRITT 2: Erstelle NEUE Chunks-Tabelle
    print("\nErstelle NEUE Chunks-Tabelle...")
    create_chunks_sql = """
    CREATE TABLE chunks (
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
    """
    conn.execute(create_chunks_sql)
    print("✓ Chunks-Tabelle erstellt")
    
    # SCHRITT 3: Lade ALLE (optional: Brexit-relevante) Speeches
    print("\nLade Speeches für Chunking...")
    start_time = time.time()
    
    # Optionales Brexit-Filter: Wenn wir bereits ein Subset kopiert haben, ist keine WHERE-Bedingung nötig
    brexit_where = ""
    if args.filter_brexit:
        print(f"  (Filter aktiv: Brexit-relevant, min. Confidence {args.min_confidence}) - Subset bereits erstellt")
    
    base_select = (
        "SELECT \n"
        "    s.speech_id,\n"
        "    s.debate_id,\n"
        "    s.speaker_name,\n"
        "    s.speaker_office,\n"
        "    s.speech_text,\n"
        "    s.time,\n"
        "    d.major_heading_text,\n"
        "    d.date\n"
        "FROM speeches s\n"
        "LEFT JOIN debates d ON s.debate_id = d.debate_id"
    )
    order_clause = "\nORDER BY s.speech_id"
    limit_clause = f"\nLIMIT {args.max_speeches}" if args.max_speeches else ""
    if args.max_speeches:
        print(f"  (Limitiert auf {args.max_speeches} Speeches für Test)")
    query = base_select + brexit_where + order_clause + limit_clause
    
    speeches = conn.execute(query).fetchall()
    print(f"✓ {len(speeches)} Speeches geladen in {time.time() - start_time:.2f} Sekunden")
    
    # SCHRITT 4: Verarbeite Speeches parallel zu Chunks
    print(f"\nVerarbeite Speeches zu Chunks (Batch-Größe: {args.batch_size})...")
    start_time = time.time()
    
    # Teile Speeches in Batches
    batches = []
    for i in range(0, len(speeches), args.batch_size):
        batch = speeches[i:i+args.batch_size]
        start_id = i * 10  # Geschätzte Start-ID für Chunks
        batches.append((batch, start_id))
    
    all_chunks = []
    
    # Parallele Verarbeitung
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for batch, start_id in batches:
            future = executor.submit(process_speech_batch, batch, start_id)
            futures.append(future)
        
        completed = 0
        for future in as_completed(futures):
            batch_chunks = future.result()
            all_chunks.extend(batch_chunks)
            completed += 1
            print(f"  Batch {completed}/{len(batches)}: {len(batch_chunks)} Chunks")
    
    print(f"✓ {len(all_chunks)} Chunks erstellt in {time.time() - start_time:.2f} Sekunden")
    
    # SCHRITT 5: Füge ALLE Chunks in Datenbank ein
    print("\nFüge ALLE Chunks in Datenbank ein (Batch-Insert)...")
    start_time = time.time()
    
    insert_sql = """
    INSERT INTO chunks (
        chunk_id, speech_id, debate_id, speaker_name, speaker_party,
        debate_title, debate_date, chunk_text, chunk_index, total_chunks,
        word_count, char_count, chunking_method, assigned_user,
        frame_label, annotation_confidence, annotation_notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    # Batch-Insert für Performance
    batch_size = 10000
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        conn.executemany(insert_sql, batch)
        print(f"  {min(i+batch_size, len(all_chunks)):,}/{len(all_chunks):,} eingefügt")
    
    print(f"✓ ALLE {len(all_chunks)} Chunks eingefügt in {time.time() - start_time:.2f} Sekunden")
    
    # SCHRITT 6: Erstelle Indizes
    print("\nErstelle Indizes...")
    start_time = time.time()
    
    indexes = [
        "CREATE INDEX idx_chunks_speech_id ON chunks(speech_id)",
        "CREATE INDEX idx_chunks_debate_id ON chunks(debate_id)", 
        "CREATE INDEX idx_chunks_speaker ON chunks(speaker_name)",
        "CREATE INDEX idx_chunks_assigned_user ON chunks(assigned_user)",
        "CREATE INDEX idx_chunks_frame_label ON chunks(frame_label)"
    ]
    
    for index_sql in indexes:
        conn.execute(index_sql)
    
    print(f"✓ Indizes erstellt in {time.time() - start_time:.2f} Sekunden")
    
    # SCHRITT 7: Weise 1.000 Chunks für Annotation zu
    annotation_count = assign_chunks_for_annotation(conn, len(all_chunks))
    
    # SCHRITT 8: Finale Statistiken
    print("\n" + "=" * 70)
    print("FINALE STATISTIKEN")
    print("=" * 70)
    
    # Chunk-Statistiken
    stats = conn.execute("""
        SELECT 
            COUNT(DISTINCT chunk_id) as unique_chunks,
            COUNT(*) as total_rows,
            AVG(word_count) as avg_words,
            AVG(char_count) as avg_chars,
            MIN(word_count) as min_words,
            MAX(word_count) as max_words
        FROM chunks
        WHERE chunk_id NOT LIKE '%_dup'
    """).fetchone()
    
    print("Chunk-Statistiken:")
    print(f"  Unique Chunks:         {stats[0]:,}")
    print(f"  Total mit Duplikaten:  {stats[1]:,}")
    print(f"  Durchschnitt Wörter:   {stats[2]:.1f}")
    print(f"  Durchschnitt Zeichen:  {stats[3]:.1f}")
    print(f"  Wort-Range:            {stats[4]}-{stats[5]}")
    
    # Tabellen-Übersicht
    print("\nTabellen in Output-Datenbank:")
    tables = conn.execute("SHOW TABLES").fetchall()
    for (table_name,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  {table_name:20}: {count:,} Zeilen")
    
    conn.close()
    
    total_elapsed = time.time() - total_start
    print(f"\n" + "=" * 70)
    print(f"✓ GESAMTPROZESS ABGESCHLOSSEN in {total_elapsed:.2f} Sekunden")
    print(f"✓ Output: {args.output_db}")
    print("=" * 70)

if __name__ == "__main__":
    main()
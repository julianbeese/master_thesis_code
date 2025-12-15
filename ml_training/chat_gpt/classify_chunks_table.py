#!/usr/bin/env python3
"""
Classify chunks from the chunks table using OpenAI API
Checks for existing classifications in multiple tables before classifying new chunks.
Works with debates_training_Kopie.duckdb -> chunks table.
"""

import duckdb
import openai
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, try manual loading
    try:
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    except Exception:
        pass
except Exception:
    pass

# Konfiguration
DB_FILE = "debates_training_Kopie.duckdb"
SOURCE_TABLE = "chunks"
OPENAI_MODEL = "gpt-4.1-mini"
MAX_RETRIES = 3
RETRY_DELAY = 1
BATCH_SIZE = 10  # Für Rate Limiting (Pause nach N Chunks)
COMMIT_BATCH_SIZE = 100  # Commit zur DB nach N Chunks (für Performance)
MAX_CHUNKS_TO_CLASSIFY = 30000  # Maximale Anzahl an Chunks, die klassifiziert werden sollen

# Die 6 möglichen Frames
FRAMES = [
    "economic",
    "moral value",
    "human impact", 
    "powerlessness",
    "conflict",
    "none"
]

# System Prompt für die Klassifizierung
SYSTEM_PROMPT = """You are an expert for the analysis of political speeches in the context of the Brexit referendum.

Your task is to classify speeches into one of the following six communicative frames:

1. Economic - Centers on financial costs, benefits, and economic implications using quantifiable measures, market dynamics, and economic terminology and business language.

2. Human Impact - Focuses on how political decisions affect individuals, families, communities, or specific demographic groups; people's daily lives and personal well-being. Describes "concrete effects" using "personal, emotional, or experiential language".

3. Conflict - Frames politics as a struggle between opposing sides, using adversarial/strategic or competitive language to depict winners and losers; zero-sum nature; incompatible interests.

4. Powerlessness - Highlights a lack of agency, control, or influence, portraying individuals, institutions, or the nation as constrained or subordinated by external authorities or structures, and emphasizing the need/action to reclaim autonomy, independence, or decision-making power.

5. Moral Value - Appeals to principles of fairness, justice, integrity, and ethical responsibility, praising actions that uphold honesty, inclusivity, and democratic or social norms, while condemning hypocrisy, discrimination, fearmongering, or unethical behavior. It often works through indirect references to societal values.

6. None - Only if no other frame is applicable. Does contain e.g. procedural or administrative content. Technical or procedural descriptions without evaluative framing.

Respond ONLY with the exact category name. No explanations or additional text."""


def setup_openai():
    """Initializes OpenAI API"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set!")
    
    client = openai.OpenAI(api_key=api_key)
    print("✓ OpenAI API configured")
    return client


def normalize_label(label: Optional[str]) -> str:
    """Normalizes labels to lowercase and validates against FRAMES"""
    if label is None:
        return "none"
    
    label_lower = str(label).lower().strip()
    
    # Check if it's a valid frame
    if label_lower in FRAMES:
        return label_lower
    
    # Try to map common variations
    mapping = {
        'economic': 'economic',
        'economics': 'economic',
        'moral value': 'moral value',
        'moral values': 'moral value',
        'moral': 'moral value',
        'human impact': 'human impact',
        'powerlessness': 'powerlessness',
        'powerless': 'powerlessness',
        'conflict': 'conflict',
        'none': 'none',
        'other': 'none',
        '': 'none'
    }
    
    if label_lower in mapping:
        return mapping[label_lower]
    
    # If unknown, return 'none'
    return "none"


def add_classification_columns(conn: duckdb.DuckDBPyConnection):
    """Adds classification columns to chunks table if they don't exist"""
    print(f"\nChecking table structure...")
    
    try:
        columns = conn.execute(f"PRAGMA table_info({SOURCE_TABLE})").fetchall()
        column_names = [col[1] for col in columns]
        
        if 'frame_label' not in column_names:
            conn.execute(f"""
                ALTER TABLE {SOURCE_TABLE}
                ADD COLUMN frame_label VARCHAR
            """)
            print(f"✓ Added column 'frame_label'")
        else:
            print(f"✓ Column 'frame_label' already exists")
        
        if 'classification_timestamp' not in column_names:
            conn.execute(f"""
                ALTER TABLE {SOURCE_TABLE}
                ADD COLUMN classification_timestamp TIMESTAMP
            """)
            print(f"✓ Added column 'classification_timestamp'")
        else:
            print(f"✓ Column 'classification_timestamp' already exists")
        
    except Exception as e:
        print(f"⚠️  Error checking/adding columns: {e}")


def get_chunks_to_classify(conn: duckdb.DuckDBPyConnection) -> List[Dict[str, Any]]:
    """
    Gets all chunks that don't have a classification yet.
    """
    print(f"\nGetting chunks that need classification...")
    
    # First, check what's already in the chunks table
    already_classified_in_chunks = conn.execute(f"""
        SELECT chunk_id, frame_label
        FROM {SOURCE_TABLE}
        WHERE frame_label IS NOT NULL 
        AND frame_label != ''
    """).fetchall()
    
    already_classified_ids = {row[0]: row[1] for row in already_classified_in_chunks}
    print(f"  Found {len(already_classified_ids)} chunks already classified in chunks table")
    
    # Get chunks without classification (limited to MAX_CHUNKS_TO_CLASSIFY)
    query = f"""
        SELECT chunk_id, chunk_text
        FROM {SOURCE_TABLE}
        WHERE chunk_text IS NOT NULL 
        AND chunk_text != ''
        AND (frame_label IS NULL OR frame_label = '')
        ORDER BY chunk_id
        LIMIT {MAX_CHUNKS_TO_CLASSIFY}
    """
    
    results = conn.execute(query).fetchall()
    chunks = []
    
    for row in results:
        chunk_id = row[0]
        chunk_text = row[1]
        
        # Skip if already classified in chunks table (shouldn't happen due to WHERE clause, but double-check)
        if chunk_id in already_classified_ids:
            continue
        
        chunks.append({
            'chunk_id': chunk_id,
            'chunk_text': chunk_text
        })
        
        # Stop if we've reached the limit
        if len(chunks) >= MAX_CHUNKS_TO_CLASSIFY:
            break
    
    print(f"✓ {len(chunks)} chunks will be classified (limit: {MAX_CHUNKS_TO_CLASSIFY})")
    return chunks


def classify_chunk_with_openai(client, text: str) -> str:
    """Classify a single chunk using OpenAI API"""
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this speech:\n\n{text}"}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        label = response.choices[0].message.content.strip()
        
        # Normalize label
        label_normalized = normalize_label(label)
        
        # Validate that the label is one of the allowed frames
        if label_normalized in FRAMES:
            return label_normalized
        else:
            print(f"⚠️  Invalid label '{label}' - using 'none' instead")
            return 'none'
        
    except Exception as e:
        print(f"❌ Error classifying chunk: {e}")
        return 'none'


def process_chunks(client, conn, chunks: List[Dict[str, Any]]):
    """
    Process all chunks and save classifications in batches.
    Saves progress after each chunk to ensure data persistence if interrupted.
    """
    print(f"\nProcessing {len(chunks)} chunks with OpenAI...")
    
    if not chunks:
        print("✓ No chunks to process")
        return
    
    total_chunks = len(chunks)
    processed_count = 0
    error_count = 0
    
    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk['chunk_id']
        text = chunk['chunk_text']
        
        # Double-check: Skip if already classified (in case of restart)
        existing = conn.execute(
            f"SELECT frame_label FROM {SOURCE_TABLE} WHERE chunk_id = ? AND frame_label IS NOT NULL AND frame_label != ''",
            [chunk_id]
        ).fetchone()
        
        if existing:
            print(f"  [{i}/{total_chunks}] Chunk {chunk_id} already classified, skipping...")
            processed_count += 1
            continue
        
        # Classify with OpenAI
        retry_count = 0
        classification = None
        
        while retry_count < MAX_RETRIES and classification is None:
            try:
                classification = classify_chunk_with_openai(client, text)
                break
            except Exception as e:
                retry_count += 1
                print(f"  ⚠️  Attempt {retry_count}/{MAX_RETRIES} failed: {e}")
                if retry_count < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    classification = 'none'
                    print(f"  ❌ All attempts failed, using 'none'")
                    error_count += 1
        
        # Save classification to database
        try:
            conn.execute(f"""
                UPDATE {SOURCE_TABLE}
                SET frame_label = ?,
                    classification_timestamp = CURRENT_TIMESTAMP
                WHERE chunk_id = ?
            """, [classification, chunk_id])
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ❌ Error saving classification for {chunk_id}: {e}")
            error_count += 1
        
        # Commit in batches for better performance
        if i % COMMIT_BATCH_SIZE == 0 or i == total_chunks:
            conn.commit()
            print(f"  💾 Progress saved: {i}/{total_chunks} chunks processed ({processed_count} successful, {error_count} errors)")
        
        # Progress update (without commit)
        elif i % 50 == 0:
            print(f"  Progress: {i}/{total_chunks} chunks processed ({processed_count} successful, {error_count} errors)")
        
        # Rate limiting - pause after N chunks
        if i % BATCH_SIZE == 0:
            time.sleep(0.5)
    
    print(f"✓ Processing complete: {processed_count} chunks processed, {error_count} errors")


def show_statistics(conn):
    """Show classification statistics"""
    print("\n" + "=" * 70)
    print("CLASSIFICATION STATISTICS")
    print("=" * 70)
    
    # Total chunks
    total_count = conn.execute(f"SELECT COUNT(*) FROM {SOURCE_TABLE}").fetchone()[0]
    classified_count = conn.execute(f"""
        SELECT COUNT(*) FROM {SOURCE_TABLE}
        WHERE frame_label IS NOT NULL AND frame_label != ''
    """).fetchone()[0]
    unclassified_count = total_count - classified_count
    
    print(f"\nTotal chunks: {total_count}")
    print(f"Classified: {classified_count} ({classified_count/total_count*100:.1f}%)")
    print(f"Unclassified: {unclassified_count} ({unclassified_count/total_count*100:.1f}%)")
    
    # Count classifications by frame
    query = f"""
        SELECT 
            frame_label,
            COUNT(*) as count
        FROM {SOURCE_TABLE}
        WHERE frame_label IS NOT NULL
        GROUP BY frame_label
        ORDER BY count DESC
    """
    
    results = conn.execute(query).fetchall()
    
    if results:
        total = sum(row[1] for row in results)
        print(f"\nFrame distribution:\n")
        print(f"{'Frame':<20s} {'Count':>10s} {'Percentage':>12s}")
        print("-" * 45)
        
        for frame, count in results:
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"{frame:<20s} {count:>10d} {percentage:>11.2f}%")


def main():
    """Main execution function"""
    print("=" * 70)
    print("CLASSIFY CHUNKS TABLE")
    print("=" * 70)
    
    # Check database
    if not Path(DB_FILE).exists():
        print(f"❌ Database {DB_FILE} not found!")
        return 1
    
    # Connect to database
    conn = duckdb.connect(DB_FILE)
    
    try:
        # Check if source table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        if SOURCE_TABLE not in table_names:
            print(f"❌ Table '{SOURCE_TABLE}' not found!")
            print(f"Available tables: {table_names}")
            return 1
        
        print(f"✓ Database: {DB_FILE}")
        print(f"✓ Source table: {SOURCE_TABLE}")
        
        # Add classification columns if needed
        add_classification_columns(conn)
        
        # Get chunks that still need classification
        chunks_to_classify = get_chunks_to_classify(conn)
        
        if not chunks_to_classify:
            print("\n✓ All chunks already have classifications!")
        else:
            # Setup OpenAI
            try:
                client = setup_openai()
            except Exception as e:
                print(f"❌ OpenAI setup failed: {e}")
                return 1
            
            # Process and classify with OpenAI
            process_chunks(client, conn, chunks_to_classify)
        
        # Show statistics
        show_statistics(conn)
        
        print("\n" + "=" * 70)
        print("✓ CLASSIFICATION COMPLETE")
        print("=" * 70)
        print(f"\nResults saved to table: '{SOURCE_TABLE}'")
        print(f"Classifications stored in column: 'frame_label'")
        print(f"Timestamps stored in column: 'classification_timestamp'")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()


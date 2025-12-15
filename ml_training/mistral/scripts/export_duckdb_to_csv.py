#!/usr/bin/env python3
"""
Exportiert annotierte Daten aus DuckDB in CSV-Dateien für Training
"""
import duckdb
import argparse
from pathlib import Path


def export_to_csv(db_path: str, output_dir: str):
    """Exportiert Training und Test Daten aus DuckDB zu CSV"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Verbinde mit DuckDB: {db_path}")
    conn = duckdb.connect(db_path)
    
    # Training Data exportieren
    train_output = output_path / "training_data_export.csv"
    print(f"Exportiere Training-Daten nach {train_output}...")
    
    conn.execute(f"""
        COPY (
            SELECT 
                chunk_text,
                frame_label,
                confidence,
                notes
            FROM annotations
            WHERE dataset_split = 'train'
            ORDER BY chunk_id
        ) TO '{train_output}' (HEADER, DELIMITER ',', QUOTE '"')
    """)
    
    # Test Data exportieren
    test_output = output_path / "test_data_export.csv"
    print(f"Exportiere Test-Daten nach {test_output}...")
    
    conn.execute(f"""
        COPY (
            SELECT 
                chunk_text,
                frame_label,
                confidence,
                notes
            FROM annotations
            WHERE dataset_split = 'test'
            ORDER BY chunk_id
        ) TO '{test_output}' (HEADER, DELIMITER ',', QUOTE '"')
    """)
    
    # Statistiken anzeigen
    train_count = conn.execute("SELECT COUNT(*) FROM annotations WHERE dataset_split = 'train'").fetchone()[0]
    test_count = conn.execute("SELECT COUNT(*) FROM annotations WHERE dataset_split = 'test'").fetchone()[0]
    
    print(f"\n✓ Export abgeschlossen!")
    print(f"  Train Samples: {train_count:,}")
    print(f"  Test Samples:  {test_count:,}")
    print(f"  Dateien: {output_path}")
    
    # Frame-Verteilung anzeigen
    print("\nFrame-Verteilung (Train):")
    train_dist = conn.execute("""
        SELECT frame_label, COUNT(*) as count 
        FROM annotations 
        WHERE dataset_split = 'train'
        GROUP BY frame_label 
        ORDER BY count DESC
    """).fetchall()
    
    for label, count in train_dist:
        percentage = (count / train_count * 100) if train_count > 0 else 0
        print(f"  {label:15}: {count:4} ({percentage:5.1f}%)")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Exportiert annotierte Daten aus DuckDB zu CSV')
    parser.add_argument('--input-db', required=True, help='Pfad zur DuckDB Datenbank')
    parser.add_argument('--output-dir', default='data', help='Ausgabeverzeichnis für CSV-Dateien')
    
    args = parser.parse_args()
    
    # Prüfe Input-Datei
    if not Path(args.input_db).exists():
        print(f"✗ DuckDB Datei {args.input_db} nicht gefunden!")
        return
    
    try:
        export_to_csv(args.input_db, args.output_dir)
    except Exception as e:
        print(f"✗ Fehler beim Export: {e}")
        raise


if __name__ == "__main__":
    main()

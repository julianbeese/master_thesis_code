#!/usr/bin/env python3
"""
Filtert Brexit-relevante Reden aus der klassifizierten Datenbank
Erstellt eine neue Datenbank mit nur den als Brexit-relevant markierten Speeches
"""

import duckdb
from pathlib import Path

# Konfiguration
INPUT_DB = "../data/processed/debates_brexit_classified.duckdb"
OUTPUT_DB = "../data/processed/debates_brexit_filtered.duckdb"

def get_table_schema(conn, table_name):
    """Holt das Schema einer Tabelle"""
    schema_info = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    columns = []
    for row in schema_info:
        col_name = row[1]
        col_type = row[2]
        is_nullable = row[3] == 0
        default_value = row[4]
        is_pk = row[5] == 1
        
        col_def = f"{col_name} {col_type}"
        if not is_nullable:
            col_def += " NOT NULL"
        if default_value is not None:
            col_def += f" DEFAULT {default_value}"
        if is_pk:
            col_def += " PRIMARY KEY"
        
        columns.append(col_def)
    
    return columns

def create_table_from_schema(conn, table_name, columns):
    """Erstellt eine Tabelle basierend auf dem Schema"""
    create_sql = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(columns) + "\n)"
    conn.execute(create_sql)

def copy_table_data(conn_source, conn_target, table_name):
    """Kopiert alle Daten einer Tabelle"""
    print(f"  Kopiere {table_name}...")
    
    # Hole alle Daten
    data = conn_source.execute(f"SELECT * FROM {table_name}").fetchall()
    
    if not data:
        print(f"    → Keine Daten in {table_name}")
        return
    
    # Hole Spaltennamen
    columns = [row[1] for row in conn_source.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
    
    # Erstelle INSERT Statement
    placeholders = ", ".join(["?" for _ in columns])
    insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
    
    # Füge alle Zeilen ein
    for row in data:
        conn_target.execute(insert_sql, row)
    
    print(f"    ✓ {len(data):,} Zeilen kopiert")

def copy_filtered_speeches(conn_source, conn_target):
    """Kopiert nur Brexit-relevante Speeches"""
    print("  Kopiere gefilterte speeches (nur Brexit-relevante)...")
    
    # Hole nur Brexit-relevante Speeches
    brexit_speeches = conn_source.execute("""
        SELECT * FROM speeches 
        WHERE brexit_related = TRUE
    """).fetchall()
    
    if not brexit_speeches:
        print("    → Keine Brexit-relevanten Speeches gefunden")
        return
    
    # Hole Spaltennamen
    columns = [row[1] for row in conn_source.execute("PRAGMA table_info('speeches')").fetchall()]
    
    # Erstelle INSERT Statement
    placeholders = ", ".join(["?" for _ in columns])
    insert_sql = f"INSERT INTO speeches VALUES ({placeholders})"
    
    # Füge alle Brexit-relevanten Speeches ein
    for row in brexit_speeches:
        conn_target.execute(insert_sql, row)
    
    print(f"    ✓ {len(brexit_speeches):,} Brexit-relevante Speeches kopiert")

def get_related_debates_and_topics(conn_source, conn_target):
    """Holt alle Debatten und Topics, die zu den Brexit-Speeches gehören"""
    print("  Kopiere zugehörige debates und topics...")
    
    # Hole alle debate_ids der Brexit-Speeches
    debate_ids = conn_source.execute("""
        SELECT DISTINCT debate_id 
        FROM speeches 
        WHERE brexit_related = TRUE
    """).fetchall()
    
    if not debate_ids:
        print("    → Keine zugehörigen Debatten gefunden")
        return
    
    debate_id_list = [row[0] for row in debate_ids]
    print(f"    → {len(debate_id_list)} zugehörige Debatten gefunden")
    
    # Kopiere relevante Debatten
    for debate_id in debate_id_list:
        # Kopiere debate
        debate_data = conn_source.execute("SELECT * FROM debates WHERE debate_id = ?", [debate_id]).fetchall()
        if debate_data:
            columns = [row[1] for row in conn_source.execute("PRAGMA table_info('debates')").fetchall()]
            placeholders = ", ".join(["?" for _ in columns])
            insert_sql = f"INSERT INTO debates VALUES ({placeholders})"
            for row in debate_data:
                try:
                    conn_target.execute(insert_sql, row)
                except:
                    # Überspringe bei Duplikaten
                    pass
        
        # Kopiere zugehörige Topics
        topics_data = conn_source.execute("SELECT * FROM topics WHERE debate_id = ?", [debate_id]).fetchall()
        if topics_data:
            columns = [row[1] for row in conn_source.execute("PRAGMA table_info('topics')").fetchall()]
            placeholders = ", ".join(["?" for _ in columns])
            insert_sql = f"INSERT INTO topics VALUES ({placeholders})"
            for row in topics_data:
                try:
                    conn_target.execute(insert_sql, row)
                except:
                    # Überspringe bei Duplikaten
                    pass

def main():
    print("=" * 70)
    print("FILTERUNG VON BREXIT-RELEVANTEN REDEN")
    print("=" * 70)
    
    # Prüfe Input-Datenbank
    if not Path(INPUT_DB).exists():
        print(f"\n✗ Input-Datenbank {INPUT_DB} nicht gefunden!")
        return
    
    print(f"\n✓ Input-Datenbank gefunden: {INPUT_DB}")
    
    # Öffne Input-Datenbank
    conn_source = duckdb.connect(INPUT_DB, read_only=True)
    
    # Erstelle Output-Datenbank
    print(f"\nErstelle Output-Datenbank: {OUTPUT_DB}")
    conn_target = duckdb.connect(OUTPUT_DB)
    
    # Hole alle Tabellennamen
    tables = [row[0] for row in conn_source.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"Gefundene Tabellen: {', '.join(tables)}")
    
    # Kopiere Schema für alle Tabellen
    print("\nErstelle Tabellenschema...")
    for table_name in tables:
        if table_name == "speeches":
            # Speeches wird separat behandelt
            continue
            
        print(f"  Erstelle Schema für {table_name}...")
        columns = get_table_schema(conn_source, table_name)
        create_table_from_schema(conn_target, table_name, columns)
    
    # Erstelle speeches Tabelle (wird separat gefüllt)
    print("  Erstelle Schema für speeches...")
    columns = get_table_schema(conn_source, "speeches")
    create_table_from_schema(conn_target, "speeches", columns)
    
    # Kopiere Daten
    print("\nKopiere Daten...")
    
    # Kopiere alle debates und topics (werden später gefiltert)
    for table_name in ["debates", "topics"]:
        if table_name in tables:
            copy_table_data(conn_source, conn_target, table_name)
    
    # Kopiere nur Brexit-relevante Speeches
    copy_filtered_speeches(conn_source, conn_target)
    
    # Filtere debates und topics auf die, die zu Brexit-Speeches gehören
    print("\nFiltere debates und topics...")
    conn_target.execute("DELETE FROM debates")
    conn_target.execute("DELETE FROM topics")
    get_related_debates_and_topics(conn_source, conn_target)
    
    # Statistiken
    print("\n" + "=" * 70)
    print("STATISTIKEN")
    print("=" * 70)
    
    # Input-Statistiken
    total_speeches_input = conn_source.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]
    brexit_speeches_input = conn_source.execute("SELECT COUNT(*) FROM speeches WHERE brexit_related = TRUE").fetchone()[0]
    
    # Output-Statistiken
    total_speeches_output = conn_target.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]
    total_debates_output = conn_target.execute("SELECT COUNT(*) FROM debates").fetchone()[0]
    total_topics_output = conn_target.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    
    print(f"\nInput-Datenbank ({INPUT_DB}):")
    print(f"  Gesamt Speeches:           {total_speeches_input:,}")
    print(f"  Brexit-relevante Speeches: {brexit_speeches_input:,}")
    print(f"  Brexit-Rate:               {brexit_speeches_input/max(total_speeches_input,1)*100:.1f}%")
    
    print(f"\nOutput-Datenbank ({OUTPUT_DB}):")
    print(f"  Speeches:                  {total_speeches_output:,}")
    print(f"  Debatten:                  {total_debates_output:,}")
    print(f"  Topics:                    {total_topics_output:,}")
    
    # Schließe Verbindungen
    conn_source.close()
    conn_target.close()
    
    print(f"\n✓ Gefilterte Datenbank erstellt: {OUTPUT_DB}")
    print(f"✓ Nur Brexit-relevante Speeches wurden übertragen")

if __name__ == "__main__":
    main()

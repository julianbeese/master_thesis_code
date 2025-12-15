#!/usr/bin/env python3
"""
Erstellt eine Analyse-Datenbank mit nur Brexit-relevanten Speeches
Führt eine grobe EDA (Exploratory Data Analysis) durch
"""

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import numpy as np

# Konfiguration
INPUT_DB = "/Users/julianbeese/Developer/Master/data_collection/data/processed/debates_brexit_keyword_filtered_final.duckdb"
OUTPUT_DB = "/Users/julianbeese/Developer/Master/data_collection/data/processed/brexit_analysis.duckdb"

def create_brexit_only_database():
    """Erstellt eine Datenbank mit nur Brexit-relevanten Speeches"""
    print("=" * 80)
    print("ERSTELLE BREXIT-ANALYSE DATENBANK")
    print("=" * 80)
    
    # Prüfe Input-Datenbank
    if not Path(INPUT_DB).exists():
        print(f"✗ Input-Datenbank {INPUT_DB} nicht gefunden!")
        return None
    
    print(f"✓ Input-Datenbank gefunden: {INPUT_DB}")
    
    # Lösche Output-Datenbank falls vorhanden
    if Path(OUTPUT_DB).exists():
        Path(OUTPUT_DB).unlink()
        print(f"✓ Alte Output-Datenbank gelöscht")
    
    # Öffne Verbindungen
    conn_source = duckdb.connect(INPUT_DB, read_only=True)
    conn_target = duckdb.connect(OUTPUT_DB)
    
    try:
        # Prüfe ob Brexit-Spalten existieren
        columns = [row[1] for row in conn_source.execute("PRAGMA table_info('speeches')").fetchall()]
        if 'brexit_related' not in columns:
            print("✗ Input-Datenbank wurde noch nicht verarbeitet!")
            return None
        
        # Hole Brexit-relevante Speeches
        print("Hole Brexit-relevante Speeches...")
        brexit_speeches = conn_source.execute("""
            SELECT * FROM speeches 
            WHERE brexit_related = TRUE
        """).fetchall()
        
        if not brexit_speeches:
            print("✗ Keine Brexit-relevanten Speeches gefunden!")
            return None
        
        print(f"✓ {len(brexit_speeches):,} Brexit-relevante Speeches gefunden")
        
        # Hole zugehörige Debatten und Topics
        print("Hole zugehörige Debatten und Topics...")
        
        # Hole alle debate_ids der Brexit-Speeches
        debate_ids = conn_source.execute("""
            SELECT DISTINCT debate_id 
            FROM speeches 
            WHERE brexit_related = TRUE
        """).fetchall()
        
        debate_id_list = [row[0] for row in debate_ids]
        print(f"✓ {len(debate_id_list)} zugehörige Debatten gefunden")
        
        # Kopiere relevante Debatten
        for debate_id in debate_id_list:
            debate_data = conn_source.execute("SELECT * FROM debates WHERE debate_id = ?", [debate_id]).fetchall()
            if debate_data:
                # Erstelle debates Tabelle falls nicht vorhanden
                if not conn_target.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'debates'").fetchone()[0]:
                    conn_target.execute("""
                        CREATE TABLE debates (
                            debate_id VARCHAR,
                            date DATE,
                            file_name VARCHAR,
                            major_heading_text VARCHAR,
                            colnum VARCHAR,
                            time VARCHAR,
                            url VARCHAR
                        )
                    """)
                
                columns = [row[1] for row in conn_source.execute("PRAGMA table_info('debates')").fetchall()]
                placeholders = ", ".join(["?" for _ in columns])
                insert_sql = f"INSERT INTO debates VALUES ({placeholders})"
                for row in debate_data:
                    try:
                        conn_target.execute(insert_sql, row)
                    except:
                        pass  # Überspringe Duplikate
        
        # Kopiere relevante Topics
        for debate_id in debate_id_list:
            topics_data = conn_source.execute("SELECT * FROM topics WHERE debate_id = ?", [debate_id]).fetchall()
            if topics_data:
                # Erstelle topics Tabelle falls nicht vorhanden
                if not conn_target.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'topics'").fetchone()[0]:
                    conn_target.execute("""
                        CREATE TABLE topics (
                            topic_id VARCHAR,
                            debate_id VARCHAR,
                            minor_heading_text VARCHAR,
                            colnum VARCHAR,
                            time VARCHAR,
                            url VARCHAR
                        )
                    """)
                
                columns = [row[1] for row in conn_source.execute("PRAGMA table_info('topics')").fetchall()]
                placeholders = ", ".join(["?" for _ in columns])
                insert_sql = f"INSERT INTO topics VALUES ({placeholders})"
                for row in topics_data:
                    try:
                        conn_target.execute(insert_sql, row)
                    except:
                        pass  # Überspringe Duplikate
        
        # Kopiere Brexit-Speeches
        print("Kopiere Brexit-Speeches...")
        conn_target.execute("""
            CREATE TABLE speeches (
                speech_id VARCHAR,
                topic_id VARCHAR,
                debate_id VARCHAR,
                speaker_name VARCHAR,
                person_id VARCHAR,
                speaker_office VARCHAR,
                speech_type VARCHAR,
                oral_qnum VARCHAR,
                colnum VARCHAR,
                time VARCHAR,
                url VARCHAR,
                speech_text VARCHAR,
                paragraph_count INTEGER,
                brexit_related BOOLEAN,
                confidence_score FLOAT,
                lists_found_keywords VARCHAR
            )
        """)
        
        columns = [row[1] for row in conn_source.execute("PRAGMA table_info('speeches')").fetchall()]
        placeholders = ", ".join(["?" for _ in columns])
        insert_sql = f"INSERT INTO speeches VALUES ({placeholders})"
        
        for row in brexit_speeches:
            conn_target.execute(insert_sql, row)
        
        print(f"✓ {len(brexit_speeches):,} Brexit-Speeches kopiert")
        
        # Statistiken
        total_debates = conn_target.execute("SELECT COUNT(*) FROM debates").fetchone()[0]
        total_topics = conn_target.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
        total_speeches = conn_target.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]
        
        print(f"\nBrexit-Analyse-Datenbank erstellt:")
        print(f"  - {total_debates:,} Debatten")
        print(f"  - {total_topics:,} Topics")
        print(f"  - {total_speeches:,} Brexit-Speeches")
        
        return conn_target
        
    finally:
        conn_source.close()

def perform_eda(conn):
    """Führt eine Exploratory Data Analysis durch"""
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # Grundlegende Statistiken
    print("\n1. GRUNDLEGENDE STATISTIKEN")
    print("-" * 40)
    
    total_speeches = conn.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]
    total_debates = conn.execute("SELECT COUNT(*) FROM debates").fetchone()[0]
    
    print(f"Gesamt Brexit-Speeches: {total_speeches:,}")
    print(f"Gesamt Debatten: {total_debates:,}")
    print(f"Durchschnitt Speeches pro Debatte: {total_speeches/total_debates:.1f}")
    
    # Confidence Score Statistiken
    confidence_stats = conn.execute("""
        SELECT 
            MIN(confidence_score) as min_conf,
            MAX(confidence_score) as max_conf,
            AVG(confidence_score) as avg_conf,
            COUNT(*) as count
        FROM speeches
    """).fetchone()
    
    print(f"\nConfidence Score Statistiken:")
    print(f"  Minimum: {confidence_stats[0]:.3f}")
    print(f"  Maximum: {confidence_stats[1]:.3f}")
    print(f"  Durchschnitt: {confidence_stats[2]:.3f}")
    
    # Datumsanalyse
    print("\n2. ZEITLICHE ANALYSE")
    print("-" * 40)
    
    # Datumsbereich
    date_range = conn.execute("""
        SELECT 
            MIN(date) as earliest_date,
            MAX(date) as latest_date
        FROM debates
    """).fetchone()
    
    print(f"Zeitraum: {date_range[0]} bis {date_range[1]}")
    
    # Speeches pro Jahr
    yearly_stats = conn.execute("""
        SELECT 
            EXTRACT(YEAR FROM d.date) as year,
            COUNT(s.speech_id) as speech_count,
            COUNT(DISTINCT d.debate_id) as debate_count
        FROM speeches s
        JOIN debates d ON s.debate_id = d.debate_id
        GROUP BY EXTRACT(YEAR FROM d.date)
        ORDER BY year
    """).fetchall()
    
    print(f"\nSpeeches pro Jahr:")
    for year, speech_count, debate_count in yearly_stats:
        print(f"  {int(year)}: {speech_count:,} Speeches in {debate_count:,} Debatten")
    
    # Top Keywords
    print("\n3. KEYWORD-ANALYSE")
    print("-" * 40)
    
    # Häufigste Keywords
    keyword_stats = conn.execute("""
        SELECT 
            lists_found_keywords,
            COUNT(*) as count
        FROM speeches 
        WHERE lists_found_keywords != '' AND lists_found_keywords IS NOT NULL
        GROUP BY lists_found_keywords
        ORDER BY count DESC
        LIMIT 10
    """).fetchall()
    
    print("Häufigste Keyword-Kombinationen:")
    for keywords, count in keyword_stats:
        print(f"  '{keywords}': {count:,} Speeches")
    
    # Speaker-Analyse
    print("\n4. SPEAKER-ANALYSE")
    print("-" * 40)
    
    # Top Speakers
    speaker_stats = conn.execute("""
        SELECT 
            speaker_name,
            COUNT(*) as speech_count,
            AVG(confidence_score) as avg_confidence
        FROM speeches 
        WHERE speaker_name IS NOT NULL AND speaker_name != ''
        GROUP BY speaker_name
        ORDER BY speech_count DESC
        LIMIT 10
    """).fetchall()
    
    print("Top 10 Speakers (nach Anzahl Brexit-Speeches):")
    for speaker, count, avg_conf in speaker_stats:
        print(f"  {speaker}: {count:,} Speeches (Ø Confidence: {avg_conf:.3f})")
    
    # Confidence Score Verteilung
    print("\n5. CONFIDENCE SCORE VERTEILUNG")
    print("-" * 40)
    
    conf_buckets = conn.execute("""
        SELECT 
            CASE 
                WHEN confidence_score < 0.1 THEN '0.0-0.1'
                WHEN confidence_score < 0.2 THEN '0.1-0.2'
                WHEN confidence_score < 0.3 THEN '0.2-0.3'
                WHEN confidence_score < 0.4 THEN '0.3-0.4'
                WHEN confidence_score < 0.5 THEN '0.4-0.5'
                WHEN confidence_score < 0.6 THEN '0.5-0.6'
                WHEN confidence_score < 0.7 THEN '0.6-0.7'
                WHEN confidence_score < 0.8 THEN '0.7-0.8'
                WHEN confidence_score < 0.9 THEN '0.8-0.9'
                ELSE '0.9-1.0'
            END as bucket,
            COUNT(*) as count
        FROM speeches
        GROUP BY bucket
        ORDER BY bucket
    """).fetchall()
    
    print("Confidence Score Verteilung:")
    for bucket, count in conf_buckets:
        percentage = (count / total_speeches) * 100
        print(f"  {bucket}: {count:,} Speeches ({percentage:.1f}%)")
    
    # Text-Längen-Analyse
    print("\n6. TEXT-LÄNGEN-ANALYSE")
    print("-" * 40)
    
    text_stats = conn.execute("""
        SELECT 
            MIN(LENGTH(speech_text)) as min_length,
            MAX(LENGTH(speech_text)) as max_length,
            AVG(LENGTH(speech_text)) as avg_length,
            AVG(paragraph_count) as avg_paragraphs
        FROM speeches
    """).fetchone()
    
    print(f"Text-Längen:")
    print(f"  Minimum: {text_stats[0]:,} Zeichen")
    print(f"  Maximum: {text_stats[1]:,} Zeichen")
    print(f"  Durchschnitt: {text_stats[2]:.0f} Zeichen")
    print(f"  Durchschnittliche Absätze: {text_stats[3]:.1f}")

def create_visualizations(conn):
    """Erstellt Visualisierungen der Daten"""
    print("\n7. ERSTELLE VISUALISIERUNGEN")
    print("-" * 40)
    
    try:
        # Setze Matplotlib Style
        plt.style.use('default')
        
        # 1. Speeches pro Jahr
        yearly_data = conn.execute("""
            SELECT 
                EXTRACT(YEAR FROM d.date) as year,
                COUNT(s.speech_id) as speech_count
            FROM speeches s
            JOIN debates d ON s.debate_id = d.debate_id
            GROUP BY EXTRACT(YEAR FROM d.date)
            ORDER BY year
        """).fetchall()
        
        if yearly_data:
            years = [int(row[0]) for row in yearly_data]
            counts = [row[1] for row in yearly_data]
            
            plt.figure(figsize=(12, 6))
            plt.plot(years, counts, marker='o', linewidth=2, markersize=6)
            plt.title('Brexit-Speeches pro Jahr', fontsize=14, fontweight='bold')
            plt.xlabel('Jahr')
            plt.ylabel('Anzahl Speeches')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('/Users/julianbeese/Developer/Master/data_collection/brexit_speeches_per_year.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Diagramm gespeichert: brexit_speeches_per_year.png")
        
        # 2. Confidence Score Histogramm
        conf_data = conn.execute("SELECT confidence_score FROM speeches").fetchall()
        if conf_data:
            scores = [row[0] for row in conf_data]
            
            plt.figure(figsize=(10, 6))
            plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
            plt.title('Verteilung der Confidence Scores', fontsize=14, fontweight='bold')
            plt.xlabel('Confidence Score')
            plt.ylabel('Anzahl Speeches')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('/Users/julianbeese/Developer/Master/data_collection/confidence_score_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Diagramm gespeichert: confidence_score_distribution.png")
        
        # 3. Top Keywords Word Cloud (falls möglich)
        try:
            from wordcloud import WordCloud
            
            # Sammle alle Keywords
            all_keywords = []
            keyword_data = conn.execute("""
                SELECT lists_found_keywords 
                FROM speeches 
                WHERE lists_found_keywords != '' AND lists_found_keywords IS NOT NULL
            """).fetchall()
            
            for row in keyword_data:
                keywords = row[0].split(', ')
                all_keywords.extend(keywords)
            
            if all_keywords:
                # Zähle Keywords
                from collections import Counter
                keyword_counts = Counter(all_keywords)
                
                # Erstelle Word Cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keyword_counts)
                
                plt.figure(figsize=(12, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Häufigste Brexit-Keywords', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig('/Users/julianbeese/Developer/Master/data_collection/brexit_keywords_wordcloud.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ Word Cloud gespeichert: brexit_keywords_wordcloud.png")
        
        except ImportError:
            print("⚠ WordCloud nicht verfügbar - überspringe Word Cloud")
        
    except Exception as e:
        print(f"⚠ Fehler beim Erstellen der Visualisierungen: {e}")

def main():
    """Hauptfunktion"""
    # Erstelle Brexit-Analyse-Datenbank
    conn = create_brexit_only_database()
    
    if conn is None:
        print("✗ Fehler beim Erstellen der Analyse-Datenbank!")
        return
    
    try:
        # Führe EDA durch
        perform_eda(conn)
        
        # Erstelle Visualisierungen
        create_visualizations(conn)
        
        print(f"\n✓ Analyse abgeschlossen!")
        print(f"✓ Analyse-Datenbank: {OUTPUT_DB}")
        print(f"✓ Visualisierungen im Projektverzeichnis gespeichert")
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()

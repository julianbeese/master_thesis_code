#!/usr/bin/env python3
"""
Beispiel-Abfragen für die Debates-Datenbank
"""

import duckdb

DB_FILE = "../data/processed/debates.duckdb"


def main():
    conn = duckdb.connect(DB_FILE, read_only=True)

    print("=" * 70)
    print("DEBATES DATENBANK - BEISPIEL-ABFRAGEN")
    print("=" * 70)

    # 1. Übersicht
    print("\n1. DATENBANK-ÜBERSICHT")
    print("-" * 70)

    debates_count = conn.execute("SELECT COUNT(*) FROM debates").fetchone()[0]
    topics_count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    speeches_count = conn.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]

    print(f"Debatten:  {debates_count:,}")
    print(f"Themen:    {topics_count:,}")
    print(f"Reden:     {speeches_count:,}")

    # 2. Zeitraum
    print("\n2. ZEITRAUM")
    print("-" * 70)

    date_range = conn.execute("""
        SELECT MIN(date) as min_date, MAX(date) as max_date
        FROM debates
    """).fetchone()

    print(f"Von: {date_range[0]}")
    print(f"Bis: {date_range[1]}")

    # 3. Top 10 Redner
    print("\n3. TOP 10 REDNER (nach Anzahl Reden)")
    print("-" * 70)

    top_speakers = conn.execute("""
        SELECT
            speaker_name,
            COUNT(*) as speech_count,
            COUNT(DISTINCT DATE_TRUNC('month', d.date)) as active_months
        FROM speeches s
        JOIN debates d ON s.debate_id = d.debate_id
        WHERE speaker_name IS NOT NULL
        GROUP BY speaker_name
        ORDER BY speech_count DESC
        LIMIT 10
    """).fetchall()

    for i, (name, count, months) in enumerate(top_speakers, 1):
        print(f"{i:2}. {name:30} {count:6,} Reden ({months} Monate aktiv)")

    # 4. Reden pro Jahr
    print("\n4. REDEN PRO JAHR")
    print("-" * 70)

    speeches_per_year = conn.execute("""
        SELECT
            YEAR(d.date) as year,
            COUNT(*) as speech_count,
            COUNT(DISTINCT d.debate_id) as debate_count
        FROM speeches s
        JOIN debates d ON s.debate_id = d.debate_id
        GROUP BY year
        ORDER BY year
    """).fetchall()

    for year, speeches, debates in speeches_per_year:
        print(f"{year}: {speeches:6,} Reden in {debates:4} Debatten")

    # 5. Häufigste Themen
    print("\n5. TOP 10 THEMEN (nach Anzahl Reden)")
    print("-" * 70)

    top_topics = conn.execute("""
        SELECT
            t.minor_heading_text,
            COUNT(s.speech_id) as speech_count
        FROM topics t
        JOIN speeches s ON s.topic_id = t.topic_id
        WHERE t.minor_heading_text IS NOT NULL
        GROUP BY t.minor_heading_text
        ORDER BY speech_count DESC
        LIMIT 10
    """).fetchall()

    for i, (topic, count) in enumerate(top_topics, 1):
        topic_short = topic[:50] + "..." if len(topic) > 50 else topic
        print(f"{i:2}. {topic_short:53} {count:5,} Reden")

    # 6. Durchschnittliche Redelänge
    print("\n6. STATISTIKEN ZUR REDELÄNGE")
    print("-" * 70)

    speech_stats = conn.execute("""
        SELECT
            AVG(LENGTH(speech_text)) as avg_length,
            MIN(LENGTH(speech_text)) as min_length,
            MAX(LENGTH(speech_text)) as max_length,
            AVG(paragraph_count) as avg_paragraphs
        FROM speeches
        WHERE speech_text IS NOT NULL AND LENGTH(speech_text) > 0
    """).fetchone()

    print(f"Durchschnittliche Länge: {speech_stats[0]:,.0f} Zeichen")
    print(f"Kürzeste Rede:          {speech_stats[1]:,} Zeichen")
    print(f"Längste Rede:           {speech_stats[2]:,} Zeichen")
    print(f"Durchschn. Paragraphen: {speech_stats[3]:.1f}")

    # 7. Beispiel einer Rede
    print("\n7. BEISPIEL EINER REDE")
    print("-" * 70)

    example_speech = conn.execute("""
        SELECT
            s.speaker_name,
            d.date,
            d.major_heading_text,
            t.minor_heading_text,
            LEFT(s.speech_text, 200) as speech_preview
        FROM speeches s
        JOIN debates d ON s.debate_id = d.debate_id
        LEFT JOIN topics t ON s.topic_id = t.topic_id
        WHERE s.speaker_name IS NOT NULL
          AND LENGTH(s.speech_text) > 100
        ORDER BY RANDOM()
        LIMIT 1
    """).fetchone()

    if example_speech:
        print(f"Redner:  {example_speech[0]}")
        print(f"Datum:   {example_speech[1]}")
        print(f"Debatte: {example_speech[2]}")
        print(f"Thema:   {example_speech[3]}")
        print(f"\nText (Auszug):\n{example_speech[4]}...")

    conn.close()
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

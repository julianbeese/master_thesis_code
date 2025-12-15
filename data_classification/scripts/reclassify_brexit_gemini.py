#!/usr/bin/env python3
"""
Brexit-Reklassifizierung mit Gemini 2.5-flash
Prüft alle Reden in debates_brexit_filtered_min20words.duckdb auf Brexit-Bezug
und erstellt eine neue Datenbank mit nur Brexit-relevanten Reden.
"""

import duckdb
import re
import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import time

# Lade .env Datei
load_dotenv()

# Konfiguration
DB_FILE = "data/processed/debates_brexit_filtered_min20words.duckdb"
OUTPUT_DB = "data/processed/debates_brexit_gemini_classified.duckdb"
GEMINI_MODEL = "gemini-2.5-flash"

# Cost Tracking (Gemini 2.5 Flash Pricing)
COST_LIMIT = 20.00  # Maximum $20
INPUT_PRICE_PER_1M = 0.075  # $0.075 per 1M input tokens
OUTPUT_PRICE_PER_1M = 0.30  # $0.30 per 1M output tokens

# Rate Limiting: 10 Requests/Minute für Gemini 2.5-flash
REQUEST_DELAY = 6.0  # Sekunden zwischen API Calls


def analyze_speech_with_gemini(speech_text, speaker_name, debate_name, date, api_key, retry_count=0):
    """
    Analysiert eine einzelne Rede mit Gemini auf Brexit-Bezug.
    
    Returns:
        tuple: (has_brexit_relation: bool, confidence: float, reasoning: str, 
                input_tokens: int, output_tokens: int)
    """
    if not api_key:
        raise ValueError("GEMINI_API_KEY nicht gesetzt")

    # Konfiguriere Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    # Erstelle Prompt
    prompt = f"""You are analyzing a UK parliamentary House of Commons speech to determine if it relates to Brexit.

**Speech Information:**
- Speaker: {speaker_name}
- Debate Topic: {debate_name}
- Date: {date}

**Speech Text:**
{speech_text[:6000]}

**Task:**
Analyze whether this speech has a significant relation to Brexit (the UK's withdrawal from the European Union).

Consider:
- Direct mentions of Brexit, EU exit, Article 50, withdrawal
- Discussions about EU membership, sovereignty, immigration from EU context
- Trade agreements in context of leaving EU
- Northern Ireland border issues related to Brexit
- EU regulations/laws in the context of Brexit
- etc.

**Response format (JSON):**
{{
  "has_brexit_relation": true/false,
  "confidence": 0.0-1.0 (0 = no relation to Brexit, 1 = very likely relation to Brexit),
  "reasoning": "One sentence explanation"
}}

Respond ONLY with the JSON object, no additional text."""

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Extrahiere Token-Zählung
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count

        # Parse JSON response
        import json
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return (
                result.get('has_brexit_relation', False),
                float(result.get('confidence', 0.0)),
                result.get('reasoning', ''),
                input_tokens,
                output_tokens
            )
        else:
            print(f"  ⚠ Konnte JSON nicht parsen: {response_text[:100]}")
            return False, 0.0, "Failed to parse response", input_tokens, output_tokens

    except Exception as e:
        error_str = str(e)

        # Rate Limit Error (429) - Retry mit exponential backoff
        if "429" in error_str or "quota" in error_str.lower():
            if retry_count < 5:  # Max 5 Retries
                wait_time = 6 * (2 ** retry_count)
                print(f"  ⚠ Rate Limit erreicht, warte {wait_time}s und versuche erneut... (Versuch {retry_count + 1}/5)")
                time.sleep(wait_time)
                return analyze_speech_with_gemini(speech_text, speaker_name, debate_name, date, api_key, retry_count + 1)
            else:
                print(f"  ✗ Rate Limit nach 5 Versuchen nicht behoben")
                return False, 0.0, "Rate Limit Error after retries", 0, 0

        print(f"  ✗ Gemini API Fehler: {e}")
        return False, 0.0, f"API Error: {str(e)}", 0, 0


def setup_output_database():
    """Erstellt die Output-Datenbank mit vereinfachtem Schema."""
    print(f"Erstelle neue Output-Datenbank: {OUTPUT_DB}")
    
    # Lösche existierende Output-DB falls vorhanden
    if Path(OUTPUT_DB).exists():
        Path(OUTPUT_DB).unlink()
        print(f"  Alte Datenbank gelöscht")
    
    conn_out = duckdb.connect(OUTPUT_DB)
    
    # Erstelle Tabellen mit vereinfachtem Schema
    conn_out.execute("""
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
    
    conn_out.execute("""
        CREATE TABLE topics (
            topic_id VARCHAR,
            debate_id VARCHAR,
            minor_heading_text VARCHAR,
            colnum VARCHAR,
            time VARCHAR,
            url VARCHAR
        )
    """)
    
    conn_out.execute("""
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
            brexit_confidence FLOAT,
            brexit_reasoning VARCHAR
        )
    """)
    
    conn_out.commit()
    print("  Tabellen erstellt")
    
    return conn_out


def copy_debate_and_topic_if_needed(conn_source, conn_out, debate_id, copied_debates, copied_topics):
    """Kopiert Debate und zugehörige Topics falls noch nicht kopiert."""
    
    # Kopiere Debate falls noch nicht kopiert
    if debate_id not in copied_debates:
        debate_data = conn_source.execute(
            "SELECT * FROM debates WHERE debate_id = ?", 
            [debate_id]
        ).fetchone()
        
        if debate_data:
            conn_out.execute(
                "INSERT INTO debates VALUES (?, ?, ?, ?, ?, ?, ?)",
                debate_data
            )
            copied_debates.add(debate_id)
    
    # Kopiere alle Topics dieser Debate falls noch nicht kopiert
    topics = conn_source.execute(
        "SELECT * FROM topics WHERE debate_id = ?",
        [debate_id]
    ).fetchall()
    
    for topic_data in topics:
        topic_id = topic_data[0]
        if topic_id not in copied_topics:
            conn_out.execute(
                "INSERT INTO topics VALUES (?, ?, ?, ?, ?, ?)",
                topic_data
            )
            copied_topics.add(topic_id)


def main():
    print("=" * 70)
    print("BREXIT-REKLASSIFIZIERUNG MIT GEMINI 2.5-FLASH")
    print("=" * 70)

    # Prüfe API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n✗ FEHLER: GEMINI_API_KEY Umgebungsvariable nicht gesetzt!")
        print("Setze den API Key mit: export GEMINI_API_KEY='your-key-here'")
        return

    print(f"\n✓ Gemini API Key gefunden")
    print(f"✓ Verwende Modell: {GEMINI_MODEL}\n")

    # Verbinde mit Quelldatenbank
    print(f"Öffne Quelldatenbank: {DB_FILE}")
    if not Path(DB_FILE).exists():
        print(f"✗ Datenbank {DB_FILE} nicht gefunden!")
        return

    conn_source = duckdb.connect(DB_FILE, read_only=True)

    # Erstelle Output-Datenbank
    conn_out = setup_output_database()

    # Hole alle Reden mit zugehörigen Informationen
    speeches = conn_source.execute("""
        SELECT 
            s.speech_id,
            s.topic_id,
            s.debate_id,
            s.speaker_name,
            s.person_id,
            s.speaker_office,
            s.speech_type,
            s.oral_qnum,
            s.colnum,
            s.time,
            s.url,
            s.speech_text,
            s.paragraph_count,
            d.major_heading_text,
            d.date
        FROM speeches s
        JOIN debates d ON s.debate_id = d.debate_id
        WHERE s.speech_text IS NOT NULL
        ORDER BY d.date, s.speech_id
    """).fetchall()

    total_speeches = len(speeches)
    print(f"\nGefunden: {total_speeches:,} Reden zum Analysieren\n")
    print(f"💰 Kosten-Limit: ${COST_LIMIT:.2f}\n")
    print("Starte Klassifizierung...\n")

    # Statistiken
    total_processed = 0
    total_brexit_related = 0
    total_non_brexit = 0

    # Cost Tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    cost_limit_reached = False

    # Tracking für kopierte Debates/Topics
    copied_debates = set()
    copied_topics = set()

    last_api_call_time = 0

    # Verarbeite jede Rede
    for i, speech_data in enumerate(speeches, 1):
        (speech_id, topic_id, debate_id, speaker_name, person_id, speaker_office,
         speech_type, oral_qnum, colnum, speech_time, url, speech_text, 
         paragraph_count, debate_name, date) = speech_data

        if i % 10 == 0 or i == 1:
            print(f"[{i}/{total_speeches}] {date} - {speaker_name}")

        # Rate Limiting
        if last_api_call_time > 0:
            time_since_last_call = time.time() - last_api_call_time
            if time_since_last_call < REQUEST_DELAY:
                wait_time = REQUEST_DELAY - time_since_last_call
                if i % 10 == 0:
                    print(f"  ⏱️  Rate Limit: Warte {wait_time:.1f}s...")
                time.sleep(wait_time)

        # Analysiere mit Gemini
        last_api_call_time = time.time()
        has_brexit_relation, confidence, reasoning, input_tokens, output_tokens = analyze_speech_with_gemini(
            speech_text,
            speaker_name or "Unknown",
            debate_name or "Unknown Debate",
            date,
            api_key
        )

        # Update Cost Tracking
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_1M
        output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
        call_cost = input_cost + output_cost
        total_cost += call_cost

        if i % 10 == 0:
            print(f"  Brexit: {has_brexit_relation}, Confidence: {confidence:.2f}")
            print(f"  💰 Kosten: ${call_cost:.4f} (Gesamt: ${total_cost:.2f})")

        # Nur Brexit-bezogene Reden speichern
        if has_brexit_relation:
            # Kopiere Debate und Topics falls noch nicht kopiert
            copy_debate_and_topic_if_needed(conn_source, conn_out, debate_id, copied_debates, copied_topics)
            
            # Füge Rede ein
            conn_out.execute("""
                INSERT INTO speeches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                speech_id, topic_id, debate_id, speaker_name, person_id, speaker_office,
                speech_type, oral_qnum, colnum, speech_time, url, speech_text, 
                paragraph_count, has_brexit_relation, confidence, reasoning
            ])
            
            total_brexit_related += 1
        else:
            total_non_brexit += 1

        total_processed += 1

        # Commit alle 100 Reden
        if i % 100 == 0:
            conn_out.commit()
            print(f"  💾 Zwischenspeicherung (Brexit: {total_brexit_related}, Non-Brexit: {total_non_brexit})")

        # Prüfe Kosten-Limit
        if total_cost >= COST_LIMIT:
            print(f"\n⚠️  KOSTEN-LIMIT ERREICHT!")
            print(f"  Aktuell: ${total_cost:.2f} / Limit: ${COST_LIMIT:.2f}")
            print(f"  Stoppe Verarbeitung und speichere bisherige Ergebnisse...\n")
            cost_limit_reached = True
            break

    # Finale Speicherung
    conn_out.commit()

    # Zusammenfassung
    print("\n" + "=" * 70)
    if cost_limit_reached:
        print("ABGEBROCHEN - KOSTEN-LIMIT ERREICHT")
    else:
        print("FERTIG!")
    print("=" * 70)
    print(f"\nStatistiken:")
    print(f"  Reden analysiert:           {total_processed:,}")
    print(f"  Brexit-bezogen (behalten):  {total_brexit_related:,} ({total_brexit_related/max(total_processed,1)*100:.1f}%)")
    print(f"  Nicht Brexit (gelöscht):    {total_non_brexit:,} ({total_non_brexit/max(total_processed,1)*100:.1f}%)")

    # Datenbank-Statistiken
    saved_speeches = conn_out.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]
    saved_debates = conn_out.execute("SELECT COUNT(*) FROM debates").fetchone()[0]
    saved_topics = conn_out.execute("SELECT COUNT(*) FROM topics").fetchone()[0]

    print(f"\nIn Datenbank gespeichert:")
    print(f"  Reden:                      {saved_speeches:,}")
    print(f"  Debatten:                   {saved_debates:,}")
    print(f"  Topics:                     {saved_topics:,}")

    # Kosten-Zusammenfassung
    print(f"\n💰 Kosten-Übersicht:")
    print(f"  API Calls:                  {total_processed:,}")
    print(f"  Input Tokens:               {total_input_tokens:,}")
    print(f"  Output Tokens:              {total_output_tokens:,}")
    print(f"  Gesamtkosten:               ${total_cost:.2f}")
    if cost_limit_reached:
        print(f"  ⚠️  Limit erreicht bei:       ${COST_LIMIT:.2f}")

    conn_source.close()
    conn_out.close()

    print(f"\n✓ Ergebnisse gespeichert in: {OUTPUT_DB}")

    if cost_limit_reached:
        print(f"\n⚠️  HINWEIS: Verarbeitung wurde wegen Kosten-Limit gestoppt.")
        print(f"   Nur {total_processed:,} von {total_speeches:,} Reden wurden verarbeitet.")
        print(f"   Führe das Script erneut aus, um fortzufahren (bereits klassifizierte werden übersprungen).")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Brexit-Klassifizierung für Parlamentsreden
Analysiert Debatten auf Brexit-Bezug mittels Keyword- und LLM-Analyse
"""

import duckdb
import re
import os
from pathlib import Path
from collections import defaultdict
import google.generativeai as genai
from dotenv import load_dotenv

# Lade .env Datei
load_dotenv()

# Konfiguration
DB_FILE = "../data/processed/debates.duckdb"
OUTPUT_DB = "../data/processed/debates_brexit_classified.duckdb"
GEMINI_MODEL = "gemini-2.5-flash"  # Gemini 2.0 Flash

# Cost Tracking (Gemini 2.0 Flash Pricing)
COST_LIMIT = 20.00  # Maximum $20
INPUT_PRICE_PER_1M = 0.075  # $0.075 per 1M input tokens
OUTPUT_PRICE_PER_1M = 0.30  # $0.30 per 1M output tokens


# Brexit-Keywords mit Gewichtung
DIRECT_KEYWORDS = [
    "brexit", "leave campaign", "remain campaign", "article 50",
    "referendum", "eu referendum", "european referendum",
    "leave the eu", "leaving the eu", "exit from europe",
    "withdrawal agreement", "divorce bill", "transition period",
    "hard brexit", "soft brexit", "british exit", "eu exit", "no-deal brexit", "brexit-related"
]

INDIRECT_KEYWORDS = [
    "european union", "european community", "eu membership",
    "brussels", "strasbourg", "european commission",
    "european parliament", "eurozone", "single market",
    "customs union", "free movement", "schengen",
    "eu law", "eu regulation", "eu directive",
    "eu budget", "eu contribution", "sovereignty",
    "independence", "british sovereignty", "take back control",
    "immigration control", "border control",
    "trade agreement", "trade deal", "wto",
    "northern ireland protocol", "backstop", "irish border", "member state", "future relationship",
    "european treaty", "maastricht treaty", "partnership agreement",
    "economic partnership", "freedom of movement", "european integration"
]

# Kombiniere alle Keywords
ALL_KEYWORDS = DIRECT_KEYWORDS + INDIRECT_KEYWORDS


def analyze_keywords(text):
    """
    Schritt 1: Keyword-basierte Analyse
    Gibt Confidence Score (0-1) und gefundene Keywords zurück
    """
    if not text:
        return 0.0, []

    text_lower = text.lower()

    # Finde alle Keywords
    found_direct = []
    found_indirect = []

    for keyword in DIRECT_KEYWORDS:
        # Case-insensitive regex search
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            found_direct.append(keyword)

    for keyword in INDIRECT_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            found_indirect.append(keyword)

    # Berechne Confidence Score
    # Direkte Keywords: höhere Gewichtung
    direct_score = min(len(found_direct) * 0.3, 1)  # Max 0.7
    indirect_score = min(len(found_indirect) * 0.05, 0.3)  # Max 0.3

    confidence = min(direct_score + indirect_score, 1.0)

    all_found = found_direct + found_indirect

    return confidence, all_found


def analyze_with_gemini(debate_name, date, speeches_text, keywords_found, api_key, retry_count=0):
    """
    Schritt 2: LLM-basierte Analyse mit Gemini
    Gibt (has_brexit_relation: bool, confidence: float, reasoning: str, input_tokens: int, output_tokens: int) zurück
    """
    import time

    if not api_key:
        raise ValueError("GEMINI_API_KEY nicht gesetzt")

    # Konfiguriere Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    # Erstelle Prompt
    prompt = f"""You are analyzing UK parliamentary House of Commons debates to determine if they relate to Brexit.

**Debate Information:**
- Topic: {debate_name}
- Date: {date}
- Keywords found: {', '.join(keywords_found[:10]) if keywords_found else 'None'}

**Speech excerpts (first 5 speeches):**
{speeches_text[:8000]}

**Task:**
Analyze whether this debate has a significant relation to Brexit (the UK's withdrawal from the European Union).

Consider:
- Direct mentions of Brexit, EU exit, Article 50, withdrawal
- Discussions about EU membership, sovereignty, immigration from EU context
- Trade agreements in context of leaving EU
- Northern Ireland border issues related to Brexit
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
        # Extrahiere JSON aus möglicherweise formatiertem Text
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
                # Exponential backoff: 6, 12, 24, 48, 96 Sekunden
                wait_time = 6 * (2 ** retry_count)
                print(f"  ⚠ Rate Limit erreicht, warte {wait_time}s und versuche erneut... (Versuch {retry_count + 1}/5)")
                time.sleep(wait_time)
                return analyze_with_gemini(debate_name, date, speeches_text, keywords_found, api_key, retry_count + 1)
            else:
                print(f"  ✗ Rate Limit nach 5 Versuchen nicht behoben")
                return False, 0.0, f"Rate Limit Error after retries", 0, 0

        print(f"  ✗ Gemini API Fehler: {e}")
        return False, 0.0, f"API Error: {str(e)}", 0, 0


def combine_results(keyword_confidence, keyword_count, llm_has_relation, llm_confidence):
    """
    Schritt 3: Kombiniere Keyword- und LLM-Ergebnisse
    Gewichtung: 30% Keywords, 70% LLM
    """
    # Wenn keine Keywords gefunden wurden, LLM wird nicht aufgerufen
    if keyword_count == 0:
        return False, 0.0

    # Gewichtete Kombination
    combined_confidence = (0.3 * keyword_confidence) + (0.7 * llm_confidence)

    # Finale Entscheidung: Brexit-Bezug wenn combined_confidence > 0.5
    has_brexit_relation = combined_confidence > 0.5

    return has_brexit_relation, combined_confidence


def setup_output_database(conn_source):
    """Erstellt (oder aktualisiert) die Output-DB und stellt benötigtes Schema sicher."""
    print(f"Initialisiere Output-Datenbank: {OUTPUT_DB}")

    def table_exists(conn, table_name):
        return conn.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table_name]).fetchone()[0] > 0

    def get_columns(conn, table_name):
        return {row[1] for row in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()}

    # Erstelle/öffne Verbindung für Output
    conn_out = duckdb.connect(OUTPUT_DB)

    # Debates
    if not table_exists(conn_out, "debates"):
        print("  Erstelle und kopiere Tabelle: debates")
        conn_out.execute(
            """
            CREATE TABLE debates (
                debate_id VARCHAR,
                date DATE,
                file_name VARCHAR,
                major_heading_text VARCHAR,
                colnum VARCHAR,
                time VARCHAR,
                url VARCHAR
            )
            """
        )
        debates_data = conn_source.execute("SELECT * FROM debates").fetchall()
        for row in debates_data:
            conn_out.execute("INSERT INTO debates VALUES (?, ?, ?, ?, ?, ?, ?)", row)
    else:
        print("  Tabelle 'debates' vorhanden – überspringe Kopie")

    # Topics
    if not table_exists(conn_out, "topics"):
        print("  Erstelle und kopiere Tabelle: topics")
        conn_out.execute(
            """
            CREATE TABLE topics (
                topic_id VARCHAR,
                debate_id VARCHAR,
                minor_heading_text VARCHAR,
                colnum VARCHAR,
                time VARCHAR,
                url VARCHAR
            )
            """
        )
        topics_data = conn_source.execute("SELECT * FROM topics").fetchall()
        for row in topics_data:
            conn_out.execute("INSERT INTO topics VALUES (?, ?, ?, ?, ?, ?)", row)
    else:
        print("  Tabelle 'topics' vorhanden – überspringe Kopie")

    # Speeches
    if not table_exists(conn_out, "speeches"):
        print("  Erstelle und kopiere Tabelle: speeches")
        conn_out.execute(
            """
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
                paragraph_count INTEGER
            )
            """
        )
        speeches_data = conn_source.execute("SELECT * FROM speeches").fetchall()
        for row in speeches_data:
            conn_out.execute("INSERT INTO speeches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
    else:
        print("  Tabelle 'speeches' vorhanden – überspringe Kopie")

    # Stelle sicher, dass Brexit-Spalten existieren
    print("  Stelle Brexit-Klassifizierungsspalten sicher…")
    existing_cols = get_columns(conn_out, "speeches")
    required_cols = [
        ("brexit_related", "BOOLEAN DEFAULT FALSE"),
        ("brexit_confidence", "FLOAT DEFAULT 0.0"),
        ("brexit_keyword_confidence", "FLOAT DEFAULT 0.0"),
        ("brexit_llm_confidence", "FLOAT DEFAULT 0.0"),
        ("brexit_keywords_found", "VARCHAR"),
        ("brexit_llm_reasoning", "VARCHAR"),
    ]

    for col_name, col_def in required_cols:
        if col_name not in existing_cols:
            conn_out.execute(f"ALTER TABLE speeches ADD COLUMN {col_name} {col_def}")
            print(f"    + Spalte hinzugefügt: {col_name}")

    conn_out.commit()
    return conn_out


def main():
    print("=" * 70)
    print("BREXIT-KLASSIFIZIERUNG VON PARLAMENTSREDEN")
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
    conn_out = setup_output_database(conn_source)

    # Hole alle einzigartigen Debatten (debate_id + date)
    debates = conn_source.execute("""
        SELECT DISTINCT
            d.debate_id,
            d.date,
            d.major_heading_text
        FROM debates d
        WHERE d.major_heading_text IS NOT NULL
        ORDER BY d.date, d.debate_id
    """).fetchall()

    print(f"\nGefunden: {len(debates)} Debatten zum Analysieren\n")
    print(f"💰 Kosten-Limit: ${COST_LIMIT:.2f}\n")
    print("Starte Klassifizierung...\n")

    # Statistiken
    total_processed = 0
    total_with_keywords = 0
    total_llm_analyzed = 0
    total_brexit_related = 0

    # Cost Tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    cost_limit_reached = False

    # Rate Limiting: 10 Requests/Minute
    # Warte 6 Sekunden zwischen API Calls (=10 Requests/Minute)
    import time
    REQUEST_DELAY = 6.0  # Sekunden
    last_api_call_time = 0

    # Verarbeite jede Debatte
    for i, (debate_id, date, debate_name) in enumerate(debates, 1):
        print(f"[{i}/{len(debates)}] {date} - {debate_name[:50]}")

        # Resume: Überspringe bereits klassifizierte Debatten
        already_processed = conn_out.execute(
            """
            SELECT 1 FROM speeches
            WHERE debate_id = ?
              AND (
                    brexit_keywords_found IS NOT NULL
                 OR brexit_llm_reasoning IS NOT NULL
                 OR brexit_confidence > 0
                 OR brexit_llm_confidence > 0
              )
            LIMIT 1
            """,
            [debate_id]
        ).fetchone()
        if already_processed:
            print("  → Bereits klassifiziert, überspringe (Resume)")
            total_processed += 1
            continue

        # Hole ersten 5 Redebeiträge dieser Debatte
        speeches = conn_source.execute("""
            SELECT speech_id, speech_text
            FROM speeches
            WHERE debate_id = ?
            AND speech_text IS NOT NULL
            ORDER BY speech_id
            LIMIT 5
        """, [debate_id]).fetchall()

        if not speeches:
            print("  → Keine Reden gefunden, überspringe")
            continue

        # Kombiniere Texte
        combined_text = "\n\n".join([s[1] for s in speeches])

        # SCHRITT 1: Keyword-Analyse
        keyword_conf, keywords_found = analyze_keywords(combined_text)

        print(f"  Keywords: {len(keywords_found)} gefunden, Confidence: {keyword_conf:.2f}")

        # Wenn keine Keywords gefunden, überspringe
        if len(keywords_found) == 0:
            total_processed += 1
            continue

        total_with_keywords += 1

        # Rate Limiting: Warte 6 Sekunden zwischen API Calls
        if last_api_call_time > 0:
            time_since_last_call = time.time() - last_api_call_time
            if time_since_last_call < REQUEST_DELAY:
                wait_time = REQUEST_DELAY - time_since_last_call
                print(f"  ⏱️  Rate Limit: Warte {wait_time:.1f}s...")
                time.sleep(wait_time)

        # SCHRITT 2: LLM-Analyse
        print(f"  Analysiere mit Gemini...")
        last_api_call_time = time.time()
        llm_has_relation, llm_conf, llm_reasoning, input_tokens, output_tokens = analyze_with_gemini(
            debate_name,
            date,
            combined_text,
            keywords_found,
            api_key
        )

        # Update Cost Tracking
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_1M
        output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
        call_cost = input_cost + output_cost
        total_cost += call_cost

        total_llm_analyzed += 1
        print(f"  LLM: {llm_has_relation}, Confidence: {llm_conf:.2f}")
        print(f"  💰 Kosten: ${call_cost:.4f} (Gesamt: ${total_cost:.2f})")

        # Prüfe Kosten-Limit
        if total_cost >= COST_LIMIT:
            print(f"\n⚠️  KOSTEN-LIMIT ERREICHT!")
            print(f"  Aktuell: ${total_cost:.2f} / Limit: ${COST_LIMIT:.2f}")
            print(f"  Stoppe Verarbeitung und speichere bisherige Ergebnisse...\n")
            cost_limit_reached = True

        # SCHRITT 3: Kombiniere Ergebnisse
        brexit_related, final_conf = combine_results(
            keyword_conf,
            len(keywords_found),
            llm_has_relation,
            llm_conf
        )

        print(f"  ✓ Final: Brexit-Bezug = {brexit_related}, Confidence = {final_conf:.2f}")

        if brexit_related:
            total_brexit_related += 1

        # Update alle Reden dieser Debatte
        conn_out.execute("""
            UPDATE speeches
            SET
                brexit_related = ?,
                brexit_confidence = ?,
                brexit_keyword_confidence = ?,
                brexit_llm_confidence = ?,
                brexit_keywords_found = ?,
                brexit_llm_reasoning = ?
            WHERE debate_id = ?
        """, [
            brexit_related,
            final_conf,
            keyword_conf,
            llm_conf,
            ', '.join(keywords_found[:10]),  # Erste 10 Keywords
            llm_reasoning,
            debate_id
        ])

        conn_out.commit()
        total_processed += 1

        print()  # Leerzeile

        # Breche ab, wenn Kosten-Limit erreicht
        if cost_limit_reached:
            break

    # Zusammenfassung
    print("=" * 70)
    if cost_limit_reached:
        print("ABGEBROCHEN - KOSTEN-LIMIT ERREICHT")
    else:
        print("FERTIG!")
    print("=" * 70)
    print(f"\nStatistiken:")
    print(f"  Debatten analysiert:        {total_processed}")
    print(f"  Mit Keywords:               {total_with_keywords}")
    print(f"  LLM-analysiert:             {total_llm_analyzed}")
    print(f"  Brexit-bezogen:             {total_brexit_related}")
    print(f"  Brexit-Rate:                {total_brexit_related/max(total_processed,1)*100:.1f}%")

    # Datenbank-Statistiken
    brexit_speeches = conn_out.execute("""
        SELECT COUNT(*) FROM speeches WHERE brexit_related = TRUE
    """).fetchone()[0]

    total_speeches = conn_out.execute("""
        SELECT COUNT(*) FROM speeches
    """).fetchone()[0]

    print(f"\nReden mit Brexit-Bezug:     {brexit_speeches:,} von {total_speeches:,}")

    # Kosten-Zusammenfassung
    print(f"\n💰 Kosten-Übersicht:")
    print(f"  API Calls:                  {total_llm_analyzed}")
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
        print(f"   Nur {total_processed} von {len(debates)} Debatten wurden verarbeitet.")


if __name__ == "__main__":
    main()

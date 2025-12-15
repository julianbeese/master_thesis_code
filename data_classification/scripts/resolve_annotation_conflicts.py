#!/usr/bin/env python3
"""
Resolve annotation conflicts using OpenAI based on provided framing guidelines.

This script connects to the Railway PostgreSQL database using the DATABASE_URL
environment variable, detects conflicts between two annotators in the table
`agreement_chunks` (where `label1 != label2`), retrieves the corresponding
`chunk_text` from `chunks`, calls OpenAI to decide which label is correct based
on strict guidelines, and writes the chosen label back to `chunks.frame_label`.

It also logs each resolution decision to an audit table `annotation_conflict_resolutions`
for traceability.

Usage examples:
  export DATABASE_URL="postgresql://..."
  export OPENAI_API_KEY="sk-..."
  python scripts/resolve_annotation_conflicts.py --limit 100 --model gpt-4o-mini

Options:
  --limit N          Process at most N conflicts (default: 50)
  --dry-run          Do not write updates, only print decisions
  --model NAME       OpenAI model name (default: gpt-4o-mini)
  --batch-size N     Decisions per transaction before committing (default: 25)

"""

import argparse
import os
import sys
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv
import psycopg2
import psycopg2.extras

try:
    # New OpenAI SDK (2024+) pattern
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

GUIDELINES_TEXT = (
    """
    Human Impact: Focuses on how political decisions affect subgroups of people; people’s daily lives and personal well-being.
    Powerlessness: Highlights lack of control, external constraints, or domination by stronger forces that limit agency or autonomy.
    Economics: Centers on financial costs, benefits, and economic implications using quantifiable measures, market dynamics, and business terminology.
    Moral Values: Appeals to ethics, fairness, integrity, or national principles such as democracy etc.
    Conflict: Frames politics as a struggle between opposing sides, using adversarial or competitive language to depict winners and losers.

    None / TBD
    Used when no frame is evident, or when content overlaps multiple frames.
    •	No identifiable framing device.
    •	Procedural, descriptive, or factual tone.
    •	TBD used when insufficient context to decide.
    """
).strip()


SYSTEM_INSTRUCTIONS = (
    """
You are an expert political text framing adjudicator. You will receive:
- The original text chunk (Text snippets from debates in the UK House of Commons)
- Two candidate frame labels from human annotators (label_a and label_b)
- Strict framing guidelines (taxonomy above)

Task:
1) Decide which ONE of the two labels best matches the text based strictly on the guidelines.
2) Never propose new labels; choose exactly one of the provided candidates.
3) Return a compact JSON object with fields: {"chosen_label": "label_a|label_b", "rationale": "<max 2 sentences>"}.
4) Keep rationale short, citing decisive criteria or required keywords.
    """
).strip()


def get_db_connection() -> psycopg2.extensions.connection:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(database_url)


def ensure_audit_table(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS annotation_conflict_resolutions (
                id SERIAL PRIMARY KEY,
                chunk_id VARCHAR(255) NOT NULL,
                label_a VARCHAR(100) NOT NULL,
                label_b VARCHAR(100) NOT NULL,
                chosen_label VARCHAR(100) NOT NULL,
                rationale TEXT,
                model VARCHAR(100) NOT NULL,
                decided_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def fetch_conflicts(conn: psycopg2.extensions.connection, limit: int) -> list:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT 
                a.chunk_id,
                a.label1 AS label_a,
                a.label2 AS label_b,
                c.chunk_text
            FROM agreement_chunks a
            JOIN chunks c ON c.chunk_id = a.chunk_id
            WHERE a.label1 IS NOT NULL 
              AND a.label2 IS NOT NULL 
              AND a.label1 <> a.label2
            ORDER BY a.updated_at DESC NULLS LAST, a.created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        return list(cur.fetchall())


def call_openai_decide(client: OpenAI, model: str, chunk_text: str, label_a: str, label_b: str) -> Tuple[str, str]:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {
            "role": "user",
            "content": (
                f"Guidelines (taxonomy):\n{GUIDELINES_TEXT}\n\n"
                f"Text chunk:\n{chunk_text}\n\n"
                f"Candidate labels:\nlabel_a: {label_a}\nlabel_b: {label_b}\n\n"
                "Respond ONLY with JSON: {\"chosen_label\": \"label_a|label_b\", \"rationale\": \"...\"}"
            ),
        },
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"OpenAI API error: {e}")

    # Very small, safe JSON parse (avoid huge deps)
    import json

    try:
        data = json.loads(content)
        chosen = data.get("chosen_label", "").strip()
        rationale = data.get("rationale", "").strip()
        if chosen not in ("label_a", "label_b"):
            raise ValueError("Model must choose 'label_a' or 'label_b'")
        return (chosen, rationale)
    except Exception as e:
        raise RuntimeError(f"Invalid JSON from model: {e}; content={content!r}")


def update_resolution(
    conn: psycopg2.extensions.connection,
    chunk_id: str,
    label_a: str,
    label_b: str,
    chosen_label: str,
    rationale: str,
    model: str,
) -> None:
    chosen_value = label_a if chosen_label == "label_a" else label_b
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE chunks
               SET frame_label = %s,
                   updated_at = CURRENT_TIMESTAMP,
                   annotation_notes = COALESCE(annotation_notes, '') || %s
             WHERE chunk_id = %s
            """,
            (
                chosen_value,
                f"\n[auto-resolution:{model}] chosen={chosen_value} (vs. {label_a}|{label_b})\nreason={rationale}\n",
                chunk_id,
            ),
        )

        cur.execute(
            """
            INSERT INTO annotation_conflict_resolutions
                (chunk_id, label_a, label_b, chosen_label, rationale, model)
            VALUES
                (%s, %s, %s, %s, %s, %s)
            """,
            (chunk_id, label_a, label_b, chosen_value, rationale, model),
        )


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Resolve annotation conflicts using OpenAI")
    parser.add_argument("--limit", type=int, default=50, help="Max conflicts to process")
    parser.add_argument("--dry-run", action="store_true", help="Do not write updates")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--batch-size", type=int, default=25, help="Transactions per commit")
    args = parser.parse_args()

    if OpenAI is None:
        print("OpenAI SDK not available. Please install 'openai' package.", file=sys.stderr)
        return 1

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set.", file=sys.stderr)
        return 1

    # Resolve repo root and data backup directory
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    backup_dir = data_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Perform pre-run backup
    database_url = os.getenv("DATABASE_URL")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dump_path = backup_dir / f"railway_backup_{timestamp}.sql"
    try:
        if shutil.which("pg_dump"):
            # Use pg_dump with URI directly
            # Avoid printing full URL to console to prevent secrets leakage
            subprocess.run(
                [
                    "pg_dump",
                    "--no-owner",
                    "--no-privileges",
                    "--format=plain",
                    f"--file={str(dump_path)}",
                    database_url,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Created SQL backup: {dump_path}")
        else:
            # CSV fallback for key tables using COPY TO STDOUT
            print("pg_dump not found. Falling back to CSV export for key tables...")
            try:
                conn_for_csv = get_db_connection()
            except Exception as e:
                print(f"❌ DB connection error (for backup): {e}", file=sys.stderr)
                return 1
            try:
                tables = ["chunks", "agreement_chunks", "annotation_conflict_resolutions"]
                with conn_for_csv.cursor() as cur:
                    for table in tables:
                        csv_path = backup_dir / f"{table}_{timestamp}.csv"
                        try:
                            with open(csv_path, "w", encoding="utf-8") as f:
                                cur.copy_expert(
                                    sql=f"COPY (SELECT * FROM {table}) TO STDOUT WITH CSV HEADER",
                                    file=f,
                                )
                            print(f"Exported {table} -> {csv_path}")
                        except Exception:
                            # Table may not exist; skip silently
                            pass
                conn_for_csv.commit()
            finally:
                conn_for_csv.close()
    except Exception as e:
        print(f"⚠️ Backup step failed: {e}", file=sys.stderr)

    try:
        conn = get_db_connection()
    except Exception as e:
        print(f"❌ DB connection error: {e}", file=sys.stderr)
        return 1

    client = OpenAI()

    try:
        conn.autocommit = False
        ensure_audit_table(conn)

        conflicts = fetch_conflicts(conn, args.limit)
        if not conflicts:
            print("No conflicts found (label1 != label2).")
            conn.commit()
            return 0

        processed = 0
        batch_count = 0
        for row in conflicts:
            chunk_id = str(row["chunk_id"])  # type: ignore
            label_a = str(row["label_a"]).strip()
            label_b = str(row["label_b"]).strip()
            chunk_text = str(row["chunk_text"]).strip()

            try:
                chosen_label, rationale = call_openai_decide(
                    client=client,
                    model=args.model,
                    chunk_text=chunk_text,
                    label_a=label_a,
                    label_b=label_b,
                )
            except Exception as e:
                print(f"Skipping chunk {chunk_id}: model error: {e}", file=sys.stderr)
                continue

            chosen_value = label_a if chosen_label == "label_a" else label_b
            print(f"chunk_id={chunk_id} -> chosen={chosen_value}  (rationale: {rationale})")

            if not args.dry_run:
                update_resolution(
                    conn=conn,
                    chunk_id=chunk_id,
                    label_a=label_a,
                    label_b=label_b,
                    chosen_label=chosen_label,
                    rationale=rationale,
                    model=args.model,
                )
                batch_count += 1

            processed += 1
            if batch_count >= args.batch_size and not args.dry_run:
                conn.commit()
                batch_count = 0

        if not args.dry_run and batch_count > 0:
            conn.commit()

        print(f"Done. Processed {processed} conflict(s).")
        return 0

    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())



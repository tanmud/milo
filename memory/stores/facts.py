from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone


FACTS_TABLE = "facts"
MIN_PROMPT_CONFIDENCE = 0.5
MAX_PROMPT_TOKENS = 200


@dataclass(slots=True)
class Fact:
    id: str
    key: str
    value: str
    category: str
    confidence: float
    last_updated: str
    source: str


def _row_to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        id=str(row["id"]),
        key=str(row["key"]),
        value=str(row["value"]),
        category=str(row["category"]),
        confidence=float(row["confidence"]),
        last_updated=str(row["last_updated"]),
        source=str(row["source"] or ""),
    )


def upsert_fact(
    conn: sqlite3.Connection,
    key: str,
    value: str,
    category: str = "general",
    confidence: float = 1.0,
    source: str = "user_explicit",
) -> str:
    existing = conn.execute(
        f"SELECT id FROM {FACTS_TABLE} WHERE key = ?",
        (key,),
    ).fetchone()
    now = datetime.now(timezone.utc).isoformat()

    if existing is not None:
        conn.execute(
            f"""
            UPDATE {FACTS_TABLE}
            SET value = ?,
                category = ?,
                confidence = ?,
                last_updated = ?,
                source = ?
            WHERE key = ?
            """,
            (value, category, confidence, now, source, key),
        )
        conn.commit()
        return str(existing["id"])

    fact_id = str(uuid.uuid4())
    conn.execute(
        f"""
        INSERT INTO {FACTS_TABLE} (id, key, value, category, confidence, last_updated, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (fact_id, key, value, category, confidence, now, source),
    )
    conn.commit()
    return fact_id


def get_fact(conn: sqlite3.Connection, key: str) -> Fact | None:
    row = conn.execute(f"SELECT * FROM {FACTS_TABLE} WHERE key = ?", (key,)).fetchone()
    return _row_to_fact(row) if row is not None else None


def get_all_facts(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute(
        f"""
        SELECT *
        FROM {FACTS_TABLE}
        WHERE confidence >= ?
        ORDER BY confidence DESC, last_updated DESC
        """,
        (MIN_PROMPT_CONFIDENCE,),
    ).fetchall()
    return {str(row["key"]): str(row["value"]) for row in rows}


def get_facts_by_category(conn: sqlite3.Connection, category: str) -> list[Fact]:
    rows = conn.execute(
        f"SELECT * FROM {FACTS_TABLE} WHERE category = ? ORDER BY confidence DESC, last_updated DESC",
        (category,),
    ).fetchall()
    return [_row_to_fact(row) for row in rows]


def delete_fact(conn: sqlite3.Connection, key: str) -> bool:
    cursor = conn.execute(f"DELETE FROM {FACTS_TABLE} WHERE key = ?", (key,))
    conn.commit()
    return cursor.rowcount > 0


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def format_facts_for_prompt(facts: dict[str, str]) -> str:
    lines = ["[KNOWN FACTS]"]
    total_tokens = _estimate_tokens(lines[0])

    for key, value in facts.items():
        entry = f"{key}: {value}"
        entry_tokens = _estimate_tokens(entry)
        if total_tokens + entry_tokens > MAX_PROMPT_TOKENS:
            break
        lines.append(entry)
        total_tokens += entry_tokens

    return "\n".join(lines)
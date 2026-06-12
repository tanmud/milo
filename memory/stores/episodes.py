from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone


EPISODES_TABLE = "episodes"
MAX_PROMPT_TOKENS = 150


@dataclass(slots=True)
class Episode:
    id: str
    description: str
    timestamp: str
    agent_id: str
    tags: list[str]
    importance: float
    session_id: str


def _row_to_episode(row: sqlite3.Row) -> Episode:
    tags_raw = row["tags"] or "[]"
    tags = json.loads(tags_raw)
    if not isinstance(tags, list):
        tags = []
    return Episode(
        id=str(row["id"]),
        description=str(row["description"]),
        timestamp=str(row["timestamp"]),
        agent_id=str(row["agent_id"]),
        tags=[str(tag) for tag in tags],
        importance=float(row["importance"]),
        session_id=str(row["session_id"]) if row["session_id"] is not None else "",
    )


def add_episode(
    conn: sqlite3.Connection,
    description: str,
    agent_id: str,
    tags: list[str] | None = None,
    importance: float = 0.5,
    session_id: str | None = None,
) -> str:
    episode_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        f"""
        INSERT INTO {EPISODES_TABLE} (id, description, timestamp, agent_id, tags, importance, session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            episode_id,
            description,
            timestamp,
            agent_id,
            json.dumps(tags or []),
            importance,
            session_id,
        ),
    )
    conn.commit()
    return episode_id


def get_episodes_by_date_range(
    conn: sqlite3.Connection,
    start: str,
    end: str,
    agent_id: str | None = None,
) -> list[Episode]:
    if agent_id is None:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {EPISODES_TABLE}
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
            """,
            (start, end),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {EPISODES_TABLE}
            WHERE timestamp BETWEEN ? AND ?
              AND agent_id = ?
            ORDER BY timestamp DESC
            """,
            (start, end, agent_id),
        ).fetchall()
    return [_row_to_episode(row) for row in rows]


def get_episodes_by_tag(conn: sqlite3.Connection, tag: str, limit: int = 10) -> list[Episode]:
    rows = conn.execute(
        f"""
        SELECT *
        FROM {EPISODES_TABLE}
        WHERE tags LIKE ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (f"%{tag}%", limit),
    ).fetchall()
    return [_row_to_episode(row) for row in rows]


def get_recent_episodes(
    conn: sqlite3.Connection,
    limit: int = 5,
    agent_id: str | None = None,
) -> list[Episode]:
    if agent_id is None:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {EPISODES_TABLE}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {EPISODES_TABLE}
            WHERE agent_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (agent_id, limit),
        ).fetchall()
    return [_row_to_episode(row) for row in rows]


def get_episodes_by_keyword(conn: sqlite3.Connection, keyword: str, limit: int = 10) -> list[Episode]:
    rows = conn.execute(
        f"""
        SELECT *
        FROM {EPISODES_TABLE}
        WHERE description LIKE ?
        ORDER BY importance DESC, timestamp DESC
        LIMIT ?
        """,
        (f"%{keyword}%", limit),
    ).fetchall()
    return [_row_to_episode(row) for row in rows]


def _format_date(iso_timestamp: str) -> str:
    normalized = iso_timestamp.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return iso_timestamp[:10]
    return parsed.strftime("%b %d")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def format_episodes_for_prompt(episodes: list[Episode]) -> str:
    lines = ["[RECENT EPISODES]"]
    total_tokens = _estimate_tokens(lines[0])

    for episode in episodes:
        entry = f"{_format_date(episode.timestamp)}: {episode.description}"
        entry_tokens = _estimate_tokens(entry)
        if total_tokens + entry_tokens > MAX_PROMPT_TOKENS:
            break
        lines.append(entry)
        total_tokens += entry_tokens

    return "\n".join(lines)
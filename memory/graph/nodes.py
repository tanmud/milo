from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


NodeType = Literal["entity", "topic", "emotion", "event", "session"]


@dataclass(slots=True)
class GraphNode:
    id: str
    name: str
    type: NodeType
    mention_count: int
    first_seen: str
    last_seen: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _row_to_node(row: sqlite3.Row) -> GraphNode:
    metadata_raw = row["metadata"] or "{}"
    metadata = json.loads(metadata_raw)
    return GraphNode(
        id=str(row["id"]),
        name=str(row["name"]),
        type=str(row["type"]),
        mention_count=int(row["mention_count"]),
        first_seen=str(row["first_seen"]),
        last_seen=str(row["last_seen"]),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def upsert_node(conn: sqlite3.Connection, node: GraphNode) -> str:
    existing = conn.execute(
        "SELECT id, mention_count FROM nodes WHERE LOWER(name) = LOWER(?) AND type = ?",
        (node.name, node.type),
    ).fetchone()

    if existing is not None:
        conn.execute(
            """
            UPDATE nodes
            SET mention_count = mention_count + 1,
                last_seen = ?
            WHERE id = ?
            """,
            (datetime.now(timezone.utc).isoformat(), existing["id"]),
        )
        conn.commit()
        return str(existing["id"])

    node_id = node.id or str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO nodes (id, name, type, mention_count, first_seen, last_seen, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            node_id,
            node.name,
            node.type,
            node.mention_count,
            node.first_seen,
            node.last_seen,
            json.dumps(node.metadata or {}),
        ),
    )
    conn.commit()
    return node_id


def get_node(conn: sqlite3.Connection, node_id: str) -> GraphNode | None:
    row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return _row_to_node(row) if row is not None else None


def get_node_by_name(conn: sqlite3.Connection, name: str, type: str) -> GraphNode | None:
    row = conn.execute(
        "SELECT * FROM nodes WHERE LOWER(name) = LOWER(?) AND type = ?",
        (name, type),
    ).fetchone()
    return _row_to_node(row) if row is not None else None


def get_all_nodes(conn: sqlite3.Connection) -> list[GraphNode]:
    rows = conn.execute("SELECT * FROM nodes ORDER BY first_seen ASC").fetchall()
    return [_row_to_node(row) for row in rows]


def delete_orphaned_nodes(conn: sqlite3.Connection, min_mentions: int = 3) -> int:
    cursor = conn.execute(
        """
        DELETE FROM nodes
        WHERE mention_count < ?
          AND NOT EXISTS (
              SELECT 1
              FROM edges
              WHERE edges.source_id = nodes.id
                 OR edges.target_id = nodes.id
          )
        """,
        (min_mentions,),
    )
    conn.commit()
    return max(cursor.rowcount, 0)
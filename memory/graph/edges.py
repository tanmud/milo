from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(slots=True)
class GraphEdge:
    id: str
    source_id: str
    target_id: str
    relation: str
    weight: float
    confidence: float
    count: int
    last_reinforced: str


def _row_to_edge(row: sqlite3.Row) -> GraphEdge:
    return GraphEdge(
        id=str(row["id"]),
        source_id=str(row["source_id"]),
        target_id=str(row["target_id"]),
        relation=str(row["relation"]),
        weight=float(row["weight"]),
        confidence=float(row["confidence"]),
        count=int(row["count"]),
        last_reinforced=str(row["last_reinforced"]),
    )


def upsert_edge(conn: sqlite3.Connection, edge: GraphEdge) -> None:
    existing = conn.execute(
        """
        SELECT *
        FROM edges
        WHERE source_id = ? AND target_id = ? AND relation = ?
        """,
        (edge.source_id, edge.target_id, edge.relation),
    ).fetchone()

    if existing is not None:
        updated_weight = min(1.0, float(existing["weight"]) + edge.confidence * 0.1)
        conn.execute(
            """
            UPDATE edges
            SET weight = ?,
                confidence = ?,
                count = count + 1,
                last_reinforced = ?
            WHERE id = ?
            """,
            (updated_weight, edge.confidence, edge.last_reinforced, existing["id"]),
        )
        conn.commit()
        return

    conn.execute(
        """
        INSERT INTO edges (
            id, source_id, target_id, relation, weight, confidence, count, last_reinforced
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            edge.id,
            edge.source_id,
            edge.target_id,
            edge.relation,
            edge.weight,
            edge.confidence,
            edge.count,
            edge.last_reinforced,
        ),
    )
    conn.commit()


def get_edges_for_node(conn: sqlite3.Connection, node_id: str) -> list[GraphEdge]:
    rows = conn.execute(
        """
        SELECT *
        FROM edges
        WHERE source_id = ? OR target_id = ?
        ORDER BY weight DESC, last_reinforced DESC
        """,
        (node_id, node_id),
    ).fetchall()
    return [_row_to_edge(row) for row in rows]


def get_all_edges(conn: sqlite3.Connection) -> list[GraphEdge]:
    rows = conn.execute("SELECT * FROM edges ORDER BY last_reinforced DESC").fetchall()
    return [_row_to_edge(row) for row in rows]


def decay_all_edges(conn: sqlite3.Connection, decay_factor: float = 0.95) -> None:
    conn.execute("UPDATE edges SET weight = weight * ?", (decay_factor,))
    conn.commit()


def prune_weak_edges(conn: sqlite3.Connection, min_weight: float = 0.1) -> int:
    cursor = conn.execute("DELETE FROM edges WHERE weight < ?", (min_weight,))
    conn.commit()
    return max(cursor.rowcount, 0)
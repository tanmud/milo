from __future__ import annotations

import json
import pickle
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional dependency fallback
    nx = None  # type: ignore[assignment]

try:
    import chromadb
except ImportError:  # pragma: no cover - optional dependency fallback
    chromadb = None  # type: ignore[assignment]

from .edges import GraphEdge, decay_all_edges, get_all_edges, prune_weak_edges, upsert_edge
from .embeddings import (
    NODE_TYPES,
    add_node_embedding,
    get_or_create_collections,
    load_embedding_model,
    remove_node_embedding,
    semantic_search as embedding_semantic_search,
)
from .nodes import GraphNode, delete_orphaned_nodes, get_all_nodes, get_node, upsert_node
from .schema import init_db


class AgentGraph:
    def __init__(self, db_path: str, chroma_path: str, embedding_model: Any | None = None):
        if nx is None:
            raise ImportError("networkx is required to construct AgentGraph")
        if chromadb is None:
            raise ImportError("chromadb is required to construct AgentGraph")

        self.db_path = db_path
        self.conn: sqlite3.Connection = init_db(db_path)
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collections = get_or_create_collections(self.chroma_client)
        self.G = nx.DiGraph()
        self.embedding_model = embedding_model or load_embedding_model()
        self.load()

    def _row_to_node(self, row: sqlite3.Row) -> GraphNode:
        metadata = row["metadata"] or "{}"
        return GraphNode(
            id=str(row["id"]),
            name=str(row["name"]),
            type=str(row["type"]),
            mention_count=int(row["mention_count"]),
            first_seen=str(row["first_seen"]),
            last_seen=str(row["last_seen"]),
            metadata=json.loads(metadata),
        )

    def _row_to_edge(self, row: sqlite3.Row) -> GraphEdge:
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

    def _get_edge_by_key(self, source_id: str, target_id: str, relation: str) -> GraphEdge | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM edges
            WHERE source_id = ? AND target_id = ? AND relation = ?
            """,
            (source_id, target_id, relation),
        ).fetchone()
        return self._row_to_edge(row) if row is not None else None

    def load(self) -> None:
        self.G.clear()
        for node in get_all_nodes(self.conn):
            self.G.add_node(node.id, **asdict(node))
        for edge in get_all_edges(self.conn):
            self.G.add_edge(edge.source_id, edge.target_id, **asdict(edge))

    def upsert_node(self, node: GraphNode) -> str:
        canonical_id = upsert_node(self.conn, node)
        canonical_node = get_node(self.conn, canonical_id)
        if canonical_node is None:
            raise RuntimeError(f"Failed to load node '{canonical_id}' after upsert")

        add_node_embedding(self.collections, canonical_node, self.embedding_model)
        self.G.add_node(canonical_node.id, **asdict(canonical_node))
        return canonical_id

    def upsert_edge(self, edge: GraphEdge) -> None:
        upsert_edge(self.conn, edge)
        stored_edge = self._get_edge_by_key(edge.source_id, edge.target_id, edge.relation)
        if stored_edge is None:
            raise RuntimeError(
                f"Failed to load edge {edge.source_id} -> {edge.target_id} ({edge.relation}) after upsert"
            )

        if stored_edge.source_id not in self.G:
            source_node = get_node(self.conn, stored_edge.source_id)
            if source_node is not None:
                self.G.add_node(source_node.id, **asdict(source_node))
        if stored_edge.target_id not in self.G:
            target_node = get_node(self.conn, stored_edge.target_id)
            if target_node is not None:
                self.G.add_node(target_node.id, **asdict(target_node))

        self.G.add_edge(stored_edge.source_id, stored_edge.target_id, **asdict(stored_edge))

    def get_neighborhood(self, node_id: str, depth: int = 2) -> tuple[list[GraphNode], list[GraphEdge]]:
        if node_id not in self.G:
            return [], []

        neighborhood = nx.ego_graph(self.G, node_id, radius=depth, undirected=False)
        nodes: list[GraphNode] = []
        for neighborhood_node_id in neighborhood.nodes:
            node = get_node(self.conn, str(neighborhood_node_id))
            if node is not None:
                nodes.append(node)

        edges: list[GraphEdge] = []
        for source_id, target_id, data in neighborhood.edges(data=True):
            if data:
                edges.append(
                    GraphEdge(
                        id=str(data["id"]),
                        source_id=str(source_id),
                        target_id=str(target_id),
                        relation=str(data["relation"]),
                        weight=float(data["weight"]),
                        confidence=float(data["confidence"]),
                        count=int(data["count"]),
                        last_reinforced=str(data["last_reinforced"]),
                    )
                )

        return nodes, edges

    def semantic_search(self, query: str, node_types: list[str] | None = None, k: int = 5) -> list[GraphNode]:
        active_node_types = node_types if node_types is not None else NODE_TYPES
        node_ids = embedding_semantic_search(self.collections, query, active_node_types, k, self.embedding_model)
        results: list[GraphNode] = []
        for node_id in node_ids:
            node = get_node(self.conn, node_id)
            if node is not None:
                results.append(node)
        return results

    def ingest_triples(self, triples: list[dict[str, Any]]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        valid_node_types = set(NODE_TYPES)

        for triple in triples:
            try:
                subject = str(triple["subject"])
                subject_type = str(triple["subject_type"])
                relation = str(triple["relation"])
                object_ = str(triple["object"])
                object_type = str(triple["object_type"])
                confidence = float(triple["confidence"])
            except KeyError as error:
                raise ValueError(f"Triple is missing required field: {error.args[0]}") from error

            if subject_type not in valid_node_types:
                raise ValueError(f"Unsupported subject_type '{subject_type}'")
            if object_type not in valid_node_types:
                raise ValueError(f"Unsupported object_type '{object_type}'")

            subject_node = GraphNode(
                id=str(uuid.uuid4()),
                name=subject,
                type=subject_type,
                mention_count=1,
                first_seen=now,
                last_seen=now,
                metadata={},
            )
            object_node = GraphNode(
                id=str(uuid.uuid4()),
                name=object_,
                type=object_type,
                mention_count=1,
                first_seen=now,
                last_seen=now,
                metadata={},
            )

            source_id = self.upsert_node(subject_node)
            target_id = self.upsert_node(object_node)

            edge = GraphEdge(
                id=str(uuid.uuid4()),
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                weight=confidence,
                confidence=confidence,
                count=1,
                last_reinforced=now,
            )
            self.upsert_edge(edge)

    def consolidate(self, decay_factor: float = 0.95, min_weight: float = 0.1) -> dict[str, int]:
        decay_all_edges(self.conn, decay_factor)

        orphan_rows = self.conn.execute(
            """
            SELECT id, type
            FROM nodes
            WHERE mention_count < ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM edges
                  WHERE edges.source_id = nodes.id
                     OR edges.target_id = nodes.id
              )
            """,
            (3,),
        ).fetchall()

        pruned_edges = prune_weak_edges(self.conn, min_weight)
        pruned_nodes = delete_orphaned_nodes(self.conn)

        for row in orphan_rows:
            remove_node_embedding(self.collections, str(row["id"]), str(row["type"]))

        self.load()
        return {"pruned_edges": pruned_edges, "pruned_nodes": pruned_nodes}

    def format_neighborhood_as_context(self, nodes: list[GraphNode], edges: list[GraphEdge]) -> str:
        if not edges:
            return ""

        node_names = {node.id: node.name for node in nodes}
        ordered_edges = sorted(edges, key=lambda edge: edge.weight, reverse=True)
        sentences: list[str] = []
        total_words = 0

        for edge in ordered_edges:
            source_name = node_names.get(edge.source_id, edge.source_id)
            target_name = node_names.get(edge.target_id, edge.target_id)
            sentence = f"{source_name} [{edge.relation}] {target_name} (weight: {edge.weight:.2f})."
            sentence_words = len(sentence.split())
            if sentences and total_words + sentence_words > 150:
                break
            sentences.append(sentence)
            total_words += sentence_words

        return " ".join(sentences)

    def serialize(self) -> None:
        self.conn.commit()
        graph_pickle_path = f"{self.db_path}.graph.pkl"
        if hasattr(nx, "write_gpickle"):
            nx.write_gpickle(self.G, graph_pickle_path)
            return

        with open(graph_pickle_path, "wb") as graph_file:
            pickle.dump(self.G, graph_file, protocol=pickle.HIGHEST_PROTOCOL)
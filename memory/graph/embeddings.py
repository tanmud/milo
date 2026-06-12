from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Protocol

try:
    from chromadb.api.models.Collection import Collection
except ImportError:  # pragma: no cover - optional dependency fallback
    Collection = Any  # type: ignore[assignment]

from sentence_transformers import SentenceTransformer


NODE_TYPES = ["entity", "topic", "emotion", "event", "session"]


class NodeLike(Protocol):
    id: str
    name: str
    type: str
    mention_count: int
    metadata: dict[str, Any]


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    if SentenceTransformer is Any:
        raise ImportError("sentence-transformers is required to load the embedding model")
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_or_create_collections(client: Any) -> dict[str, Collection]:
    collections: dict[str, Collection] = {}
    for node_type in NODE_TYPES:
        collections[node_type] = client.get_or_create_collection(
            name=f"graph_{node_type}",
            metadata={"hnsw:space": "cosine"},
        )
    return collections


def _embedding_text(node: NodeLike) -> str:
    if node.metadata:
        metadata_text = json.dumps(node.metadata, sort_keys=True, ensure_ascii=False)
        return f"{node.name} {metadata_text}".strip()
    return node.name


def add_node_embedding(
    collections: dict[str, Collection],
    node: NodeLike,
    model: SentenceTransformer,
) -> None:
    collection = collections[node.type]
    embedding_text = _embedding_text(node)
    embedding = model.encode(
        [embedding_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    collection.upsert(
        ids=[node.id],
        embeddings=[embedding.tolist()],
        documents=[node.name],
        metadatas=[{"type": node.type, "mention_count": node.mention_count}],
    )


def remove_node_embedding(collections: dict[str, Collection], node_id: str, node_type: str) -> None:
    collection = collections.get(node_type)
    if collection is not None:
        collection.delete(ids=[node_id])


def semantic_search(
    collections: dict[str, Collection],
    query: str,
    node_types: list[str],
    k: int,
    model: SentenceTransformer,
) -> list[str]:
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    candidate_scores: dict[str, float] = {}
    for node_type in node_types:
        collection = collections.get(node_type)
        if collection is None:
            continue
        if collection.count() == 0:
            continue

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["distances"],
        )
        ids = results.get("ids", [[]])
        distances = results.get("distances", [[]])
        if not ids:
            continue

        for index, node_id in enumerate(ids[0]):
            if not node_id:
                continue
            distance = float(distances[0][index]) if distances and distances[0][index] is not None else float("inf")
            previous_best = candidate_scores.get(node_id)
            if previous_best is None or distance < previous_best:
                candidate_scores[node_id] = distance

    ordered_ids = sorted(candidate_scores.items(), key=lambda item: item[1])
    return [node_id for node_id, _distance in ordered_ids[:k]]
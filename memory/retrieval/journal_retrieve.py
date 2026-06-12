from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from memory.graph.agent_graph import AgentGraph
from memory.stores.episodes import get_recent_episodes, format_episodes_for_prompt


JOURNAL_ENTRIES_COLLECTION = "journal_entries"
JOURNAL_METADATA_TABLE = "journal_metadata"
JOURNAL_AGENT_ID = "agent_1"
DEFAULT_COLLECTION_RESULTS = 2


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _as_list(embedding: Any) -> list[float]:
    if hasattr(embedding, "tolist"):
        return list(embedding.tolist())
    return list(embedding)


def _get_semantic_entries(agent_graph: AgentGraph, transcript: str, n_results: int) -> list[str]:
    collection = agent_graph.chroma_client.get_or_create_collection(name=JOURNAL_ENTRIES_COLLECTION)
    if collection.count() == 0:
        return []

    query_embedding = agent_graph.embedding_model.encode(
        [transcript],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    results = collection.query(
        query_embeddings=[_as_list(query_embedding)],
        n_results=n_results,
        include=["documents", "distances"],
    )
    documents = results.get("documents", [[]])
    return [str(document) for document in documents[0] if document]


def _get_last_session_meta(conn) -> dict[str, Any] | None:
    row = conn.execute(
        f"""
        SELECT last_session_date, duration_secs, themes
        FROM {JOURNAL_METADATA_TABLE}
        WHERE agent_id = ?
        """,
        (JOURNAL_AGENT_ID,),
    ).fetchone()
    if row is None:
        return None

    themes_raw = row["themes"] or "[]"
    try:
        themes = json.loads(themes_raw)
    except json.JSONDecodeError:
        themes = []
    if not isinstance(themes, list):
        themes = []

    return {
        "date": str(row["last_session_date"] or ""),
        "duration_secs": int(row["duration_secs"] or 0),
        "themes": [str(theme) for theme in themes],
    }


async def journal_retrieve(
    transcript: str,
    agent_graph: AgentGraph,
    conn,
    config: dict,
) -> dict[str, Any]:
    recency_episodes = get_recent_episodes(conn, limit=2, agent_id=JOURNAL_AGENT_ID)
    recent_prompt = format_episodes_for_prompt(recency_episodes)
    recent_episodes = [line for line in recent_prompt.splitlines()[1:] if line.strip()]

    semantic_limit = int(config.get("retrieval_k", DEFAULT_COLLECTION_RESULTS))
    date_context = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    recency_query = f"{date_context} {transcript}".strip()
    recency_semantic = _get_semantic_entries(agent_graph, recency_query, semantic_limit)
    pure_semantic = _get_semantic_entries(agent_graph, transcript, semantic_limit)

    semantic_summaries: list[str] = []
    seen: set[str] = set()
    for item in recency_semantic + pure_semantic:
        if item in seen:
            continue
        seen.add(item)
        semantic_summaries.append(item)
        if len(semantic_summaries) >= 4:
            break

    last_session_meta = _get_last_session_meta(conn)
    token_estimate = sum(_estimate_tokens(text) for text in recent_episodes + semantic_summaries)

    return {
        "recent_episodes": recent_episodes,
        "semantic_summaries": semantic_summaries,
        "last_session_meta": last_session_meta,
        "token_estimate": token_estimate,
    }
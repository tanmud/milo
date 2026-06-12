from __future__ import annotations

import re
from collections import Counter
from typing import Any

from memory.graph.agent_graph import AgentGraph
from memory.stores.episodes import get_episodes_by_keyword


CONVERSATION_SUMMARIES_COLLECTION = "conversation_summaries"
DEFAULT_RETRIEVAL_K = 3
DEFAULT_MIN_SIMILARITY_THRESHOLD = 0.0
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _as_list(embedding: Any) -> list[float]:
    if hasattr(embedding, "tolist"):
        return list(embedding.tolist())
    return list(embedding)


def _extract_keywords(transcript: str, max_keywords: int = 3) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_'-]*", transcript.lower())
    filtered = [word for word in words if word not in STOPWORDS and len(word) > 2]
    if not filtered:
        return []

    counts = Counter(filtered)
    keywords: list[str] = []
    for word, _count in counts.most_common():
        if word not in keywords:
            keywords.append(word)
        if len(keywords) >= max_keywords:
            break
    return keywords


async def assistant_retrieve(
    transcript: str,
    agent_graph: AgentGraph,
    conn,
    config: dict,
    token_budget: int = 500,
) -> dict[str, Any]:
    retrieval_k = int(config.get("retrieval_k", DEFAULT_RETRIEVAL_K))
    min_similarity_threshold = float(config.get("min_similarity_threshold", DEFAULT_MIN_SIMILARITY_THRESHOLD))
    if token_budget < 200:
        retrieval_k = 1

    summaries: list[str] = []
    collection = agent_graph.chroma_client.get_or_create_collection(name=CONVERSATION_SUMMARIES_COLLECTION)
    if collection.count() > 0:
        query_embedding = agent_graph.embedding_model.encode(
            [transcript],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        results = collection.query(
            query_embeddings=[_as_list(query_embedding)],
            n_results=retrieval_k,
            include=["documents", "distances"],
        )
        documents = results.get("documents", [[]])
        distances = results.get("distances", [[]])
        if documents:
            for index, document in enumerate(documents[0]):
                if not document:
                    continue
                distance = float(distances[0][index]) if distances and distances[0][index] is not None else 1.0
                similarity = 1.0 - distance
                if similarity >= min_similarity_threshold:
                    summaries.append(str(document))

    keywords = _extract_keywords(transcript)
    episodes: list[str] = []
    seen_descriptions: set[str] = set()
    for keyword in keywords:
        for episode in get_episodes_by_keyword(conn, keyword, limit=3):
            if episode.description in seen_descriptions:
                continue
            seen_descriptions.add(episode.description)
            episodes.append(f"{episode.timestamp[:10]}: {episode.description}")
            if len(episodes) >= 3:
                break
        if len(episodes) >= 3:
            break

    token_estimate = sum(_estimate_tokens(text) for text in summaries + episodes)

    return {
        "summaries": summaries,
        "episodes": episodes,
        "token_estimate": token_estimate,
    }
from __future__ import annotations

from typing import Any

from memory.graph.agent_graph import AgentGraph


DEFAULT_GRAPH_K = 3
NODES_TABLE = "nodes"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _normalize_text(value: str) -> str:
    return value.lower().strip()


def _extract_entity_matches(agent_graph: AgentGraph, transcript: str) -> list[str]:
    transcript_lower = _normalize_text(transcript)
    matched_node_ids: list[str] = []
    for node_id, node_data in agent_graph.G.nodes(data=True):
        node_name = str(node_data.get("name", "")).strip()
        if not node_name:
            continue
        normalized_name = _normalize_text(node_name)
        if normalized_name in transcript_lower or transcript_lower in normalized_name:
            matched_node_ids.append(str(node_id))
    return matched_node_ids


async def graph_retrieve(
    transcript: str,
    agent_graph: AgentGraph,
    config: dict,
    max_tokens: int = 200,
) -> dict[str, Any]:
    if not bool(config.get("agent_graph_enabled", True)):
        return {"graph_context": "", "matched_nodes": [], "token_estimate": 0}

    entity_matches = _extract_entity_matches(agent_graph, transcript)
    semantic_nodes = agent_graph.semantic_search(transcript, node_types=["topic", "emotion", "entity"], k=DEFAULT_GRAPH_K)

    ordered_node_ids: list[str] = []
    seen_ids: set[str] = set()
    for node_id in entity_matches + [node.id for node in semantic_nodes]:
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        ordered_node_ids.append(node_id)
        if len(ordered_node_ids) >= DEFAULT_GRAPH_K:
            break

    collected_nodes: dict[str, Any] = {}
    collected_edges: dict[str, Any] = {}
    matched_names: list[str] = []
    for node_id in ordered_node_ids:
        node = agent_graph.conn.execute(f"SELECT name FROM {NODES_TABLE} WHERE id = ?", (node_id,)).fetchone()
        if node is not None:
            name = str(node["name"])
            if name not in matched_names:
                matched_names.append(name)

        neighborhood_nodes, neighborhood_edges = agent_graph.get_neighborhood(node_id, depth=2)
        for neighborhood_node in neighborhood_nodes:
            collected_nodes[neighborhood_node.id] = neighborhood_node
        for edge in neighborhood_edges:
            collected_edges[edge.id] = edge

    graph_context = agent_graph.format_neighborhood_as_context(list(collected_nodes.values()), list(collected_edges.values()))
    words = graph_context.split()
    if len(words) > max_tokens:
        graph_context = " ".join(words[:max_tokens])

    return {
        "graph_context": graph_context,
        "matched_nodes": matched_names,
        "token_estimate": _estimate_tokens(graph_context),
    }
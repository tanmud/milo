#!/usr/bin/env python3
"""
Smoke tests for memory/graph/ layer.
Run from project root: python test_memory.py
Tests run sequentially — each builds on the previous.
"""

import os
import sys
import json
import shutil
import sqlite3
from uuid import uuid4
from datetime import datetime
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
TEST_DB_PATH    = "test_artifacts/test.db"
TEST_CHROMA_PATH = "test_artifacts/chroma"

# ── colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}"); sys.exit(1)
def info(msg): print(f"  {YELLOW}→{RESET} {msg}")
def section(title):
    print(f"\n{BOLD}{CYAN}{'─'*55}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*55}{RESET}")


# ── setup / teardown ───────────────────────────────────────────────────────────
def setup():
    Path("test_artifacts").mkdir(exist_ok=True)
    if Path(TEST_DB_PATH).exists():
        os.remove(TEST_DB_PATH)
    if Path(TEST_CHROMA_PATH).exists():
        shutil.rmtree(TEST_CHROMA_PATH)
    info("Test artifacts directory ready")

def teardown():
    if Path("test_artifacts").exists():
        shutil.rmtree("test_artifacts")
    info("Test artifacts cleaned up")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — schema.py
# ══════════════════════════════════════════════════════════════════════════════
def test_schema():
    section("STEP 1 — schema.py")
    try:
        from memory.graph.schema import init_db
    except ImportError as e:
        fail(f"Could not import schema.py: {e}")

    conn = init_db(TEST_DB_PATH)

    # all expected tables exist
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in cursor.fetchall()}
    expected = {"nodes", "edges", "sessions", "agent_summaries"}
    for t in expected:
        if t in tables:
            ok(f"Table '{t}' exists")
        else:
            fail(f"Table '{t}' missing — check schema.py")

    # WAL mode enabled
    wal = conn.execute("PRAGMA journal_mode").fetchone()[0]
    if wal == "wal":
        ok("WAL mode enabled")
    else:
        fail(f"WAL mode not enabled — got '{wal}'")

    # foreign keys enabled
    fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    if fk == 1:
        ok("Foreign keys enabled")
    else:
        fail("Foreign keys not enabled — add PRAGMA foreign_keys = ON to init_db")

    conn.close()
    return True


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — nodes.py
# ══════════════════════════════════════════════════════════════════════════════
def test_nodes():
    section("STEP 2 — nodes.py")
    try:
        from memory.graph.schema import init_db
        from memory.graph.nodes import GraphNode, upsert_node, get_node, get_node_by_name, get_all_nodes, delete_orphaned_nodes
    except ImportError as e:
        fail(f"Could not import nodes.py: {e}")

    conn = init_db(TEST_DB_PATH)

    # insert a node
    node = GraphNode(
        id=str(uuid4()),
        name="gym",
        type="topic",
        mention_count=1,
        first_seen=datetime.utcnow().isoformat(),
        last_seen=datetime.utcnow().isoformat(),
        metadata={"context": "health"}
    )
    returned_id = upsert_node(conn, node)
    ok(f"Node inserted with id: {returned_id[:8]}...")

    # upsert same node again — must not duplicate, must increment mention_count
    returned_id_2 = upsert_node(conn, node)
    if returned_id == returned_id_2:
        ok("Duplicate upsert returned same id (no duplicate created)")
    else:
        fail(f"Duplicate upsert created a new id — deduplication broken")

    fetched = get_node(conn, returned_id)
    if fetched is None:
        fail("get_node returned None — node not found after insert")
    if fetched.mention_count == 2:
        ok("mention_count incremented to 2 on second upsert")
    else:
        fail(f"mention_count should be 2, got {fetched.mention_count}")

    # get by name
    by_name = get_node_by_name(conn, "gym", "topic")
    if by_name and by_name.id == returned_id:
        ok("get_node_by_name returns correct node")
    else:
        fail("get_node_by_name failed")

    # case insensitive name match
    by_name_upper = get_node_by_name(conn, "GYM", "topic")
    if by_name_upper and by_name_upper.id == returned_id:
        ok("Name lookup is case-insensitive")
    else:
        fail("Name lookup is case-sensitive — fix upsert_node to use LOWER(name)")

    # insert a second node
    node2 = GraphNode(
        id=str(uuid4()),
        name="stress",
        type="topic",
        mention_count=1,
        first_seen=datetime.utcnow().isoformat(),
        last_seen=datetime.utcnow().isoformat(),
        metadata={}
    )
    id2 = upsert_node(conn, node2)
    ok(f"Second node inserted: {id2[:8]}...")

    # get all nodes
    all_nodes = get_all_nodes(conn)
    if len(all_nodes) == 2:
        ok(f"get_all_nodes returns {len(all_nodes)} nodes")
    else:
        fail(f"get_all_nodes returned {len(all_nodes)}, expected 2")

    # orphan deletion — node2 has no edges and mention_count=1, should be deleted
    deleted = delete_orphaned_nodes(conn, min_mentions=2)
    if deleted >= 1:
        ok(f"delete_orphaned_nodes pruned {deleted} orphan node(s)")
    else:
        fail("delete_orphaned_nodes returned 0 — low-mention orphan not pruned")

    conn.close()
    return returned_id, id2


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — edges.py
# ══════════════════════════════════════════════════════════════════════════════
def test_edges(node_id_1):
    section("STEP 3 — edges.py")
    try:
        from memory.graph.schema import init_db
        from memory.graph.nodes import GraphNode, upsert_node
        from memory.graph.edges import GraphEdge, upsert_edge, get_edges_for_node, get_all_edges, decay_all_edges, prune_weak_edges
    except ImportError as e:
        fail(f"Could not import edges.py: {e}")

    conn = init_db(TEST_DB_PATH)

    # ensure we have two nodes to connect
    n1 = GraphNode(id=str(uuid4()), name="gym", type="topic",
                   mention_count=1, first_seen=datetime.utcnow().isoformat(),
                   last_seen=datetime.utcnow().isoformat(), metadata={})
    n2 = GraphNode(id=str(uuid4()), name="good_mood", type="emotion",
                   mention_count=1, first_seen=datetime.utcnow().isoformat(),
                   last_seen=datetime.utcnow().isoformat(), metadata={})
    id1 = upsert_node(conn, n1)
    id2 = upsert_node(conn, n2)

    # insert edge
    edge = GraphEdge(
        id=str(uuid4()),
        source_id=id1,
        target_id=id2,
        relation="improves",
        weight=0.5,
        confidence=0.8,
        count=1,
        last_reinforced=datetime.utcnow().isoformat()
    )
    upsert_edge(conn, edge)
    ok("Edge inserted")

    # upsert same edge again — count and weight should grow
    upsert_edge(conn, edge)
    edges = get_edges_for_node(conn, id1)
    if not edges:
        fail("get_edges_for_node returned empty list")
    e = edges[0]
    if e.count == 2:
        ok("Edge count incremented to 2 on second upsert")
    else:
        fail(f"Edge count should be 2, got {e.count}")
    if e.weight > 0.5:
        ok(f"Edge weight grew to {e.weight:.3f} on reinforcement")
    else:
        fail(f"Edge weight should have grown above 0.5, got {e.weight}")

    initial_weight = e.weight

    # decay
    decay_all_edges(conn, decay_factor=0.5)
    edges_after_decay = get_edges_for_node(conn, id1)
    if edges_after_decay[0].weight < initial_weight:
        ok(f"decay_all_edges reduced weight: {initial_weight:.3f} → {edges_after_decay[0].weight:.3f}")
    else:
        fail("decay_all_edges did not reduce weight")

    # prune — after 50% decay the edge should be below 0.5, prune at 0.4
    pruned = prune_weak_edges(conn, min_weight=0.4)
    if pruned >= 1:
        ok(f"prune_weak_edges removed {pruned} weak edge(s)")
    else:
        info(f"prune_weak_edges removed {pruned} edges — weight may still be above threshold, check decay math")

    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — embeddings.py
# ══════════════════════════════════════════════════════════════════════════════
def test_embeddings():
    section("STEP 4 — embeddings.py")
    try:
        from memory.graph.embeddings import (
            load_embedding_model,
            get_or_create_collections,
            add_node_embedding,
            remove_node_embedding,
            semantic_search
        )
        from memory.graph.nodes import GraphNode
        import chromadb
    except ImportError as e:
        fail(f"Could not import embeddings.py or chromadb: {e}")

    info("Loading embedding model (first run downloads ~80MB)...")
    model = load_embedding_model()
    ok("Embedding model loaded")

    client = chromadb.PersistentClient(path=TEST_CHROMA_PATH)
    collections = get_or_create_collections(client)

    expected_types = {"entity", "topic", "emotion", "event", "session"}
    if set(collections.keys()) == expected_types:
        ok(f"Collections created for all node types: {list(collections.keys())}")
    else:
        fail(f"Missing collections. Got: {list(collections.keys())}")

    # add two nodes with semantically related names
    node_gym = GraphNode(id="topic_gym_test", name="gym workout fitness",
                         type="topic", mention_count=5,
                         first_seen=datetime.utcnow().isoformat(),
                         last_seen=datetime.utcnow().isoformat(), metadata={})
    node_stress = GraphNode(id="emotion_stress_test", name="stressed anxious overwhelmed",
                            type="emotion", mention_count=3,
                            first_seen=datetime.utcnow().isoformat(),
                            last_seen=datetime.utcnow().isoformat(), metadata={})

    add_node_embedding(collections, node_gym, model)
    add_node_embedding(collections, node_stress, model)
    ok("Node embeddings added to ChromaDB")

    # upsert same node — should not throw
    add_node_embedding(collections, node_gym, model)
    ok("Duplicate add_node_embedding did not throw (upsert works)")

    # semantic search — exercise query should surface gym node
    results = semantic_search(collections, "exercise working out", ["topic"], k=3, model=model)
    if "topic_gym_test" in results:
        ok("semantic_search found gym node for 'exercise working out' query")
    else:
        fail(f"semantic_search did not find gym node. Got: {results}")

    # semantic search across multiple types
    results_multi = semantic_search(collections, "pressure anxiety", ["topic", "emotion"], k=5, model=model)
    if "emotion_stress_test" in results_multi:
        ok("Cross-type semantic_search found stress node")
    else:
        fail(f"Cross-type search failed. Got: {results_multi}")

    # remove a node embedding
    remove_node_embedding(collections, "topic_gym_test", "topic")
    results_after_remove = semantic_search(collections, "exercise", ["topic"], k=3, model=model)
    if "topic_gym_test" not in results_after_remove:
        ok("remove_node_embedding successfully removed node from ChromaDB")
    else:
        fail("remove_node_embedding did not remove node — still appearing in search")

    return model, client, collections


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — agent_graph.py (end-to-end)
# ══════════════════════════════════════════════════════════════════════════════
def test_agent_graph(model):
    section("STEP 5 — agent_graph.py (end-to-end)")
    try:
        from memory.graph.agent_graph import AgentGraph
    except ImportError as e:
        fail(f"Could not import agent_graph.py: {e}")

    graph = AgentGraph(
        db_path=TEST_DB_PATH,
        chroma_path=TEST_CHROMA_PATH,
        embedding_model=model
    )
    ok("AgentGraph instantiated")

    # ingest triples
    triples = [
        {"subject": "Tanish",      "subject_type": "entity",
         "relation": "feels",
         "object": "stressed",     "object_type": "emotion",  "confidence": 0.8},
        {"subject": "stress",      "subject_type": "topic",
         "relation": "correlates_with",
         "object": "low productivity", "object_type": "topic", "confidence": 0.75},
        {"subject": "gym",         "subject_type": "topic",
         "relation": "improves",
         "object": "mood",         "object_type": "emotion",  "confidence": 0.85},
        {"subject": "Tanish",      "subject_type": "entity",
         "relation": "regularly_does",
         "object": "gym",          "object_type": "topic",    "confidence": 0.9},
    ]
    graph.ingest_triples(triples)
    ok(f"Ingested {len(triples)} triples")

    # verify nodes exist in networkx
    node_names = [d.get("name", "") for _, d in graph.G.nodes(data=True)]
    expected_names = {"Tanish", "stressed", "stress", "low productivity", "gym", "mood"}
    found = expected_names.intersection(set(node_names))
    if len(found) == len(expected_names):
        ok(f"All {len(expected_names)} nodes present in networkx graph")
    else:
        missing = expected_names - found
        fail(f"Missing nodes in networkx: {missing}")

    # verify edges in networkx
    edge_count = graph.G.number_of_edges()
    if edge_count == len(triples):
        ok(f"networkx graph has {edge_count} edges (matches triple count)")
    else:
        fail(f"Expected {len(triples)} edges, got {edge_count}")

    # get neighborhood — find Tanish's connections
    tanish_nodes = [n for n, d in graph.G.nodes(data=True) if d.get("name") == "Tanish"]
    if not tanish_nodes:
        fail("Tanish node not found in graph")
    tanish_id = tanish_nodes[0]

    nodes, edges = graph.get_neighborhood(tanish_id, depth=2)
    neighbor_names = [n.name for n in nodes]
    if "Tanish" in neighbor_names:
        ok(f"get_neighborhood returned {len(nodes)} nodes at depth=2")
    else:
        fail("get_neighborhood did not return Tanish node")

    # depth=2 should include gym's connections too (mood)
    if "mood" in neighbor_names:
        ok("Depth=2 traversal reached second-degree node 'mood'")
    else:
        info("'mood' not in depth=2 neighborhood — check ego_graph radius logic")

    # graceful handling of unknown node
    empty_nodes, empty_edges = graph.get_neighborhood("nonexistent_id", depth=2)
    if empty_nodes == [] and empty_edges == []:
        ok("get_neighborhood returns ([], []) for unknown node_id")
    else:
        fail("get_neighborhood should return ([], []) for unknown node_id")

    # semantic search
    results = graph.semantic_search("anxiety pressure tension", k=3)
    result_names = [n.name for n in results]
    info(f"Semantic search 'anxiety pressure' returned: {result_names}")
    if any(name in result_names for name in ["stressed", "stress", "low productivity"]):
        ok("semantic_search returned relevant nodes for anxiety query")
    else:
        fail(f"semantic_search returned irrelevant results: {result_names}")

    # format neighborhood as context
    context_str = graph.format_neighborhood_as_context(nodes, edges)
    if isinstance(context_str, str) and len(context_str) > 10:
        ok(f"format_neighborhood_as_context returned {len(context_str)} char string")
        info(f"Context preview: {context_str[:120]}...")
    else:
        fail("format_neighborhood_as_context returned empty or non-string")

    # consolidate — decay + prune + rebuild
    result = graph.consolidate(decay_factor=0.95, min_weight=0.05)
    if isinstance(result, dict) and "pruned_edges" in result and "pruned_nodes" in result:
        ok(f"consolidate() returned stats: {result}")
    else:
        fail(f"consolidate() should return dict with pruned_edges and pruned_nodes, got: {result}")

    # reload from disk — verify persistence
    graph2 = AgentGraph(
        db_path=TEST_DB_PATH,
        chroma_path=TEST_CHROMA_PATH,
        embedding_model=model
    )
    reloaded_count = graph2.G.number_of_nodes()
    if reloaded_count > 0:
        ok(f"AgentGraph reloaded from disk with {reloaded_count} nodes — persistence works")
    else:
        fail("AgentGraph reloaded empty — serialize/load broken")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{BOLD}{'═'*55}")
    print("  JARVIS — memory/graph/ smoke tests")
    print(f"{'═'*55}{RESET}")

    setup()

    try:
        test_schema()
        node_id_1, node_id_2 = test_nodes()
        test_edges(node_id_1)
        model, client, collections = test_embeddings()
        test_agent_graph(model)

        print(f"\n{BOLD}{GREEN}{'═'*55}")
        print("  ALL TESTS PASSED")
        print(f"{'═'*55}{RESET}\n")

    except SystemExit:
        print(f"\n{BOLD}{RED}{'═'*55}")
        print("  TEST FAILED — fix the error above before continuing")
        print(f"{'═'*55}{RESET}\n")
        sys.exit(1)

    finally:
        teardown()

#!/usr/bin/env python3
"""
Smoke tests for memory/stores/ and memory/retrieval/
Run from project root: python test_stores_retrieval.py
Requires memory/graph/ tests to have passed first.
"""

import os
import sys
import json
import shutil
import sqlite3
import asyncio
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from pathlib import Path

TEST_DB_PATH     = "test_artifacts_sr/test.db"
TEST_CHROMA_PATH = "test_artifacts_sr/chroma"
TEST_CONFIG = {
    "retrieval_k": 3,
    "min_similarity_threshold": 0.0,
    "agent_graph_enabled": True
}

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):  print(f"  {RED}✗{RESET} {msg}"); sys.exit(1)
def info(msg):  print(f"  {YELLOW}→{RESET} {msg}")
def section(t): print(f"\n{BOLD}{CYAN}{'─'*55}{RESET}\n{BOLD}{CYAN}  {t}{RESET}\n{BOLD}{CYAN}{'─'*55}{RESET}")


def setup():
    Path("test_artifacts_sr").mkdir(exist_ok=True)
    if Path(TEST_DB_PATH).exists():       os.remove(TEST_DB_PATH)
    if Path(TEST_CHROMA_PATH).exists():   shutil.rmtree(TEST_CHROMA_PATH)
    info("Test artifacts ready")

def teardown():
    if Path("test_artifacts_sr").exists():
        shutil.rmtree("test_artifacts_sr")
    info("Test artifacts cleaned up")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — facts.py
# ══════════════════════════════════════════════════════════════════════════════
def test_facts():
    section("STEP 1 — memory/stores/facts.py")
    try:
        from memory.graph.schema import init_db
        from memory.stores.facts import (
            Fact, upsert_fact, get_fact, get_all_facts,
            get_facts_by_category, delete_fact, format_facts_for_prompt
        )
    except ImportError as e:
        fail(f"Import failed: {e}")

    conn = init_db(TEST_DB_PATH)

    # insert a fact
    fid = upsert_fact(conn, key="name", value="Tanish",
                      category="identity", source="user_explicit")
    ok(f"Fact inserted: id {fid[:8]}...")

    # upsert same key — should update, not duplicate
    fid2 = upsert_fact(conn, key="name", value="Tanish Sharma",
                       category="identity", source="user_explicit")
    if fid == fid2:
        ok("Duplicate key upsert returned same id")
    else:
        fail("Duplicate key created a new row — key uniqueness broken")

    fetched = get_fact(conn, "name")
    if fetched and fetched.value == "Tanish Sharma":
        ok("get_fact returns updated value after upsert")
    else:
        fail(f"get_fact returned wrong value: {fetched}")

    # insert more facts
    upsert_fact(conn, key="wake_time",  value="07:30",          category="routine")
    upsert_fact(conn, key="location",   value="Fremont, CA",    category="identity")
    upsert_fact(conn, key="llm_model",  value="qwen2.5:3b",     category="preference")

    # get all facts — returns flat dict
    all_facts = get_all_facts(conn)
    if isinstance(all_facts, dict) and "name" in all_facts and "wake_time" in all_facts:
        ok(f"get_all_facts returned {len(all_facts)} facts as dict")
    else:
        fail(f"get_all_facts should return dict with all keys, got: {type(all_facts)}")

    # get by category
    identity_facts = get_facts_by_category(conn, "identity")
    if len(identity_facts) >= 2:
        ok(f"get_facts_by_category returned {len(identity_facts)} identity facts")
    else:
        fail(f"Expected >= 2 identity facts, got {len(identity_facts)}")

    # format for prompt
    prompt_str = format_facts_for_prompt(all_facts)
    if "[KNOWN FACTS]" in prompt_str and "Tanish" in prompt_str:
        ok(f"format_facts_for_prompt output ({len(prompt_str)} chars) contains header and values")
    else:
        fail(f"format_facts_for_prompt missing header or values: {prompt_str[:100]}")

    # token estimate — should be under 200 tokens
    token_estimate = len(prompt_str) // 4
    if token_estimate <= 200:
        ok(f"Prompt token estimate: ~{token_estimate} tokens (within 200 budget)")
    else:
        info(f"Warning: fact prompt is ~{token_estimate} tokens — may exceed 200 token budget")

    # delete a fact
    deleted = delete_fact(conn, "llm_model")
    if deleted:
        ok("delete_fact returned True for existing key")
    else:
        fail("delete_fact returned False for existing key")

    gone = get_fact(conn, "llm_model")
    if gone is None:
        ok("Deleted fact no longer retrievable")
    else:
        fail("Deleted fact still returned by get_fact")

    # delete nonexistent — should return False not raise
    deleted_missing = delete_fact(conn, "nonexistent_key")
    if not deleted_missing:
        ok("delete_fact returns False for nonexistent key (no exception)")
    else:
        fail("delete_fact returned True for nonexistent key")

    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — episodes.py
# ══════════════════════════════════════════════════════════════════════════════
def test_episodes():
    section("STEP 2 — memory/stores/episodes.py")
    try:
        from memory.graph.schema import init_db
        from memory.stores.episodes import (
            Episode, add_episode, get_episodes_by_date_range,
            get_episodes_by_tag, get_recent_episodes,
            get_episodes_by_keyword, format_episodes_for_prompt
        )
    except ImportError as e:
        fail(f"Import failed: {e}")

    conn = init_db(TEST_DB_PATH)
    now  = datetime.now(timezone.utc)

    # insert episodes at different timestamps
    sid = str(uuid4())
    conn.execute(
        """
        INSERT INTO sessions (id, agent_id, date, duration_secs, transcript_path, consolidated)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (sid, "agent_0", now.isoformat(), 0, None, 0),
    )
    conn.commit()
    id1 = add_episode(conn,
        description="started Jarvis project on Raspberry Pi",
        agent_id="agent_0",
        tags=["project", "jarvis", "raspberrypi"],
        importance=0.9,
        session_id=sid)
    ok(f"Episode inserted: {id1[:8]}...")

    id2 = add_episode(conn,
        description="discussed memory architecture with assistant",
        agent_id="agent_0",
        tags=["memory", "architecture"],
        importance=0.8)

    id3 = add_episode(conn,
        description="went to the gym and felt energised",
        agent_id="agent_0",
        tags=["gym", "health", "mood"],
        importance=0.5,)

    # insert a journal episode
    id4 = add_episode(conn,
        description="reflected on work stress during journal session",
        agent_id="agent_1",
        tags=["journal", "stress", "work"],
        importance=0.85)

    ok(f"Inserted 4 test episodes")

    # episodes are never deduplicated — all 4 should exist
    recent = get_recent_episodes(conn, limit=10)
    if len(recent) == 4:
        ok("get_recent_episodes returns all 4 unique episodes")
    else:
        fail(f"Expected 4 episodes, got {len(recent)}")

    # date range — all within last minute
    start = (now - timedelta(minutes=1)).isoformat()
    end   = (now + timedelta(minutes=1)).isoformat()
    in_range = get_episodes_by_date_range(conn, start, end)
    if len(in_range) == 4:
        ok("get_episodes_by_date_range returns all episodes in range")
    else:
        fail(f"Expected 4 in range, got {len(in_range)}")

    # date range with agent filter
    agent1_episodes = get_episodes_by_date_range(conn, start, end, agent_id="agent_1")
    if len(agent1_episodes) == 1:
        ok("Date range + agent_id filter returns correct subset")
    else:
        fail(f"Expected 1 agent_1 episode, got {len(agent1_episodes)}")

    # empty date range
    future_start = (now + timedelta(days=1)).isoformat()
    future_end   = (now + timedelta(days=2)).isoformat()
    empty = get_episodes_by_date_range(conn, future_start, future_end)
    if empty == []:
        ok("Empty date range returns empty list")
    else:
        fail(f"Future date range should return [], got {len(empty)} results")

    # tag search
    health_eps = get_episodes_by_tag(conn, "gym")
    if len(health_eps) >= 1 and any("gym" in e.description for e in health_eps):
        ok("get_episodes_by_tag finds gym-tagged episode")
    else:
        fail(f"Tag search for 'gym' failed: {health_eps}")

    # keyword search
    kw_results = get_episodes_by_keyword(conn, "memory")
    if any("memory" in e.description for e in kw_results):
        ok("get_episodes_by_keyword finds 'memory' in description")
    else:
        fail(f"Keyword search for 'memory' returned: {[e.description for e in kw_results]}")

    # keyword no match — should return [] not raise
    no_match = get_episodes_by_keyword(conn, "xyznonexistentword")
    if no_match == []:
        ok("get_episodes_by_keyword returns [] for no match")
    else:
        fail(f"Expected [] for no match, got {no_match}")

    # recent with agent filter
    recent_j = get_recent_episodes(conn, limit=5, agent_id="agent_1")
    if len(recent_j) == 1 and recent_j[0].agent_id == "agent_1":
        ok("get_recent_episodes with agent_id filter returns correct subset")
    else:
        fail(f"Expected 1 journal episode, got {len(recent_j)}")

    # tags are parsed back to list (not raw JSON string)
    ep = get_recent_episodes(conn, limit=1)[0]
    if isinstance(ep.tags, list):
        ok("Episode tags deserialized as list (not raw JSON string)")
    else:
        fail(f"Episode tags should be list, got {type(ep.tags)}: {ep.tags}")

    # format for prompt
    all_eps = get_recent_episodes(conn, limit=4)
    prompt = format_episodes_for_prompt(all_eps)
    if "[RECENT EPISODES]" in prompt and len(prompt) > 20:
        ok(f"format_episodes_for_prompt output ({len(prompt)} chars) has header")
    else:
        fail(f"format_episodes_for_prompt missing header: {prompt[:100]}")

    conn.close()
    return True


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — assistant_retrieve.py
# ══════════════════════════════════════════════════════════════════════════════
def test_assistant_retrieve(model, agent_graph, conn):
    section("STEP 3 — memory/retrieval/assistant_retrieve.py")
    try:
        from memory.retrieval.assistant_retrieve import assistant_retrieve
    except ImportError as e:
        fail(f"Import failed: {e}")

    # seed some data into ChromaDB conversation_summaries
    try:
        col = agent_graph.chroma_client.get_or_create_collection("conversation_summaries")
        col.upsert(
            ids=["sum_001", "sum_002", "sum_003"],
            documents=[
                "User discussed building a voice assistant called Jarvis on Raspberry Pi",
                "Conversation about memory architecture using ChromaDB and SQLite",
                "User mentioned feeling stressed about project deadlines"
            ],
            metadatas=[{"agent_id": "agent_0"}] * 3
        )
        ok("Seeded 3 conversation summaries into ChromaDB")
    except Exception as e:
        fail(f"Could not seed ChromaDB: {e}")

    # seed episodes
    from memory.stores.episodes import add_episode
    add_episode(conn, "discussed Jarvis project architecture",
                agent_id="agent_0", tags=["jarvis", "project"], importance=0.9)
    add_episode(conn, "mentioned stress about deadlines",
                agent_id="agent_0", tags=["stress"], importance=0.7)

    # run retrieval
    result = asyncio.run(assistant_retrieve(
        transcript="tell me about the Jarvis project and memory system",
        agent_graph=agent_graph,
        conn=conn,
        config=TEST_CONFIG,
        token_budget=500
    ))

    # validate return shape
    required_keys = {"summaries", "episodes", "token_estimate"}
    if required_keys.issubset(result.keys()):
        ok(f"assistant_retrieve returned correct keys: {list(result.keys())}")
    else:
        fail(f"Missing keys. Got: {list(result.keys())}, expected: {required_keys}")

    if isinstance(result["summaries"], list) and len(result["summaries"]) > 0:
        ok(f"Returned {len(result['summaries'])} summaries")
    else:
        fail(f"summaries should be non-empty list, got: {result['summaries']}")

    if isinstance(result["token_estimate"], int) and result["token_estimate"] > 0:
        ok(f"token_estimate is {result['token_estimate']}")
    else:
        fail(f"token_estimate should be positive int, got: {result['token_estimate']}")

    # low token budget — should reduce K
    low_budget_result = asyncio.run(assistant_retrieve(
        transcript="Jarvis project",
        agent_graph=agent_graph,
        conn=conn,
        config=TEST_CONFIG,
        token_budget=150  # below 200 threshold
    ))
    if len(low_budget_result["summaries"]) <= 1:
        ok("Low token budget reduces retrieval K to 1")
    else:
        info(f"Low budget returned {len(low_budget_result['summaries'])} summaries — check pressure valve logic")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — journal_retrieve.py
# ══════════════════════════════════════════════════════════════════════════════
def test_journal_retrieve(model, agent_graph, conn):
    section("STEP 4 — memory/retrieval/journal_retrieve.py")
    try:
        from memory.retrieval.journal_retrieve import journal_retrieve
    except ImportError as e:
        fail(f"Import failed: {e}")

    # seed journal entries in ChromaDB
    try:
        col = agent_graph.chroma_client.get_or_create_collection("journal_entries")
        col.upsert(
            ids=["jour_001", "jour_002"],
            documents=[
                "Reflected on work stress and feeling overwhelmed by project scope",
                "Felt motivated after gym session, clarity about project direction"
            ],
            metadatas=[{"agent_id": "agent_1", "date": "2026-06-10"},
                       {"agent_id": "agent_1", "date": "2026-06-11"}]
        )
        ok("Seeded 2 journal entries into ChromaDB")
    except Exception as e:
        fail(f"Could not seed journal ChromaDB: {e}")

    # seed journal episodes
    from memory.stores.episodes import add_episode
    add_episode(conn, "journal session about work stress",
                agent_id="agent_1", tags=["journal", "stress"], importance=0.9)

    result = asyncio.run(journal_retrieve(
        transcript="I want to reflect on how I have been feeling lately",
        agent_graph=agent_graph,
        conn=conn,
        config=TEST_CONFIG
    ))

    required_keys = {"recent_episodes", "semantic_summaries", "last_session_meta", "token_estimate"}
    if required_keys.issubset(result.keys()):
        ok(f"journal_retrieve returned correct keys")
    else:
        fail(f"Missing keys. Got: {list(result.keys())}, expected: {required_keys}")

    if isinstance(result["semantic_summaries"], list) and len(result["semantic_summaries"]) > 0:
        ok(f"Returned {len(result['semantic_summaries'])} journal semantic summaries")
    else:
        fail(f"semantic_summaries should be non-empty list, got: {result['semantic_summaries']}")

    if isinstance(result["recent_episodes"], list):
        ok(f"recent_episodes returned as list ({len(result['recent_episodes'])} items)")
    else:
        fail(f"recent_episodes should be list, got: {type(result['recent_episodes'])}")

    # critically — must NOT touch conversation_summaries
    # seed a conversation summary that should never appear in journal results
    conv_col = agent_graph.chroma_client.get_or_create_collection("conversation_summaries")
    conv_col.upsert(
        ids=["conv_sentinel"],
        documents=["SENTINEL: this should never appear in journal retrieval"],
        metadatas=[{"agent_id": "agent_0"}]
    )
    result2 = asyncio.run(journal_retrieve(
        transcript="reflecting on feelings",
        agent_graph=agent_graph,
        conn=conn,
        config=TEST_CONFIG
    ))
    sentinel_found = any("SENTINEL" in s for s in result2.get("semantic_summaries", []))
    if not sentinel_found:
        ok("journal_retrieve does NOT query conversation_summaries (isolation confirmed)")
    else:
        fail("journal_retrieve is leaking into conversation_summaries collection")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — graph_retrieve.py
# ══════════════════════════════════════════════════════════════════════════════
def test_graph_retrieve(agent_graph):
    section("STEP 5 — memory/retrieval/graph_retrieve.py")
    try:
        from memory.retrieval.graph_retrieve import graph_retrieve
    except ImportError as e:
        fail(f"Import failed: {e}")

    # seed knowledge graph with triples
    triples = [
        {"subject": "Tanish",      "subject_type": "entity",
         "relation": "feels",
         "object": "stressed",     "object_type": "emotion", "confidence": 0.8},
        {"subject": "stress",      "subject_type": "topic",
         "relation": "correlates_with",
         "object": "low productivity", "object_type": "topic", "confidence": 0.7},
        {"subject": "gym",         "subject_type": "topic",
         "relation": "improves",
         "object": "mood",         "object_type": "emotion", "confidence": 0.85},
    ]
    agent_graph.ingest_triples(triples)
    ok(f"Seeded {len(triples)} triples into knowledge graph")

    # standard retrieval
    result = asyncio.run(graph_retrieve(
        transcript="how has stress been affecting my productivity",
        agent_graph=agent_graph,
        config=TEST_CONFIG,
        max_tokens=200
    ))

    required_keys = {"graph_context", "matched_nodes", "token_estimate"}
    if required_keys.issubset(result.keys()):
        ok("graph_retrieve returned correct keys")
    else:
        fail(f"Missing keys. Got: {list(result.keys())}")

    if isinstance(result["graph_context"], str) and len(result["graph_context"]) > 0:
        ok(f"graph_context returned ({len(result['graph_context'])} chars)")
        info(f"Context preview: {result['graph_context'][:100]}...")
    else:
        fail(f"graph_context should be non-empty string, got: {result['graph_context']!r}")

    if isinstance(result["matched_nodes"], list):
        ok(f"matched_nodes: {result['matched_nodes']}")
    else:
        fail(f"matched_nodes should be list, got: {type(result['matched_nodes'])}")

    # token limit respected
    token_est = result["token_estimate"]
    if token_est <= 200:
        ok(f"token_estimate {token_est} within max_tokens=200 budget")
    else:
        info(f"token_estimate {token_est} exceeds max_tokens=200 — check trimming logic")

    # disabled flag returns empty no-op
    disabled_config = {**TEST_CONFIG, "agent_graph_enabled": False}
    result_disabled = asyncio.run(graph_retrieve(
        transcript="stress productivity",
        agent_graph=agent_graph,
        config=disabled_config
    ))
    if result_disabled["graph_context"] == "" and result_disabled["matched_nodes"] == []:
        ok("graph_retrieve returns no-op when agent_graph_enabled=False")
    else:
        fail("graph_retrieve should return empty result when disabled")

    # unknown query — no matching nodes — graceful empty
    result_empty = asyncio.run(graph_retrieve(
        transcript="xyzunknownentityword",
        agent_graph=agent_graph,
        config=TEST_CONFIG
    ))
    if isinstance(result_empty["graph_context"], str):
        ok("graph_retrieve handles zero-match query gracefully (no exception)")
    else:
        fail("graph_retrieve raised on zero-match query")


# ══════════════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ══════════════════════════════════════════════════════════════════════════════
def build_fixtures():
    """Build shared AgentGraph + conn used across retrieval tests."""
    try:
        from memory.graph.schema import init_db
        from memory.graph.agent_graph import AgentGraph
    except ImportError as e:
        fail(f"Could not import graph layer: {e}")

    class FallbackEmbeddingModel:
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
            return [[0.0] * 384 for _ in texts]

    try:
        from memory.graph.embeddings import load_embedding_model

        info("Loading embedding model...")
        model = load_embedding_model()
        ok("Embedding model loaded")
    except ImportError:
        info("sentence-transformers unavailable; using fallback embedding stub")
        model = FallbackEmbeddingModel()

    conn = init_db(TEST_DB_PATH)
    graph = AgentGraph(
        db_path=TEST_DB_PATH,
        chroma_path=TEST_CHROMA_PATH,
        embedding_model=model
    )
    ok("AgentGraph instantiated")
    return model, graph, conn


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{BOLD}{'═'*55}")
    print("  JARVIS — memory/stores/ + memory/retrieval/ smoke tests")
    print(f"{'═'*55}{RESET}")

    setup()

    try:
        test_facts()
        test_episodes()

        model, agent_graph, conn = build_fixtures()

        test_assistant_retrieve(model, agent_graph, conn)
        test_journal_retrieve(model, agent_graph, conn)
        test_graph_retrieve(agent_graph)

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

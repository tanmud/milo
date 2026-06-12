from __future__ import annotations

import sqlite3


NODE_TABLE = "nodes"
EDGE_TABLE = "edges"
SESSION_TABLE = "sessions"
AGENT_SUMMARIES_TABLE = "agent_summaries"
FACTS_TABLE = "facts"
EPISODES_TABLE = "episodes"
JOURNAL_METADATA_TABLE = "journal_metadata"


def init_db(db_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('entity', 'topic', 'emotion', 'event', 'session')),
            mention_count INTEGER DEFAULT 1,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            metadata TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
            target_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
            relation TEXT NOT NULL,
            weight REAL DEFAULT 0.5,
            confidence REAL DEFAULT 0.5,
            count INTEGER DEFAULT 1,
            last_reinforced TEXT NOT NULL,
            UNIQUE(source_id, target_id, relation)
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            date TEXT NOT NULL,
            duration_secs INTEGER DEFAULT 0,
            transcript_path TEXT,
            consolidated INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS agent_summaries (
            agent_id TEXT PRIMARY KEY,
            last_active TEXT,
            current_theme TEXT,
            recent_insight TEXT
        );

        CREATE TABLE IF NOT EXISTS facts (
            id TEXT PRIMARY KEY,
            key TEXT NOT NULL UNIQUE,
            value TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            confidence REAL DEFAULT 1.0,
            last_updated TEXT NOT NULL,
            source TEXT
        );

        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            tags TEXT DEFAULT '[]',
            importance REAL DEFAULT 0.5,
            session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS journal_metadata (
            agent_id TEXT PRIMARY KEY,
            last_session_date TEXT,
            duration_secs INTEGER DEFAULT 0,
            themes TEXT DEFAULT '[]'
        );
        """
    )
    connection.commit()
    return connection
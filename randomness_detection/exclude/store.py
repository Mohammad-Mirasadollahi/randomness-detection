"""SQLite-backed exclusion and score-cache storage."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Literal

RuleType = Literal["exact", "domain", "suffix", "prefix", "glob"]

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA mmap_size=268435456;

CREATE TABLE IF NOT EXISTS exact_rules (
    pattern TEXT PRIMARY KEY NOT NULL,
    rule_type TEXT NOT NULL,
    created_at INTEGER NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS wildcard_rules (
    pattern TEXT PRIMARY KEY NOT NULL,
    rule_type TEXT NOT NULL,
    normalized TEXT NOT NULL,
    created_at INTEGER NOT NULL
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_wildcard_type ON wildcard_rules(rule_type);

CREATE TABLE IF NOT EXISTS score_cache (
    cache_key TEXT PRIMARY KEY NOT NULL,
    score INTEGER NOT NULL,
    label TEXT NOT NULL,
    confidence TEXT NOT NULL,
    breakdown_json TEXT NOT NULL,
    updated_at INTEGER NOT NULL
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_score_cache_score ON score_cache(score);
"""


class ExcludeStore:
    """Persistent store for exact rules and score cache."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure()

    def _configure(self) -> None:
        with self._lock:
            for statement in _SCHEMA.strip().split(";"):
                sql = statement.strip()
                if sql:
                    self._conn.execute(sql)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def count_exact(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM exact_rules").fetchone()
            return int(row["c"])

    def count_wildcards(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM wildcard_rules").fetchone()
            return int(row["c"])

    def count_score_cache(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM score_cache").fetchone()
            return int(row["c"])

    def has_exact(self, pattern: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM exact_rules WHERE pattern = ? LIMIT 1",
                (pattern,),
            ).fetchone()
            return row is not None

    def find_exact_domains(self, domains: list[str]) -> set[str]:
        if not domains:
            return set()
        placeholders = ",".join("?" for _ in domains)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT pattern FROM exact_rules WHERE rule_type = 'domain' AND pattern IN ({placeholders})",
                domains,
            ).fetchall()
        return {str(row["pattern"]) for row in rows}

    def find_exact_many(self, patterns: list[str]) -> set[str]:
        if not patterns:
            return set()
        placeholders = ",".join("?" for _ in patterns)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT pattern FROM exact_rules WHERE pattern IN ({placeholders})",
                patterns,
            ).fetchall()
        return {str(row["pattern"]) for row in rows}

    def add_exact_many(self, items: Iterable[tuple[str, RuleType]]) -> tuple[int, int]:
        now = int(time.time())
        added = 0
        duplicates = 0
        with self._lock:
            cursor = self._conn.cursor()
            for pattern, rule_type in items:
                try:
                    cursor.execute(
                        "INSERT INTO exact_rules(pattern, rule_type, created_at) VALUES (?, ?, ?)",
                        (pattern, rule_type, now),
                    )
                    added += 1
                except sqlite3.IntegrityError:
                    duplicates += 1
            self._conn.commit()
        return added, duplicates

    def remove_exact_many(self, patterns: Iterable[str]) -> int:
        patterns = list(patterns)
        if not patterns:
            return 0
        placeholders = ",".join("?" for _ in patterns)
        with self._lock:
            cursor = self._conn.execute(
                f"DELETE FROM exact_rules WHERE pattern IN ({placeholders})",
                patterns,
            )
            self._conn.commit()
            return cursor.rowcount

    def list_wildcards(self) -> list[tuple[str, RuleType, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT pattern, rule_type, normalized FROM wildcard_rules"
            ).fetchall()
        return [(str(row["pattern"]), row["rule_type"], str(row["normalized"])) for row in rows]

    def add_wildcards_many(self, items: Iterable[tuple[str, RuleType, str]]) -> tuple[int, int]:
        now = int(time.time())
        added = 0
        duplicates = 0
        with self._lock:
            cursor = self._conn.cursor()
            for pattern, rule_type, normalized in items:
                try:
                    cursor.execute(
                        "INSERT INTO wildcard_rules(pattern, rule_type, normalized, created_at) VALUES (?, ?, ?, ?)",
                        (pattern, rule_type, normalized, now),
                    )
                    added += 1
                except sqlite3.IntegrityError:
                    duplicates += 1
            self._conn.commit()
        return added, duplicates

    def remove_wildcards_many(self, patterns: Iterable[str]) -> int:
        patterns = list(patterns)
        if not patterns:
            return 0
        placeholders = ",".join("?" for _ in patterns)
        with self._lock:
            cursor = self._conn.execute(
                f"DELETE FROM wildcard_rules WHERE pattern IN ({placeholders})",
                patterns,
            )
            self._conn.commit()
            return cursor.rowcount

    def get_score_cache(self, key: str) -> dict[str, object] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT score, label, confidence, breakdown_json FROM score_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "score": int(row["score"]),
            "label": str(row["label"]),
            "confidence": str(row["confidence"]),
            "breakdown_json": str(row["breakdown_json"]),
        }

    def get_score_cache_many(self, keys: list[str]) -> dict[str, dict[str, object]]:
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT cache_key, score, label, confidence, breakdown_json FROM score_cache WHERE cache_key IN ({placeholders})",
                keys,
            ).fetchall()
        return {
            str(row["cache_key"]): {
                "score": int(row["score"]),
                "label": str(row["label"]),
                "confidence": str(row["confidence"]),
                "breakdown_json": str(row["breakdown_json"]),
            }
            for row in rows
        }

    def put_score_cache_many(self, items: Iterable[tuple[str, int, str, str, str]]) -> None:
        now = int(time.time())
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO score_cache(cache_key, score, label, confidence, breakdown_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    score=excluded.score,
                    label=excluded.label,
                    confidence=excluded.confidence,
                    breakdown_json=excluded.breakdown_json,
                    updated_at=excluded.updated_at
                """,
                [(key, score, label, confidence, breakdown_json, now) for key, score, label, confidence, breakdown_json in items],
            )
            self._conn.commit()

    def vacuum(self) -> None:
        with self._lock:
            self._conn.execute("VACUUM")
            self._conn.commit()

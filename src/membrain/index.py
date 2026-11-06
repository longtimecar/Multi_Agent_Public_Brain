"""Derived SQLite index: FTS5 for lexical, brute-force cosine for semantic.

Nothing here is authoritative. Delete `.index/` and everything rebuilds from
Markdown. That is the durability property the whole design rests on, so the
index is written to be trivially disposable.

Brute-force cosine over float32 blobs is used instead of a native ANN
extension: at personal scale (thousands of facts) a full scan costs single-digit
milliseconds, and it avoids a compiled dependency that breaks on Windows.
The `search()` interface is narrow enough to swap in sqlite-vec later.

Scheme A adds uri / sensitive / peer / abstract columns (schema_version 2).
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .models import Episode, Fact

SCHEMA_VERSION = 2
_TOKEN_RE = re.compile(r"[\w一-鿿]+", re.UNICODE)


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 query.

    Quoting every token avoids syntax errors from punctuation, and OR-ing them
    stops a single rare word from excluding otherwise good matches. Ranking is
    left to bm25().
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return ""
    uniq = list(dict.fromkeys(tokens))[:32]
    return " OR ".join(f'"{t}"' for t in uniq)


# CJK runs of 2+ characters and latin words of 3+. FTS5's unicode61 tokenizer
# treats a whole CJK run as a single token, so Chinese questions need substring
# matching rather than token matching.
_MATCH_TERM_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{3,}")
# With no time phrase in the question, the term scan looks back this far instead
# of reading every episode ever captured.
EPISODE_SCAN_DAYS = 30


def _match_terms(text: str) -> list[str]:
    """Substrings worth looking for in a raw turn, longest CJK run split to bigrams."""
    terms: list[str] = []
    for token in _MATCH_TERM_RE.findall((text or "").casefold()):
        if "\u4e00" <= token[0] <= "\u9fff" and len(token) > 2:
            # A long CJK run rarely appears verbatim in a different phrasing, so
            # also try its bigrams: "昨天的任务" still matches "任务" and "昨天".
            terms.extend(token[i:i + 2] for i in range(len(token) - 1))
        terms.append(token)
    return list(dict.fromkeys(terms))[:40]


@dataclass
class Hit:
    fact: Fact
    lexical: float = 0.0
    semantic: float = 0.0
    # A structured match on the type and day the question names ("我昨晚说的
    # 任务"). It outranks topical similarity, so it needs its own field rather
    # than being folded into lexical score where one strong semantic neighbour
    # can push it off the first page.
    typed: float = 0.0


class Index:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create()
        self._migrate()

    # ------------------------------------------------------------- schema

    def _create(self) -> None:
        c = self._conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                subject TEXT NOT NULL,
                claim TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                n_episodes INTEGER NOT NULL,
                body TEXT NOT NULL,
                uri TEXT,
                sensitive INTEGER NOT NULL DEFAULT 0,
                peer TEXT,
                abstract TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                id UNINDEXED,
                subject,
                claim,
                body,
                uri,
                abstract,
                tokenize = 'unicode61 remove_diacritics 2'
            );
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                dim INTEGER NOT NULL,
                vec BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                day TEXT NOT NULL,
                ts TEXT NOT NULL,
                ts_source TEXT NOT NULL,
                session TEXT NOT NULL,
                agent TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS episodes_day ON episodes(day);
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                id UNINDEXED,
                text,
                tokenize = 'unicode61 remove_diacritics 2'
            );
            """
        )
        c.commit()

    def _migrate(self) -> None:
        """Additive column migration for pre-Scheme-A indexes."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}
        alters = []
        if "uri" not in cols:
            alters.append("ALTER TABLE facts ADD COLUMN uri TEXT")
        if "sensitive" not in cols:
            alters.append("ALTER TABLE facts ADD COLUMN sensitive INTEGER NOT NULL DEFAULT 0")
        if "peer" not in cols:
            alters.append("ALTER TABLE facts ADD COLUMN peer TEXT")
        if "abstract" not in cols:
            alters.append("ALTER TABLE facts ADD COLUMN abstract TEXT")
        for sql in alters:
            self._conn.execute(sql)
        if alters:
            self._set_meta("schema_version", str(SCHEMA_VERSION))
            self._conn.commit()
        elif self._get_meta("schema_version") != str(SCHEMA_VERSION):
            self._set_meta("schema_version", str(SCHEMA_VERSION))
            self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        """The shared connection, for other derived tables such as the archive."""
        return self._conn

    def _get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def model_matches(self, model: str | None) -> bool:
        if model is None:
            return True
        return self._get_meta("embed_model") == model

    def claim_model(self, model: str) -> bool:
        """Record which embedding model this index's vectors come from.

        The first model to write a vector claims the index; later writes from a
        different model are refused so two vector spaces never mix. A full
        reindex resets the claim.
        """
        current = self._get_meta("embed_model")
        if current is None:
            self._set_meta("embed_model", model)
            return True
        return current == model

    # -------------------------------------------------------------- writes

    def upsert(self, fact: Fact, body: str = "") -> None:
        self._conn.execute(
            """
            INSERT INTO facts(id, type, subject, claim, status, confidence, valid_from, valid_to,
                              n_episodes, body, uri, sensitive, peer, abstract)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                type=excluded.type, subject=excluded.subject, claim=excluded.claim,
                status=excluded.status, confidence=excluded.confidence,
                valid_from=excluded.valid_from, valid_to=excluded.valid_to,
                n_episodes=excluded.n_episodes, body=excluded.body,
                uri=excluded.uri, sensitive=excluded.sensitive, peer=excluded.peer,
                abstract=excluded.abstract
            """,
            (
                fact.id,
                fact.type.value,
                fact.subject,
                fact.claim,
                fact.status.value,
                fact.confidence,
                fact.valid_from.isoformat(),
                fact.valid_to.isoformat() if fact.valid_to else None,
                fact.n_episodes,
                body,
                fact.uri,
                1 if fact.sensitive else 0,
                fact.peer,
                fact.abstract,
            ),
        )
        self._conn.execute("DELETE FROM facts_fts WHERE id = ?", (fact.id,))
        try:
            self._conn.execute(
                "INSERT INTO facts_fts(id, subject, claim, body, uri, abstract) VALUES(?,?,?,?,?,?)",
                (
                    fact.id,
                    fact.subject.replace("-", " "),
                    fact.claim,
                    body,
                    fact.uri or "",
                    fact.abstract or "",
                ),
            )
        except sqlite3.OperationalError:
            # Older FTS table without uri/abstract columns — recreate on next reset/reindex.
            self._conn.execute(
                "INSERT INTO facts_fts(id, subject, claim, body) VALUES(?,?,?,?)",
                (fact.id, fact.subject.replace("-", " "), fact.claim, body),
            )

    def put_vector(self, fact_id: str, vec: np.ndarray) -> None:
        self._conn.execute(
            "INSERT INTO vectors(id, dim, vec) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
            (fact_id, int(vec.shape[0]), vec.astype(np.float32).tobytes()),
        )

    def commit(self) -> None:
        self._conn.commit()

    def reset(self) -> None:
        """Drop derived state. Called before a full rebuild."""
        self._conn.executescript(
            "DELETE FROM facts; DELETE FROM facts_fts; DELETE FROM vectors; DELETE FROM meta;"
            " DELETE FROM episodes; DELETE FROM episodes_fts;"
        )
        # Recreate FTS with Scheme A columns if an old virtual table shape remains.
        self._conn.execute("DROP TABLE IF EXISTS facts_fts")
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE facts_fts USING fts5(
                id UNINDEXED,
                subject,
                claim,
                body,
                uri,
                abstract,
                tokenize = 'unicode61 remove_diacritics 2'
            )
            """
        )
        self._set_meta("schema_version", str(SCHEMA_VERSION))
        self._conn.commit()

    # --------------------------------------------------------------- reads

    def lexical(self, query: str, limit: int = 50) -> dict[str, float]:
        q = fts_query(query)
        if not q:
            return {}
        try:
            rows = self._conn.execute(
                "SELECT id, bm25(facts_fts) AS score FROM facts_fts WHERE facts_fts MATCH ? ORDER BY score LIMIT ?",
                (q, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        # bm25() is negative and lower-is-better; invert so higher is better.
        return {r["id"]: -float(r["score"]) for r in rows}

    # ------------------------------------------------------- episode recall
    # Facts are the curated layer. When they come up thin, the raw captured
    # turns still answer "what did I say last night" — so episodes need their
    # own searchable index instead of being write-only.

    def upsert_episode(self, ep: Episode) -> None:
        day = ep.local_date.isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO episodes (id, day, ts, ts_source, session, agent, role, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ep.id, day, ep.ts.isoformat(), ep.ts_source, ep.session, ep.agent, ep.role, ep.text),
        )
        self._conn.execute("DELETE FROM episodes_fts WHERE id = ?", (ep.id,))
        self._conn.execute("INSERT INTO episodes_fts (id, text) VALUES (?, ?)", (ep.id, ep.text))

    def search_episodes(self, query: str, start: date | None = None, end: date | None = None,
                        limit: int = 5, role: str | None = None) -> list[dict[str, Any]]:
        """Search raw turns, optionally bounded to a date window. Newest first.

        Two matchers are merged. FTS5 handles latin text well but its unicode61
        tokenizer treats a whole CJK run as one token, so a Chinese question
        would find nothing; the term scan covers that. The scan is bounded by the
        day index (and by recent days when the question carries no time phrase),
        so it reads a slice of the table rather than all of it.

        Empty results are normal here, so this never raises: the caller falls
        back to fact-only recall.
        """
        found: dict[str, dict[str, Any]] = {}
        for row in self._fts_episodes(query, start, end, limit * 3, role):
            found[row["episode"]] = row
        for row in self._scan_episodes(query, start, end, limit * 3, role):
            previous = found.get(row["episode"])
            if previous is None or row["score"] > previous["score"]:
                found[row["episode"]] = row
        ranked = sorted(found.values(), key=lambda r: (r["score"], r["date"]), reverse=True)
        return ranked[:limit]

    def _fts_episodes(self, query: str, start: date | None, end: date | None,
                      limit: int, role: str | None) -> list[dict[str, Any]]:
        q = fts_query(query)
        if not q:
            return []
        sql = ("SELECT e.id, e.day, e.ts, e.ts_source, e.session, e.agent, e.role, e.text, "
               "bm25(episodes_fts) AS score FROM episodes_fts JOIN episodes e ON e.id = episodes_fts.id "
               "WHERE episodes_fts MATCH ?")
        args: list[Any] = [q]
        if start is not None:
            sql += " AND e.day >= ?"
            args.append(start.isoformat())
        if end is not None:
            sql += " AND e.day <= ?"
            args.append(end.isoformat())
        if role:
            sql += " AND e.role = ?"
            args.append(role)
        sql += " ORDER BY score LIMIT ?"
        args.append(limit)
        try:
            rows = self._conn.execute(sql, args).fetchall()
        except sqlite3.OperationalError:
            return []
        return [self._episode_row(r, -float(r["score"])) for r in rows]

    def _scan_episodes(self, query: str, start: date | None, end: date | None,
                       limit: int, role: str | None) -> list[dict[str, Any]]:
        terms = _match_terms(query)
        if not terms:
            return []
        if start is None:
            start = date.today() - timedelta(days=EPISODE_SCAN_DAYS)
        sql = ("SELECT id, day, ts, ts_source, session, agent, role, text FROM episodes "
               "WHERE day >= ? AND day <= ?")
        args: list[Any] = [start.isoformat(), (end or date.today()).isoformat()]
        if role:
            sql += " AND role = ?"
            args.append(role)
        try:
            rows = self._conn.execute(sql, args).fetchall()
        except sqlite3.OperationalError:
            return []
        scored: list[tuple[float, Any]] = []
        for r in rows:
            low = (r["text"] or "").casefold()
            hits = sum(1 for t in terms if t in low)
            if hits:
                scored.append((float(hits), r))
        scored.sort(key=lambda item: (item[0], item[1]["day"]), reverse=True)
        return [self._episode_row(r, score) for score, r in scored[:limit]]

    @staticmethod
    def _episode_row(r: Any, score: float) -> dict[str, Any]:
        return {
            "episode": r["id"],
            "role": r["role"],
            "session": r["session"],
            "agent": r["agent"],
            "date": r["day"],
            "ts": r["ts"],
            "ts_source": r["ts_source"],
            "text": (r["text"] or "")[:300],
            "score": score,
            "origin": "episode",
        }

    def vector_matrix(self) -> tuple[list[str], np.ndarray | None]:
        rows = self._conn.execute("SELECT id, dim, vec FROM vectors").fetchall()
        if not rows:
            return [], None
        dims = {r["dim"] for r in rows}
        if len(dims) != 1:
            return [], None  # mixed embedding spaces: refuse to compare them
        ids = [r["id"] for r in rows]
        mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        return ids, mat

    def stats(self) -> dict[str, int]:
        f = self._conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()["n"]
        v = self._conn.execute("SELECT COUNT(*) AS n FROM vectors").fetchone()["n"]
        sensitive = self._conn.execute(
            "SELECT COUNT(*) AS n FROM facts WHERE sensitive = 1"
        ).fetchone()["n"]
        e = self._conn.execute("SELECT COUNT(*) AS n FROM episodes").fetchone()["n"]
        return {"facts": f, "vectors": v, "sensitive": sensitive, "episodes": e}

    def close(self) -> None:
        self._conn.close()

"""Layered archive of long notes: the on-demand tier beside the fact store.

Facts are short claims that get injected into prompts. The archive keeps whole
notes (session records, lessons, reports) that are too long or too
task-specific to inject, and finds them when asked. Layers, after OpenViking:

  L0  abstract   one line in each entry's front matter, shown in result lists
  L1  overview   per source and category, computed from the index on demand
  L2  body       the full note, an immutable Markdown file

Layout: archive/<source>/<category>/<id>.md. Files are never edited. A newer
import of the same note adds a new file and the index shows only the newest
per (source, scope, origin). Hiding an entry appends an `archive_retract`
event to the ledger. The SQLite tables are derived and rebuild from the files.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml
from pydantic import ValidationError

from .embeddings import EmbeddingError
from .store import FM_RE, Store, StoreError, _atomic_write

ID_RE = re.compile(r"^a_\d{8}_[a-z0-9]{6}$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
EMBED_CHARS = 1200
EMBED_BATCH = 16

CATEGORY_LABELS = {
    "profile": "个人档案", "preferences": "偏好笔记", "events": "事件记录", "experiences": "经验与教训",
    "code": "程序与项目", "study": "学习资料", "reports": "报告", "sessions": "对话记录", "misc": "其他",
}


@dataclass(frozen=True)
class ArchiveEntry:
    id: str
    source: str
    category: str
    title: str
    abstract: str
    body: str
    date: str
    origin: str
    scope: str = ""
    batch: str = ""
    content_hash: str = ""


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def entry_id(source: str, scope: str, origin: str, chash: str, day: str) -> str:
    digest = hashlib.sha256(f"{source}|{scope}|{origin}|{chash}".encode("utf-8")).hexdigest()[:6]
    return f"a_{day.replace('-', '')}_{digest}"


_COMMENT = re.compile(r"<!--.*?-->", re.S)


def make_abstract(body: str, limit: int = 160) -> str:
    """First plain sentences of a note, without headings or metadata comments."""
    lines = [ln.strip() for ln in _COMMENT.sub("", body).splitlines()]
    flat = re.sub(r"\s+", " ", " ".join(ln for ln in lines if ln and not ln.startswith("#")))
    return flat.strip()[:limit]


def render(entry: ArchiveEntry) -> str:
    meta = asdict(entry)
    meta.pop("body")
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=10000) + "---\n" \
        + entry.body.rstrip() + "\n"


def parse(text: str) -> ArchiveEntry:
    m = FM_RE.match(text)
    if not m:
        raise StoreError("archive file missing front-matter")
    meta = yaml.safe_load(m.group(1)) or {}
    fields = {k: str(meta.get(k, "")) for k in ("id", "source", "category", "title", "abstract", "date",
                                                  "origin", "scope", "batch", "content_hash")}
    if not ID_RE.fullmatch(fields["id"]):
        raise StoreError("invalid archive id")
    return ArchiveEntry(body=m.group(2).rstrip("\n"), **fields)


_TOKEN = re.compile(r"[㐀-鿿]+|[a-z0-9_]+")


def tokens(text: str) -> list[str]:
    """Latin words as they are; Chinese runs as overlapping character pairs.

    The default FTS tokenizer treats a whole run of Chinese as one word, so a
    query could only match a note that contains the exact same run.
    """
    out: list[str] = []
    for piece in _TOKEN.findall(text.lower()):
        if "㐀" <= piece[0] <= "鿿" and len(piece) > 1:
            out.extend(piece[i:i + 2] for i in range(len(piece) - 1))
        else:
            out.append(piece)
    return out


def fts_match(query: str) -> str:
    uniq = list(dict.fromkeys(tokens(query)))[:40]
    return " OR ".join(f'"{t}"' for t in uniq)


class ArchiveStore:
    """The canonical files. Append-only."""

    def __init__(self, home: Path) -> None:
        self.root = Path(home) / "archive"

    def path(self, entry: ArchiveEntry) -> Path:
        return self.root / entry.source / entry.category / f"{entry.id}.md"

    def write(self, entry: ArchiveEntry) -> Path:
        if not (ID_RE.fullmatch(entry.id) and SLUG_RE.fullmatch(entry.source) and SLUG_RE.fullmatch(entry.category)):
            raise StoreError("invalid archive entry")
        path = self.path(entry)
        if path.exists():
            raise StoreError(f"archive entry {entry.id} already exists; the archive is append-only")
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, render(entry))
        return path

    def count(self) -> int:
        return sum(1 for _ in self.root.glob("*/*/a_*.md"))

    def exists(self, entry_id: str) -> bool:
        return bool(ID_RE.fullmatch(entry_id)) and any(self.root.glob(f"*/*/{entry_id}.md"))

    def find(self, entry_id: str) -> ArchiveEntry | None:
        if not ID_RE.fullmatch(entry_id):
            return None
        for path in self.root.glob(f"*/*/{entry_id}.md"):
            try:
                return parse(path.read_text(encoding="utf-8"))
            except (OSError, StoreError, yaml.YAMLError):
                return None
        return None

    def iter_entries(self) -> Iterator[ArchiveEntry]:
        for path in sorted(self.root.glob("*/*/a_*.md")):
            try:
                yield parse(path.read_text(encoding="utf-8"))
            except (OSError, StoreError, yaml.YAMLError, ValidationError):
                continue  # one damaged file must not hide the rest



class Archive:
    """Files plus a derived SQLite index (full text and vectors) in the shared db."""

    def __init__(self, store: Store, conn: sqlite3.Connection, embedder: Any, settings: Any) -> None:
        self.store = store
        self.files = ArchiveStore(store.home)
        self.conn = conn
        self.embedder = embedder
        self.settings = settings
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS archive (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, category TEXT NOT NULL, title TEXT NOT NULL,
                abstract TEXT NOT NULL, date TEXT NOT NULL, origin TEXT NOT NULL, scope TEXT NOT NULL,
                batch TEXT NOT NULL, chars INTEGER NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS archive_fts USING fts5(id UNINDEXED, text);
            CREATE TABLE IF NOT EXISTS archive_vectors (id TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS archive_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        conn.commit()

    # ------------------------------------------------------------ helpers

    def _meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM archive_meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT INTO archive_meta(key,value) VALUES(?,?) "
                          "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def hidden_ids(self) -> set[str]:
        return {e["id"] for e in self.store.ledger_events("archive_retract") if "id" in e}

    def visible(self) -> list[ArchiveEntry]:
        """Entries not retracted, newest per (source, scope, origin)."""
        hidden = self.hidden_ids()
        newest: dict[tuple[str, str, str], ArchiveEntry] = {}
        for entry in self.files.iter_entries():
            if entry.id in hidden:
                continue
            key = (entry.source, entry.scope, entry.origin)
            if key not in newest or (entry.batch, entry.id) > (newest[key].batch, newest[key].id):
                newest[key] = entry
        return list(newest.values())

    @staticmethod
    def _embed_text(entry: ArchiveEntry) -> str:
        return f"{entry.title}\n{entry.abstract}\n{entry.body}"[:EMBED_CHARS]

    def _put_rows(self, entries: list[ArchiveEntry]) -> None:
        for e in entries:
            self.conn.execute(
                "INSERT INTO archive(id,source,category,title,abstract,date,origin,scope,batch,chars) "
                "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "category=excluded.category,title=excluded.title,abstract=excluded.abstract,chars=excluded.chars",
                (e.id, e.source, e.category, e.title, e.abstract, e.date, e.origin, e.scope, e.batch, len(e.body)))
            self.conn.execute("DELETE FROM archive_fts WHERE id=?", (e.id,))
            self.conn.execute("INSERT INTO archive_fts(id,text) VALUES(?,?)",
                              (e.id, " ".join(tokens(f"{e.title} {e.abstract} {e.body}"))))

    def _put_vectors(self, entries: list[ArchiveEntry]) -> int:
        """Embed in small batches. Returns how many vectors were stored."""
        if not (self.embedder.enabled and entries):
            return 0
        model = self._meta("embed_model")
        if model not in (None, self.embedder.model):
            return 0  # another model's vectors live here; a reindex replaces them
        stored = 0
        try:
            for i in range(0, len(entries), EMBED_BATCH):
                chunk = entries[i:i + EMBED_BATCH]
                matrix = self.embedder.embed([self._embed_text(e) for e in chunk])
                if matrix is None:
                    break
                for e, vec in zip(chunk, matrix):
                    self.conn.execute(
                        "INSERT INTO archive_vectors(id,dim,vec) VALUES(?,?,?) "
                        "ON CONFLICT(id) DO UPDATE SET dim=excluded.dim,vec=excluded.vec",
                        (e.id, int(vec.shape[0]), vec.astype(np.float32).tobytes()))
                    stored += 1
            self._set_meta("embed_model", self.embedder.model)
        except EmbeddingError:
            pass  # lexical search still works; the next reindex fills the vectors
        return stored

    def _drop(self, ids: list[str]) -> None:
        for table in ("archive", "archive_fts", "archive_vectors"):
            self.conn.executemany(f"DELETE FROM {table} WHERE id=?", [(i,) for i in ids])

    # ------------------------------------------------------------- writes

    def sync(self, embed: bool = True) -> dict[str, int]:
        """Bring the index in line with the files without re-embedding what is indexed."""
        live = {e.id: e for e in self.visible()}
        indexed = {r[0] for r in self.conn.execute("SELECT id FROM archive")}
        self._drop(sorted(indexed - set(live)))
        fresh = [live[i] for i in sorted(set(live) - indexed)]
        self._put_rows(fresh)
        have_vec = {r[0] for r in self.conn.execute("SELECT id FROM archive_vectors")}
        embedded = self._put_vectors([e for e in live.values() if e.id not in have_vec]) if embed else 0
        self._set_meta("file_count", str(self.files.count()))
        self.conn.commit()
        return {"entries": len(live), "added": len(fresh), "embedded": embedded}

    def reindex(self) -> dict[str, int]:
        """Rebuild everything from the files."""
        self.conn.executescript("DELETE FROM archive; DELETE FROM archive_fts; "
                                "DELETE FROM archive_vectors; DELETE FROM archive_meta;")
        return self.sync()

    def ensure_current(self) -> None:
        """Cheap start-up check: index the text of files the index has not seen."""
        if self._meta("file_count") != str(self.files.count()):
            self.sync(embed=False)

    def retract(self, entry_id: str, reason: str) -> None:
        if not self.files.exists(entry_id):
            raise StoreError(f"no such archive entry: {entry_id}")
        self.store.append_event({"op": "archive_retract", "id": entry_id, "reason": reason})

    # -------------------------------------------------------------- reads

    SEMANTIC_FLOOR = 0.35  # long notes score lower against short queries than facts do

    def _rows(self, ids: list[str]) -> dict[str, sqlite3.Row]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        return {r["id"]: r for r in self.conn.execute(f"SELECT * FROM archive WHERE id IN ({marks})", ids)}

    def search(self, query: str, source: str | None = None, category: str | None = None,
               top_k: int = 8) -> tuple[list[dict[str, Any]], list[str]]:
        """Hybrid search. Returns (results, notes)."""
        notes: list[str] = []
        lexical: dict[str, float] = {}
        match = fts_match(query)
        if match:
            try:
                rows = self.conn.execute("SELECT id, bm25(archive_fts) FROM archive_fts WHERE archive_fts MATCH ? "
                                         "ORDER BY 2 LIMIT 100", (match,)).fetchall()
                lexical = {r[0]: -float(r[1]) for r in rows}
            except sqlite3.OperationalError:
                pass
        semantic: dict[str, float] = {}
        semantic_ran = False
        if self.embedder.enabled and self._meta("embed_model") == self.embedder.model:
            vectors = self.conn.execute("SELECT id, vec FROM archive_vectors").fetchall()
            if vectors:
                try:
                    q = self.embedder.embed_queries([query])[0]
                    ids = [r[0] for r in vectors]
                    sims = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in vectors]) @ q
                    semantic = {i: float(s) for i, s in zip(ids, sims) if s >= self.SEMANTIC_FLOOR}
                    semantic_ran = True
                except (EmbeddingError, ValueError):
                    notes.append("embedding failed; text match only")
        else:
            notes.append("no semantic index for the archive yet; text match only")
        # A text-only match must cover most of the query once meaning is available to judge;
        # otherwise one shared common word lets an unrelated note through.
        strong = sorted(lexical, key=lexical.get, reverse=True)[:top_k]
        if semantic_ran:
            wanted = set(tokens(query))
            marks = ",".join("?" * len(strong))
            have = {r[0]: set(r[1].split()) for r in self.conn.execute(
                f"SELECT id, text FROM archive_fts WHERE id IN ({marks})", strong)} if strong else {}
            strong = [i for i in strong if i in semantic or (wanted and len(wanted & have.get(i, set())) / len(wanted) >= 0.5)]
        admitted = set(semantic) | set(strong)
        score: dict[str, float] = {}
        for ranking in (lexical, semantic):
            for rank, i in enumerate(sorted((i for i in ranking if i in admitted), key=ranking.get, reverse=True)):
                score[i] = score.get(i, 0.0) + 1.0 / (60 + rank + 1)
        rows = self._rows(list(score))
        out = []
        for i in sorted(score, key=score.get, reverse=True):
            r = rows.get(i)
            if r is None or (source and r["source"] != source) or (category and r["category"] != category):
                continue
            out.append({"id": i, "source": r["source"], "category": r["category"], "title": r["title"],
                        "abstract": r["abstract"], "date": r["date"], "chars": r["chars"],
                        "similarity": round(semantic.get(i, 0.0), 3)})
            if len(out) >= top_k:
                break
        return out, notes

    def browse(self, source: str | None = None, category: str | None = None, limit: int = 50) -> dict[str, Any]:
        """Top level: counts per source and category. Below it: entries, newest first."""
        if not source and not category:
            tree: dict[str, dict[str, int]] = {}
            for r in self.conn.execute("SELECT source, category, COUNT(*) n FROM archive GROUP BY source, category"):
                tree.setdefault(r["source"], {})[r["category"]] = r["n"]
            return {"tree": tree, "labels": CATEGORY_LABELS}
        where: list[str] = []
        args: list[str] = []
        for column, value in (("source", source), ("category", category)):
            if value:
                where.append(f"{column}=?")
                args.append(value)
        rows = self.conn.execute(
            f"SELECT id,source,category,title,abstract,date,chars FROM archive WHERE {' AND '.join(where)} "
            "ORDER BY date DESC, id DESC", args).fetchall()
        items = [{k: r[k] for k in r.keys()} for r in rows[:limit]]
        overview = {"count": len(rows), "first_date": min((r["date"] for r in rows), default=""),
                    "last_date": max((r["date"] for r in rows), default="")}
        return {"source": source, "category": category, "overview": overview, "items": items}

    def read(self, entry_id: str, offset: int = 0, limit: int = 4000) -> dict[str, Any]:
        entry = self.files.find(entry_id)
        if entry is None or entry_id in self.hidden_ids():
            raise StoreError(f"no such archive entry: {entry_id}")
        offset = max(0, offset)
        return {"id": entry.id, "source": entry.source, "category": entry.category, "title": entry.title,
                "abstract": entry.abstract, "date": entry.date, "origin": entry.origin, "scope": entry.scope,
                "total_chars": len(entry.body), "offset": offset,
                "text": entry.body[offset:offset + max(1, min(limit, 20000))]}

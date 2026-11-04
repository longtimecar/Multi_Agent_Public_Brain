"""RAG knowledge library: upload → parse → chunk → enrich → embed → search.

A RAGFlow-style ingestion pipeline scoped to personal scale:

    parse (txt/md/html/docx/pdf) → chunk (heading-aware, ~800 chars) →
    transform (one LLM pass for abstract/keywords) → index (FTS5 + vectors).

The markdown document is the source of truth and lives in
memory/resources/<category>/<slug>.md (same L0/L2 layout the rest of the
product uses). Chunks and vectors live in memory/.index/rag.sqlite — derived
and disposable, exactly like the facts index. Deleting .index/ and re-feeding
the documents rebuilds everything.
"""

from __future__ import annotations

import base64
import hashlib
import re
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .resources import import_resource, list_resources, parse_resource, resources_root

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MAX_DOC_CHARS = 400_000
SUPPORTED = {".txt", ".md", ".markdown", ".log", ".csv", ".json", ".html", ".htm", ".docx", ".pdf"}


# ----------------------------------------------------------------- parsing

_TAG_RE = re.compile(r"<(script|style)[\s\S]*?</\1>|<[^>]+>", re.I)
_WS_RE = re.compile(r"[ \t\r\f\v]+")


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    return _clean_text(_TAG_RE.sub("", xml))


def _html_text(raw: str) -> str:
    raw = _TAG_RE.sub(" ", raw)
    return _clean_text(raw)


def extract_text(path: Path) -> str:
    """Pull plain text out of a supported file. Raises ValueError for the rest."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"暂不支持这种文件类型：{suffix or path.name}（支持 txt/md/csv/json/html/docx/pdf）")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("解析 PDF 需要 pypdf，先运行 uv sync 安装依赖。") from exc
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return _clean_text(text)[:MAX_DOC_CHARS]
    if suffix == ".docx":
        return _docx_text(path)[:MAX_DOC_CHARS]
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        return _html_text(raw)[:MAX_DOC_CHARS]
    return _clean_text(raw)[:MAX_DOC_CHARS]


# ----------------------------------------------------------------- chunking

_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|.{1,40}\n[-=]{3,})$", re.M)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on headings/paragraph boundaries, then pack into ~size windows.

    RAGFlow's rule applies here too: chunking decides retrieval precision, so
    we never cut inside a paragraph when a boundary is available.
    """
    text = text.strip()
    if not text:
        return []
    parts: list[str] = []
    cursor = 0
    boundaries = [m.start() for m in _HEADING_RE.finditer(text)] + [len(text)]
    for boundary in boundaries:
        piece = text[cursor:boundary].strip()
        if piece:
            parts.append(piece)
        cursor = boundary
    chunks: list[str] = []
    current = ""
    for part in parts:
        while len(part) > size:
            head, part = part[:size], part[size - overlap:]
            merged = (current + "\n" + head).strip()
            chunks.append(merged[: size + overlap])
            current = head[-overlap:]
        merged = (current + "\n" + part).strip()
        if len(merged) <= size:
            current = merged
        else:
            if current:
                chunks.append(current)
            current = part
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c][:400]


# ------------------------------------------------------------------- store

def rag_db(home: Path) -> sqlite3.Connection:
    path = Path(home) / ".index" / "rag.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id TEXT PRIMARY KEY,
            resource_id TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            seq INTEGER NOT NULL,
            text TEXT NOT NULL,
            vec BLOB
        );
        CREATE TABLE IF NOT EXISTS rag_docs (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            uri TEXT NOT NULL,
            source_path TEXT,
            abstract TEXT,
            chunks INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    return conn


def _pack_vec(vector: Any) -> bytes | None:
    if vector is None:
        return None
    try:
        return vector.astype("float32").tobytes()
    except AttributeError:
        return None


def _cosine_rank(conn: sqlite3.Connection, query_vec: Any) -> dict[str, float]:
    if query_vec is None:
        return {}
    rows = conn.execute("SELECT id, vec FROM rag_chunks").fetchall()
    scores: dict[str, float] = {}
    import numpy as np

    q = np.asarray(query_vec, dtype="float32").reshape(-1)
    qn = float(np.linalg.norm(q)) or 1.0
    for row in rows:
        if not row["vec"]:
            continue
        v = np.frombuffer(row["vec"], dtype="float32")
        if v.shape != q.shape:
            continue
        scores[row["id"]] = float(v @ q / ((float(np.linalg.norm(v)) or 1.0) * qn))
    return scores


def _lexical_rank(conn: sqlite3.Connection, query: str, top_k: int) -> dict[str, float]:
    tokens = re.findall(r"[\w一-鿿]+", query.lower())
    if not tokens:
        return {}
    fts = " OR ".join(f'"{t}"' for t in list(dict.fromkeys(tokens))[:32])
    try:
        rows = conn.execute(
            "SELECT c.id AS id, bm25(rag_chunks_fts) AS s FROM rag_chunks_fts "
            "JOIN rag_chunks c ON c.rowid = rag_chunks_fts.rowid "
            "WHERE rag_chunks_fts MATCH ? LIMIT ?", (fts, top_k * 3)).fetchall()
    except sqlite3.Error:
        return {}
    return {row["id"]: 1.0 / (1.0 + abs(float(row["s"]))) for row in rows}


def ensure_fts(conn: sqlite3.Connection) -> None:
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(text, content='rag_chunks', content_rowid='rowid')")
        conn.execute("INSERT INTO rag_chunks_fts(rag_chunks_fts) VALUES('rebuild')")
        conn.commit()
    except sqlite3.Error:
        pass


# ---------------------------------------------------------------- pipeline

def ingest_text(*, home: Path, body: str, category: str = "docs", title: str | None = None,
                source_path: str = "", embedder: Any = None, llm: Any = None) -> dict[str, Any]:
    """Chunk → enrich → embed → store one document. Idempotent by content hash."""
    text = (body or "").strip()
    if not text:
        raise ValueError("文档内容为空。")
    import_resource(home, body=text, category=category, title=title, source_path=source_path, origin="upload")
    # Re-read what import_resource persisted to get the canonical id/uri.
    newest = sorted((resources_root(home) / category.lstrip("/")).glob("*.md"),
                    key=lambda p: p.stat().st_mtime, reverse=True)[:5]
    rid = title or ""
    uri = ""
    abstract = ""
    for path in newest:
        try:
            entry = parse_resource(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if entry.body.strip() == text.strip() or (title and entry.title == title):
            rid, uri, abstract = entry.id, entry.uri, entry.abstract
            break
    if not rid or rid == (title or ""):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
        rid = f"r_{datetime.now(timezone.utc):%Y%m%d}_{digest}"
        uri = f"mb://user/resources/{category}/{digest}"
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("没有提取到可检索的正文。")
    keywords = ""
    if llm is not None:
        try:
            result = llm.complete_json(
                "Summarize a document for a knowledge base index. "
                'Return {"title": short title, "keywords": [..up to 8..]}.',
                text[:2000])
            kw = result.get("keywords")
            if isinstance(kw, list):
                keywords = " ".join(str(k) for k in kw[:8])
            if not title and isinstance(result.get("title"), str):
                title = result["title"][:80]
        except Exception:
            pass
    vectors = None
    if embedder is not None:
        try:
            vectors = embedder.embed(chunks)
        except Exception:
            vectors = None
    conn = rag_db(home)
    try:
        conn.execute("DELETE FROM rag_chunks WHERE resource_id = ?", (rid,))
        conn.execute("DELETE FROM rag_docs WHERE id = ?", (rid,))
        for seq, chunk in enumerate(chunks):
            cid = hashlib.sha256(f"{rid}|{seq}|{chunk}".encode("utf-8")).hexdigest()[:16]
            vec = _pack_vec(vectors[seq]) if vectors is not None and seq < len(vectors) else None
            conn.execute(
                "INSERT OR REPLACE INTO rag_chunks(id, resource_id, title, category, seq, text, vec) "
                "VALUES(?,?,?,?,?,?,?)",
                (cid, rid, title or rid, category, seq, chunk, vec))
        conn.execute(
            "INSERT OR REPLACE INTO rag_docs(id, title, category, uri, source_path, abstract, chunks, created_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (rid, title or rid, category, uri, source_path, abstract, len(chunks),
             datetime.now(timezone.utc).isoformat()))
        conn.commit()
        ensure_fts(conn)
    finally:
        conn.close()
    return {"id": rid, "title": title or rid, "category": category, "uri": uri,
            "chunks": len(chunks), "embedded": vectors is not None, "keywords": keywords}


def ingest_file(*, home: Path, path: Path | str, category: str = "docs",
                title: str | None = None, embedder: Any = None, llm: Any = None) -> dict[str, Any]:
    path = Path(path)
    text = extract_text(path)
    return ingest_text(home=home, body=text, category=category,
                       title=title or path.stem, source_path=str(path),
                       embedder=embedder, llm=llm)


def ingest_upload(*, home: Path, filename: str, data_b64: str, category: str = "docs",
                  embedder: Any = None, llm: Any = None) -> dict[str, Any]:
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"暂不支持这种文件类型：{suffix or name}")
    import tempfile

    data = base64.b64decode(data_b64 or "")
    if not data:
        raise ValueError("文件内容为空。")
    if len(data) > 30 * 1024 * 1024:
        raise ValueError("文件超过 30MB 上限。")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return ingest_file(home=home, path=tmp_path, category=category or "docs",
                           title=Path(name).stem, embedder=embedder, llm=llm)
    finally:
        tmp_path.unlink(missing_ok=True)


def list_documents(home: Path) -> list[dict[str, Any]]:
    conn = rag_db(home)
    try:
        rows = conn.execute(
            "SELECT d.*, (SELECT COUNT(*) FROM rag_chunks c WHERE c.resource_id = d.id) AS chunks "
            "FROM rag_docs d ORDER BY d.created_at DESC").fetchall()
        docs = [dict(row) for row in rows]
    finally:
        conn.close()
    for item in docs:
        item["size"] = None
    return docs


def get_document(home: Path, doc_id: str) -> dict[str, Any]:
    conn = rag_db(home)
    try:
        row = conn.execute("SELECT * FROM rag_docs WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            raise KeyError(doc_id)
        chunks = [dict(r) for r in conn.execute(
            "SELECT id, seq, text FROM rag_chunks WHERE resource_id = ? ORDER BY seq", (doc_id,))]
        out = dict(row)
        out["chunk_list"] = chunks
        out.pop("vec", None)
        return out
    finally:
        conn.close()


def delete_document(home: Path, doc_id: str) -> bool:
    conn = rag_db(home)
    try:
        row = conn.execute("SELECT uri FROM rag_docs WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM rag_chunks WHERE resource_id = ?", (doc_id,))
        conn.execute("DELETE FROM rag_docs WHERE id = ?", (doc_id,))
        conn.commit()
        ensure_fts(conn)
    finally:
        conn.close()
    uri = str(row["uri"] or "")
    if uri.startswith("mb://user/resources/"):
        from .resources import parse_resource_uri

        parsed = parse_resource_uri(uri)
        candidate = resources_root(home) / str(parsed["category"] or "") / f"{parsed['subject'] or ''}.md"
        if candidate.exists():
            candidate.unlink(missing_ok=True)
    return True


def search(home: Path, query: str, *, top_k: int = 8, embedder: Any = None) -> list[dict[str, Any]]:
    """Hybrid retrieval over chunks: vector + lexical, fused 0.6/0.4."""
    query = (query or "").strip()
    if not query:
        return []
    conn = rag_db(home)
    try:
        ensure_fts(conn)
        query_vec = None
        if embedder is not None:
            try:
                vectors = embedder.embed_queries([query]) if hasattr(embedder, "embed_queries") else embedder.embed([query])
                if vectors is not None and len(vectors):
                    query_vec = vectors[0]
            except Exception:
                query_vec = None
        sem = _cosine_rank(conn, query_vec)
        lex = _lexical_rank(conn, query, top_k)
        scores: dict[str, float] = {}
        for cid, s in sem.items():
            scores[cid] = 0.6 * s + 0.4 * lex.get(cid, 0.0)
        for cid, s in lex.items():
            scores.setdefault(cid, 0.6 * 0.0 + 0.4 * s)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        if not ranked:
            return []
        hits = []
        for cid, score in ranked:
            row = conn.execute(
                "SELECT c.id, c.resource_id, c.title, c.category, c.seq, c.text FROM rag_chunks c WHERE c.id = ?",
                (cid,)).fetchone()
            if row is not None:
                item = dict(row)
                item["score"] = round(score, 4)
                hits.append(item)
        return hits
    finally:
        conn.close()

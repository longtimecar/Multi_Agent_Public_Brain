"""Resources namespace + import pipeline (OpenViking gap 11 thought).

URI: mb://user/resources/<category>/<slug>
Storage: MEMBRAIN_HOME/resources/<category>/<slug>.md with L0 abstract in
front matter and L2 body. L1 overview is computed per category on demand.
These are NOT memory facts — additive reference material beside the fact store
(similar spirit to archive, separate namespace).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .overview import make_abstract
from .uri import slugify

RESOURCE_SCHEME_PREFIX = "mb://user/resources/"
_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.S)
_CAT = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


def resources_root(home: Path) -> Path:
    return Path(home) / "resources"


def build_resource_uri(category: str, subject: str) -> str:
    cat = slugify(category)
    sub = slugify(subject)
    return f"{RESOURCE_SCHEME_PREFIX}{cat}/{sub}"


def parse_resource_uri(uri: str) -> dict[str, str | None]:
    raw = (uri or "").strip()
    if not raw.startswith(RESOURCE_SCHEME_PREFIX):
        return {"category": None, "subject": None}
    rest = raw[len(RESOURCE_SCHEME_PREFIX) :].split("?", 1)[0].strip("/")
    parts = [p for p in rest.split("/") if p]
    return {
        "category": parts[0] if parts else None,
        "subject": parts[1] if len(parts) > 1 else None,
    }


@dataclass
class ResourceEntry:
    id: str
    category: str
    title: str
    abstract: str
    body: str
    uri: str
    origin: str = "imported"
    created_at: str = ""
    source_path: str = ""

    def render(self) -> str:
        meta = {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "abstract": self.abstract,
            "uri": self.uri,
            "origin": self.origin,
            "created_at": self.created_at,
            "source_path": self.source_path,
        }
        return (
            "---\n"
            + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=10000)
            + "---\n"
            + self.body.rstrip()
            + "\n"
        )


def _entry_id(category: str, title: str, body: str) -> str:
    digest = hashlib.sha256(f"{category}|{title}|{body}".encode("utf-8")).hexdigest()[:10]
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"r_{day}_{digest}"


def parse_resource(text: str) -> ResourceEntry:
    m = _FM.match(text)
    if not m:
        raise ValueError("resource missing front-matter")
    meta = yaml.safe_load(m.group(1)) or {}
    return ResourceEntry(
        id=str(meta.get("id") or ""),
        category=str(meta.get("category") or "misc"),
        title=str(meta.get("title") or "untitled"),
        abstract=str(meta.get("abstract") or ""),
        body=m.group(2).rstrip("\n"),
        uri=str(meta.get("uri") or ""),
        origin=str(meta.get("origin") or "imported"),
        created_at=str(meta.get("created_at") or ""),
        source_path=str(meta.get("source_path") or ""),
    )


def import_resource(
    home: Path,
    *,
    body: str,
    category: str = "misc",
    title: str | None = None,
    source_path: str = "",
    origin: str = "imported",
) -> dict[str, Any]:
    """Write one resource markdown (L0 abstract + L2 body). Does not create a Fact."""
    cat = slugify(category)
    if not _CAT.fullmatch(cat):
        raise ValueError("invalid resource category")
    text = (body or "").lstrip("\ufeff").strip()
    if not text:
        raise ValueError("empty resource body")
    title_s = (title or "").strip() or make_abstract(text, limit=60) or "untitled"
    abstract = make_abstract(text, limit=160)
    uri = build_resource_uri(cat, title_s)
    rid = _entry_id(cat, title_s, text)
    entry = ResourceEntry(
        id=rid,
        category=cat,
        title=title_s,
        abstract=abstract,
        body=text,
        uri=uri,
        origin=origin,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_path=source_path,
    )
    dest_dir = resources_root(home) / cat
    dest_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(title_s)
    path = dest_dir / f"{slug}.md"
    # Append-ish: if same slug exists with different hash, write id-suffixed file.
    if path.exists():
        try:
            old = parse_resource(path.read_text(encoding="utf-8"))
            if old.abstract == abstract and old.body.strip() == text:
                return {"written": False, "reason": "duplicate", "path": str(path), "uri": old.uri, "id": old.id}
        except Exception:
            pass
        path = dest_dir / f"{slug}-{rid[-6:]}.md"
    path.write_text(entry.render(), encoding="utf-8")
    return {"written": True, "path": str(path), "uri": uri, "id": rid, "abstract": abstract}


def import_resource_file(home: Path, file_path: Path | str, *, category: str = "misc", title: str | None = None) -> dict[str, Any]:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    return import_resource(
        home,
        body=text,
        category=category,
        title=title or path.stem,
        source_path=str(path),
        origin="imported",
    )


def list_resources(home: Path, *, category: str | None = None) -> list[dict[str, Any]]:
    root = resources_root(home)
    if not root.exists():
        return []
    pattern = f"{slugify(category)}/*.md" if category else "*/*.md"
    items: list[dict[str, Any]] = []
    for path in sorted(root.glob(pattern)):
        try:
            entry = parse_resource(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append(
            {
                "id": entry.id,
                "category": entry.category,
                "title": entry.title,
                "abstract": entry.abstract,
                "uri": entry.uri,
                "path": str(path),
            }
        )
    return items


def read_resource(home: Path, uri_or_path: str, *, offset: int = 0, limit: int = 4000) -> dict[str, Any]:
    raw = (uri_or_path or "").strip()
    path: Path | None = None
    if raw.startswith(RESOURCE_SCHEME_PREFIX):
        parsed = parse_resource_uri(raw)
        cat, sub = parsed.get("category"), parsed.get("subject")
        if cat and sub:
            candidate = resources_root(home) / cat / f"{sub}.md"
            if candidate.exists():
                path = candidate
            else:
                # fuzzy: any file in category whose uri matches
                for p in (resources_root(home) / cat).glob("*.md") if (resources_root(home) / cat).exists() else []:
                    try:
                        e = parse_resource(p.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if e.uri == raw or slugify(e.title) == sub:
                        path = p
                        break
    else:
        path = Path(raw)
    if path is None or not path.exists():
        raise FileNotFoundError(f"resource not found: {uri_or_path}")
    entry = parse_resource(path.read_text(encoding="utf-8"))
    offset = max(0, offset)
    limit = max(1, min(limit, 20000))
    return {
        "id": entry.id,
        "uri": entry.uri,
        "category": entry.category,
        "title": entry.title,
        "abstract": entry.abstract,
        "total_chars": len(entry.body),
        "offset": offset,
        "text": entry.body[offset : offset + limit],
        "path": str(path),
    }


def category_overview(home: Path, category: str, *, limit: int = 20) -> dict[str, Any]:
    """L1-style overview: abstracts under one category."""
    items = list_resources(home, category=category)[:limit]
    lines = [f"- {it['title']}: {it['abstract']}" for it in items]
    overview = make_abstract(" ".join(it["abstract"] for it in items), limit=240) if items else ""
    return {
        "category": slugify(category),
        "count": len(items),
        "overview": overview,
        "items": items,
        "lines": lines,
    }

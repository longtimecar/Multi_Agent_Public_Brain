"""Agent Evolution thin layer (OpenViking gap 9 thought).

Pipeline: case fact -> trajectory (append-only JSONL) -> experience note.
Default OFF (Settings.evolution_enabled / MEMBRAIN_EVOLUTION). Never disables
Agent built-in memory; never hard-deletes facts; experiences stay under
MEMBRAIN_HOME/evolution/ (not mixed into memory facts unless a human promotes).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Fact, FactType
from .overview import make_abstract

_SLUG = re.compile(r"[^a-z0-9]+")


def evolution_root(home: Path) -> Path:
    return Path(home) / "evolution"


def trajectories_dir(home: Path) -> Path:
    return evolution_root(home) / "trajectories"


def experiences_dir(home: Path) -> Path:
    return evolution_root(home) / "experiences"


def is_enabled(settings: Any) -> bool:
    return bool(getattr(settings, "evolution_enabled", False))


def _slug(text: str) -> str:
    return _SLUG.sub("-", (text or "").strip().lower()).strip("-")[:80] or "case"


@dataclass
class TrajectoryEvent:
    ts: str
    kind: str  # case_seed | step | note
    text: str
    fact_id: str | None = None
    meta: dict[str, Any] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def trajectory_path(home: Path, case_id: str) -> Path:
    return trajectories_dir(home) / f"{_slug(case_id)}.jsonl"


def seed_trajectory_from_case(home: Path, fact: Fact) -> dict[str, Any]:
    """Append a case_seed event from a CASE fact. Idempotent per fact_id."""
    type_value = fact.type.value if isinstance(fact.type, FactType) else str(fact.type)
    if type_value != FactType.CASE.value:
        raise ValueError("evolution seed requires FactType.CASE")
    path = trajectory_path(home, fact.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("fact_id"):
                existing_ids.add(str(row["fact_id"]))
    if fact.id in existing_ids:
        return {"path": str(path), "appended": False, "reason": "already_seeded"}
    event = TrajectoryEvent(
        ts=_now(),
        kind="case_seed",
        text=fact.claim,
        fact_id=fact.id,
        meta={
            "subject": fact.subject,
            "uri": fact.uri,
            "abstract": fact.abstract or make_abstract(fact.claim),
        },
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    return {"path": str(path), "appended": True, "fact_id": fact.id}


def append_step(
    home: Path,
    case_id: str,
    text: str,
    *,
    kind: str = "step",
    meta: dict | None = None,
) -> dict[str, Any]:
    path = trajectory_path(home, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = TrajectoryEvent(
        ts=_now(),
        kind=kind,
        text=text.strip(),
        fact_id=None,
        meta=meta or {},
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    return {"path": str(path), "appended": True}


def read_trajectory(home: Path, case_id: str) -> list[dict[str, Any]]:
    path = trajectory_path(home, case_id)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def distill_experience(home: Path, case_id: str, *, title: str | None = None) -> dict[str, Any]:
    """Write an experience markdown from trajectory (L0 abstract + body). Not a Fact."""
    events = read_trajectory(home, case_id)
    if not events:
        return {"written": False, "reason": "empty_trajectory"}
    texts = [str(e.get("text") or "").strip() for e in events if str(e.get("text") or "").strip()]
    body_lines = [f"- ({e.get('kind')}) {e.get('text')}" for e in events]
    abstract = make_abstract(" ".join(texts), limit=160)
    slug = _slug(case_id)
    exp_dir = experiences_dir(home)
    exp_dir.mkdir(parents=True, exist_ok=True)
    path = exp_dir / f"{slug}.md"
    title_line = title or f"Experience from case {slug}"
    content = (
        f"---\n"
        f"id: exp_{slug}\n"
        f"case_id: {slug}\n"
        f"title: {title_line}\n"
        f"abstract: {abstract}\n"
        f"created_at: {_now()}\n"
        f"---\n\n"
        f"# {title_line}\n\n"
        f"{abstract}\n\n"
        f"## Trajectory\n\n"
        + "\n".join(body_lines)
        + "\n"
    )
    path.write_text(content, encoding="utf-8")
    return {"written": True, "path": str(path), "abstract": abstract, "events": len(events)}


def list_experiences(home: Path) -> list[dict[str, Any]]:
    root = experiences_dir(home)
    if not root.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        abstract = ""
        title = path.stem
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                for line in parts[1].splitlines():
                    if line.startswith("abstract:"):
                        abstract = line[len("abstract:") :].strip()
                    if line.startswith("title:"):
                        title = line[len("title:") :].strip()
        items.append({"id": path.stem, "path": str(path), "title": title, "abstract": abstract})
    return items


def promote_live_cases(store: Any, settings: Any, *, limit: int = 50) -> dict[str, Any]:
    """When enabled, seed trajectories for live CASE facts. No-op if disabled."""
    if not is_enabled(settings):
        return {"enabled": False, "seeded": 0, "skipped": "evolution_disabled"}
    home = Path(settings.home)
    seeded = 0
    details = []
    cases = [
        f
        for f in store.live_facts()
        if (f.type == FactType.CASE) or (getattr(f.type, "value", None) == "case")
    ]
    for fact in cases[: max(0, limit)]:
        result = seed_trajectory_from_case(home, fact)
        if result.get("appended"):
            seeded += 1
        details.append(result)
    return {"enabled": True, "seeded": seeded, "cases": len(cases), "details": details}

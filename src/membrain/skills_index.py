"""Skills directory index + SessionStart budgeted checklist (OV gap 10 thought).

Scans MEMBRAIN_HOME/skills and optional extra roots for skill cards
(SKILL.md or *.md with name + one-line summary). Injection is additive and
token-budgeted; missing dirs yield an empty list (fail-open).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .budget import estimate_tokens
from .overview import make_abstract

_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.S)


@dataclass(frozen=True)
class SkillCard:
    name: str
    summary: str
    path: str
    source_root: str

    def line(self) -> str:
        return f"- [skill {self.name}] {self.summary}"


def skills_dir(home: Path) -> Path:
    return Path(home) / "skills"


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    m = _FM.match(text.lstrip("\ufeff"))
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip().lower()] = v.strip().strip("\"'")
    return meta, m.group(2)


def _card_from_file(path: Path, source_root: Path) -> SkillCard | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta, body = _parse_front_matter(text)
    name = meta.get("name") or meta.get("title") or path.parent.name if path.name.lower() == "skill.md" else path.stem
    if path.name.lower() == "skill.md" and path.parent.name:
        name = meta.get("name") or meta.get("title") or path.parent.name
    summary = meta.get("summary") or meta.get("description") or meta.get("abstract") or ""
    if not summary:
        # First non-heading prose line / abstract of body
        for ln in body.splitlines():
            s = ln.strip()
            if not s or s.startswith("#") or s.startswith("```"):
                continue
            summary = make_abstract(s, limit=120)
            break
    if not summary:
        summary = make_abstract(body, limit=120) or name
    return SkillCard(name=str(name).strip()[:80], summary=str(summary).strip()[:160], path=str(path), source_root=str(source_root))


def iter_skill_files(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return []
    out: list[Path] = []
    # Prefer SKILL.md in subdirs; also accept top-level *.md (not README alone unless only).
    for skill_md in sorted(root.glob("*/SKILL.md")) + sorted(root.glob("*/skill.md")):
        out.append(skill_md)
    for md in sorted(root.glob("*.md")):
        if md.name.lower() in {"readme.md", "license.md", "notice.md"}:
            continue
        out.append(md)
    # de-dupe preserving order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def scan_skills(*roots: Path | str | None) -> list[SkillCard]:
    cards: list[SkillCard] = []
    seen_names: set[str] = set()
    for raw in roots:
        if not raw:
            continue
        root = Path(raw)
        for path in iter_skill_files(root):
            card = _card_from_file(path, root)
            if card is None:
                continue
            key = card.name.casefold()
            if key in seen_names:
                continue
            seen_names.add(key)
            cards.append(card)
    return cards


def default_roots(home: Path, extra: list[Path | str] | None = None) -> list[Path]:
    roots = [skills_dir(home)]
    for item in extra or []:
        if item:
            roots.append(Path(item))
    return roots


def skills_injection_lines(
    home: Path,
    *,
    budget_tokens: int,
    existing_block: str = "",
    extra_roots: list[Path | str] | None = None,
    max_skills: int = 24,
) -> tuple[list[str], list[str]]:
    """Return (lines, notes) that fit under remaining budget for SessionStart."""
    notes: list[str] = []
    cards = scan_skills(*default_roots(home, extra_roots))
    if not cards:
        notes.append("skills:none")
        return [], notes
    notes.append(f"skills:found:{len(cards)}")
    lines: list[str] = ["- [skills index]"]
    used = 0
    for card in cards[:max_skills]:
        line = card.line()
        trial_lines = (existing_block.splitlines() if existing_block else []) + lines + [line]
        candidate = "<personal_memory>\n" + "\n".join(trial_lines) + "\n</personal_memory>"
        if estimate_tokens(candidate) > budget_tokens:
            notes.append("skills:budget_stop")
            break
        lines.append(line)
        used += 1
    if used == 0:
        return [], notes + ["skills:no_fit"]
    notes.append(f"skills:injected:{used}")
    return lines, notes


def ensure_skills_dir(home: Path) -> Path:
    path = skills_dir(home)
    path.mkdir(parents=True, exist_ok=True)
    readme = path / "README.md"
    if not readme.exists():
        readme.write_text(
            "# membrain skills\n\n"
            "Drop skill folders with `SKILL.md` (name/summary front matter) here.\n"
            "SessionStart injects a budgeted name + one-line checklist.\n",
            encoding="utf-8",
        )
    return path

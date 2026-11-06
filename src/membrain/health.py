"""Memory health scanner: duplicates, conflicts, noise, and decay candidates.

The scanner only reports. Every mutating suggestion (supersede, retract) is
executed by the dashboard after the owner presses a button — the same rule the
conflict resolver follows: detection is mechanical, resolution is human.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Iterable

from .models import Fact, FactType
from .agents import is_disabled

# Types that carry the user's long-term identity. They never become decay
# candidates no matter how long they sit unused.
_NEVER_FORGET_TYPES = {FactType.PREFERENCE, FactType.CONSTRAINT}
# Types whose value is naturally short-lived; they decay into forget
# candidates even at moderate confidence.
_EPHEMERAL_TYPES = {FactType.EVENT, FactType.CASE, FactType.TOOL}

_DUPLICATE_THRESHOLD = 0.88
_NOISE_MAX_CONFIDENCE = 0.42
_STALE_MAX_CONFIDENCE = 0.50
_FORGET_DAYS = 90
_NOISE_DAYS = 90
# One embedding request per 16 claims: the whole-store request timed out.
EMBED_BATCH = 16


@dataclass(frozen=True)
class Finding:
    kind: str            # duplicate | conflict | noisy | forget
    severity: int        # 3 = act soon, 1 = informational
    fact_ids: tuple[str, ...]
    summary: str
    suggestion: str
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "severity": self.severity, "fact_ids": list(self.fact_ids),
                "summary": self.summary, "suggestion": self.suggestion,
                "fingerprint": self.fingerprint}


def _fingerprint(kind: str, fact_ids: Iterable[str]) -> str:
    raw = "|".join((kind, *sorted(fact_ids)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _fact_age_days(fact: Fact) -> int:
    created = fact.created_at.date() if fact.created_at else fact.valid_from
    return max(0, (date.today() - created).days)


def _lexical_similarity(a: str, b: str) -> float:
    # Character-level ratio: robust to punctuation splits and works for
    # Chinese where whitespace tokenisation does not apply.
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()


class HealthScanner:
    def __init__(self, service: Any, since_use: Callable[[Fact], int]) -> None:
        """`since_use(fact)` returns days since creation or last recall."""
        self.service = service
        self._since_use = since_use

    def scan(self, kinds: set[str] | None = None) -> list[Finding]:
        """Findings for the requested kinds, or everything.

        Some detectors are cheap; duplicates embed every live claim. Asking for
        one kind must not pay for the others — a targeted cleanup waited minutes
        for an embedding pass it never used.
        """
        wanted = set(kinds) if kinds else {"duplicate", "conflict", "noise", "decay",
                                           "command_misfiled"}
        disabled = self.service.disabled_agents()
        facts = [fact for fact in self.service.store.all_facts()
                 if fact.is_live and not is_disabled(fact.source.agent, disabled)]
        findings: list[Finding] = []
        if "duplicate" in wanted:
            findings.extend(self._find_duplicates(facts))
        if "conflict" in wanted:
            findings.extend(self._find_conflicts())
        if "noise" in wanted:
            findings.extend(self._find_noisy(facts))
        if "decay" in wanted:
            findings.extend(self._find_decay_candidates(facts))
        if "command_misfiled" in wanted:
            findings.extend(self._find_misfiled_commands(facts))
        return findings

    def _find_misfiled_commands(self, facts: list[Fact]) -> list[Finding]:
        """Preferences and facts whose evidence is an instruction to the assistant.

        "把重复的清理掉" is work to do, not a trait of the owner. Storing it as a
        preference is how one-off commands become durable noise; the admin can
        retract these because the report names them.
        """
        from .extract import _is_agent_command

        episodes = {ep.id: ep for ep in self.service.store.iter_episodes()}
        out: list[Finding] = []
        for fact in facts:
            if fact.type not in {FactType.PREFERENCE, FactType.FACT, FactType.BELIEF}:
                continue
            turns = [episodes.get(item.episode) for item in fact.evidence]
            turns = [turn for turn in turns if turn is not None]
            if not turns or not all(_is_agent_command(turn.text) for turn in turns):
                continue
            out.append(Finding(
                kind="command_misfiled", severity=3, fact_ids=(fact.id,),
                summary=f"证据是给你的指令，不是用户的陈述：“{fact.claim[:70]}”",
                suggestion="这不是偏好或事实：撤回，或改记为任务/目标。",
                fingerprint=_fingerprint("command_misfiled", (fact.id,)),
            ))
        return out

    def _find_duplicates(self, facts: list[Fact]) -> list[Finding]:
        if len(facts) < 2:
            return []
        matrix = self._embed_claims(facts)
        findings: list[Finding] = []
        seen_pairs: set[tuple[str, str]] = set()
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                a, b = facts[i], facts[j]
                if matrix is not None:
                    score = float(matrix[i] @ matrix[j])
                else:
                    score = _lexical_similarity(a.claim, b.claim)
                if score < _DUPLICATE_THRESHOLD:
                    continue
                pair_key = tuple(sorted((a.id, b.id)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                keep, drop = (a, b) if len(a.evidence) >= len(b.evidence) else (b, a)
                findings.append(Finding(
                    kind="duplicate", severity=3, fact_ids=(a.id, b.id),
                    summary=f"疑似重复（相似度 {score:.2f}）：“{a.claim}” vs “{b.claim}”",
                    suggestion=f"建议保留证据更多的“{keep.claim}”，把另一条标记为已替代。",
                    fingerprint=_fingerprint("duplicate", pair_key),
                ))
        return findings

    def _embed_claims(self, facts: list[Fact]):
        """Normalised claim vectors, or None when embeddings are unavailable.

        Embedded in batches: one request with every claim in the store timed out
        on a real endpoint, and the timeout was swallowed, so the report quietly
        fell back to word overlap while still looking authoritative.
        """
        embedder = self.service.embedder
        if not embedder.enabled:
            return None
        try:
            import numpy as np
            chunks = []
            for start in range(0, len(facts), EMBED_BATCH):
                batch = [fact.claim for fact in facts[start:start + EMBED_BATCH]]
                matrix = embedder.embed(batch)
                if matrix is None or len(matrix) != len(batch):
                    return None
                chunks.append(matrix)
            matrix = np.vstack(chunks)
            if not len(matrix):
                return None
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return matrix / norms
        except Exception:
            return None

    def _find_conflicts(self) -> list[Finding]:
        contested = [fact for fact in self.service.retriever.contested()
                     if not is_disabled(fact.source.agent, self.service.disabled_agents())]
        by_subject: dict[str, list[Fact]] = {}
        for fact in contested:
            by_subject.setdefault(fact.subject, []).append(fact)
        findings: list[Finding] = []
        for subject, group in by_subject.items():
            ids = tuple(fact.id for fact in group)
            claims = " vs ".join(fact.claim for fact in group)
            findings.append(Finding(
                kind="conflict", severity=3, fact_ids=ids,
                summary=f"待裁决冲突（{subject}）：{claims}",
                suggestion="在冲突页比较两条证据后选择保留哪一条。",
                fingerprint=_fingerprint("conflict", ids),
            ))
        return findings

    def _find_noisy(self, facts: list[Fact]) -> list[Finding]:
        findings: list[Finding] = []
        for fact in facts:
            if fact.sensitive or fact.type == FactType.CONSTRAINT:
                continue
            if (fact.confidence < _NOISE_MAX_CONFIDENCE and len(fact.evidence) <= 1
                    and _fact_age_days(fact) > _NOISE_DAYS):
                findings.append(Finding(
                    kind="noisy", severity=2, fact_ids=(fact.id,),
                    summary=f"疑似噪声：低置信 {fact.confidence:.2f}、单条证据、已存在 "
                            f"{_fact_age_days(fact)} 天——“{fact.claim}”",
                    suggestion="如果这不是长期有效信息，建议撤回；否则保留即可。",
                    fingerprint=_fingerprint("noisy", (fact.id,)),
                ))
        return findings

    def _find_decay_candidates(self, facts: list[Fact]) -> list[Finding]:
        findings: list[Finding] = []
        for fact in facts:
            if fact.type in _NEVER_FORGET_TYPES or fact.sensitive:
                continue
            # Unfinished work is not clutter. A task retires when the owner says
            # it is done (closed_at), never because the calendar ran out —
            # otherwise "记住进度，明天继续" decays exactly while it still matters.
            if fact.is_open_task:
                continue
            days = self._since_use(fact)
            low_value = (fact.confidence < _STALE_MAX_CONFIDENCE
                         or fact.type in _EPHEMERAL_TYPES
                         or fact.type == FactType.TASK)
            if days <= _FORGET_DAYS or not low_value:
                continue
            findings.append(Finding(
                kind="forget", severity=1, fact_ids=(fact.id,),
                summary=f"遗忘候选：{days} 天未被召回、置信 {fact.confidence:.2f}——“{fact.claim}”",
                suggestion="它已经只以摘要形式注入。如果确认不再需要，可以撤回归档。",
                fingerprint=_fingerprint("forget", (fact.id,)),
            ))
        return findings


class DismissalStore:
    """Findings the owner chose to ignore, persisted by fingerprint."""

    def __init__(self, path: Any) -> None:
        self.path = path

    def load(self) -> set[str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(v) for v in data.get("dismissed", ()) if isinstance(v, str)}
        except (OSError, ValueError):
            pass
        return set()

    def save(self, dismissed: set[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"dismissed": sorted(dismissed)}, ensure_ascii=False, indent=2),
                             encoding="utf-8")

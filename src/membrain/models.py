"""Schemas. The ontology is deliberately small.

There is no `trait` type and no `personality` type. Inferred dispositions,
decision styles and behavioural archetypes are not representable, so the
extractor cannot write them regardless of what it believes about the user.
That is the primary anti-sycophancy control: capability removal, not judgment.

Scheme A (OpenViking-port) adds optional URI / L0 abstract / peer / sensitive
fields. Old fact files without them still parse; defaults are filled on write.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import local_date as _local_date

ID_RE = re.compile(r"^f_\d{8}_[a-z0-9]{6}$")


class FactType(str, Enum):
    CONSTRAINT = "constraint"
    FACT = "fact"
    PREFERENCE = "preference"
    GOAL = "goal"
    PROJECT = "project"
    # Scheme A extensions (OpenViking-inspired)
    ENTITY = "entity"
    EVENT = "event"
    CASE = "case"
    TOOL = "tool"
    BELIEF = "belief"
    # Work the user handed to a later session: a handoff, plan, commitment or
    # progress snapshot. Distinct from `goal` (what you want to become) because
    # its useful life is "the next session", not "a month from now".
    TASK = "task"


# Types agents may write via MCP (never constraint).
AGENT_WRITABLE_TYPES = frozenset({
    FactType.FACT,
    FactType.PREFERENCE,
    FactType.GOAL,
    FactType.PROJECT,
    FactType.ENTITY,
    FactType.EVENT,
    FactType.CASE,
    FactType.TOOL,
    FactType.BELIEF,
    FactType.TASK,
})


class Status(str, Enum):
    ACTIVE = "active"
    CONTESTED = "contested"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


# "imported" marks a claim distilled from another memory system's notes at the
# owner's request. It is not the user's own typed text, so it never counts as
# evidence for the extractor or MCP writes; the owner chose to trust it.
Origin = Literal["stated", "inferred", "imported"]

# Context layers (OpenViking ContextLevel): 0=abstract, 1=overview, 2=detail/claim
ContextLevel = Literal[0, 1, 2]


class Evidence(BaseModel):
    """A pointer back to something the user actually said or did.

    `quote` must be verbatim text from the referenced episode. A claim with no
    quote is rejected by validation rather than by the model's conscience.
    """

    model_config = ConfigDict(extra="forbid")

    episode: str
    quote: str
    date: date
    origin: Origin = "stated"

    @field_validator("quote")
    @classmethod
    def _quote_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("evidence quote must not be empty")
        if len(v) > 2000:
            raise ValueError("evidence quote too long (max 2000 chars)")
        return v.strip()


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str = "unknown"
    session: str | None = None


class Fact(BaseModel):
    """One claim about the user. One file on disk.

    Never mutated in place once written, except for lifecycle fields
    (`status`, `valid_to`, `supersedes`, `retracted_reason`). Claim text and
    evidence are immutable — changing what you believe means writing a new fact.

    Optional Scheme A fields (uri, abstract, peer, sensitive) are additive:
    missing values are filled with defaults so older Markdown still loads.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    type: FactType
    subject: str = Field(min_length=1, max_length=120)
    claim: str = Field(min_length=1, max_length=1000)
    evidence: list[Evidence] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    valid_from: date
    valid_to: date | None = None
    supersedes: str | None = None
    source: Provenance = Field(default_factory=Provenance)
    status: Status = Status.ACTIVE
    retracted_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # When the work a `task` describes was finished. Only the owner's own words
    # (or a session summary confirming it) may set this — never a calendar rule,
    # because an unfinished task must not disappear just because nobody mentioned
    # it for a while. Null means "not finished, or not a task".
    closed_at: date | None = None
    # --- Scheme A (optional / backward compatible) ---
    uri: str | None = None
    abstract: str | None = None  # L0 one-liner
    peer: str | None = None  # optional project/git identity; default recall ignores
    sensitive: bool = False  # religion/politics/belief: store but never dump every turn

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not ID_RE.match(v):
            raise ValueError(f"id must match {ID_RE.pattern}, got {v!r}")
        return v

    @field_validator("subject")
    @classmethod
    def _subject_slug(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("subject must not be empty")
        return cleaned

    @field_validator("evidence")
    @classmethod
    def _distinct_episodes(cls, v: list[Evidence]) -> list[Evidence]:
        seen: set[str] = set()
        for e in v:
            if e.episode in seen:
                raise ValueError(f"duplicate episode {e.episode!r} in evidence")
            seen.add(e.episode)
        return v

    @model_validator(mode="before")
    @classmethod
    def _fill_scheme_a(cls, data):
        """Fill Scheme A defaults on dict input (compatible with older Markdown)."""
        if not isinstance(data, dict):
            return data
        from .overview import looks_sensitive, make_abstract
        from .uri import build_uri

        ftype = data.get("type")
        type_value = ftype.value if isinstance(ftype, FactType) else str(ftype or "fact")
        subject = str(data.get("subject") or "general")
        claim = str(data.get("claim") or "")
        peer = data.get("peer")
        if not data.get("uri"):
            data["uri"] = build_uri(type_value, subject, peer if isinstance(peer, str) else None)
        if not data.get("abstract") and claim:
            data["abstract"] = make_abstract(claim)
        sensitive = bool(data.get("sensitive", False))
        if not sensitive and (
            type_value == FactType.BELIEF.value
            or looks_sensitive(f"{subject} {claim}", type_value)
        ):
            data["sensitive"] = True
        return data

    @property
    def n_episodes(self) -> int:
        return len(self.evidence)

    @property
    def is_live(self) -> bool:
        return self.status in (Status.ACTIVE, Status.CONTESTED) and self.valid_to is None

    @property
    def is_open_task(self) -> bool:
        """A task the user has not said is finished. Drives the to-do query."""
        return self.type == FactType.TASK and self.closed_at is None and self.is_live


class Episode(BaseModel):
    """A captured turn. Append-only, never edited, never indexed destructively."""

    model_config = ConfigDict(extra="allow")

    id: str
    ts: datetime
    session: str
    agent: str
    role: Literal["user", "assistant", "system", "tool"]
    text: str
    project: str | None = None
    # How much the timestamp should be trusted. "exact" = read from the agent's
    # own per-turn log time. "estimated" = the source kept no per-turn time, so
    # ts is a session-level or day-level stand-in. Defaults to "estimated":
    # a capture path that forgets to declare its accuracy must degrade to
    # "we are not sure", never to "trust me".
    ts_source: Literal["exact", "estimated"] = "estimated"

    @field_validator("ts")
    @classmethod
    def _ts_must_be_aware(cls, value: datetime) -> datetime:
        # A naive timestamp makes every day-window filter ("昨晚") silently
        # wrong by the local UTC offset, so it is rejected at the boundary.
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("episode timestamps must be timezone-aware")
        return value

    @property
    def local_date(self) -> date:
        """The calendar day this turn belongs to for the operator."""
        return _local_date(self.ts)

    @property
    def time_is_exact(self) -> bool:
        return self.ts_source == "exact"

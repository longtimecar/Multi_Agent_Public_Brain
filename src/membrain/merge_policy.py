"""Lightweight type → merge strategy table (OpenViking gap 8 thought).

Sits above the ledger: facts remain append-only; this table only advises which
operation family extract/conflict should prefer. Never hard-deletes files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import FactType

# Strategies (product semantics, not OV runtime):
# - corroborate_same_identity: same subject → add evidence / keep both if uncertain
# - supersede_same_subject: explicit change on same subject → supersede via ledger
# - contest_on_conflict: contradictions stay contested for human resolve
# - append_distinct: treat as new (events/cases); do not collapse
# - never_auto_merge: sensitive/high-stakes; only explicit human correct/retract


@dataclass(frozen=True)
class MergeRule:
    fact_type: str
    strategy: str
    notes: str

    def as_dict(self) -> dict[str, str]:
        return {
            "type": self.fact_type,
            "strategy": self.strategy,
            "notes": self.notes,
        }


TYPE_MERGE_TABLE: dict[str, MergeRule] = {
    FactType.PREFERENCE.value: MergeRule(
        FactType.PREFERENCE.value,
        "supersede_same_subject",
        "Same owner + choice dimension: prefer supersede on explicit change; else corroborate.",
    ),
    FactType.GOAL.value: MergeRule(
        FactType.GOAL.value,
        "supersede_same_subject",
        "Goals evolve; explicit change supersedes prior live goal on same subject.",
    ),
    FactType.TOOL.value: MergeRule(
        FactType.TOOL.value,
        "supersede_same_subject",
        "Tool prefs/version notes supersede prior on same subject.",
    ),
    FactType.FACT.value: MergeRule(
        FactType.FACT.value,
        "contest_on_conflict",
        "Contradictory facts stay contested; never silent pick-a-side.",
    ),
    FactType.CONSTRAINT.value: MergeRule(
        FactType.CONSTRAINT.value,
        "contest_on_conflict",
        "Constraints are high-stakes; conflicts surface as contested.",
    ),
    FactType.BELIEF.value: MergeRule(
        FactType.BELIEF.value,
        "never_auto_merge",
        "Sensitive beliefs: store/mark only; merge/correct is human-driven.",
    ),
    FactType.ENTITY.value: MergeRule(
        FactType.ENTITY.value,
        "corroborate_same_identity",
        "Same real-world identity → corroborate; uncertain identity → keep separate.",
    ),
    FactType.PROJECT.value: MergeRule(
        FactType.PROJECT.value,
        "corroborate_same_identity",
        "Same project identity → corroborate; forks/aliases stay separate until explicit.",
    ),
    FactType.EVENT.value: MergeRule(
        FactType.EVENT.value,
        "append_distinct",
        "Distinct occurrences stay distinct even with shared participants/themes.",
    ),
    FactType.CASE.value: MergeRule(
        FactType.CASE.value,
        "append_distinct",
        "Cases/trajectories append; do not collapse into a single blob.",
    ),
}


def rule_for(fact_type: str | FactType | None) -> MergeRule:
    if isinstance(fact_type, FactType):
        key = fact_type.value
    else:
        key = str(fact_type or "").strip().lower()
    if key in TYPE_MERGE_TABLE:
        return TYPE_MERGE_TABLE[key]
    return MergeRule(
        key or "unknown",
        "contest_on_conflict",
        "Unknown type: default to contest_on_conflict (safe).",
    )


def preferred_operation(fact_type: str | FactType | None, *, explicit_change: bool = False) -> str:
    """Map strategy → extract-facing operation hint."""
    rule = rule_for(fact_type)
    if rule.strategy == "never_auto_merge":
        return "none"
    if rule.strategy == "append_distinct":
        return "add"
    if rule.strategy == "supersede_same_subject":
        return "supersede" if explicit_change else "corroborate"
    if rule.strategy == "corroborate_same_identity":
        return "corroborate"
    return "none"  # contest_on_conflict: leave to conflict scanner / human


def table_as_list() -> list[dict[str, str]]:
    return [TYPE_MERGE_TABLE[k].as_dict() for k in sorted(TYPE_MERGE_TABLE)]


def advise_skip(op_name: str, fact_type: str | FactType | None) -> dict[str, Any] | None:
    """Optional skipped_operations entry when an op fights the type policy."""
    rule = rule_for(fact_type)
    preferred = preferred_operation(fact_type, explicit_change=(op_name == "supersede"))
    if rule.strategy == "never_auto_merge" and op_name in {"supersede", "corroborate"}:
        return {
            "operation": op_name,
            "reason": f"type_policy:{rule.strategy}",
            "type": rule.fact_type,
            "advice": "use human correct/retract for belief/sensitive",
        }
    if rule.strategy == "append_distinct" and op_name == "supersede":
        return {
            "operation": op_name,
            "reason": f"type_policy:{rule.strategy}",
            "type": rule.fact_type,
            "advice": "prefer add for distinct events/cases",
            "preferred": preferred,
        }
    return None

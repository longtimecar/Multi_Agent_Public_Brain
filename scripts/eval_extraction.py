"""Evaluate extraction quality and safety guards on a labelled conversation suite.

Offline deterministic verification:
    uv run python scripts/eval_extraction.py --fake

Live LLM evaluation (reads credentials from .env or environment):
    uv run python scripts/eval_extraction.py

All facts and indexes live in an isolated temporary directory; real memory/
files are never read or modified.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from membrain.confidence import is_promoted
from membrain.config import Settings, load_settings
from membrain.embeddings import Embedder
from membrain.extract import Extractor
from membrain.index import Index
from membrain.llm import LLMClient
from membrain.models import Episode, Evidence, Fact, FactType, Status
from membrain.retrieval import Retriever
from membrain.store import Store

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "extraction_eval.json"


class _ScriptedEvalLLM:
    """Deterministic fake LLM for offline verification of Extractor guards and pipeline."""

    def __init__(self, mock_ops: list[dict[str, Any]], ref_map: dict[str, str]) -> None:
        self.mock_ops = mock_ops
        self.ref_map = ref_map

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        if "Compare personal facts" in system:
            return {"duplicate_id": None}
        if "evidence quote repair" in system:
            return {"quote": ""}
        payload = json.loads(user)
        user_turns = [t for t in payload.get("turns", []) if t.get("evidence") is True and t.get("episode")]
        default_ep = user_turns[-1]["episode"] if user_turns else ""
        resolved: list[dict[str, Any]] = []
        for raw in self.mock_ops:
            item = dict(raw)
            if "episode" not in item:
                item["episode"] = default_ep
            fact_ref = item.pop("fact_ref", None)
            if fact_ref and "fact_id" not in item:
                item["fact_id"] = self.ref_map.get(fact_ref, fact_ref)
            resolved.append(item)
        return {"operations": resolved}


def _make_settings(home: Path, live: Settings | None = None, use_embeddings: bool = False) -> Settings:
    if live is None:
        return Settings(
            home=home,
            embed_base_url=None,
            embed_api_key=None,
            embed_model="text-embedding-3-small",
            embed_timeout=20.0,
            turn_token_budget=1500,
            session_token_budget=2500,
        )
    return dataclasses.replace(
        live,
        home=home,
        embed_base_url=live.embed_base_url if use_embeddings else None,
        embed_api_key=live.embed_api_key if use_embeddings else None,
    )


def evaluate_case(
    case: dict[str, Any],
    *,
    fake: bool,
    live_settings: Settings | None = None,
    use_embeddings: bool = False,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="membrain-extract-eval-") as tmp:
        home = Path(tmp)
        settings = _make_settings(home, live=None if fake else live_settings, use_embeddings=use_embeddings)
        store = Store(home)
        index = Index(settings.db_path)
        embedder = Embedder(settings)
        retriever = Retriever(store, index, embedder, settings)
        try:
            ref_map: dict[str, str] = {}
            now = datetime.now(timezone.utc)
            seed_ids: set[str] = set()
            for idx, seed in enumerate(case.get("existing_facts") or []):
                ep = store.append_episodes([
                    Episode(
                        id=f"ep_seed_{idx}",
                        ts=now,
                        session="seed-session",
                        agent="eval-seed",
                        role="user",
                        text=seed["quote"],
                        ts_source="exact",
                    )
                ])[0]
                fact = store.draft_fact(
                    type=FactType(seed["type"]),
                    subject=seed["subject"],
                    claim=seed["claim"],
                    evidence=[Evidence(episode=ep.id, quote=seed["quote"], date=ep.local_date, origin="stated")],
                    source_agent="eval-seed",
                    source_session="seed-session",
                )
                stored = store.write_fact(fact)
                retriever.sync_fact(stored)
                seed_ids.add(stored.id)
                if seed.get("ref"):
                    ref_map[str(seed["ref"])] = stored.id

            session_id = f"eval-{case['id']}"
            episodes_to_save = []
            for t_idx, turn in enumerate(case.get("turns") or []):
                episodes_to_save.append(
                    Episode(
                        id=f"ep_{case['id']}_{t_idx}",
                        ts=now,
                        session=session_id,
                        agent="eval",
                        role=turn["role"],
                        text=turn["text"],
                        ts_source="exact",
                    )
                )
            store.append_episodes(episodes_to_save)

            llm: Any = (
                _ScriptedEvalLLM(case.get("mock_operations") or [], ref_map)
                if fake
                else LLMClient(settings)
            )
            extractor = Extractor(store, retriever, llm)
            report = extractor.run(session=session_id)

            all_facts = store.all_facts()
            new_facts: list[Fact] = [f for f in all_facts if f.id not in seed_ids]
            action = case["expected_action"]
            failures: list[str] = []

            if any(f.type == FactType.CONSTRAINT for f in new_facts):
                failures.append("invariant violated: extractor wrote a constraint")
            if any(f.type.value == "trait" for f in all_facts):
                failures.append("invariant violated: trait type appeared")

            min_w = int(case.get("expected_min_writes", 0))
            max_w = int(case.get("expected_max_writes", 99))
            if not min_w <= len(new_facts) <= max_w:
                failures.append(f"wrote {len(new_facts)} facts, expected [{min_w}, {max_w}]")

            if action == "no_write" and new_facts:
                failures.append(f"expected no writes, got {[f.claim for f in new_facts]}")
            elif action in {"write", "write_non_constraint"} and new_facts:
                expected_type = case.get("expected_type")
                if expected_type and all(f.type.value != expected_type for f in new_facts):
                    failures.append(
                        f"expected type {expected_type}, got {[f.type.value for f in new_facts]}"
                    )
                if case.get("expected_asserted") is True and not any(is_promoted(f) for f in new_facts):
                    failures.append(f"expected promoted/asserted fact, got conf {[f.confidence for f in new_facts]}")
            elif action == "write_unasserted" and new_facts:
                if any(is_promoted(f) for f in new_facts):
                    failures.append(f"hedged turn promoted unexpectedly: {[f.confidence for f in new_facts]}")
            elif action == "supersede":
                if report.superseded < 1:
                    failures.append("expected supersede operation")
                for ref, old_id in ref_map.items():
                    old_fact = store.get_fact(old_id)
                    if old_fact.status != Status.SUPERSEDED:
                        failures.append(f"seed {ref} ({old_id}) status={old_fact.status.value}, expected superseded")
            elif action == "retract":
                if report.retracted < 1:
                    failures.append("expected retract operation")
                for ref, old_id in ref_map.items():
                    old_fact = store.get_fact(old_id)
                    if old_fact.status != Status.RETRACTED:
                        failures.append(f"seed {ref} ({old_id}) status={old_fact.status.value}, expected retracted")

            return {
                "id": case["id"],
                "category": case.get("category", "general"),
                "language": case.get("language", ""),
                "expected_action": action,
                "passed": not failures,
                "failures": failures,
                "written": len(new_facts),
                "skipped": report.skipped,
                "rejected": report.rejected,
                "repaired": report.repaired,
                "superseded": report.superseded,
                "retracted": report.retracted,
                "claims": [{"type": f.type.value, "claim": f.claim, "confidence": f.confidence} for f in new_facts],
            }
        finally:
            index.close()


def run_eval(
    fixture_path: Path = DEFAULT_FIXTURE,
    *,
    fake: bool = False,
    use_embeddings: bool = False,
) -> dict[str, Any]:
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("extraction_eval fixture must be a non-empty JSON list")
    live_settings = None if fake else load_settings()
    if not fake and (live_settings is None or not live_settings.llm_enabled):
        raise RuntimeError("No LLM configured in .env/environment; run with --fake for offline guard checks.")

    results = [
        evaluate_case(case, fake=fake, live_settings=live_settings, use_embeddings=use_embeddings)
        for case in cases
    ]
    positive_cases = [r for r in results if r["expected_action"] != "no_write"]
    negative_cases = [r for r in results if r["expected_action"] == "no_write"]
    passed_count = sum(1 for r in results if r["passed"])
    tp = sum(1 for r in positive_cases if r["passed"])
    false_writes = sum(1 for r in negative_cases if r["written"] > 0)
    fp_rate = round(false_writes / len(negative_cases), 4) if negative_cases else 0.0
    pos_pass_rate = round(tp / len(positive_cases), 4) if positive_cases else 0.0
    overall_rate = round(passed_count / len(results), 4) if results else 0.0

    by_category: dict[str, dict[str, int]] = {}
    for row in results:
        cat = by_category.setdefault(row["category"], {"total": 0, "passed": 0})
        cat["total"] += 1
        if row["passed"]:
            cat["passed"] += 1

    return {
        "mode": "fake" if fake else "live",
        "fixture": str(fixture_path),
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "pass_rate": overall_rate,
        "positive_pass_rate": pos_pass_rate,
        "false_write_rate": fp_rate,
        "total_rejected": sum(r["rejected"] for r in results),
        "total_skipped": sum(r["skipped"] for r in results),
        "total_repaired": sum(r["repaired"] for r in results),
        "by_category": by_category,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--fake", action="store_true", help="use deterministic scripted LLM to test extractor rules offline")
    parser.add_argument("--with-embeddings", action="store_true", help="enable configured embeddings during live eval")
    parser.add_argument("--json", action="store_true", help="emit full JSON summary")
    parser.add_argument("--verbose", action="store_true", help="print per-case outcomes")
    args = parser.parse_args(argv)

    try:
        summary = run_eval(args.fixture, fake=args.fake, use_embeddings=args.with_embeddings)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(
            f"Extraction eval ({summary['mode']}): "
            f"{summary['passed']}/{summary['total']} passed ({summary['pass_rate']:.1%}) | "
            f"positive={summary['positive_pass_rate']:.1%} | "
            f"false_write_rate={summary['false_write_rate']:.1%} | "
            f"skipped={summary['total_skipped']} rejected={summary['total_rejected']}"
        )
        for cat, counts in summary["by_category"].items():
            print(f"  {cat:<22} {counts['passed']}/{counts['total']}")
        for row in summary["results"]:
            if not row["passed"] or args.verbose:
                status = "PASS" if row["passed"] else "FAIL"
                detail = "; ".join(row["failures"]) if row["failures"] else f"writes={row['written']} skipped={row['skipped']} rejected={row['rejected']}"
                print(f"  [{status}] {row['id']}: {detail}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

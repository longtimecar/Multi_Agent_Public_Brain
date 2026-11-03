from __future__ import annotations

import json
from types import SimpleNamespace

from membrain.hooks import _prompt_context, _session_start_context
from membrain.models import Evidence, FactType
from test_core import world, write
from datetime import date


def test_injection_skips_unchanged_facts_and_repeats_changed_facts(world):
    settings, store, _, retriever = world
    fact = write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    retriever.reindex()
    svc = SimpleNamespace(settings=settings, store=store, retriever=retriever)
    payload = {"session_id": "session", "prompt": "spicy food"}
    assert "spicy" in _prompt_context(svc, payload)
    assert _prompt_context(svc, payload) == ""
    store.corroborate(fact.id, Evidence(episode="another", quote="Still no spicy food", date=date.today()))
    assert "spicy" in _prompt_context(svc, payload)
    rows = [json.loads(line) for line in (settings.home / ".state" / "injection-log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [len(r["fact_ids"]) for r in rows] == [1, 0, 1]
    assert all(r["tokens"] <= settings.turn_token_budget for r in rows)


def test_compact_resets_injected_history_and_strips_profile_comments(world):
    settings, store, _, retriever = world
    settings.profile_path.write_text("# Profile\n<!-- secret comment -->\nName: test", encoding="utf-8")
    write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    retriever.reindex()
    svc = SimpleNamespace(settings=settings, store=store, retriever=retriever)
    prompt = {"session_id": "session", "prompt": "spicy food"}
    assert _prompt_context(svc, prompt)
    assert not _prompt_context(svc, prompt)
    block = _session_start_context(svc, {"session_id": "session", "source": "compact"})
    assert "secret comment" not in block and "Name: test" in block
    assert _prompt_context(svc, prompt)

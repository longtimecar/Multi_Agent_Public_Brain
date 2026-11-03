from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta

import pytest

from membrain.confidence import is_promoted
from membrain.models import Evidence, FactType, Status
from membrain.store import Store, StoreError
from test_core import write


def test_corroboration_replays_confidence_without_changing_fact_file(tmp_path):
    store = Store(tmp_path / "memory")
    fact = write(store, "Prefers concise answers", "style", FactType.PREFERENCE, days_ago=28)
    original = store.fact_path(fact.id).read_bytes()
    for i, days in enumerate((14, 0)):
        ev = Evidence(episode=f"independent_{i}", quote="Keep it short", date=date.today() - timedelta(days=days))
        assert store.corroborate(fact.id, ev)
        assert not store.corroborate(fact.id, ev)
    folded = Store(store.home).get_fact(fact.id)
    assert folded.n_episodes == 3 and is_promoted(folded)
    assert folded.confidence >= 0.6
    assert store.fact_path(fact.id).read_bytes() == original
    store.contest([fact.id], "contradiction")
    assert store.get_fact(fact.id).status == Status.CONTESTED
    assert store.get_fact(fact.id).confidence <= 0.45
    store.retract(fact.id, "wrong")
    with pytest.raises(StoreError):
        store.corroborate(fact.id, Evidence(episode="late", quote="x", date=date.today()))


def test_concurrent_processes_append_complete_ledger_entries(tmp_path):
    store = Store(tmp_path / "memory")
    script = (
        "import sys; from pathlib import Path; from membrain.store import Store; "
        "s=Store(Path(sys.argv[1])); n=int(sys.argv[2]); "
        "[s._append_ledger({'op':'test','id':str(n)+'-'+str(i),'payload':'x'*4096}) for i in range(50)]"
    )
    workers = [subprocess.Popen([sys.executable, "-c", script, str(store.home), str(i)]) for i in range(3)]
    for worker in workers:
        assert worker.wait(timeout=60) == 0
    lines = store.ledger_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    assert len(events) == 150
    assert len({event["id"] for event in events}) == 150

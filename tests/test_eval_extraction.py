"""Offline smoke test for the extraction evaluation harness and dataset."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_eval_module():
    path = ROOT / "scripts" / "eval_extraction.py"
    spec = importlib.util.spec_from_file_location("eval_extraction", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extraction_eval_fixture_structure_and_fake_run(capsys):
    fixture = ROOT / "tests" / "fixtures" / "extraction_eval.json"
    items = json.loads(fixture.read_text(encoding="utf-8"))
    assert len(items) >= 15
    assert any(c.get("language") == "zh" for c in items)
    assert any(c.get("language") == "en" for c in items)
    assert any(c.get("expected_action") == "no_write" for c in items)

    mod = _load_eval_module()
    summary = mod.run_eval(fixture, fake=True)
    assert summary["mode"] == "fake"
    assert summary["failed"] == 0, [r for r in summary["results"] if not r["passed"]]
    assert summary["false_write_rate"] == 0.0
    assert summary["positive_pass_rate"] == 1.0
    assert mod.main(["--fake", "--json"]) == 0
    cli_out = json.loads(capsys.readouterr().out)
    assert cli_out["passed"] == len(items)

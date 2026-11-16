# Project status

membrain 0.1.0 is a local-first, file-backed memory layer shared by multiple AI
agents. It is usable for daily work, but it is still an early release: verify it
with a disposable memory directory before relying on it for important data.

## Implemented

- **Canonical memory files.** Facts, episodes, profile, ledger and archive notes
  live as plain files under `memory/`. SQLite is a rebuildable index.
- **Lifecycle without destructive edits.** Facts are append-only. Correction,
  retraction, contest and supersede are ledger events.
- **Evidence requirement.** A stored claim must point to a verbatim user turn;
  model output and tool results cannot become evidence.
- **Multiple retrieval modes.** Lexical search, optional embeddings, optional
  intent expansion and optional reranking. Optional model-dependent stages fail
  open so a bad provider cannot block recall.
- **Session archive.** Sessions are captured in two phases and produce a
  human-readable `memory_diff` showing what was written or skipped.
- **Session history import.** Claude Code, Codex, OpenCode, WorkBuddy, Cursor and
  DeepSeek Harness histories can be discovered and imported through a read-only
  adapter. Other sources can use live capture through MCP where supported.
- **External web AI import.** Copy-paste guided export prompt for ChatGPT,
  Gemini, Claude, Kimi, Doubao, and DeepSeek with selectable `memory`
  distillation or `rag` library ingestion and batch undo (`external_import.py`).
- **Archive & RAG library.** Imported conversations and notes are browsable as L0
  summary, L1 source/category tree and L2 full text, alongside uploaded documents
  in `rag.py`.
- **Reversible 4-tier conflict resolution.** Unambiguous non-constraint
  contradictions follow deterministic precedence in `conflict.py` (`auto_resolve`
  with full ledger undo in Dashboard / Admin Agent), while `constraint` facts and
  exact ties stay `contested` for the owner.
- **Modular local dashboard.** `src/membrain/dashboard/` package (split into
  `runtime.py`, `admin_agent.py`, `http.py`) providing memory overview, search and
  correction, capture review, session diffs, external/session import, health scan,
  and model config on loopback only (guarded against DNS rebinding on GET and POST).
- **Windows-first setup.** `install.ps1` one-click installer, `membrain shortcut`
  (plus Dashboard Settings button) for windowless `.lnk` launch, and optional
  silent logon autostart (off by default).
- **Agent detection and connection.** Installed agents are detected per machine;
  safe config files can be connected from the dashboard and protected files get
  a copy-paste prompt instead.
- **Quality evaluation harnesses.** `scripts/calibrate.py` (82 ZH/EN recall pairs)
  and `scripts/eval_extraction.py` (`--fake` rule-guard baseline and live `.env`
  evaluation over `tests/fixtures/extraction_eval.json`).

## Architecture boundaries

1. **Add, never disable.** membrain is a shared layer. An agent's built-in memory
   remains active.
2. **No silent authority.** MCP tools, extractors, and external importers never
   create `constraint` facts, and imported notes cannot grant standing permission.
3. **Fail open.** One damaged source, model timeout or reranker failure must not
   stop capture or basic recall.
4. **Local by default.** The dashboard binds to loopback only and rejects foreign
   `Host` / `Origin` headers.
5. **No web-chat scraping.** Use official exports, prompt-guided copy/paste backfill,
   or local agent histories.

## Known limitations

- History adapters depend on each product's private local format and can break
  when that product changes its storage.
- Cursor history comes from its local search index, so per-turn detail can be
  coarser than JSONL sources.
- Model-based extraction can occasionally over-generalize. Review the library
  and session diffs; use retract, conflict undo, or batch undo when needed.
- Semantic recall quality depends on the chosen embedding model. Recalibrate the
  floor after changing models.
- Optional intent, rerank and evolution features remain off by default.
- Windows `install.ps1` and desktop `.lnk` shortcut behavior are unit-tested on
  Linux with mocks and queued for Windows CI / local PowerShell verification.

## Current work

Completed in earlier milestone (2026-11-05, pinned by `tests/test_temporal_recall.py`):

1. **Time is real.** Session parsers keep each turn's own timestamp (`exact` vs
   `estimated`); episodes are filed under the operator's calendar day (`MEMBRAIN_TZ`).
2. **Tasks are stored.** `FactType.TASK` is wired through extraction and durability
   is judged per type; claims use the owner's language.
3. **Ingestion standards agree across paths.** Extraction, MCP writes, session
   import distillation and the dashboard admin loop share the same per-type
   durability rule.
4. **Recall reads time and falls back.** `parse_time_window` resolves coarse
   phrases deterministically; FTS + CJK bigram scans raw episodes when needed.
5. **Capture does not need the dashboard.** `membrain watch` runs the loop in
   the foreground and `membrain autostart` installs it at login.
6. **Session summaries.** `membrain consolidate` writes `memory/summaries/*.md`
   as a derived view only.
7. **Pipeline resilience.** Fixed `agents._is_running` exact matching with
   injectable process lists, Qoder path detection, extraction lock deferral,
   newest-first bounded batch ordering, isolated operation failures with
   capped retries, and ordered fragment citation recovery in `locate_quote`.

Completed in open-source & usability milestone (2026-11-16):

1. Guarded dashboard GET/POST against DNS rebinding, pinned `mcp>=2`, added
   `ubuntu-latest` + `windows-latest` CI matrix, removed unrelated skill &
   legacy OpenViking import, and extracted shared `import_batches.py`.
2. Added external AI export import (`external_import.py` supporting `memory` and
   `rag` modes + batch undo) and 4-tier reversible conflict resolution (`conflict.py`,
   `store.py`, dashboard Health UI, and admin agent tools).
3. Split monolithic `dashboard.py` into `src/membrain/dashboard/{runtime,admin_agent,http}.py`.
4. Added Windows `membrain shortcut`, Dashboard Settings desktop shortcut button,
   and pure-ASCII `install.ps1` (autostart remains off by default).
5. Added ZH/EN extraction quality dataset (`tests/fixtures/extraction_eval.json`),
   evaluation runner (`scripts/eval_extraction.py`), and 1-week comparison guide.

## Next work

1. **Windows acceptance & live `.env` baseline.** Run `.\install.ps1`, double-click
   `membrain 记忆后台.lnk`, verify `windows-latest` CI, and run
   `uv run python scripts/eval_extraction.py` against the owner's configured model.
2. **One-week real-workflow evaluation.** Execute the three-way comparison protocol
   in [TESTING.md](TESTING.md) (`AGENTS.md` vs `membrain` vs competitor) and
   record concrete precision/recall and manual cleanup numbers.
3. **Cross-agent coverage expansion.** Externalize `agents.py` registry specs so
   community contributors can add new IDE/CLI adapters via JSON without code edits.
4. **Unified profile & portable backup.** Add evidence-linked `profile.md` draft
   generation (human-approved before write) and `membrain backup` / `restore` /
   `verify` CLI commands with ledger integrity checks.

## Verification

Run the complete suite from the repository root:

```bash
uv run pytest tests -q
node --check src/membrain/web/app.js
uv run python scripts/eval_extraction.py --fake
```

For a disposable manual test, set `MEMBRAIN_HOME` to a new temporary directory,
run `uv run membrain init`, and start the dashboard. Do not point destructive
tests at a real memory tree.

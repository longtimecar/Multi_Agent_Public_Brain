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
- **Archive library.** Imported conversations and notes are browsable as L0
  summary, L1 source/category tree and L2 full text.
- **Local dashboard.** Memory overview, search and correction, capture review,
  session diffs, library import and model configuration on loopback only.
- **Agent detection and connection.** Installed agents are detected per machine;
  safe config files can be connected from the dashboard and protected files get
  a copy-paste prompt instead.
- **Optional OpenViking import.** Disabled until the owner sets
  `MEMBRAIN_OPENVIKING_ROOT`; nothing is scanned from a built-in machine path.

## Architecture boundaries

1. **Add, never disable.** membrain is a shared layer. An agent's built-in memory
   remains active.
2. **No silent authority.** MCP tools cannot create `constraint` facts, and
   imported notes cannot grant standing permission.
3. **Fail open.** One damaged source, model timeout or reranker failure must not
   stop capture or basic recall.
4. **Local by default.** The dashboard binds to loopback and has no remote
   account system.
5. **No web-chat scraping.** Use official exports or local agent histories.

## Known limitations

- History adapters depend on each product's private local format and can break
  when that product changes its storage.
- Cursor history comes from its local search index, so per-turn detail can be
  coarser than JSONL sources.
- Model-based extraction can occasionally over-generalize. Review the library
  and session diffs; use retract or batch undo when needed.
- Semantic recall quality depends on the chosen embedding model. Recalibrate the
  floor after changing models.
- Optional intent, rerank and evolution features remain off by default.
- OpenViking is an optional legacy source, not the storage backend.

## Current work

Done on 2026-11-05, after the owner reported that "记住现在的进度，明天继续"
could not be asked back the next day. All four layers of that failure are
closed, and `tests/test_temporal_recall.py` pins each one:

1. **Time is real.** Session parsers keep each turn's own timestamp (per-turn
   where the source has one, session-level otherwise); `Episode.ts_source`
   records `exact` vs `estimated` and defaults to `estimated`, so a capture path
   that forgets to declare accuracy cannot claim precision. Naive timestamps are
   rejected. Episodes are filed under the operator's calendar day
   (`MEMBRAIN_TZ`), which is what stops a one-day-wide "昨晚" window from
   missing by a whole day in UTC+8.
2. **Tasks are stored.** `FactType.TASK` is wired through the type tables, the
   extraction prompt no longer says `Ignore tasks`, and durability is judged per
   type: "would this matter at the start of the next session" for tasks,
   "a month from now in an unrelated task" for everything else. The one-off-spec
   gate now runs *after* the type is known, so a turn that is both a deliverable
   spec and a handoff survives. Claims are written in the owner's language —
   an English claim could not be found by a Chinese question.
3. **Ingestion standards agree across paths.** Extraction, MCP writes, session
   import distillation and the dashboard admin loop were each applying a
   different bar; session import was the strictest and explicitly dropped task
   status. They now share the same per-type durability rule.
4. **Recall reads time and falls back.** `parse_time_window` resolves coarse
   phrases deterministically (no model in the hot path). Out-of-window evidence
   is demoted, never hard-filtered. Episodes have their own FTS index plus a CJK
   term scan, because unicode61 treats a whole Chinese run as one token. When
   curated facts cannot answer, raw turns are returned and labelled as such.
   `membrain tasks` lists open tasks; `Fact.closed_at` plus a ledger `close`
   event retires finished work, and an open task is never a forget candidate.
5. **Capture does not need the dashboard.** `membrain watch` runs the loop in
   the foreground and `membrain autostart` installs it at login.
6. **Session summaries.** `membrain consolidate` writes
   `memory/summaries/*.md` (done / decided / open) with pointers back to its
   episodes. Derived view only: never indexed as facts, never evidence.

Also fixed: `agents._is_running` matched process names by substring, so a
running `deepseek harness.exe` reported the DeepSeek desktop app as installed —
the same confusion the uninstall-list guard already handled. `scan` now takes an
injectable process list, so those tests no longer depend on what the owner
happens to have open.

## Next work

- Add adapters for additional local agent histories where the format is stable.
- Expand recall and extraction quality fixtures, especially multilingual cases.
- Improve first-run guidance and empty-state onboarding.
- Add deployment examples for common desktop environments.
- Continue security review of local HTTP endpoints and path handling.

## Verification

Run the complete suite from the repository root:

```bash
uv run pytest tests -q
node --check src/membrain/web/app.js
```

For a disposable manual test, set `MEMBRAIN_HOME` to a new temporary directory,
run `uv run membrain init`, and start the dashboard. Do not point destructive
tests at a real memory tree.

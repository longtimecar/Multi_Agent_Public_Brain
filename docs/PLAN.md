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

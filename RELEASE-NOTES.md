# membrain 0.1.0 — release notes

Personal, portable, cross-agent long-term memory for one owner. Your memories
live in plain files under a local memory tree; every agent you connect reads
and writes the same store.

## Install and run

```bash
git clone <this repo>
cd membrain
uv sync
uv run membrain init          # create the memory tree and profile
uv run membrain dashboard     # local dashboard on http://127.0.0.1:1966
```

Copy `.env.example` to `.env` and fill in your model / embedding endpoints.
Nothing is required to start; without a model the store still works read-only.

## Connect an agent

```bash
uv run python scripts/install_harness.py          # dry run: prints the plan
uv run python scripts/install_harness.py --apply   # writes the hook configs
```

The script fills the `<PROJECT>` placeholder in `harness/` with your checkout
path and installs into the agent's own config directory. It only **adds**
hooks — an agent's built-in memory is never disabled.

For MCP clients, copy `.mcp.json.example` to your MCP config and replace
`<PROJECT>` with your checkout path.

## What is in this bundle

- `src/membrain/` — the library: capture hooks, facts, retrieval, session
  archive (two-phase), archive library (L0/L1/L2), importers, dashboard server
- `tests/` — the test suite (`uv run pytest tests -q`)
- `scripts/` — session importers, harness installer, archive noise review,
  release builder
- `harness/` — thin agent hook plugins (Cursor, OpenCode, WorkBuddy)
- `docs/` — DESIGN.md (goals and decisions), PLAN.md (status), TESTING.md,
  OPEN_SOURCE_MEMORY_RESEARCH.md

## What is deliberately not in this bundle

- **Your memory.** The `memory/` tree is gitignored and shipped empty. It is
  created by `membrain init` on your machine. Nothing you write is ever
  committed.
- **Your keys.** Only `.env.example` (key names, no values) ships. `.env` is
  gitignored.
- **Machine paths.** Every absolute path in `harness/` is the `<PROJECT>`
  placeholder, resolved by the installer.

## Verified state (2026-11-03)

- The test suite passes; the concurrency case is stable across repeated runs
- Live end-to-end: session archive → LLM clean/classify → library note, with
  injected context stripped and duplicates blocked by content hash
- Six session sources are read (read-only, never modified): Claude Code,
  Codex, OpenCode, WorkBuddy, Cursor, DeepSeek Harness — plus OpenViking
  note trees

## Known limits

- Session history import covers the agents listed above; other IDEs can connect
  for live capture and on-demand recall, but their history is not imported yet.
- Cursor history comes from its local search index, so per-turn detail may be
  coarser than the other sources.
- DeepSeek Harness transcripts are zstd-compressed. Python 3.14+ reads them
  out of the box; older interpreters need `uv sync --extra dsh`.
- OpenViking history import is opt-in: set `MEMBRAIN_OPENVIKING_ROOT` to enable
  it. With no value set, no OpenViking path is scanned and the dashboard shows a
  concise optional-source empty state.

## License

MIT — see [LICENSE](LICENSE).

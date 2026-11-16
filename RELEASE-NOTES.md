# membrain 0.1.0 — release notes

Personal, portable, cross-agent long-term memory for one owner. Your memories
live in plain files under a local memory tree; every agent you connect reads
and writes the same store.

## Install and run

### Windows one-click installer

```powershell
git clone <this repo>
cd Multi_Agent_Public_Brain
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

`install.ps1` checks/installs `uv`, syncs dependencies, seeds `.env` from
`.env.example` if missing, runs `uv run membrain init`, and creates the
windowless desktop launcher `membrain 记忆后台.lnk` via
`uv run membrain shortcut` (without enabling logon autostart).

### Manual setup (Windows / Linux / macOS)

```bash
git clone <this repo>
cd Multi_Agent_Public_Brain
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

- `src/membrain/` — the core library: capture hooks, immutable facts, ledger,
  retrieval, session archive (L0/L1/L2), external AI export import
  (`external_import.py`), shared `import_batches.py`, 4-tier conflict handling
  (`conflict.py`), and the modular local dashboard (`dashboard/`)
- `install.ps1` — pure-ASCII Windows one-click bootstrap script
- `tests/` — offline pytest suite (`uv run pytest tests -q`) and labelled ZH/EN
  fixtures (`recall_pairs.json`, `extraction_eval.json`)
- `scripts/` — session importers, harness installer, Windows shortcut helper
  (`create_shortcut.py`), recall calibrator (`calibrate.py`), extraction eval
  runner (`eval_extraction.py`), and release builder (`build_release.py`)
- `harness/` — thin agent hook plugins (Cursor, OpenCode, WorkBuddy)
- `docs/` — `DESIGN.md`, `PLAN.md`, `TESTING.md`, `ONBOARDING.md`,
  `OPEN_SOURCE_MEMORY_RESEARCH.md`

## What is deliberately not in this bundle

- **Your memory.** The `memory/` tree is gitignored and shipped empty. It is
  created by `membrain init` on your machine. Nothing you write is ever
  committed.
- **Your keys.** Only `.env.example` (key names, no values) ships. `.env` is
  gitignored.
- **Machine paths.** Every absolute path in `harness/` is the `<PROJECT>`
  placeholder, resolved by the installer.

## Verified state (2026-11-16)

- Full offline test suite passes in Linux CI (`ubuntu-latest`), covering
  immutability, ledger folding, confidence calculation, reversible 4-tier
  conflict resolution, external AI `memory`/`rag` import & batch undo,
  dashboard DNS-rebinding guards on GET/POST, and mocked Windows shortcut checks
- `scripts/eval_extraction.py --fake` passes 18/18 ZH/EN rule-guard checks with
  `false_write_rate = 0.0`
- Live local capture & read-only session import cover six local session
  sources: Claude Code, Codex, OpenCode, WorkBuddy, Cursor, and DeepSeek Harness
- External web AI backfill works via a copyable bilingual export prompt for
  ChatGPT, Claude web, Gemini, Kimi, Doubao, and DeepSeek (`memory` or `rag`
  mode, with batch undo)

## Known limits

- Session history import covers the local agents listed above; web chat UIs are
  never scraped — use the prompt-guided external AI import card or official
  exports instead.
- Cursor history comes from its local search index, so per-turn detail may be
  coarser than JSONL sources.
- DeepSeek Harness transcripts are zstd-compressed. Python 3.14+ reads them
  out of the box; older interpreters use the `zstandard` dependency bundled in
  `pyproject.toml`.
- Windows runtime checks (`install.ps1` and real `.lnk` desktop shortcut launch)
  and live `.env` extraction/comparison benchmarks (`scripts/eval_extraction.py`
  and [docs/TESTING.md](docs/TESTING.md)) are verified via mocks in Linux and
  await execution on a Windows machine with the owner's model endpoint.

## License

MIT — see [LICENSE](LICENSE).

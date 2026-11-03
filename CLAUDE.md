# membrain

Cross-agent personal memory for one owner. Before changing anything, read
[docs/DESIGN.md](docs/DESIGN.md) (goals and decisions) and
[docs/PLAN.md](docs/PLAN.md) (status, known issues, next tasks). Work through
the tasks in PLAN.md in order and update it as each one finishes.

## Rules that must hold

- Fact files are immutable after writing. Lifecycle changes go to `memory/ledger.jsonl`.
- No trait type and no personality inference.
- Evidence is only text the user typed: never assistant replies, tool output or injected memory blocks.
- Confidence is computed in `confidence.py`, never taken from a model.
- Contradictions are marked contested and never resolved automatically.
- No MCP tool or extractor writes a `constraint`.
- Hooks always exit 0. A memory failure must never break the user's turn.
- `memory/.index/` is disposable. Store nothing there that cannot be rebuilt.

## Commands

```bash
uv sync
```

```bash
uv run pytest -q
```

```bash
uv run membrain stats
```

## Environment

- Windows. The username contains non-ASCII characters, and Git Bash mangles them in paths. Build paths inside Python, not in shell strings.
- Bare `python` is the Windows Store stub. Use `uv run` or `.venv/Scripts/python.exe`.
- No git repo, by the owner's choice. Do not run `git init`.
- The MCP SDK is 2.x: `from mcp.server.mcpserver import MCPServer`. Result fields are snake_case (`structured_content`, `input_schema`).

## Secrets

- `.env` holds live API keys. Never print it. To check it, print key names and whether each is set.
- `~/.claude/settings.json` holds the owner's auth token. Never print it. Change it with a script that reads and writes JSON.
- `.env.example` is the committed template and must stay free of real values.

## Ask the owner first

- Editing `.claude/settings.json`, `~/.claude/settings.json` or `~/.codex/config.toml`.
- Writing to `memory/profile.md` or adding constraint facts.
- Deleting anything under `memory/` other than `.index/`.

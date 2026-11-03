# harness/ — thin agent plugins

Ready-made hook wiring that calls membrain. Source of truth for the
capture/inject logic remains `scripts/harness_hook.py` + `src/membrain/hooks.py`.

| Agent | Live install path | Repo snapshot |
|---|---|---|
| Cursor | `~/.cursor/hooks.json` | `harness/cursor/hooks.json` |
| OpenCode | `~/.config/opencode/plugins/membrain.ts` | `harness/opencode/membrain.ts` |
| WorkBuddy | `~/.workbuddy/settings.json` → `hooks` | `harness/workbuddy/hooks.snippet.json` |
| Claude Code | `~/.claude/settings.json` → `hooks` | `scripts/hook.py` (event names only differ) |

## The `<PROJECT>` and `<PYTHON>` placeholders

`<PROJECT>` stands for the absolute path of your checkout. `<PYTHON>` is the
interpreter inside its virtual environment (`.venv/Scripts/python.exe` on
Windows, `.venv/bin/python` elsewhere). The installer resolves both for the
current machine, so nothing here points at the author's paths.

Fill it in automatically:

```bash
uv run python scripts/install_harness.py          # dry run — prints the plan
uv run python scripts/install_harness.py --apply   # writes the configs
```

The installer merges into an existing config file instead of overwriting it,
and it only **adds** hooks — an agent's built-in memory is never disabled.

Hard constraint: ADD ONLY — never disable agent built-in memory.

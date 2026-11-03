# membrain

Portable, user-owned long-term memory shared across AI agents.

Design and reasoning: [docs/DESIGN.md](docs/DESIGN.md). Status and next tasks:
[docs/PLAN.md](docs/PLAN.md).

Your memory lives in **Markdown and JSONL files you own**. The database is a
disposable index. Any MCP client can read and write it; agents with lifecycle
hooks capture and recall automatically.

## Why files first

| Layer | What | Disposable? |
|---|---|---|
| Canonical | `memory/facts/*.md`, `memory/episodes/*.jsonl`, `memory/profile.md`, `memory/ledger.jsonl` | **No — this is the product** |
| Derived | `memory/.index/memory.db` | Yes, rebuilt by `membrain reindex` |
| Access | MCP server (`membrain-mcp`) + agent hooks (`scripts/hook.py`) | Yes |

Delete the entire `.index/` directory and nothing is lost. That is the property
that makes this memory yours rather than a vendor's.

## Design rules

1. **Append-only.** Fact files are never edited after write. Lifecycle changes
   (`retract`, `contest`, `resolve`, `supersede`) are appended to
   `ledger.jsonl` and folded at read time.
2. **No `trait` type.** The ontology has `constraint`, `fact`, `preference`,
   `goal`, `project`. Inferred personality and decision-style claims are not
   representable, so no prompt can drift the system into a flattering caricature.
3. **Evidence or nothing.** Every claim carries a verbatim quote and the
   episode it came from. No quote, no write.
4. **Confidence is computed in code.** Models cluster at 0.8-0.9 regardless of
   truth, so the score comes from evidence count, time spread, stated-vs-inferred,
   and contradiction status.
5. **A single observation is never asserted.** One-episode claims are capped at
   0.40 and injected as questions. Silence never counts as confirmation.
6. **Conflicts are surfaced, never auto-resolved.** Detection is automatic and
   mechanical; resolution is `membrain resolve`, by you.
7. **Constraints are human-only.** No MCP tool can write a `constraint`, so a
   poisoned tool result cannot become a standing instruction.

## Install

```bash
git clone https://github.com/<you>/membrain.git
cd membrain
uv sync
uv run membrain init
```

`membrain init` creates an empty `memory/` tree on your machine. The repository
ships **no memories and no keys** — there is nothing of the author's in it to
accidentally inherit.

Credentials go in `.env` at the repository root. It is gitignored; copy the
template and fill it in:

```bash
cp .env.example .env
```

Real environment variables override `.env`. `uv run membrain stats` reports
which file was loaded and what is configured, without printing any key.

Embeddings are optional: without them retrieval falls back to full-text.
Any OpenAI-compatible `/embeddings` endpoint works, including a local server.
After setting or changing the embedding model, run `uv run membrain reindex`.
Qwen3 retrieval queries get an instruction by default; override it with
`MEMBRAIN_EMBED_QUERY_INSTRUCTION`. `MEMBRAIN_SEMANTIC_FLOOR` controls the
minimum similarity (default `0.41`). Calibrate for the configured model with
`uv run python scripts/calibrate.py`. The labelled cases include hard negatives;
the command reports unrelated injections and fails on missed related cases. Add
`--strict` to fail on unrelated injections too. One threshold cannot
guarantee that every unrelated query is excluded. When embeddings are unavailable
or a fact has no vector, lexical fallback can also return unrelated matches.
The configured LLM extracts facts in the background after captured turns.
Run `uv run membrain extract` to process pending turns manually; extraction
errors are reported and pending turns can be retried.

OpenViking history is optional and disabled by default. Point
`MEMBRAIN_OPENVIKING_ROOT` at the OpenViking user folder only if you want the
legacy import card to scan it. With no value set, membrain never looks for a
built-in OpenViking path on your machine.

## Use

```bash
uv run membrain add "Prefers terse answers" --type preference --subject communication --quote "keep it short"
uv run membrain add "Never use em dashes" --type constraint --subject style
uv run membrain list
uv run membrain search "any dietary considerations?"
uv run membrain show f_20261031_abcdef
uv run membrain correct f_20261031_abcdef --reason "user says this is outdated"
uv run membrain resolve f_NEW --drop f_OLD
uv run membrain reindex
uv run membrain stats
uv run membrain extract
uv run membrain doctor
```

For a browser-based memory manager, run `uv run membrain dashboard`. It opens
the local dashboard at `http://127.0.0.1:1966/` and uses the same memory files
as the CLI and MCP server. The service binds to loopback only. Its interface
can add facts from owner-entered quotes, replace or retract memories, review
conflicts, trigger extraction, and rebuild the disposable index. Fact Markdown
files remain immutable; lifecycle changes are appended to the ledger.

`add --type constraint` is the only way to create a hard constraint. That is
deliberate.

## Agent integration

**MCP (portable floor).** Copy `.mcp.json.example` to `.mcp.json` and replace
`<PROJECT>` with the absolute path of your checkout; point your MCP client at
that file. Tools: `search_memory`, `write_memory`, `correct_memory`,
`list_contested`, `get_profile`, `memory_stats`, `find_user_evidence`. Agent
writes require a captured user episode and an exact quote; `find_user_evidence`
locates its ID.

**Hooks (automatic).** `harness/` holds one ready-made hook config per agent.
The installer fills in your path and copies the config into place:

```bash
uv run python scripts/install_harness.py          # dry run — prints the plan
uv run python scripts/install_harness.py --apply   # writes the configs
uv run python scripts/install_harness.py --apply --agent cursor
```

Session start injects profile + constraints + goals; each prompt injects a
budgeted relevant-memory block; each stop appends new turns to a raw episode
file with no model call.

**Claude Code** wires `scripts/hook.py` to `SessionStart`, `UserPromptSubmit`,
`Stop`, `PreCompact`, `SessionEnd` through `.claude/settings.json`.

Other agents use the same shim with their own event names — Codex and Cursor
expose hooks, Gemini CLI uses `SessionStart` / `BeforeAgent` / `SessionEnd`,
OpenCode uses a plugin. Anything without hooks still gets memory through MCP.

**What is not supported, on purpose:** scraping web chat UIs and HTTPS
interception. They break on every redesign, violate terms of service at every
vendor simultaneously, and lose fidelity. Use the official data export for
backfill instead.

## Hygiene

```bash
uv run pytest -q
```

Covers: fact round-trip, append-only immutability, ledger folding, confidence
capping, same-day corroboration not promoting, lexical and semantic recall,
budget enforcement without mid-item truncation, conflict surfacing without
resolution, explicit resolution, index rebuild from Markdown, and evidence
validation.

For live workflow checks see [docs/TESTING.md](docs/TESTING.md). Project hooks
are not yet installed globally and cross-agent behavior still requires manual
acceptance testing.

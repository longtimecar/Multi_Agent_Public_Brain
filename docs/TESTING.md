# MVP manual checks

## Browser dashboard

Run `uv run membrain dashboard` to open the local manager at
`http://127.0.0.1:1966/`. Review the real counts, select a fact to inspect its
quotes and source, and use the filters or search box. The dashboard supports
owner-entered facts, replacements with a typed quote, conflict marking and
resolution, retraction, background extraction and index rebuild. Replacement
and retraction preserve the original fact file and append lifecycle events.
The profile is shown read-only because it contains owner-authored constraints.

The default address is loopback-only. On Windows, `127.0.0.0` cannot be bound;
the command normalizes that requested address to `127.0.0.1` while keeping port
1966.

The dashboard also contains an Agent workspace. The conversation window calls
the configured LLM through the backend, retrieves relevant memory for each user
message, and shows the returned source facts beside the answer. The model card
can test the current connection or save a new base URL, API format, model name
and timeout. Leave the API Key field blank to keep the existing key; the key is
never displayed back in the page or API response.

Run these from the repository root after `uv sync`. The commands inspect and
modify your real memory. Use synthetic examples only if you are comfortable
keeping them in the append-only history; `correct` retracts rather than erases.
Use an isolated `MEMBRAIN_HOME` for disposable testing.

1. Run `uv run membrain doctor` and `uv run membrain stats`. Check model
   reachability and the five project hook events. Restart the agent session
   after changing `.env`.
2. In the dashboard, open **配置或更换模型**, confirm the model name and
   interface format, then click **测试模型连接**. Send a short question in
   **与 Agent 对话** and verify that an answer appears, the source panel lists
   any recalled facts, and a user plus assistant episode is present in the
   capture list. Assistant text must never appear as evidence on a fact.
3. In a Claude Code session opened in this project, state a durable synthetic
   fact. After the response completes, run `uv run membrain list` within a minute.
   Inspect it with `uv run membrain show <id>` and verify the quote is your
   actual message. Run `uv run membrain extract` for a manual retry and inspect
   `memory/.state/extract.log` if no fact appears.
4. Open a Codex session configured with the same MCP server. Use
   `search_memory` to find the fact and `membrain show <id>` to inspect its
   source. Repeat in the opposite direction only with a captured user episode:
   MCP writes require a real episode ID and a verbatim quote.
   To test automatic capture in Codex, first review and trust this project's
   `SessionStart` and `UserPromptSubmit` hooks in Codex's hook settings (or
   `/hooks` in the CLI). Start a new project session, state a real durable fact
   in an ordinary user message, then check the dashboard's capture and memory
   lists. Confirm its source is `codex` and its evidence is your exact text.
   This captures new messages only; it does not import historical chats.
5. For a dietary example, search in both languages with `uv run membrain search
   "any dietary considerations?"` and its Chinese equivalent. Confirm the
   relevant fact is present. Some irrelevant hits are acceptable under the
   recall-first decision; `uv run python scripts/calibrate.py --strict` reports
   them as failures for measurement.
6. Start a new Claude Code session and confirm the profile appears. Ask two
   questions that retrieve the same fact; verify the second prompt does not
   inject the unchanged fact again. Inspect `memory/.state/injection-log.jsonl`
   for session IDs, fact IDs and token counts (at most 2500 for session start,
   normally at most 1500 per prompt, with a hard per-prompt cap of 2000).
7. Repeat the original claim in another user turn. After extraction, inspect
   the fact: it should have another evidence episode rather than a duplicate.
   State an explicit change or correction and confirm the prior entry is
   superseded or retracted and the new one is retrieved.
8. Verify conflict handling:
   - Ambiguous tie or `constraint` conflict stays `contested` until resolved via
     `uv run membrain resolve <keep> --drop <id>` or the dashboard Health view.
   - An unambiguous non-constraint contradiction (e.g., local firm `stated` vs
     `imported`, explicit change cue, or strictly more evidence days/episodes)
     records a reversible `auto_resolve` decision visible in the Health view
     (**自动冲突裁决记录**) and undoable back to `contested`.
9. In Dashboard -> **资料库 -> 数据导入**, open **从其他 AI 导入记忆 / 资料**
   (ChatGPT / Claude / Gemini / Kimi / 豆包 / DeepSeek): copy the guided export
   prompt, test pasting sample structured text into both `rag` (reference
   library only) and `memory` (distilled `origin="imported"` facts), then verify
   batch undo reverts the import cleanly.
10. On Windows, run `uv run membrain shortcut` (or click **创建桌面快捷方式** in
    Settings) and verify double-clicking `membrain 记忆后台.lnk` opens the
    dashboard with no console window and without starting a second daemon.
11. With all memory processes stopped, move the disposable index directory
    elsewhere or delete it, then run `uv run membrain reindex` and verify facts
    and statuses remain. Never remove the canonical facts, episodes or ledger.
12. After every project-level criterion passes and the owner approves any
    requested settings changes, review per-project vs global hook setup carefully
    before touching user-home settings files.

To run with an isolated home in PowerShell, set `$env:MEMBRAIN_HOME` to a new
temporary folder before launching the agent and CLI processes. Clear that
environment override afterward. An isolated home needs its own profile if
profile injection is being tested.

## Automated quality evaluation

- **Offline unit & guard suite**: `uv run pytest tests -q`
- **Extraction rule-guard check (offline)**:
  `uv run python scripts/eval_extraction.py --fake --verbose`
- **Live LLM extraction benchmark** (uses `.env` LLM against an isolated temp home):
  `uv run python scripts/eval_extraction.py --verbose`
  Targets: `positive_pass_rate >= 85%`, `false_write_rate == 0%` (no command misfiled as preference, no assistant quote accepted, no `constraint`/`trait` written).
- **Embedding recall calibration**:
  `uv run python scripts/calibrate.py` (`--strict` to fail on hard negatives).

## 1-Week three-way real-workflow comparison protocol

To verify whether `membrain` measurably beats plain Markdown rules or a
commercial/open-source alternative on your own daily work rather than synthetic
benchmarks, run a structured 7-day comparison across the same kinds of tasks:

### Three arms

1. **Arm A (`AGENTS.md` / native rules baseline)**: static repository `AGENTS.md` /
   `CLAUDE.md` + manually pasted project context, no external memory service.
2. **Arm B (`membrain`)**: cross-agent shared memory + hooks/watcher + external
   import + 4-tier reversible conflict handling.
3. **Arm C (competitor of choice)**: e.g. OpenViking, Mem0/OpenMemory, Basic
   Memory, or vendor built-in cloud memory (e.g. Claude/ChatGPT native memory).

### Protocol (7 days, same daily tasks)

- **Seed identically on Day 0**: seed 15–20 real personal preferences, project
  constraints, active tech stacks, and 3 pending task handoffs into each arm
  (in `membrain`, put hard constraints in `profile.md` or `membrain add --type constraint`,
  and optionally import prior AI context via **外部 AI 导入**).
- **Exercise at least 2 agents per day**: e.g. code or plan in Claude Code /
  Codex in the morning, continue or review in Cursor / Qoder / Dashboard in the afternoon.
- **Inject 3 realistic updates mid-week**:
  1. One explicit preference/tool switch ("我现在改用 X 了").
  2. One accidental contradiction against an earlier imported note.
  3. One overnight task handoff ("记住进度，明天继续 Y").

### Metrics to log daily

| Metric | How to measure | Pass bar for `membrain` |
|---|---|---|
| Repeat-explanation count | Times you had to restate a known preference/constraint manually | Lower than Arm A & <= Arm C |
| Cross-agent handoff hit rate | Overnight / cross-IDE task resumed accurately in the next session | >= 80% without manual paste |
| False/polluting memory writes | Commands/assistant replies stored as preferences, or stale values resurfacing | Lower than Arm C; 0 unrecoverable mutations |
| Conflict resolution accuracy | Contradictions caught cleanly; ties paused instead of silently picking wrong side | <= 1 manual undo needed per week |
| Maintenance time (minutes/week) | Total minutes spent editing `profile.md`, reviewing Health Center, undoing bad items | <= 10 min/week |

At the end of Day 7, keep whichever system wins on **(repeat explanations saved +
handoff accuracy) minus (pollution cleanup cost)**. If Arm A (`AGENTS.md`) wins on
your actual workflow with near-zero maintenance cost, simplify down to static docs
and file-backed notes without feeling obligated to keep background extraction running.

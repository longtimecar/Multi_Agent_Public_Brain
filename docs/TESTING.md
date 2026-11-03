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
8. State two contradictory facts without saying either one supersedes the
   other. Verify both are retained, flagged contested, and never automatically
   resolved. Resolve only through `uv run membrain resolve <keep> --drop <id>`.
9. With all memory processes stopped, move the disposable index directory
   elsewhere or delete it, then run `uv run membrain reindex` and verify facts
   and statuses remain. Never remove the canonical facts, episodes or ledger.
10. After every project-level criterion passes and the owner approves the
   requested settings changes, install globally and repeat in another folder.
   Global installation is not yet implemented.

To run with an isolated home in PowerShell, set `$env:MEMBRAIN_HOME` to a new
temporary folder before launching the agent and CLI processes. Clear that
environment override afterward. An isolated home needs its own profile if
profile injection is being tested.

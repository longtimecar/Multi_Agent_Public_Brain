# membrain design

The source of truth for what membrain is for and why it is built this way. It
replaces the planning file from the first design conversation. Current status
and next tasks are in [PLAN.md](PLAN.md). Parts marked *(MVP)* are designed but
not built yet.

## Goal

A portable, user-owned memory of facts and constraints about one person, shared
by every AI agent they use, so they stop repeating themselves.

What the owner asked for, in priority order:
1. Usability and accuracy: agents know the relevant facts without being told again.
2. Several agents read and write the same memory.
3. Writing is automatic: no "remember this" commands and no review of each entry.
4. Recall works by meaning, not keywords, across Chinese and English.
5. Prefer relevant memory and keep it within a small budget. If relevance and recall cannot both be perfect, the owner prefers occasional irrelevant injection over missing related memories.
6. The memory outlives any vendor: plain files the owner can read and edit.

Privacy is not a concern for the owner. Cloud APIs are used for the LLM and embeddings.

Not the goal: modelling personality, decision style or other inferred traits.
The owner chose the fact-and-constraint foundation after agreeing that inference
from chat suffers from biased samples, flattering bias and confirmation loops.

## Capture boundary

Automatic capture is possible only where an agent offers lifecycle hooks or
plugins. OpenViking, Basic Memory and similar projects all work this way; none
scrape web UIs.

- Project-scoped automatic capture is implemented for Claude Code (transcript
  hooks) and Codex (the top-level `UserPromptSubmit` hook). Both still require
  live acceptance; Codex also requires the owner to review and trust the hooks.
  Cursor, Gemini CLI and OpenCode adapters remain future work.
- Shared without auto-capture: any MCP client, through the MCP tools.
- Not supported, on purpose: scraping ChatGPT, Claude or Gemini web apps, and HTTPS interception. Official data exports are the route for backfill.

## Architecture

| Layer | Contents | Role |
|---|---|---|
| Canonical | `memory/profile.md`, `memory/facts/*.md`, `memory/episodes/*.jsonl`, `memory/ledger.jsonl` | The product. Plain files. No git repo (owner's choice); history comes from immutable fact files and the append-only ledger. |
| Index | `memory/.index/memory.db`: SQLite FTS5 plus float32 vectors | Derived. Delete it and `membrain reindex` rebuilds it. |
| Access | MCP server `membrain-mcp`; hooks through `scripts/hook.py` | MCP is the portable floor; hooks make it automatic. |
| Extraction | Background LLM pass over new episodes | Turns what the user said into facts. |

## Data model

- **Fact**: one claim per file with YAML front-matter. Types: `constraint`, `fact`, `preference`, `goal`, `project`. There is no `trait` type, so inferred personality cannot be stored.
- **Language**: claims are written in English (owner's choice). Quotes stay verbatim in whatever language the user typed.
- **Evidence**: each fact carries one or more `{episode, quote, date, origin}`. The quote must be text the user actually typed.
- **Ledger**: fact files are never edited after writing. `retract`, `contest`, `resolve`, `supersede`, `link` and `corroborate` are appended to `ledger.jsonl` and folded at read time.
- **Episode**: one captured turn, user or assistant, appended to `episodes/YYYY-MM-DD.jsonl`. Stored raw, never interpreted at capture time.
- **Confidence**: computed in `confidence.py` from evidence count, spread over time, stated vs inferred, and contested status. Never taken from a model. A plain first-person statement ("I'm allergic to peanuts") reaches 0.62 to 0.70 from one episode and is asserted. A statement containing a hedge ("maybe", "也许", "if") or inferred evidence stays under 0.40 until it recurs. Hedges are found by a fixed word list in `text.py`, and the extractor widens a quote to its clause so a dropped "也许" is not lost. A fact is asserted at 0.60. Changed 2026-11-01 at the owner's request; before that, one episode capped at 0.40 whatever it said.

## Write path

Archive tier (2026-11-01): long notes that are too long or task-specific to inject live in
`memory/archive/` as immutable files with an L0 abstract in the front matter, an L1 overview
computed per category, and the L2 body. They are found by `search_archive` and never injected
automatically. Facts distilled from imported notes carry evidence origin `imported`: not the
owner's typed text, so never accepted by the extractor or MCP writes as evidence, but trusted by
the owner's choice and treated as explicit for confidence. Imports are reversible by batch.

0. Claude Code `UserPromptSubmit` records the typed prompt immediately, so an
   agent can cite it as evidence in the same reply. The later transcript pass
   skips turns already recorded. Agents without hooks call the MCP tool
   `record_user_message` with the user's words; those episodes carry
   `relayed=true` because the server cannot check that the text is genuine.
1. Claude Code `Stop`, `PreCompact` and `SessionEnd` hooks append new turns;
   the Codex `UserPromptSubmit` hook checks the current user message against
   the local desktop transcript in a background process before recording it.
   Missing or unfamiliar provenance is skipped. Neither capture path calls the
   model synchronously.
2. A background extractor reads unprocessed user turns, retrieves similar existing facts, and asks the LLM for operations: `add`, `corroborate`, `supersede`, `retract`, `none`. Code validates each operation before writing.
3. After each write, a mechanical conflict scan (`conflict.py`) marks contradictions `contested`.

## Read path

- `SessionStart`: inject `profile.md`, constraints and confident goals, capped at 2,500 tokens.
- `UserPromptSubmit`: hybrid retrieval (FTS5 plus vectors, combined by rank fusion), top 8, capped at 1,500 tokens (hard cap 2,000). Explicit requests for a memory or preference overview instead enumerate the relevant live category, newest first, subject to the same caps; this avoids requiring a broad question to resemble each individual fact. Low-confidence items are phrased as unconfirmed notes; contested ones are labelled. Items are dropped whole, never cut off. Matches with an available vector below the similarity floor are dropped, including lexical matches. When the embedding service or a fact's vector is unavailable, lexical fallback has no semantic guard. Overview recognition currently uses limited phrase patterns, not general intent understanding.
- MCP `search_memory` for agents that want to look something up themselves.

The budgets follow OpenViking's defaults (session-start block about 9,500 bytes,
per-turn recall 2,000 tokens) and Anthropic's context-engineering guidance:
over-retrieval degrades agents well before the context limit.

## Who decides what

| Task | Owner |
|---|---|
| What is worth remembering | Backend LLM, at write time |
| Duplicate or new | Code selects embedding candidates; LLM compares their meaning |
| Explicit change ("I've switched to X") | Backend LLM, recorded as `supersede` *(MVP)* |
| Correction ("that memory is wrong") | Backend LLM watching the turn *(MVP)*, or the agent through `correct_memory` |
| Confidence | Code |
| Detecting contradictions | Code, recorded as `contested` |
| Resolving contradictions | The owner, with `membrain resolve`. Never automatic. |
| Hard constraints | The owner only, through the CLI or `profile.md`. No MCP tool or extractor writes one. |

The owner does not review entries one by one. Review at write time is the
backend LLM plus code validation. The owner corrects mistakes when a wrong
memory shows up in a conversation, by telling the active agent.

## Rules against flattering drift and confirmation loops

Enforced by schema and code, not by asking the model to be careful:
1. No trait type exists.
2. No quote, no write. Quotes are checked against the user's own turn *(MVP)*.
3. Only text the user typed counts as evidence. Assistant replies, tool output and injected memory blocks never do *(MVP: enforced in capture and extraction)*. This stops the system confirming itself.
4. Silence is never confirmation. An unchallenged hedge adds no evidence.
5. Single observations are injected as unconfirmed notes, not as facts.
6. A correction retracts permanently and outweighs any number of earlier mentions.
7. Conflicts are surfaced, never resolved automatically.

## Prompt-injection safety

Memory persists, so a poisoned entry would come back in every future session.
- No MCP tool writes constraints.
- Evidence comes only from user-typed text, so a web page or tool result cannot become a memory.
- Injected memory is wrapped in `<memory>` or `<personal_memory>` tags, which capture strips *(MVP)*.

## Models

Set in `.env` (gitignored; the template is `.env.example`).
- Embeddings: `Qwen/Qwen3-Embedding-8B` through SiliconFlow, 4,096 dimensions. Chosen over bge-m3 for retrieval accuracy: multilingual MTEB retrieval 70.88 against 54.60, from the table on Qwen's model card. Queries should carry Qwen's instruction prefix; stored facts do not need one.
- LLM: `agnes-2.5-flash` at `api.agnes-ai.cn`, OpenAI-compatible chat completions. Accepts `response_format: json_object`.

Measured on 2026-10-31 with 5 facts and 5 queries in Chinese and English:
- Embedding one query takes 0.5 to 0.6 s; a batch of 5 takes 2.6 s.
- The right fact ranked first for 5 of 5 queries, with or without the instruction.
- With the instruction, right matches scored at least 0.47 (mean 0.52) and unrelated pairs averaged 0.32, peaking at 0.42. Without it, unrelated pairs peaked at 0.56, above the lowest right match (0.52). So the instruction is what makes a similarity floor possible.
- An LLM JSON call takes 0.55 to 0.9 s and parses cleanly. The model paraphrases and translates Chinese into English, so code must enforce verbatim quotes.

The broader 37-pair calibration on 2026-10-31 found overlapping scores: one
related Chinese dietary query scored 0.4443 while an unrelated jazz-work query
scored 0.5018. At the former 0.43 floor, 36 of 37 cases passed; the default is
now 0.41 (owner's recall-first choice), which injects 2 unrelated pairs. A floor alone
cannot separate those two pairs. Run `uv run python scripts/calibrate.py` after
changing the embedding model, query instruction, or floor; it uses synthetic
data and isolated temporary stores.

## References

- OpenViking, https://github.com/volcengine/OpenViking: hook plus MCP capture, injection budgets.
- Basic Memory, https://github.com/basicmachines-co/basic-memory: Markdown as source of truth, SQLite as index.
- Mem0 / OpenMemory, https://github.com/mem0ai/mem0: v3 extraction is ADD-only; its issue tracker reports stale and contradictory facts piling up, which is why membrain flags conflicts.
- Zep / Graphiti: time-bounded validity on facts, the model for supersession.
- Anthropic, "Effective context engineering for AI agents", https://www.anthropic.com/research/effective-context-engineering-for-ai-agents

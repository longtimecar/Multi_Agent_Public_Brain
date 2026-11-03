/**
 * membrain OpenCode thin plugin (Phase 4)
 * ADD ONLY: does not disable OpenCode / provider / MCP / built-in memory.
 *
 * Dual-shape (same pattern as rtk.ts):
 * - V2: setup() → ctx.session.hook("prompt"|"context"|"compaction")
 * - V1: server() → event + experimental.chat.system.transform + chat.message
 *
 * Capture/inject by shelling this checkout's scripts/harness_hook.py so Python
 * remains the single source of truth for L0/L1 + sensitive gating.
 */
import { execFile } from "node:child_process"
import { existsSync } from "node:fs"
import { promisify } from "node:util"

const execFileAsync = promisify(execFile)

const PY =
  "<PYTHON>"
const HARNESS =
  "<PROJECT>/scripts/harness_hook.py"

type Json = Record<string, unknown>

async function runHarness(
  event: string,
  payload: Json,
): Promise<{ context: string; raw: string }> {
  if (!existsSync(PY) || !existsSync(HARNESS)) {
    return { context: "", raw: "" }
  }
  try {
    const { stdout } = await execFileAsync(
      PY,
      [HARNESS, "opencode", event, "--json"],
      {
        input: JSON.stringify(payload),
        windowsHide: true,
        timeout: 25_000,
        maxBuffer: 2 * 1024 * 1024,
        encoding: "utf8",
      },
    )
    const text = String(stdout || "").trim()
    try {
      const parsed = JSON.parse(text) as { context?: string }
      return { context: String(parsed.context || ""), raw: text }
    } catch {
      return { context: text, raw: text }
    }
  } catch (err) {
    console.warn("[membrain] harness_hook failed", err)
    return { context: "", raw: "" }
  }
}

const injectedSessions = new Set<string>()

async function injectOnce(sessionID: string, systemPush?: (text: string) => void) {
  if (!sessionID || injectedSessions.has(sessionID)) return
  const { context } = await runHarness("session-start", {
    session_id: sessionID,
    agent: "opencode",
  })
  injectedSessions.add(sessionID)
  if (context && systemPush) systemPush(context)
  return context
}

const MembrainOpenCodePlugin = {
  id: "membrain",

  async setup(ctx: any) {
    // Capture user prompt text at admission (V2).
    await ctx?.session?.hook?.("prompt", async (event: any) => {
      const sessionID = String(event?.sessionID || event?.session_id || "")
      const text = String(event?.prompt?.text || "")
      if (!text.trim()) return
      await runHarness("prompt-submit", {
        session_id: sessionID,
        prompt: text,
        agent: "opencode",
      })
      // Best-effort session inject on first prompt if context hook missed.
      await injectOnce(sessionID)
    })

    // Inject profile+L0+L1 (non-sensitive) into system context.
    await ctx?.session?.hook?.("context", async (event: any) => {
      const sessionID = String(event?.sessionID || "")
      if (!sessionID || injectedSessions.has(sessionID)) return
      const { context } = await runHarness("session-start", {
        session_id: sessionID,
        agent: "opencode",
      })
      injectedSessions.add(sessionID)
      if (context) {
        event.system = event.system || []
        event.system.push({ type: "text", text: context })
      }
    })

    // Auto-commit / extract on compaction.
    await ctx?.session?.hook?.("compaction", async (event: any) => {
      const sessionID = String(event?.sessionID || "")
      await runHarness("precompact", {
        session_id: sessionID,
        agent: "opencode",
        prompt: "",
      })
    })
  },

  async server() {
    return {
      event: async ({ event }: any) => {
        if (event?.type === "session.created") {
          const id = String(event?.properties?.info?.id || event?.properties?.sessionID || "")
          if (id) await injectOnce(id)
        }
        if (event?.type === "session.compacted") {
          const id = String(event?.properties?.sessionID || event?.properties?.info?.id || "")
          await runHarness("precompact", { session_id: id, agent: "opencode", prompt: "" })
        }
      },
      "experimental.chat.system.transform": async (input: any, output: any) => {
        const sessionID = String(input?.sessionID || "")
        if (!sessionID || injectedSessions.has(sessionID)) return
        const { context } = await runHarness("session-start", {
          session_id: sessionID,
          agent: "opencode",
        })
        injectedSessions.add(sessionID)
        if (context) {
          output.system = output.system || []
          output.system.push(context)
        }
      },
      "chat.message": async (input: any, output: any) => {
        const sessionID = String(input?.sessionID || "")
        const parts = Array.isArray(output?.parts) ? output.parts : []
        const text = parts
          .map((p: any) => (typeof p?.text === "string" ? p.text : ""))
          .filter(Boolean)
          .join("\n")
        // Only record likely user turns (role heuristics).
        const role = String(output?.message?.role || input?.role || "user")
        if (role === "user" && text.trim()) {
          await runHarness("prompt-submit", {
            session_id: sessionID,
            prompt: text,
            agent: "opencode",
          })
        }
      },
    }
  },
}

export default MembrainOpenCodePlugin
export { MembrainOpenCodePlugin }
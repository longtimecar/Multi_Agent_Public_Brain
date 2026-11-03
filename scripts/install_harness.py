"""Install the agent hook wiring for this machine.

Usage:
    uv run python scripts/install_harness.py            # dry run, prints the plan
    uv run python scripts/install_harness.py --apply     # write the files

What it does:
  * fills the <PROJECT> placeholder in harness/* with the absolute path of
    this checkout, so the hook configs point at your own interpreter
  * copies the hook config into the agent's live config directory
  * merges into an existing config file rather than overwriting it
  * never disables an agent's built-in memory — it only adds hooks

Supported agents: cursor, opencode, workbuddy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLACEHOLDER = "<PROJECT>"
PYTHON_PLACEHOLDER = "<PYTHON>"

# agent -> source file inside harness/, live config path relative to the home
# directory, and the format of that file
AGENTS = {
    "cursor": {
        "src": "harness/cursor/hooks.json",
        "live_rel": Path(".cursor") / "hooks.json",
        "format": "json",
    },
    "opencode": {
        "src": "harness/opencode/membrain.ts",
        "live_rel": Path(".config") / "opencode" / "plugins" / "membrain.ts",
        "format": "text",
    },
    "workbuddy": {
        "src": "harness/workbuddy/hooks.snippet.json",
        "live_rel": Path(".workbuddy") / "settings.json",
        "format": "workbuddy",
    },
}


def local_python(project: Path) -> Path:
    venv = project / ".venv"
    candidates = (venv / "Scripts" / "python.exe", venv / "bin" / "python")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def fill(text: str, project: Path) -> str:
    return (text.replace(PYTHON_PLACEHOLDER, local_python(project).as_posix())
            .replace(PLACEHOLDER, project.as_posix()))


def merge_workbuddy(snippet: dict, live_path: Path) -> dict:
    """Merge the membrain hook block into ~/.workbuddy/settings.json."""
    live: dict = {}
    if live_path.is_file():
        try:
            live = json.loads(live_path.read_text(encoding="utf-8"))
        except ValueError:
            live = {}
    hooks = live.setdefault("hooks", {})
    membrain = hooks.setdefault("membrain", {})
    membrain.update(snippet.get("membrain", snippet))
    return live


def resolve_home() -> Path:
    """Home directory for the agent configs.

    Path.home() follows USERPROFILE on Windows and ignores HOME, which makes it
    impossible to test the installer against a throwaway directory. HOME wins
    when it is set, so the script can be pointed at a scratch directory.
    """
    env_home = os.environ.get("HOME")
    if env_home:
        return Path(env_home)
    return Path.home()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the files (default: dry run)")
    ap.add_argument("--agent", action="append", choices=sorted(AGENTS), help="limit to one agent")
    ap.add_argument("--project", default=str(REPO), help="absolute path of this checkout")
    ap.add_argument("--home", default=None,
                    help="home directory to install into (default: $HOME, else the user home)")
    args = ap.parse_args()

    project = Path(args.project).resolve()
    home = Path(args.home).resolve() if args.home else resolve_home()
    targets = list(args.agent) if args.agent else sorted(AGENTS)

    for name in targets:
        spec = AGENTS[name]
        src = REPO / spec["src"]
        if not src.is_file():
            print(f"[{name}] skipped: {spec['src']} not found")
            continue
        live: Path = home / spec["live_rel"]
        print(f"[{name}] {spec['src']} -> {live}")
        if not args.apply:
            print("          (dry run — pass --apply to write)")
            continue

        text = src.read_text(encoding="utf-8")
        filled = fill(text, project)
        live.parent.mkdir(parents=True, exist_ok=True)

        # A live config may already hold the owner's other settings. Keep a
        # copy before touching it, so a mistaken run is always recoverable.
        backup: Path | None = None
        if live.is_file():
            backup = live.with_name(live.name + ".bak-membrain")
            shutil.copy2(live, backup)

        if spec["format"] == "json":
            data = json.loads(filled)
            if live.is_file():
                try:
                    existing = json.loads(live.read_text(encoding="utf-8"))
                except ValueError:
                    existing = {}
                # Keep every key of the live file; only the hooks block is
                # extended with the membrain entries.
                merged_hooks = existing.get("hooks", {})
                merged_hooks.update(data.get("hooks", {}))
                existing["hooks"] = merged_hooks
                data = existing
            live.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        elif spec["format"] == "workbuddy":
            snippet = json.loads(filled)
            live.write_text(json.dumps(merge_workbuddy(snippet, live), indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        else:
            live.write_text(filled, encoding="utf-8")
        print(f"          written" + (f" (backup: {backup})" if backup else ""))

    if not args.apply:
        print("\nNothing was written. Re-run with --apply to install.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

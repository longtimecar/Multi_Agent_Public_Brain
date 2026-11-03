"""Build a clean, publishable release bundle: no personal data, no secrets.

Usage:  uv run python scripts/build_release.py
Output: dist/membrain-0.1.0/membrain-0.1.0/  +  dist/membrain-0.1.0.zip

What it does:
  * copies an allowlist of code/docs (never memory/, .env, backups, evidence)
  * scrubs absolute paths from text files
  * scans the staged tree for secret-looking strings and refuses to package on a hit
  * writes RELEASE-NOTES.md and a zip

The bundle ships an empty memory tree: the library creates its own on
`membrain init`, so a downloader never inherits somebody else's memories.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "dist" / "membrain-0.1.0"
STAGE = DIST / "membrain-0.1.0"

INCLUDE_FILES = ["pyproject.toml", "README.md", "AGENTS.md", "CLAUDE.md", ".env.example",
                 ".gitignore", "LICENSE", "RELEASE-NOTES.md", ".mcp.json.example"]
INCLUDE_DIRS = ["src", "tests", "scripts", "harness", "skills"]
DOC_ALLOW = {"DESIGN.md", "PLAN.md", "TESTING.md", "OPEN_SOURCE_MEMORY_RESEARCH.md"}
SKIP_DIRS = {"__pycache__", ".venv", ".pytest_cache", ".mypy_cache", "node_modules",
             "backups", "evidence", "memory", "dist", "promo_membrain_20261031",
             "reference_materials", "content_directions", ".edge-check"}
SKIP_SUFFIX = {".pyc", ".pyo", ".log", ".db", ".sqlite", ".sqlite3", ".zip", ".whl"}

# Remove machine-local absolute paths from the staged text. The release must
# not depend on knowing the author's directory names in advance.
SANITIZE = [
    (re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"'<>`|]+"), "<PATH>"),
    (re.compile(r"(?<![A-Za-z0-9])/home/(?![\\[()])[^\\/\s\"'<>`]+"), "<HOME>"),
]

# Sanitization should have removed these. Keep a second guard so a future file
# type or an exception cannot silently publish machine identity or local paths.
PRIVATE_PATTERNS = [
    (re.compile(r"(?i)(?<![A-Za-z0-9])[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]"),
     "absolute user path"),
    (re.compile(r"(?<![\w.])/home/(?![\\[()])[^\\/\s\"'<>`]+"), "absolute home path"),
]

SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "openai-style key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws access key"),
    (re.compile(r"eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}"), "jwt"),
    (re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"), "private key"),
    (re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{24,}['\"]"), "inline secret"),
]

TEXT_SUFFIX = {".py", ".md", ".toml", ".txt", ".json", ".yaml", ".yml", ".cfg", ".ini",
               ".html", ".css", ".js", ".ts", ".ps1", ".sh", ".example"}


def sanitize(text: str) -> str:
    for pattern, repl in SANITIZE:
        text = pattern.sub(repl, text)
    return text


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in TEXT_SUFFIX or src.name.endswith(".example"):
        try:
            dst.write_text(sanitize(src.read_text(encoding="utf-8")), encoding="utf-8")
        except UnicodeDecodeError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def stage() -> list[Path]:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    staged: list[Path] = []
    for name in INCLUDE_FILES:
        src = REPO / name
        if src.is_file():
            copy_file(src, STAGE / name)
            staged.append(STAGE / name)
    for name in INCLUDE_DIRS:
        src_dir = REPO / name
        if not src_dir.is_dir():
            continue
        for path in src_dir.rglob("*"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.is_dir() or path.suffix.lower() in SKIP_SUFFIX:
                continue
            rel = path.relative_to(REPO)
            copy_file(path, STAGE / rel)
            staged.append(STAGE / rel)
    docs = REPO / "docs"
    if docs.is_dir():
        for path in docs.iterdir():
            if path.is_file() and path.name in DOC_ALLOW:
                copy_file(path, STAGE / "docs" / path.name)
                staged.append(STAGE / "docs" / path.name)
    (STAGE / ".gitignore").write_text(
        "memory/\n.env\n.venv/\n__pycache__/\n*.pyc\ndist/\ndocs/backups/\ndocs/evidence/\n",
        encoding="utf-8")
    return staged


def scan_tree(root: Path, patterns: list[tuple[re.Pattern[str], str]]) -> list[str]:
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIX:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern, label in patterns:
            if pattern.search(text):
                hits.append(f"{path.relative_to(root)}: {label}")
                break
    return hits


def main() -> int:
    staged = stage()
    hits = scan_tree(STAGE, SECRET_PATTERNS)
    if hits:
        print("SECRET SCAN FAILED — nothing packaged:")
        for hit in hits:
            print("  -", hit)
        return 1
    private_hits = scan_tree(STAGE, PRIVATE_PATTERNS)
    if private_hits:
        print("PRIVATE-DATA SCAN FAILED — nothing packaged:")
        for hit in private_hits:
            print("  -", hit)
        return 1
    zip_path = REPO / "dist" / "membrain-0.1.0.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                zf.write(path, Path("membrain-0.1.0") / path.relative_to(STAGE))
    print(f"staged files: {len(staged)}")
    print("secret scan: clean")
    print("private-data scan: clean")
    print(f"bundle: {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")
    print(f"tree:   {STAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

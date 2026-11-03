"""Peer / workspace identity helpers (OpenViking gap 7 thought).

Default recall stays cross-project (peer ignored). Optional peer_scope=actor
filters to matching peer OR facts with no peer. Writing may attach a detected
workspace peer without changing default recall.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

from .uri import parse_uri

_PEER_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def normalize_peer(raw: str | None) -> str | None:
    text = (raw or "").strip()
    if not text:
        return None
    # Normalize common git remote shapes toward host-org-repo
    if "://" in text or text.startswith("git@"):
        text = _git_remote_to_peer(text) or text
    cleaned = _PEER_SAFE.sub("-", text).strip("-").lower()
    return cleaned[:120] or None


def _git_remote_to_peer(remote: str) -> str | None:
    raw = remote.strip()
    if raw.startswith("git@"):
        # git@github.com:org/repo.git
        raw = raw.replace(":", "/", 1).replace("git@", "https://", 1)
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    path = unquote(parsed.path or "").strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if not host and not path:
        return None
    joined = f"{host}-{path}".replace("/", "-")
    return _PEER_SAFE.sub("-", joined).strip("-").lower() or None


def detect_workspace_peer(cwd: str | Path | None = None) -> str | None:
    """Best-effort peer from git origin, else repo root name. Never raises."""
    root = Path(cwd) if cwd else Path.cwd()
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode != 0:
            return None
        top = Path((proc.stdout or "").strip())
        if not top.exists():
            return None
        remote = subprocess.run(
            ["git", "-C", str(top), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if remote.returncode == 0 and remote.stdout.strip():
            return normalize_peer(remote.stdout.strip())
        return normalize_peer(top.name)
    except Exception:
        return None


def fact_peer(fact_uri: str | None, fact_peer_field: str | None = None) -> str | None:
    if fact_peer_field and str(fact_peer_field).strip():
        return normalize_peer(str(fact_peer_field))
    parsed = parse_uri(fact_uri or "")
    return normalize_peer(parsed.get("peer") if isinstance(parsed.get("peer"), str) else None)


def peer_allows(
    fact_uri: str | None,
    fact_peer_field: str | None,
    *,
    peer: str | None,
    peer_scope: str = "all",
) -> bool:
    """Default peer_scope=all → always True. actor → match peer or empty peer."""
    scope = (peer_scope or "all").strip().lower()
    if scope in {"", "all", "any", "cross"}:
        return True
    want = normalize_peer(peer)
    if scope not in {"actor", "peer", "strict"}:
        return True
    if not want:
        return True
    have = fact_peer(fact_uri, fact_peer_field)
    if have is None:
        return True  # unscoped facts remain visible
    return have == want

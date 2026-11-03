"""URI namespaces for membrain facts (OpenViking-port Scheme A).

Canonical form: mb://user/memories/<type>/<subject>
Optional peer suffix is a query param only for filtering; default recall
is cross-project (peer ignored unless scope asks for it).
"""

from __future__ import annotations

import re
from urllib.parse import quote, unquote, urlparse

URI_SCHEME = "mb"
URI_PREFIX = "mb://user/memories/"
RESOURCE_PREFIX = "mb://user/resources/"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(subject: str) -> str:
    cleaned = _SLUG_RE.sub("-", subject.strip().lower()).strip("-")
    return cleaned or "general"


def build_uri(fact_type: str, subject: str, peer: str | None = None) -> str:
    """Build mb://user/memories/<type>/<subject>[?peer=...]."""
    path = f"{fact_type.strip().lower()}/{slugify(subject)}"
    uri = f"{URI_PREFIX}{path}"
    if peer and peer.strip():
        uri += f"?peer={quote(peer.strip(), safe='')}"
    return uri


def parse_uri(uri: str) -> dict[str, str | None]:
    """Parse a fact URI into type / subject / peer. Tolerant of missing pieces."""
    raw = (uri or "").strip()
    if not raw:
        return {"type": None, "subject": None, "peer": None, "prefix": ""}
    parsed = urlparse(raw)
    peer = None
    if parsed.query:
        for part in parsed.query.split("&"):
            if part.startswith("peer="):
                peer = unquote(part[5:]) or None
    path = parsed.path or ""
    # urlparse("mb://user/memories/x/y") -> netloc=user, path=/memories/x/y
    if parsed.scheme == URI_SCHEME and parsed.netloc == "user":
        parts = [p for p in path.split("/") if p]
        if parts and parts[0] == "memories":
            parts = parts[1:]
        fact_type = parts[0] if parts else None
        subject = parts[1] if len(parts) > 1 else None
        prefix = f"{URI_PREFIX}{fact_type}/" if fact_type else URI_PREFIX
        return {"type": fact_type, "subject": subject, "peer": peer, "prefix": prefix}
    # Fallback: treat as path-like prefix for scope matching
    return {"type": None, "subject": None, "peer": peer, "prefix": raw.rstrip("/")}


def uri_matches_scope(uri: str, scope: str | None) -> bool:
    """True if uri is under scope prefix (or scope is empty = all)."""
    if not scope or not scope.strip():
        return True
    scope = scope.strip()
    # Allow shorthand: type name or type/subject without scheme
    if "://" not in scope:
        if "/" in scope:
            scope = URI_PREFIX + scope.lstrip("/")
        else:
            scope = URI_PREFIX + scope.strip().lower() + "/"
    uri_n = (uri or "").split("?", 1)[0].rstrip("/")
    scope_n = scope.split("?", 1)[0].rstrip("/")
    return uri_n == scope_n or uri_n.startswith(scope_n + "/")


def ensure_uri(fact_type: str, subject: str, uri: str | None = None, peer: str | None = None) -> str:
    if uri and uri.strip():
        return uri.strip()
    return build_uri(fact_type, subject, peer)


def build_resource_uri(category: str, subject: str) -> str:
    """mb://user/resources/<category>/<subject> (not a memory fact)."""
    return f"{RESOURCE_PREFIX}{slugify(category)}/{slugify(subject)}"

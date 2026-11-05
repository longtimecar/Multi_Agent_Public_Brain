"""Configuration, resolved once from environment.

Precedence: real environment variable > `.env` file > built-in default. The
`.env` file sits at the repository root (override with MEMBRAIN_ENV_FILE) and
is gitignored, so keys stay out of the repo and out of chat logs. Nothing is
hardcoded per machine.
"""

from __future__ import annotations

import math
import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[2]

# Running from a source checkout: the repository root is two levels above
# src/membrain. Running from an installed wheel: that path is inside
# site-packages, which is not a place to keep memories or read a .env from.
_IN_CHECKOUT = (_PKG_ROOT / "pyproject.toml").is_file() and (_PKG_ROOT / "src").is_dir()
IN_CHECKOUT = _IN_CHECKOUT

REPO_ROOT = _PKG_ROOT if _IN_CHECKOUT else Path.home() / ".membrain"
DEFAULT_HOME = REPO_ROOT / "memory"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
QWEN_RETRIEVAL_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
DEFAULT_SEMANTIC_FLOOR = 0.41


def load_dotenv(path: Path, environ: MutableMapping[str, str] | None = None) -> bool:
    """Read KEY=VALUE lines into `environ`, never overriding a non-empty value.

    Supports comments, blank lines, an optional `export ` prefix, single or
    double quotes, and ` #` inline comments on unquoted values. Returns whether
    the file existed.
    """
    environ = os.environ if environ is None else environ
    if not path.is_file():
        return False
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        # An empty variable counts as unset, matching _env() below.
        if not environ.get(key):
            environ[key] = value
    return True


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@lru_cache(maxsize=8)
def _tzinfo_for(name: str):
    """Resolve an IANA timezone name; fall back to the system default."""
    if name:
        try:
            from zoneinfo import ZoneInfo

            return ZoneInfo(name)
        except Exception:
            pass
    return datetime.now(timezone.utc).astimezone().tzinfo


def resolve_tzinfo():
    """The operator's timezone: MEMBRAIN_TZ (IANA name) or the system default.

    Time-aware recall compares a user's phrase ("昨晚") against stored turns,
    so both sides must be the same calendar. Reading it here keeps `.env`
    handling in one place instead of scattered os.environ lookups.
    """
    return _tzinfo_for((_env("MEMBRAIN_TZ") or "").strip())


def local_date(ts: datetime) -> date:
    """Calendar day of a timestamp in the operator's timezone."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(resolve_tzinfo()).date()


@dataclass(frozen=True)
class Settings:
    home: Path
    embed_base_url: str | None
    embed_api_key: str | None
    embed_model: str
    embed_timeout: float
    turn_token_budget: int
    session_token_budget: int
    # Background extraction (Phase 2). Read now so `.env` has one shape.
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_api_format: str = "openai"  # "openai" or "anthropic"
    env_file: Path | None = None  # the .env that was loaded, if any
    embed_query_instruction: str | None = None
    semantic_floor: float = DEFAULT_SEMANTIC_FLOOR
    llm_timeout: float = 30.0
    duplicate_floor: float = 0.70
    # Scheme A auto-commit (tokens OR idle minutes). LLM quota OK.
    auto_commit_token_threshold: int = 8000
    auto_commit_idle_minutes: int = 30
    library_auto_import: bool = True
    # OV gaps 2+3 feature flags (default off — SessionStart stays fast).
    use_intent_default: bool = False
    use_rerank_default: bool = False
    intent_max_queries: int = 5
    rerank_candidate_factor: int = 2
    # Gap 9/10/11/13 (evolution/rerank default off; skills inject on if present)
    evolution_enabled: bool = False
    skills_inject_session_start: bool = True
    rerank_provider: str = "heuristic"
    rerank_base_url: str | None = None
    rerank_api_key: str | None = None
    rerank_model: str | None = None
    rerank_timeout: float = 20.0
    # Optional legacy source. It is unset by default so a fresh install never
    # scans a path that belonged to the machine that built the release.
    openviking_root: Path | None = None
    # Operator timezone (IANA name, "" = system default). Drives which calendar
    # day a captured turn belongs to and how "昨晚" resolves on recall.
    tz: str = ""

    @property
    def facts_dir(self) -> Path:
        return self.home / "facts"

    @property
    def episodes_dir(self) -> Path:
        return self.home / "episodes"

    @property
    def index_dir(self) -> Path:
        return self.home / ".index"

    @property
    def db_path(self) -> Path:
        return self.index_dir / "memory.db"

    @property
    def profile_path(self) -> Path:
        return self.home / "profile.md"

    @property
    def embeddings_enabled(self) -> bool:
        return bool(self.embed_base_url)

    @property
    def query_instruction(self) -> str:
        if self.embed_query_instruction is not None:
            return self.embed_query_instruction.strip()
        if "qwen3-embedding" in self.embed_model.lower():
            return QWEN_RETRIEVAL_INSTRUCTION
        return ""

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)

    def describe(self) -> dict[str, str]:
        """Configuration summary that is safe to print. Never includes keys."""
        return {
            "env_file": str(self.env_file) if self.env_file else "not found",
            "embeddings": f"enabled ({self.embed_model})" if self.embeddings_enabled else "disabled (lexical only)",
            "llm": f"configured ({self.llm_api_format}, {self.llm_model})" if self.llm_enabled else "not configured",
            "evolution": "enabled" if getattr(self, "evolution_enabled", False) else "disabled",
            "rerank_provider": getattr(self, "rerank_provider", "heuristic"),
            "skills_inject": "on" if getattr(self, "skills_inject_session_start", True) else "off",
            "timezone": self.tz or "system default",
        }


def load_settings() -> Settings:
    env_file = Path(_env("MEMBRAIN_ENV_FILE") or DEFAULT_ENV_FILE).expanduser()
    found = load_dotenv(env_file)
    home = Path(_env("MEMBRAIN_HOME") or DEFAULT_HOME).expanduser().resolve()
    openviking_raw = _env("MEMBRAIN_OPENVIKING_ROOT")
    openviking_root = Path(openviking_raw).expanduser().resolve() if openviking_raw else None
    semantic_floor = _env_float("MEMBRAIN_SEMANTIC_FLOOR", DEFAULT_SEMANTIC_FLOOR)
    if not math.isfinite(semantic_floor) or not 0.0 <= semantic_floor <= 1.0:
        semantic_floor = DEFAULT_SEMANTIC_FLOOR
    llm_timeout = _env_float("MEMBRAIN_LLM_TIMEOUT", 30.0)
    if not math.isfinite(llm_timeout) or llm_timeout <= 0:
        llm_timeout = 30.0
    duplicate_floor = _env_float("MEMBRAIN_DUPLICATE_FLOOR", 0.70)
    if not math.isfinite(duplicate_floor) or not 0.0 <= duplicate_floor <= 1.0:
        duplicate_floor = 0.70
    return Settings(
        home=home,
        embed_base_url=_env("MEMBRAIN_EMBED_BASE_URL"),
        embed_api_key=_env("MEMBRAIN_EMBED_API_KEY"),
        embed_model=_env("MEMBRAIN_EMBED_MODEL", "text-embedding-3-small") or "text-embedding-3-small",
        embed_timeout=_env_float("MEMBRAIN_EMBED_TIMEOUT", 20.0),
        turn_token_budget=_env_int("MEMBRAIN_TURN_TOKEN_BUDGET", 1500),
        session_token_budget=_env_int("MEMBRAIN_SESSION_TOKEN_BUDGET", 2500),
        llm_base_url=_env("MEMBRAIN_LLM_BASE_URL"),
        llm_api_key=_env("MEMBRAIN_LLM_API_KEY"),
        llm_model=_env("MEMBRAIN_LLM_MODEL"),
        llm_api_format=(_env("MEMBRAIN_LLM_API_FORMAT", "openai") or "openai").strip().lower(),
        env_file=env_file if found else None,
        embed_query_instruction=_env("MEMBRAIN_EMBED_QUERY_INSTRUCTION"),
        semantic_floor=semantic_floor,
        llm_timeout=llm_timeout,
        duplicate_floor=duplicate_floor,
        auto_commit_token_threshold=_env_int("MEMBRAIN_AUTO_COMMIT_TOKENS", 8000),
        auto_commit_idle_minutes=_env_int("MEMBRAIN_AUTO_COMMIT_IDLE_MINUTES", 30),
        library_auto_import=(_env("MEMBRAIN_LIBRARY_AUTO_IMPORT", "1") or "1").strip().lower() in {"1", "true", "yes", "on"},
        use_intent_default=(_env("MEMBRAIN_USE_INTENT", "0") or "0").strip().lower() in {"1", "true", "yes", "on"},
        use_rerank_default=(_env("MEMBRAIN_USE_RERANK", "0") or "0").strip().lower() in {"1", "true", "yes", "on"},
        intent_max_queries=_env_int("MEMBRAIN_INTENT_MAX_QUERIES", 5),
        rerank_candidate_factor=_env_int("MEMBRAIN_RERANK_CANDIDATE_FACTOR", 2),
        evolution_enabled=(_env("MEMBRAIN_EVOLUTION", "0") or "0").strip().lower() in {"1", "true", "yes", "on"},
        skills_inject_session_start=(_env("MEMBRAIN_SKILLS_INJECT", "1") or "1").strip().lower() in {"1", "true", "yes", "on"},
        rerank_provider=(_env("MEMBRAIN_RERANK_PROVIDER", "heuristic") or "heuristic").strip().lower(),
        rerank_base_url=_env("MEMBRAIN_RERANK_BASE_URL"),
        rerank_api_key=_env("MEMBRAIN_RERANK_API_KEY"),
        rerank_model=_env("MEMBRAIN_RERANK_MODEL"),
        rerank_timeout=_env_float("MEMBRAIN_RERANK_TIMEOUT", 20.0),
        openviking_root=openviking_root,
        tz=(_env("MEMBRAIN_TZ") or "").strip(),
    )

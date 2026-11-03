"""D2.7 noise review: list (and optionally retract) artifact/nonsense archive entries.

Read-only by default. --apply hides candidates via the ledger (archive_retract);
files stay on disk and the list printed here is the recovery record.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("MEMBRAIN_HOME", str(REPO / "memory"))

from membrain.server import MemoryService  # noqa: E402
from membrain.session_import import find_noise_entries  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="retract candidates (ledgered, files kept)")
    parser.add_argument("--reason", default="d2.7 noise review 2026-11-03")
    args = parser.parse_args()
    svc = MemoryService()
    candidates = find_noise_entries(svc)
    print(json.dumps(candidates, ensure_ascii=False, indent=2))
    print(f"candidates: {len(candidates)}")
    if args.apply and candidates:
        for item in candidates:
            svc.archive.retract(item["id"], args.reason)
        svc.archive.sync(embed=False)
        print(f"retracted: {len(candidates)} (files kept; ids above are the recovery record)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

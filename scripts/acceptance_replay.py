#!/usr/bin/env python3
"""Replay postmortem fixture assertions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.postmortem.checker import Status, check_fixture


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", action="append", required=True, help="Fixture clip name or path")
    parser.add_argument("--fixtures-dir", default="tests/postmortem/fixtures")
    args = parser.parse_args()

    failed = False
    for clip in args.clip:
        root = Path(clip)
        if not root.exists():
            root = Path(args.fixtures_dir) / clip
        results = check_fixture(_load(root / "pre_p1.json"), _load(root / "post_p1.json"), _load(root / "expected.json"))
        print(f"\n{root}:")
        for result in results:
            print(f"  {result.status:19s} {result.stage:10s} {result.assertion}")
            failed = failed or result.status == Status.FAIL
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

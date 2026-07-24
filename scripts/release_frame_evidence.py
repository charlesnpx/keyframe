#!/usr/bin/env python3
"""Repository entry point for standalone release-frame evidence."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from keyframe.release_evidence import main


if __name__ == "__main__":
    raise SystemExit(main())

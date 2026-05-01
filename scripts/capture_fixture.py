#!/usr/bin/env python3
"""Placeholder fixture capture entrypoint.

Real clip capture needs the source media. Synthetic fixtures can be created by
writing the pre_p1/post_p1/expected JSON files directly under tests/postmortem/fixtures.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.parse_args()
    print("Fixture capture requires a source video and is intentionally not automatic yet.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

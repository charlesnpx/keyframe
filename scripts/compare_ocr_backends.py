#!/usr/bin/env python3
"""Placeholder OCR backend comparison entrypoint."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", nargs="?")
    parser.parse_args()
    print("OCR backend provenance is ignored by fixture checks; comparison is reserved for manual audits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

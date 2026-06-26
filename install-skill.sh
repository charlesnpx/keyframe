#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

choose_python() {
  local candidates=()
  if [[ -n "${PYTHON:-}" ]]; then
    candidates+=("$PYTHON")
  fi
  candidates+=(python3.13 python3.12 python3.11 python3)

  local candidate
  for candidate in "${candidates[@]}"; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    if "$candidate" - "$candidate" <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 11) <= sys.version_info < (3, 14) else 1)
PY
    then
      command -v "$candidate"
      return 0
    fi
  done

  return 1
}

PYTHON_BIN="$(choose_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  cat >&2 <<'EOF'
Error: keyframe requires Python >=3.11,<3.14.
Install Python 3.12 or set PYTHON=/path/to/python3.12, then rerun install-skill.sh.
EOF
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/keyframe/cli.py" install-skills "$@"

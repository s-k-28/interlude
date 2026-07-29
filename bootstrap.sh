#!/usr/bin/env bash
set -euo pipefail

# bootstrap.sh — one-time repair and setup for a fresh Codespace or local clone.
# Safe to run more than once.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> Step 1/9: removing superseded and junk files"
rm -f "app/storage/b2.py"
rm -f "app/storage/.probe.py"
rm -f "app/storage/.probe"

echo "==> Step 2/9: relocating pending API test into tests/"
mkdir -p "tests"
if [ -f "app/api/_test_api_pending_move.py" ]; then
  mv "app/api/_test_api_pending_move.py" "tests/test_api.py"
  echo "    moved app/api/_test_api_pending_move.py -> tests/test_api.py"
else
  echo "    nothing to move (already relocated or absent)"
fi

echo "==> Step 3/9: dropping the unused Sequence import from app/pipeline/runner.py"
if [ -f "app/pipeline/runner.py" ]; then
  sed -i.bak 's/from collections.abc import Callable, Sequence/from collections.abc import Callable/' "app/pipeline/runner.py" && rm -f "app/pipeline/runner.py.bak"
  echo "    runner.py imports normalized"
else
  echo "    app/pipeline/runner.py not present, skipping"
fi

echo "==> Step 4/9: ensuring package __init__.py files exist"
for pkg in "app" "app/api" "app/pipeline" "app/storage" "app/adapters" "tests"; do
  mkdir -p "$pkg"
  touch "$pkg/__init__.py"
  echo "    $pkg/__init__.py"
done

echo "==> Step 5/9: installing Python dependencies"
pip install -r "requirements.txt"

echo "==> Step 6/9: preparing .env"
if [ -f ".env" ]; then
  echo "    .env already exists, leaving it alone"
else
  cp ".env.example" ".env"
  echo "    created .env from .env.example — REMINDER: fill in your credentials in .env before running the server"
fi

echo "==> Step 7/9: running module self-checks"
SELFCHECK_PASS=0
SELFCHECK_FAIL=0
for mod in \
  "app.adapters.paritok_adapter" \
  "app.adapters.genblaze_adapter" \
  "app.adapters.gemini_adapter" \
  "app.api.schemas" \
  "app.api.jobs"
do
  echo "    self-check: $mod"
  if python -m "$mod"; then
    SELFCHECK_PASS=$((SELFCHECK_PASS + 1))
    echo "      PASS $mod"
  else
    SELFCHECK_FAIL=$((SELFCHECK_FAIL + 1))
    echo "      FAIL $mod"
  fi
done

echo "==> Step 8/9: running the test suite"
PYTEST_EXIT=0
pytest -q || PYTEST_EXIT=$?

echo "==> Step 9/9: summary"
echo "    self-checks passed: $SELFCHECK_PASS"
echo "    self-checks failed: $SELFCHECK_FAIL"
echo "    pytest exit code:   $PYTEST_EXIT"
echo ""
echo "    Next: fill in .env, then run"
echo "      uvicorn app.main:app --reload --port 8000"

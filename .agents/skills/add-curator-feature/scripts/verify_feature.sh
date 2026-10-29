#!/usr/bin/env bash
# Automated verification script for dataset-curator feature implementations
set -eo pipefail

REPO_DIR="/home/jmn/Devel/Python/list-dataset"
VENV_BIN="${REPO_DIR}/.venv/bin"

if [ ! -d "${VENV_BIN}" ]; then
  echo "Error: Virtual environment not found at ${VENV_BIN}" >&2
  exit 1
fi

cd "${REPO_DIR}"

echo "========================================="
echo "Step 1: Checking code style with Ruff"
echo "========================================="
"${VENV_BIN}/ruff" check src tests

echo ""
echo "========================================="
echo "Step 2: Checking formatting with Ruff"
echo "========================================="
"${VENV_BIN}/ruff" format --check src tests

echo ""
echo "========================================="
echo "Step 3: Executing Test Suite & Coverage"
echo "========================================="
"${VENV_BIN}/python" -m pytest tests/ --cov=src/dataset --cov-report=term-missing

echo ""
echo "========================================="
echo "All quality checks PASSED successfully!  "
echo "========================================="

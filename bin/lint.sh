#!/bin/bash

# Script to check code formatting and linting using ruff (without fixing)

set -e

uv run --frozen --only-dev ruff --version

echo "🔍 Running ruff check (without auto-fix)..."
uv run --frozen --only-dev ruff check llm_punctuator scripts tests

echo "🔍 Checking code formatting..."
uv run --frozen --only-dev ruff format --check llm_punctuator scripts tests

echo "✅ All linting checks passed!"

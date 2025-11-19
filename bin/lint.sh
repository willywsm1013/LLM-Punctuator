#!/bin/bash

# Script to check code formatting and linting using ruff (without fixing)

set -e

echo "🔍 Running ruff check (without auto-fix)..."
ruff check .

echo "🔍 Checking code formatting..."
ruff format --check .

echo "✅ All linting checks passed!"

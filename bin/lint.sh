#!/bin/bash

# Script to check code formatting and linting using ruff (without fixing)

set -e

echo "🔍 Running ruff check (without auto-fix)..."
ruff check llm_punctuator

echo "🔍 Checking code formatting..."
ruff format --check llm_punctuator

echo "✅ All linting checks passed!"

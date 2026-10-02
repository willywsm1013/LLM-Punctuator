#!/bin/bash

# Script to check and fix code formatting using ruff

set -e

echo "🔍 Running ruff check..."
ruff check llm_punctuator scripts tests --fix

echo "✨ Running ruff format..."
ruff format llm_punctuator scripts tests

echo "✅ Code formatting complete!"

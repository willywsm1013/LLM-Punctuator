#!/bin/bash

# Script to check and fix code formatting using ruff

set -e

echo "🔍 Running ruff check..."
ruff check llm_punctuator --fix

echo "✨ Running ruff format..."
ruff format llm_punctuator

echo "✅ Code formatting complete!"

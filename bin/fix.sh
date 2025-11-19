#!/bin/bash

# Script to check and fix code formatting using ruff

set -e

echo "🔍 Running ruff check..."
ruff check . --fix

echo "✨ Running ruff format..."
ruff format .

echo "✅ Code formatting complete!"

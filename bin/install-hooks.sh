#!/bin/bash

# Install git hooks from .githooks directory

echo "📦 Installing git hooks..."

# Set git hooks path to .githooks
git config core.hooksPath .githooks

echo "✅ Git hooks installed successfully!"
echo ""
echo "The following hooks are now active:"
echo "  - commit-msg: Validates Conventional Commits format"

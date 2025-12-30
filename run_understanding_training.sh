#!/bin/bash

# ALEN Understanding-Based Training Runner
# Builds, runs, and trains the AI with understanding (not memorization)

set -e

echo "======================================================================"
echo "ALEN UNDERSTANDING-BASED TRAINING"
echo "======================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 is available"

# Check if requests library is available
if ! python3 -c "import requests" 2>/dev/null; then
    echo "Installing requests library..."
    pip3 install requests --quiet
fi

echo "✓ Python requests library is available"
echo ""

# Check if server is already running
if curl -s http://localhost:3000/health > /dev/null 2>&1; then
    echo "✓ Server is already running"
    echo ""
else
    echo "Server is not running. Please start it in another terminal:"
    echo "  cargo run --release"
    echo ""
    echo "Or run this script with --start-server flag"
    exit 1
fi

# Run training
echo "Starting training..."
echo ""
python3 train_understanding.py

echo ""
echo "======================================================================"
echo "TRAINING COMPLETE"
echo "======================================================================"
echo ""
echo "The AI has been trained with UNDERSTANDING, not MEMORIZATION."
echo ""
echo "Test it with:"
echo "  curl -X POST http://localhost:3000/chat \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"message\": \"What is 7 plus 8?\"}'"
echo ""

#!/bin/bash

# Build and train ALEN with understanding-based learning
# This script builds the project, starts the server, and trains from all data files

set -e

echo "======================================================================"
echo "ALEN BUILD AND TRAIN"
echo "======================================================================"
echo ""

# Build the project
echo "Building ALEN..."
cargo build --release --example train_from_files 2>&1 | grep -E "Compiling|Finished|error|warning" || true

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "✗ Build failed!"
    exit 1
fi

echo "✓ Build successful"
echo ""

# Run the training
echo "======================================================================"
echo "TRAINING FROM FILES"
echo "======================================================================"
echo ""

./target/release/examples/train_from_files

echo ""
echo "======================================================================"
echo "TRAINING COMPLETE"
echo "======================================================================"
echo ""
echo "To start the server and test:"
echo "  cargo run --release"
echo ""
echo "Then in another terminal:"
echo "  curl -X POST http://localhost:3000/chat \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"message\": \"What is 7 plus 8?\"}'"
echo ""

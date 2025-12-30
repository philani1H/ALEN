#!/bin/bash

echo "=========================================="
echo "ALEN Complete System Training"
echo "=========================================="
echo ""

# Train on all data
echo "Training on story understanding..."
cargo run --release -- train \
    --data training_data/story_understanding.txt \
    --epochs 5 \
    --batch-size 8 \
    --learning-rate 0.001

echo ""
echo "Training on reasoning patterns..."
cargo run --release -- train \
    --data training_data/reasoning_patterns.txt \
    --epochs 5 \
    --batch-size 8 \
    --learning-rate 0.001

echo ""
echo "Training on enhanced conversations..."
cargo run --release -- train \
    --data training_data/enhanced_conversations.txt \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 0.001

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="

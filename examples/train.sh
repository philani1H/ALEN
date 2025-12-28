#!/bin/bash
# Example: Training the Deliberative AI on various problems
# Run: ./examples/train.sh

BASE_URL="${BASE_URL:-http://localhost:3000}"

echo "=== Deliberative AI Training Examples ==="
echo ""

# Example 1: Simple math
echo "1. Training on math problem..."
curl -s -X POST "$BASE_URL/train" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is 7 times 8?",
    "expected_answer": "56",
    "constraints": ["mathematical", "multiplication"]
  }' | jq .

echo ""

# Example 2: Factual knowledge
echo "2. Training on factual knowledge..."
curl -s -X POST "$BASE_URL/train" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is the chemical symbol for gold?",
    "expected_answer": "Au",
    "constraints": ["chemistry", "element"],
    "context": ["periodic table", "metals"]
  }' | jq .

echo ""

# Example 3: Logic problem
echo "3. Training on logic problem..."
curl -s -X POST "$BASE_URL/train" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "If all cats are mammals and all mammals breathe, do cats breathe?",
    "expected_answer": "yes",
    "constraints": ["logical", "deduction"]
  }' | jq .

echo ""

# Example 4: Batch training
echo "4. Batch training on multiple problems..."
curl -s -X POST "$BASE_URL/train/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "problems": [
      {
        "input": "What is 5 + 5?",
        "expected_answer": "10"
      },
      {
        "input": "What is the square root of 16?",
        "expected_answer": "4"
      },
      {
        "input": "How many sides does a hexagon have?",
        "expected_answer": "6"
      }
    ]
  }' | jq .

echo ""
echo "=== Training Complete ==="

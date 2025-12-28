#!/bin/bash
# Example: Performing inference with the Deliberative AI
# Run: ./examples/infer.sh

BASE_URL="${BASE_URL:-http://localhost:3000}"

echo "=== Deliberative AI Inference Examples ==="
echo ""

# Example 1: Simple question
echo "1. Simple question inference..."
curl -s -X POST "$BASE_URL/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is the meaning of democracy?"
  }' | jq .

echo ""

# Example 2: With context
echo "2. Inference with context..."
curl -s -X POST "$BASE_URL/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What are the best practices for error handling?",
    "context": ["programming", "Rust", "software engineering"]
  }' | jq .

echo ""

# Example 3: With constraints
echo "3. Inference with constraints..."
curl -s -X POST "$BASE_URL/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Design a data structure for a cache",
    "constraints": ["memory-efficient", "O(1) lookup", "LRU eviction"]
  }' | jq .

echo ""

# Example 4: Complex reasoning
echo "4. Complex reasoning task..."
curl -s -X POST "$BASE_URL/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Compare the trade-offs between consistency and availability in distributed systems",
    "context": ["CAP theorem", "distributed databases"],
    "constraints": ["technical", "balanced analysis"]
  }' | jq .

echo ""
echo "=== Inference Complete ==="

#!/bin/bash
# Example: Managing knowledge in the Deliberative AI
# Run: ./examples/knowledge.sh

BASE_URL="${BASE_URL:-http://localhost:3000}"

echo "=== Deliberative AI Knowledge Management ==="
echo ""

# Add semantic facts
echo "1. Adding semantic facts..."

curl -s -X POST "$BASE_URL/facts" \
  -H "Content-Type: application/json" \
  -d '{
    "concept": "Rust",
    "content": "Rust is a systems programming language focused on safety, speed, and concurrency",
    "category": "programming"
  }' | jq .

curl -s -X POST "$BASE_URL/facts" \
  -H "Content-Type: application/json" \
  -d '{
    "concept": "ownership",
    "content": "Rust ownership system ensures memory safety without garbage collection",
    "category": "programming"
  }' | jq .

curl -s -X POST "$BASE_URL/facts" \
  -H "Content-Type: application/json" \
  -d '{
    "concept": "borrowing",
    "content": "Borrowing allows references to data without taking ownership",
    "category": "programming"
  }' | jq .

echo ""

# Search facts
echo "2. Searching for related facts..."
curl -s -X POST "$BASE_URL/facts/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "memory safety programming",
    "limit": 5
  }' | jq .

echo ""

# Get episodic memory stats
echo "3. Episodic memory statistics..."
curl -s "$BASE_URL/memory/episodic/stats" | jq .

echo ""

# Get top episodes
echo "4. Top verified episodes..."
curl -s "$BASE_URL/memory/episodic/top/5" | jq .

echo ""
echo "=== Knowledge Management Complete ==="

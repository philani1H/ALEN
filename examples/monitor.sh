#!/bin/bash
# Example: Monitoring and controlling the Deliberative AI
# Run: ./examples/monitor.sh

BASE_URL="${BASE_URL:-http://localhost:3000}"

echo "=== Deliberative AI Monitoring & Control ==="
echo ""

# Health check
echo "1. Health check..."
curl -s "$BASE_URL/health" | jq .

echo ""

# System statistics
echo "2. System statistics..."
curl -s "$BASE_URL/stats" | jq .

echo ""

# Operator performance
echo "3. Operator performance..."
curl -s "$BASE_URL/operators" | jq .

echo ""

# Set bias for creative exploration
echo "4. Setting bias for creative exploration..."
curl -s -X POST "$BASE_URL/bias" \
  -H "Content-Type: application/json" \
  -d '{
    "risk_tolerance": 0.8,
    "exploration": 0.9,
    "creativity": 0.85,
    "urgency": 0.3
  }' | jq .

echo ""

# Do some inference with creative bias
echo "5. Inference with creative bias..."
curl -s -X POST "$BASE_URL/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What novel approaches could solve climate change?"
  }' | jq .

echo ""

# Reset bias to defaults
echo "6. Resetting bias to defaults..."
curl -s -X POST "$BASE_URL/bias/reset" | jq .

echo ""

# Set bias for analytical precision
echo "7. Setting bias for analytical precision..."
curl -s -X POST "$BASE_URL/bias" \
  -H "Content-Type: application/json" \
  -d '{
    "risk_tolerance": 0.2,
    "exploration": 0.3,
    "creativity": 0.2,
    "urgency": 0.1
  }' | jq .

echo ""

# Do inference with analytical bias
echo "8. Inference with analytical bias..."
curl -s -X POST "$BASE_URL/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Calculate the time complexity of quicksort"
  }' | jq .

echo ""

# Reset learning rate
echo "9. Resetting learning rate..."
curl -s -X POST "$BASE_URL/learning/reset" | jq .

echo ""
echo "=== Monitoring Complete ==="

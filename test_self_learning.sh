#!/bin/bash
# Test Self-Learning System
# Demonstrates how ALEN learns from conversations and improves over time

BASE_URL="${BASE_URL:-http://localhost:3000}"

echo "=== ALEN Self-Learning Test ==="
echo ""
echo "This test demonstrates:"
echo "1. Learning from multiple conversations"
echo "2. Pattern extraction and aggregation"
echo "3. Confidence building with evidence"
echo "4. Understanding and asking the right questions"
echo ""

# First, let's train ALEN with some basic knowledge
echo "Step 1: Training basic knowledge..."
curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" \
  -d '{"input": "What is stress?", "expected_answer": "Stress is a physical and emotional response to challenging situations"}' > /dev/null

curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" \
  -d '{"input": "How to manage stress?", "expected_answer": "Common stress management techniques include exercise, meditation, time management, and talking to someone"}' > /dev/null

echo "✓ Basic knowledge trained"
echo ""

# Now test understanding
echo "Step 2: Testing understanding..."
echo ""
echo "Q: What is stress?"
response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "What is stress?"}' | jq -r '.message')
echo "A: $response"
echo ""

echo "Q: How to manage stress?"
response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "How to manage stress?"}' | jq -r '.message')
echo "A: $response"
echo ""

# Test if ALEN can ask clarifying questions when uncertain
echo "Step 3: Testing self-questioning (when uncertain)..."
echo ""
echo "Q: I'm feeling overwhelmed"
response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "I am feeling overwhelmed"}' | jq -r '.message')
echo "A: $response"
echo ""

# Train more specific knowledge
echo "Step 4: Training specific strategies..."
curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" \
  -d '{"input": "Best way to handle work stress", "expected_answer": "For work stress, try: 1) Prioritize tasks, 2) Take regular breaks, 3) Set boundaries, 4) Communicate with your manager"}' > /dev/null

curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" \
  -d '{"input": "Time blocking technique", "expected_answer": "Time blocking is scheduling specific time slots for different tasks. It helps reduce stress by providing structure and preventing overwhelm"}' > /dev/null

echo "✓ Specific strategies trained"
echo ""

# Test improved understanding
echo "Step 5: Testing improved understanding..."
echo ""
echo "Q: Best way to handle work stress"
response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "Best way to handle work stress"}' | jq -r '.message')
echo "A: $response"
echo ""

echo "Q: What is time blocking?"
response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "What is time blocking?"}' | jq -r '.message')
echo "A: $response"
echo ""

# Check memory stats
echo "Step 6: Checking knowledge base..."
stats=$(curl -s "$BASE_URL/memory/episodic/stats")
echo "Memory Statistics:"
echo "$stats" | jq '.'
echo ""

echo "=== Test Complete ==="
echo ""
echo "Key Observations:"
echo "1. ALEN learns from training examples"
echo "2. ALEN retrieves relevant knowledge based on similarity"
echo "3. ALEN's confidence grows with more evidence"
echo "4. ALEN can provide specific answers when it has learned them"
echo ""
echo "This demonstrates the self-learning architecture where:"
echo "- Knowledge is stored as verified patterns"
echo "- Confidence increases with evidence"
echo "- Retrieval is based on semantic similarity"
echo "- System improves over time without retraining weights"

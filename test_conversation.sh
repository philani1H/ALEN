#!/bin/bash
# Test Natural Conversation with ALEN
# This tests how ALEN behaves in a real conversation

BASE_URL="${BASE_URL:-http://localhost:3000}"

echo "=== ALEN Conversation Test ==="
echo ""
echo "First, let's teach ALEN some knowledge..."
echo ""

# Train conversational knowledge
echo "Training conversational responses..."
curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" \
  -d '{"input": "Hi", "expected_answer": "Hello! How can I help you today?"}' > /dev/null

curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" \
  -d '{"input": "How are you?", "expected_answer": "I am functioning well, thank you for asking! How are you doing?"}' > /dev/null

curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" \
  -d '{"input": "What can you do?", "expected_answer": "I can help you with questions, provide information, and have conversations. I learn from our interactions and improve over time. What would you like to know?"}' > /dev/null

# Train some domain knowledge
curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" \
  -d '{"input": "What is machine learning?", "expected_answer": "Machine learning is a type of artificial intelligence where computers learn from data and improve their performance without being explicitly programmed for every task."}' > /dev/null

curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" \
  -d '{"input": "How does AI learn?", "expected_answer": "AI learns by finding patterns in data. It adjusts its internal parameters based on examples, similar to how humans learn from experience."}' > /dev/null

curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" \
  -d '{"input": "What is the difference between AI and ML?", "expected_answer": "AI is the broader concept of machines being able to carry out tasks intelligently. ML is a subset of AI that focuses on machines learning from data."}' > /dev/null

# Train problem-solving knowledge
curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" \
  -d '{"input": "I am stressed about work", "expected_answer": "I understand work stress can be challenging. Some strategies that help: taking breaks, prioritizing tasks, and talking to someone. What specifically is causing you stress?"}' > /dev/null

curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" \
  -d '{"input": "How to manage time better?", "expected_answer": "Good time management involves: 1) Prioritizing important tasks, 2) Breaking large tasks into smaller ones, 3) Using time blocking, 4) Minimizing distractions. Which area would you like to focus on?"}' > /dev/null

echo "✓ Training complete"
echo ""
echo "=== Starting Conversation ==="
echo ""

# Conversation 1: Greeting
echo "You: Hi"
response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "Hi"}' | jq -r '.message')
echo "ALEN: $response"
echo ""

# Conversation 2: Follow-up
echo "You: How are you?"
response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "How are you?"}' | jq -r '.message')
echo "ALEN: $response"
echo ""

# Conversation 3: Capability question
echo "You: What can you do?"
response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "What can you do?"}' | jq -r '.message')
echo "ALEN: $response"
echo ""

# Conversation 4: Technical question
echo "You: What is machine learning?"
response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "What is machine learning?"}' | jq -r '.message')
echo "ALEN: $response"
echo ""

# Conversation 5: Related question
echo "You: How does AI learn?"
response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "How does AI learn?"}' | jq -r '.message')
echo "ALEN: $response"
echo ""

# Conversation 6: Comparison question
echo "You: What is the difference between AI and ML?"
response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "What is the difference between AI and ML?"}' | jq -r '.message')
echo "ALEN: $response"
echo ""

# Conversation 7: Personal problem
echo "You: I am stressed about work"
response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "I am stressed about work"}' | jq -r '.message')
echo "ALEN: $response"
echo ""

# Conversation 8: Practical advice
echo "You: How to manage time better?"
response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "How to manage time better?"}' | jq -r '.message')
echo "ALEN: $response"
echo ""

# Conversation 9: Something not trained
echo "You: Tell me about quantum computing"
response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "Tell me about quantum computing"}' | jq -r '.message')
echo "ALEN: $response"
echo ""

# Check stats
echo "=== Conversation Statistics ==="
stats=$(curl -s "$BASE_URL/memory/episodic/stats")
echo "$stats" | jq '.'
echo ""

echo "=== Analysis ==="
echo ""
echo "Observations:"
echo "1. Does ALEN respond naturally to greetings?"
echo "2. Does ALEN maintain context across questions?"
echo "3. Does ALEN provide accurate information when trained?"
echo "4. Does ALEN ask clarifying questions when appropriate?"
echo "5. How does ALEN handle questions it hasn't been trained on?"
echo ""
echo "Key Behaviors to Note:"
echo "- Confidence in trained topics"
echo "- Uncertainty in untrained topics"
echo "- Natural conversation flow"
echo "- Ability to ask follow-up questions"

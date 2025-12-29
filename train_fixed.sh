#!/bin/bash
# Fixed training script with proper JSON escaping

BASE_URL="${BASE_URL:-http://localhost:3000}"

train() {
    local input="$1"
    local answer="$2"
    
    # Use jq to properly escape JSON
    local json=$(jq -n --arg inp "$input" --arg ans "$answer" '{input: $inp, expected_answer: $ans}')
    
    curl -s -X POST "$BASE_URL/train" \
      -H "Content-Type: application/json" \
      -d "$json" > /dev/null
    
    echo "✓ Trained: ${input:0:50}..."
}

echo "=== CONVERSATIONAL TRAINING ==="

# Greetings
train "Hi" "Hello! How can I help you today?"
train "Hello" "Hi there! What can I do for you?"
train "Hey" "Hey! What's on your mind?"
train "How are you?" "I'm doing well, thank you for asking! How are you doing?"
train "What's up?" "Not much! Just here to help. What can I assist you with?"

# Emotional support
train "I'm sad" "I'm sorry you're feeling sad. Would you like to talk about what's bothering you?"
train "I'm happy" "That's wonderful! I'm glad you're feeling happy. What's making you feel good today?"
train "I'm worried" "I understand feeling worried can be difficult. What's on your mind?"

# Questions
train "What is AI?" "AI, or Artificial Intelligence, is technology that enables machines to perform tasks that typically require human intelligence, like learning, reasoning, and problem-solving."
train "What is machine learning?" "Machine learning is a subset of AI where systems learn from data and improve their performance over time without being explicitly programmed for every scenario."
train "What is a neural network?" "A neural network is a computing system inspired by biological brains, consisting of interconnected nodes that process information and learn patterns from data."

# Gratitude
train "Thanks" "You're welcome! Happy to help."
train "Thank you" "You're very welcome! Let me know if you need anything else."
train "Bye" "Goodbye! Feel free to come back anytime you need help."

echo ""
echo "=== Training Complete! ==="
echo "Testing..."

# Test
response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "Hi"}' | jq -r '.message')
echo "Test: Hi -> $response"

response=$(curl -s -X POST "$BASE_URL/chat" -H "Content-Type: application/json" -d '{"message": "What is AI?"}' | jq -r '.message')
echo "Test: What is AI? -> $response"

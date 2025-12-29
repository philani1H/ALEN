#!/bin/bash
# Test script for integrated ALEN system with all three fixes
# Tests: Space separation, adaptive thresholds, episodic integration

set -e

API_URL="http://localhost:3000"
TRAIN_URL="$API_URL/train"
CHAT_URL="$API_URL/chat"

echo "========================================="
echo "ALEN INTEGRATED SYSTEM TEST"
echo "Testing all three engineering fixes:"
echo "1. Space Separation (input embeddings)"
echo "2. Adaptive Thresholds (domain-specific)"
echo "3. Episodic Integration (confidence boost)"
echo "========================================="
echo ""

# Function to train
train() {
    local input="$1"
    local target="$2"
    echo "Training: '$input' -> '$target'"
    curl -s -X POST "$TRAIN_URL" \
        -H "Content-Type: application/json" \
        -d "{\"input\": \"$input\", \"expected_answer\": \"$target\", \"dimension\": 128}" \
        | jq -r '.success, .confidence, .iterations' | paste -sd ' ' - | awk '{print "  Success: " $1 ", Confidence: " $2 ", Iterations: " $3}'
    echo ""
}

# Function to chat
chat() {
    local message="$1"
    echo "Chat: '$message'"
    curl -s -X POST "$CHAT_URL" \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"$message\"}" \
        | jq -r '.message, .confidence, .operator_used' | paste -sd '|' - | awk -F'|' '{print "  Response: " $1 "\n  Confidence: " $2 "\n  Operator: " $3}'
    echo ""
}

echo "========================================="
echo "PHASE 1: TRAINING MATH EXAMPLES"
echo "========================================="
echo ""

train "What is 2+2?" "4"
train "What is 3+3?" "6"
train "What is 5+5?" "10"
train "What is 10+10?" "20"
train "Calculate 7+8" "15"

echo "========================================="
echo "PHASE 2: TRAINING CONVERSATION EXAMPLES"
echo "========================================="
echo ""

train "Hello, how are you?" "I'm doing well, thank you for asking! How can I help you today?"
train "What's your name?" "I'm ALEN, an AI that learns through reasoning and verification."
train "Tell me about yourself" "I'm ALEN, an adaptive learning system that uses mathematical reasoning to understand and respond to questions."

echo "========================================="
echo "PHASE 3: TRAINING LOGIC EXAMPLES"
echo "========================================="
echo ""

train "If it rains, the ground gets wet. It rained. What happened?" "The ground got wet."
train "All humans are mortal. Socrates is human. What can we conclude?" "Socrates is mortal."

echo "========================================="
echo "PHASE 4: TESTING SIMILAR MATH QUESTIONS"
echo "Testing Fix #1: Input embedding similarity"
echo "========================================="
echo ""

chat "What is 4+4?"
chat "Calculate 6+6"
chat "What is 9+9?"

echo "========================================="
echo "PHASE 5: TESTING CONVERSATION"
echo "Testing Fix #2: Adaptive thresholds (conversation domain)"
echo "========================================="
echo ""

chat "Hi there!"
chat "Who are you?"
chat "Can you introduce yourself?"

echo "========================================="
echo "PHASE 6: TESTING LOGIC"
echo "Testing Fix #2: Adaptive thresholds (logic domain)"
echo "========================================="
echo ""

chat "If all cats are animals, and Fluffy is a cat, what is Fluffy?"

echo "========================================="
echo "PHASE 7: TESTING UNKNOWN QUESTIONS"
echo "Testing Fix #2: Threshold-based refusal"
echo "========================================="
echo ""

chat "What is the meaning of life?"
chat "Explain quantum mechanics"

echo "========================================="
echo "PHASE 8: TESTING EPISODIC BOOST"
echo "Testing Fix #3: Confidence boost from similar episodes"
echo "========================================="
echo ""

chat "What is 2+2?"  # Should have HIGH confidence (exact match)
chat "What is 3+3?"  # Should have HIGH confidence (exact match)
chat "What is 8+8?"  # Should have MEDIUM confidence (similar pattern)

echo "========================================="
echo "TEST COMPLETE"
echo "========================================="
echo ""
echo "Summary:"
echo "- Trained 10 examples across 3 domains"
echo "- Tested similarity-based retrieval"
echo "- Tested domain-specific thresholds"
echo "- Tested episodic confidence boost"
echo "- Tested refusal on unknown questions"
echo ""
echo "Check the responses above to verify:"
echo "1. Similar questions get similar answers (Fix #1)"
echo "2. Different domains have different confidence thresholds (Fix #2)"
echo "3. Confidence is higher for questions similar to training data (Fix #3)"
echo "4. System refuses to answer when confidence is too low (Fix #2)"

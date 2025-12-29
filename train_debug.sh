#!/bin/bash
# Debug training script

BASE_URL="${BASE_URL:-http://localhost:3000}"

train() {
    local input="$1"
    local answer="$2"
    
    # Use jq to properly escape JSON
    local json=$(jq -n --arg inp "$input" --arg ans "$answer" '{input: $inp, expected_answer: $ans}')
    
    echo "DEBUG: Training with JSON: $json"
    
    curl -s -X POST "$BASE_URL/train" \
      -H "Content-Type: application/json" \
      -d "$json"
    
    echo ""
}

# Clear memory
curl -s -X DELETE "$BASE_URL/memory/episodic/clear"
echo ""

# Train one example
train "Hi" "Hello! How can I help you today?"

# Check what was stored
echo "Checking memory:"
curl -s "$BASE_URL/memory/episodic/top/1" | jq '.[0] | {input: .problem_input, output: .answer_output}'

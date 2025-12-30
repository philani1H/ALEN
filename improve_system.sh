#!/bin/bash
# System Improvement Script
# Adds more training data and tests improvements

BASE_URL="${BASE_URL:-http://localhost:3000}"

echo "================================================================="
echo "  ALEN System Improvement"
echo "================================================================="
echo ""

# Check server
if ! curl -s "$BASE_URL/health" > /dev/null 2>&1; then
    echo "❌ Server not running. Please start: cargo run --release"
    exit 1
fi

echo "✓ Server is healthy"
echo ""

# Clear old memory
echo "Step 1: Clearing old episodic memory..."
curl -s -X DELETE "$BASE_URL/memory/episodic/clear" > /dev/null
echo "✓ Memory cleared"
echo ""

# Train with new advanced data
echo "Step 2: Training with advanced Q&A data..."
echo ""

count=0
success=0

while IFS= read -r line; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    
    # Look for -> pattern
    if [[ "$line" =~ "->" ]]; then
        input="${line%% ->*}"
        output="${line#*-> }"
        
        # Trim whitespace
        input=$(echo "$input" | xargs)
        output=$(echo "$output" | xargs)
        
        ((count++))
        
        # Train
        result=$(curl -s -X POST "$BASE_URL/train" \
          -H "Content-Type: application/json" \
          -d "{\"input\": $(echo "$input" | jq -Rs .), \"expected_answer\": $(echo "$output" | jq -Rs .)}" | \
          jq -r '.success')
        
        if [ "$result" == "true" ]; then
            ((success++))
            echo -n "✓"
        else
            echo -n "✗"
        fi
        
        # Progress indicator
        if [ $((count % 10)) -eq 0 ]; then
            echo " [$count trained]"
        fi
        
        sleep 0.05
    fi
done < training_data/advanced_qa.txt

echo ""
echo "✓ Training complete: $success/$count successful"
echo ""

# Test improvements
echo "Step 3: Testing improvements..."
echo ""

test_questions=(
    "Hello"
    "What is 2+2?"
    "What is the capital of France?"
    "What is Python?"
    "What color is the sky?"
    "How many hours in a day?"
    "Thank you"
)

correct=0
total=${#test_questions[@]}

for question in "${test_questions[@]}"; do
    echo "Q: $question"
    response=$(curl -s -X POST "$BASE_URL/chat" \
      -H "Content-Type: application/json" \
      -d "{\"message\": \"$question\"}" | jq -r '.message')
    
    echo "A: ${response:0:80}"
    
    # Check if response is not a refusal
    if [[ ! "$response" =~ "don't have enough confidence" ]] && \
       [[ ! "$response" =~ "Unable to find" ]]; then
        ((correct++))
        echo "✓ Answered"
    else
        echo "✗ No answer"
    fi
    echo ""
done

# Summary
echo "================================================================="
echo "  Improvement Summary"
echo "================================================================="
echo ""
echo "Training:"
echo "  Total pairs: $count"
echo "  Successful: $success"
echo "  Success rate: $((success * 100 / count))%"
echo ""
echo "Testing:"
echo "  Questions: $total"
echo "  Answered: $correct"
echo "  Answer rate: $((correct * 100 / total))%"
echo ""

# Get final stats
stats=$(curl -s "$BASE_URL/stats")
episodes=$(echo "$stats" | jq -r '.episodic_memory.total_episodes')
avg_conf=$(echo "$stats" | jq -r '.episodic_memory.average_confidence')

echo "System Stats:"
echo "  Episodes: $episodes"
echo "  Avg confidence: $(echo "$avg_conf * 100" | bc -l | cut -c1-5)%"
echo ""

if [ $correct -ge $((total * 7 / 10)) ]; then
    echo "Status: ✅ System improved! Answering $((correct * 100 / total))% of questions"
else
    echo "Status: ⚠️  System needs more training"
fi
echo ""

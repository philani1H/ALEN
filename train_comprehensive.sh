#!/bin/bash

# Comprehensive Training Script
# Trains the model with all new training data

API_URL="http://localhost:3000"

echo "============================================================"
echo "🧠 ALEN Comprehensive Training"
echo "============================================================"

# Check if server is running
echo ""
echo "🔍 Checking server..."
if ! curl -s "${API_URL}/health" > /dev/null 2>&1; then
    echo "❌ Server not running! Please start the server first:"
    echo "   cargo run --release"
    exit 1
fi
echo "✅ Server is running"

# Function to train from a file
train_file() {
    local file=$1
    local filename=$(basename "$file")
    
    echo ""
    echo "📖 Training from: $filename"
    
    # Parse Q&A pairs and send to API
    local count=0
    local in_question=false
    local question=""
    local answer=""
    
    while IFS= read -r line; do
        # Skip comments and empty lines
        if [[ "$line" =~ ^#.*$ ]] || [[ -z "$line" ]]; then
            continue
        fi
        
        # Check for Q: or Question:
        if [[ "$line" =~ ^Q:\ (.*)$ ]] || [[ "$line" =~ ^Question:\ (.*)$ ]]; then
            question="${BASH_REMATCH[1]}"
            in_question=true
        # Check for A: or Answer:
        elif [[ "$line" =~ ^A:\ (.*)$ ]] || [[ "$line" =~ ^Answer:\ (.*)$ ]]; then
            answer="${BASH_REMATCH[1]}"
            
            if [ -n "$question" ] && [ -n "$answer" ]; then
                # Escape quotes for JSON
                question_escaped=$(echo "$question" | sed 's/"/\\"/g')
                answer_escaped=$(echo "$answer" | sed 's/"/\\"/g')
                
                # Send training request
                curl -s -X POST "${API_URL}/train" \
                    -H "Content-Type: application/json" \
                    -d "{\"input\": \"$question_escaped\", \"expected_answer\": \"$answer_escaped\"}" \
                    > /dev/null 2>&1
                
                ((count++))
                
                # Show progress every 10 examples
                if [ $((count % 10)) -eq 0 ]; then
                    echo "   ✓ Trained $count examples..."
                fi
            fi
            
            question=""
            answer=""
            in_question=false
        fi
    done < "$file"
    
    echo "   ✅ Completed: $count examples from $filename"
    return $count
}

# Train from priority files
echo ""
echo "🚀 Starting training..."

total_trained=0

# Priority files
priority_files=(
    "training_data/neural_question_generation.txt"
    "training_data/neural_followup_generation.txt"
    "training_data/neural_state_expression.txt"
    "training_data/comprehensive_all_patterns.txt"
    "training_data/self_questioning_help.txt"
)

for file in "${priority_files[@]}"; do
    if [ -f "$file" ]; then
        train_file "$file"
        total_trained=$((total_trained + $?))
    fi
done

# Train from other files
for file in training_data/*.txt; do
    # Skip if already trained
    skip=false
    for priority in "${priority_files[@]}"; do
        if [ "$file" = "$priority" ]; then
            skip=true
            break
        fi
    done
    
    if [ "$skip" = false ] && [ -f "$file" ]; then
        train_file "$file"
        total_trained=$((total_trained + $?))
    fi
done

echo ""
echo "============================================================"
echo "✅ Training complete! Trained on $total_trained examples"
echo "============================================================"

# Get stats
echo ""
echo "📊 System Statistics:"
curl -s "${API_URL}/stats" | grep -o '"episodic_memory_size":[0-9]*' | cut -d':' -f2 | xargs -I {} echo "   Episodes in memory: {}"
curl -s "${API_URL}/stats" | grep -o '"semantic_memory_size":[0-9]*' | cut -d':' -f2 | xargs -I {} echo "   Facts in memory: {}"

echo ""
echo "🎉 Ready to chat! Try:"
echo "   curl -X POST ${API_URL}/infer -H 'Content-Type: application/json' -d '{\"input\": \"What is 2+2?\"}'"

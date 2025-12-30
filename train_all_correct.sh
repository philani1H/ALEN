#!/bin/bash
# Train ALEN with ALL training data files (correct format)

BASE_URL="${BASE_URL:-http://localhost:3000}"

echo "======================================================================"
echo "  ALEN - Training with ALL Data Files"
echo "======================================================================"
echo ""

# Function to train a Q&A pair
train_pair() {
    local question="$1"
    local answer="$2"
    
    curl -s -X POST "$BASE_URL/train" \
      -H "Content-Type: application/json" \
      -d "{\"input\": $(echo "$question" | jq -Rs .), \"expected_answer\": $(echo "$answer" | jq -Rs .)}" | \
      jq -r 'if .success then "✓" else "✗" end' | tr -d '\n'
}

# Function to process a training file
process_file() {
    local file="$1"
    local filename=$(basename "$file")
    
    echo "======================================================================"
    echo "Processing: $filename"
    echo "======================================================================"
    
    local count=0
    local success=0
    
    # Read file line by line
    while IFS= read -r line; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        
        # Look for -> pattern (input -> output)
        if [[ "$line" =~ "->" ]]; then
            input="${line%% ->*}"
            output="${line#*-> }"
            
            # Trim whitespace
            input=$(echo "$input" | xargs)
            output=$(echo "$output" | xargs)
            
            ((count++))
            echo -n "[$count] ${input:0:40}... "
            result=$(train_pair "$input" "$output")
            echo " $result"
            [[ "$result" == "✓" ]] && ((success++))
            
            sleep 0.05  # Small delay
        fi
        
    done < "$file"
    
    if [ $count -gt 0 ]; then
        local rate=$((success * 100 / count))
        echo "✓ Completed: $success/$count pairs ($rate% success)"
    else
        echo "⚠️  No training pairs found"
    fi
    echo ""
}

# Check server health
echo "Checking server health..."
if ! curl -s "$BASE_URL/health" > /dev/null 2>&1; then
    echo "❌ Server not responding at $BASE_URL"
    echo "Please start the server first: cargo run --release"
    exit 1
fi
echo "✓ Server is healthy"
echo ""

# Find and process all training files
training_files=$(find training_data -name "*.txt" 2>/dev/null | sort)

if [ -z "$training_files" ]; then
    echo "❌ No training files found in training_data/"
    exit 1
fi

file_count=$(echo "$training_files" | wc -l)
echo "Found $file_count training files"
echo ""

# Process each file
for file in $training_files; do
    process_file "$file"
done

# Final statistics
echo "======================================================================"
echo "  TRAINING COMPLETE"
echo "======================================================================"

# Get system stats
echo ""
echo "Fetching system statistics..."
stats=$(curl -s "$BASE_URL/stats")

if [ $? -eq 0 ]; then
    echo ""
    echo "System Statistics:"
    echo "$stats" | jq -r '
        "  Episodic Memory:",
        "    Total episodes: \(.episodic_memory.total_episodes)",
        "    Verified: \(.episodic_memory.verified_episodes)",
        "    Avg confidence: \((.episodic_memory.average_confidence * 100) | floor)%",
        "",
        "  Semantic Memory:",
        "    Total facts: \(.semantic_memory.total_facts)",
        "",
        "  Control State:",
        "    Confidence: \((.control_state.confidence * 100) | floor)%",
        "    Learning rate: \(.learning_rate)",
        "",
        "  Top Operators:",
        (.operator_stats | sort_by(-.usage_count) | .[:5] | .[] | "    • \(.operator_type): \(.usage_count) uses, \((.success_rate * 100) | floor)% success")
    '
fi

echo ""
echo "✓ Training complete! Test the system:"
echo "  curl -X POST $BASE_URL/chat \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"message\": \"Hello! How are you?\"}' | jq ."
echo ""

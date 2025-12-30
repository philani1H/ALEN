#!/bin/bash
# Train ALEN with ALL training data files

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
      -d "{\"input\": \"$question\", \"expected_answer\": \"$answer\"}" | \
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
        
        # Look for Q: or Question: patterns
        if [[ "$line" =~ ^Q:|^Question: ]]; then
            question="${line#*:}"
            question="${question# }"  # Trim leading space
            
            # Read next line for answer
            read -r next_line
            if [[ "$next_line" =~ ^A:|^Answer: ]]; then
                answer="${next_line#*:}"
                answer="${answer# }"  # Trim leading space
                
                ((count++))
                echo -n "[$count] Training: ${question:0:50}... "
                result=$(train_pair "$question" "$answer")
                echo "$result"
                [[ "$result" == "✓" ]] && ((success++))
                
                sleep 0.1  # Small delay
            fi
        fi
        
        # Look for Input:/Output: patterns
        if [[ "$line" =~ ^Input: ]]; then
            input="${line#*:}"
            input="${input# }"
            
            read -r next_line
            if [[ "$next_line" =~ ^Output: ]]; then
                output="${next_line#*:}"
                output="${output# }"
                
                ((count++))
                echo -n "[$count] Training: ${input:0:50}... "
                result=$(train_pair "$input" "$output")
                echo "$result"
                [[ "$result" == "✓" ]] && ((success++))
                
                sleep 0.1
            fi
        fi
        
    done < "$file"
    
    if [ $count -gt 0 ]; then
        local rate=$((success * 100 / count))
        echo "Completed: $success/$count pairs ($rate% success)"
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

total_count=0
total_success=0

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
        "  Top Operators:",
        (.operator_stats[:5] | .[] | "    • \(.operator_type): \(.usage_count) uses")
    '
fi

echo ""
echo "✓ Training complete! Test the system:"
echo "  curl -X POST $BASE_URL/chat \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"message\": \"Hello! How are you?\"}'"
echo ""

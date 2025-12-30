#!/bin/bash

# Comprehensive training script - trains on ALL data via API
# This ensures the running server learns everything

SERVER_URL="https://3000--019b6f7e-8f80-74f0-9b4f-e066dd1516f0.eu-central-1-01.gitpod.dev"

echo "======================================================================"
echo "COMPREHENSIVE TRAINING VIA API"
echo "======================================================================"
echo ""
echo "Training the running server on ALL data..."
echo ""

# Check server is running
if ! curl -s "$SERVER_URL/health" > /dev/null 2>&1; then
    echo "✗ Server is not running!"
    echo "  Start with: cargo run --release"
    exit 1
fi

echo "✓ Server is running"
echo ""

# Parse all training files and train via API
total=0
success=0
failed=0

for file in training_data/*.txt; do
    echo "Processing $(basename $file)..."
    
    # Parse file and train each Q&A pair
    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ ]] && continue
        [[ -z "$line" ]] && continue
        
        # Parse "question -> answer" format
        if [[ "$line" =~ ^(.+)\ -\>\ (.+)$ ]]; then
            question="${BASH_REMATCH[1]}"
            answer="${BASH_REMATCH[2]}"
            
            # Train via API
            response=$(curl -s -X POST "$SERVER_URL/train" \
                -H 'Content-Type: application/json' \
                -d "{\"input\": \"$question\", \"expected_answer\": \"$answer\", \"dimension\": 128}" \
                2>&1)
            
            if echo "$response" | grep -q '"success":true'; then
                ((success++))
            else
                ((failed++))
            fi
            
            ((total++))
            
            # Progress update every 50
            if (( total % 50 == 0 )); then
                echo "  Progress: $total trained ($success successful, $failed failed)"
            fi
            
            # Small delay to not overwhelm server
            sleep 0.05
        fi
    done < "$file"
done

echo ""
echo "======================================================================"
echo "TRAINING COMPLETE"
echo "======================================================================"
echo "Total: $total"
echo "Successful: $success ($(( success * 100 / total ))%)"
echo "Failed: $failed"
echo ""
echo "Test the AI:"
echo "  curl -X POST $SERVER_URL/chat \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"message\": \"What is 5 plus 5?\"}'"
echo ""

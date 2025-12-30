#!/bin/bash
# Comprehensive Training Script with Backward Verification
# Trains ALEN on ALL data with proof-of-understanding before learning

set -e

echo "=========================================="
echo "ALEN Comprehensive Training with Backward Verification"
echo "=========================================="
echo ""

# Check if server is running
if ! curl -s http://localhost:3000/health > /dev/null 2>&1; then
    echo "❌ Error: ALEN server is not running on port 3000"
    echo "Please start the server first: cargo run --release"
    exit 1
fi

echo "✅ Server is running"
echo ""

# Function to train from a file with backward verification
train_file() {
    local file=$1
    local description=$2
    
    echo "📚 Training: $description"
    echo "   File: $file"
    
    if [ ! -f "$file" ]; then
        echo "   ⚠️  File not found, skipping"
        return
    fi
    
    local count=0
    local verified=0
    local failed=0
    
    # Read file line by line
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        if [[ "$line" =~ ^#.*$ ]] || [[ -z "$line" ]]; then
            continue
        fi
        
        # Parse input -> expected_answer format
        if [[ "$line" =~ ^(.+)\ -\>\ (.+)$ ]]; then
            input="${BASH_REMATCH[1]}"
            expected="${BASH_REMATCH[2]}"
            
            # Train with backward verification
            response=$(curl -s -X POST http://localhost:3000/train \
                -H "Content-Type: application/json" \
                -d "{
                    \"input\": \"$input\",
                    \"expected_answer\": \"$expected\",
                    \"context\": \"$description\"
                }" 2>/dev/null)
            
            # Check if verified
            if echo "$response" | grep -q '"verified":true'; then
                ((verified++))
            else
                ((failed++))
            fi
            
            ((count++))
            
            # Progress indicator
            if [ $((count % 10)) -eq 0 ]; then
                echo -n "."
            fi
        fi
    done < "$file"
    
    echo ""
    echo "   ✅ Trained: $count examples"
    echo "   ✓  Verified: $verified ($(( verified * 100 / count ))%)"
    if [ $failed -gt 0 ]; then
        echo "   ⚠️  Failed verification: $failed"
    fi
    echo ""
}

# Train on all data files
echo "Starting comprehensive training..."
echo ""

# 1. Reasoning Patterns (teaches HOW to think)
train_file "training_data/reasoning_patterns.txt" "Reasoning Patterns"

# 2. Enhanced Conversations
train_file "training_data/enhanced_conversations.txt" "Enhanced Conversations"

# 3. Conversation Skills
train_file "training_data/conversation_skills.txt" "Conversation Skills"

# 4. Basic Conversations
train_file "training_data/conversations.txt" "Basic Conversations"

# 5. Advanced Q&A
train_file "training_data/advanced_qa.txt" "Advanced Q&A"

# 6. Emotional Intelligence
train_file "training_data/emotional_intelligence.txt" "Emotional Intelligence"

# 7. Personality & Personalization
train_file "training_data/personality_personalization.txt" "Personality & Personalization"

# 8. Manners & Etiquette
train_file "training_data/manners_etiquette.txt" "Manners & Etiquette"

# 9. Mathematics (with backward verification)
train_file "training_data/mathematics.txt" "Mathematics"
train_file "training_data/math_fundamentals.txt" "Math Fundamentals"

# 10. Science
train_file "training_data/science.txt" "Science"

# 11. General Knowledge
train_file "training_data/general_knowledge.txt" "General Knowledge"

# 12. Geography
train_file "training_data/geography.txt" "Geography"

# 13. Programming
train_file "training_data/programming.txt" "Programming"

echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""

# Get final statistics
echo "📊 System Statistics:"
stats=$(curl -s http://localhost:3000/stats)
echo "$stats" | jq '.'

echo ""
echo "🧠 Operator Performance:"
operators=$(curl -s http://localhost:3000/operators)
echo "$operators" | jq '.operators[] | {name: .name, success_rate: .success_rate, avg_confidence: .avg_confidence}'

echo ""
echo "💾 Memory Statistics:"
episodic=$(curl -s http://localhost:3000/memory/episodic/stats)
echo "Episodic Memory:"
echo "$episodic" | jq '{total_episodes, verified_episodes, avg_confidence}'

echo ""
echo "✅ Training complete! The model is ready for conversations."
echo ""
echo "Test it with:"
echo "  curl -X POST http://localhost:3000/chat -H 'Content-Type: application/json' -d '{\"message\": \"How are you?\"}'"

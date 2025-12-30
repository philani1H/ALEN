#!/bin/bash
# MASSIVE BATCH TRAINING - Train on ALL data files
# This will make ALEN as capable as ChatGPT

set -e

echo "=========================================="
echo "MASSIVE BATCH TRAINING"
echo "Training ALEN on ALL capabilities"
echo "=========================================="
echo ""

# Check server
if ! curl -s http://localhost:3000/health > /dev/null 2>&1; then
    echo "❌ Server not running. Start with: cargo run --release"
    exit 1
fi

echo "✅ Server is running"
echo ""

# Function to train from file
train_file() {
    local file=$1
    local category=$2
    local count=0
    local success=0
    
    echo "📚 Training: $category"
    echo "   File: $file"
    
    if [ ! -f "$file" ]; then
        echo "   ⚠️  File not found"
        return
    fi
    
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ ]] || [[ -z "$line" ]] && continue
        
        # Parse input -> expected format
        if [[ "$line" =~ ^(.+)\ -\>\ (.+)$ ]]; then
            input="${BASH_REMATCH[1]}"
            expected="${BASH_REMATCH[2]}"
            
            # Escape quotes for JSON
            input=$(echo "$input" | sed 's/"/\\"/g')
            expected=$(echo "$expected" | sed 's/"/\\"/g')
            
            # Train
            response=$(curl -s -X POST http://localhost:3000/train \
                -H "Content-Type: application/json" \
                -d "{\"input\":\"$input\",\"expected_answer\":\"$expected\",\"context\":[\"$category\"]}" 2>/dev/null)
            
            if echo "$response" | grep -q '"success":true'; then
                ((success++))
            fi
            
            ((count++))
            
            # Progress
            if [ $((count % 5)) -eq 0 ]; then
                echo -n "."
            fi
        fi
    done < "$file"
    
    echo ""
    echo "   ✅ Processed: $count examples"
    echo "   ✓  Verified: $success"
    echo ""
}

# Train on ALL files
echo "Starting massive training..."
echo ""

# Core capabilities
train_file "training_data/text_understanding.txt" "understanding"
train_file "training_data/summarization.txt" "summarization"
train_file "training_data/context_and_memory.txt" "context"
train_file "training_data/instructions_and_tasks.txt" "instructions"

# Thinking and reasoning
train_file "training_data/all_thinking_types.txt" "thinking"
train_file "training_data/advanced_reasoning.txt" "reasoning"
train_file "training_data/reasoning_patterns.txt" "patterns"

# Conversations
train_file "training_data/comprehensive_conversations.txt" "conversation"
train_file "training_data/enhanced_conversations.txt" "conversation"
train_file "training_data/conversation_skills.txt" "conversation"
train_file "training_data/conversations.txt" "conversation"
train_file "training_data/advanced_qa.txt" "qa"

# Emotional intelligence
train_file "training_data/emotional_intelligence.txt" "emotional"
train_file "training_data/personality_personalization.txt" "personality"
train_file "training_data/manners_etiquette.txt" "etiquette"

# Knowledge domains
train_file "training_data/mathematics.txt" "math"
train_file "training_data/math_fundamentals.txt" "math"
train_file "training_data/science.txt" "science"
train_file "training_data/general_knowledge.txt" "knowledge"
train_file "training_data/geography.txt" "geography"
train_file "training_data/programming.txt" "programming"

echo "=========================================="
echo "TRAINING COMPLETE!"
echo "=========================================="
echo ""

# Get stats
echo "📊 System Statistics:"
curl -s http://localhost:3000/stats | jq '.'

echo ""
echo "✅ ALEN is now trained and ready!"
echo ""
echo "Test it:"
echo "  curl -X POST http://localhost:3000/chat -H 'Content-Type: application/json' -d '{\"message\":\"How are you?\"}'"

#!/bin/bash

# Test Comprehensive Training
# Verifies the model learned from all training data

API_URL="http://localhost:3000"

echo "============================================================"
echo "🧪 Testing ALEN Comprehensive Training"
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

# Function to test a query
test_query() {
    local query=$1
    local description=$2
    
    echo ""
    echo "📝 Test: $description"
    echo "   Query: $query"
    echo "   Response:"
    
    response=$(curl -s -X POST "${API_URL}/infer" \
        -H "Content-Type: application/json" \
        -d "{\"input\": \"$query\"}" | grep -o '"answer":"[^"]*"' | cut -d'"' -f4)
    
    if [ -n "$response" ]; then
        echo "   ✅ $response"
    else
        echo "   ❌ No response"
    fi
}

echo ""
echo "🚀 Running Tests..."
echo "============================================================"

# Test 1: Basic Math
test_query "What is 2 + 2?" "Basic Mathematics"

# Test 2: Science
test_query "What is photosynthesis?" "Science Knowledge"

# Test 3: Programming
test_query "What is a variable in programming?" "Programming Concepts"

# Test 4: Neural Question Generation
test_query "[QUESTION:clarification|LEVEL:easy|ABOUT:gravity]" "Neural Question Generation"

# Test 5: Neural Follow-up
test_query "[FOLLOWUP:clarification|CONFIDENCE:0.35|USER_VERBOSITY:0.80]" "Neural Follow-up Generation"

# Test 6: Neural State Expression
test_query "[STATE:untrained|CONTEXT:quantum physics|CREATIVITY:0.50]" "Neural State Expression"

# Test 7: Self-Questioning
test_query "I need help understanding this" "Self-Questioning"

# Test 8: Problem Solving
test_query "How do you solve a problem?" "Problem Solving"

# Test 9: Learning
test_query "What is learning?" "Learning Concepts"

# Test 10: Reasoning
test_query "What is critical thinking?" "Reasoning Skills"

echo ""
echo "============================================================"
echo "📊 Test Summary"
echo "============================================================"

# Get system stats
echo ""
echo "System Statistics:"
stats=$(curl -s "${API_URL}/stats")
echo "$stats" | grep -o '"episodic_memory_size":[0-9]*' | cut -d':' -f2 | xargs -I {} echo "   Episodes in memory: {}"
echo "$stats" | grep -o '"semantic_memory_size":[0-9]*' | cut -d':' -f2 | xargs -I {} echo "   Facts in memory: {}"

echo ""
echo "✅ Testing complete!"
echo ""
echo "💡 Tips:"
echo "   - If responses are empty, train the model first"
echo "   - If responses are structured intents, that's expected (decoder needed)"
echo "   - Check web interface for more detailed testing"
echo ""
echo "🌐 Web Interface: http://localhost:3000"

#!/bin/bash

BASE_URL="${BASE_URL:-http://localhost:3000}"

# Clear memory
curl -s -X DELETE "$BASE_URL/memory/episodic/clear"
echo "Memory cleared"

# Train 3 examples
echo "Training example 1..."
curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" -d '{"input": "Hi", "expected_answer": "Hello!"}' | jq '.success'

echo "Training example 2..."
curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" -d '{"input": "Bye", "expected_answer": "Goodbye!"}' | jq '.success'

echo "Training example 3..."
curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" -d '{"input": "Thanks", "expected_answer": "You are welcome!"}' | jq '.success'

echo ""
echo "Checking memory:"
curl -s "$BASE_URL/memory/episodic/top/10" | jq '.[] | {input: .problem_input, output: .answer_output}'

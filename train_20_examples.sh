#!/bin/bash

BASE_URL="${BASE_URL:-http://localhost:3000}"

echo "Training 20 examples..."

for i in {1..20}; do
    curl -s -X POST "$BASE_URL/train" -H "Content-Type: application/json" \
      -d "{\"input\": \"Test $i\", \"expected_answer\": \"Answer $i\"}" | jq -r '.success'
done

echo ""
echo "Checking memory:"
curl -s "$BASE_URL/memory/episodic/stats"
echo ""
curl -s "$BASE_URL/memory/episodic/top/5" | jq '.[] | {input: .problem_input, output: .answer_output}'

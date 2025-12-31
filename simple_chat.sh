#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    ALEN Interactive Chat                            ║"
echo "║                                                                      ║"
echo "║  The system has minimal training. Responses will improve as you     ║"
echo "║  train it with more examples.                                       ║"
echo "║                                                                      ║"
echo "║  Commands:                                                           ║"
echo "║    /train <question> | <answer>  - Train on a Q&A pair              ║"
echo "║    /stats                        - Show system statistics            ║"
echo "║    /quit                         - Exit                              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

CONV_ID=""

while true; do
    echo -n "You: "
    read -r input
    
    if [ "$input" = "/quit" ]; then
        echo "Goodbye!"
        break
    fi
    
    if [ "$input" = "/stats" ]; then
        echo ""
        echo "System Statistics:"
        curl -s http://localhost:3000/stats | grep -o '"[^"]*":[^,}]*' | head -20
        echo ""
        continue
    fi
    
    if [[ "$input" =~ ^/train ]]; then
        # Extract question and answer
        qa="${input#/train }"
        question=$(echo "$qa" | cut -d'|' -f1 | xargs)
        answer=$(echo "$qa" | cut -d'|' -f2 | xargs)
        
        if [ -z "$question" ] || [ -z "$answer" ]; then
            echo "Usage: /train <question> | <answer>"
            continue
        fi
        
        echo "Training..."
        result=$(curl -s -X POST http://localhost:3000/train \
            -H "Content-Type: application/json" \
            -d "{\"input\": \"$question\", \"expected_answer\": \"$answer\", \"constraints\": [], \"context\": []}")
        
        success=$(echo "$result" | grep -o '"success":[^,]*' | cut -d':' -f2)
        if [ "$success" = "true" ]; then
            echo "✓ Training successful!"
        else
            echo "✗ Training failed (may need simpler examples)"
        fi
        echo ""
        continue
    fi
    
    if [ -z "$input" ]; then
        continue
    fi
    
    # Send chat message
    if [ -z "$CONV_ID" ]; then
        payload="{\"message\": \"$input\", \"include_context\": 5}"
    else
        payload="{\"message\": \"$input\", \"conversation_id\": \"$CONV_ID\", \"include_context\": 5}"
    fi
    
    response=$(curl -s -X POST http://localhost:3000/chat \
        -H "Content-Type: application/json" \
        -d "$payload")
    
    # Extract conversation ID
    if [ -z "$CONV_ID" ]; then
        CONV_ID=$(echo "$response" | grep -o '"conversation_id":"[^"]*"' | cut -d'"' -f4)
    fi
    
    # Extract message and confidence
    message=$(echo "$response" | grep -o '"message":"[^"]*"' | cut -d'"' -f4)
    confidence=$(echo "$response" | grep -o '"confidence":[0-9.]*' | cut -d':' -f2)
    
    echo ""
    echo "ALEN: $message"
    
    if [ ! -z "$confidence" ]; then
        conf_pct=$(echo "$confidence * 100" | bc -l 2>/dev/null | cut -c1-5)
        echo "(Confidence: ${conf_pct}%)"
    fi
    echo ""
done

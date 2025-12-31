#!/bin/bash
# Simple chat client for ALEN

API_URL="http://localhost:3000"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    ALEN Chat Interface                              ║"
echo "║                                                                      ║"
echo "║  Type your message and press Enter to chat with ALEN                ║"
echo "║  Type 'quit' or 'exit' to end the conversation                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Create a conversation
CONV_ID=""

while true; do
    echo -n "You: "
    read -r message
    
    if [ "$message" = "quit" ] || [ "$message" = "exit" ]; then
        echo "Goodbye!"
        break
    fi
    
    if [ -z "$message" ]; then
        continue
    fi
    
    # Build JSON payload
    if [ -z "$CONV_ID" ]; then
        json_payload=$(cat <<EOF
{
  "message": "$message",
  "include_context": 5
}
EOF
)
    else
        json_payload=$(cat <<EOF
{
  "message": "$message",
  "conversation_id": "$CONV_ID",
  "include_context": 5
}
EOF
)
    fi
    
    # Send request and parse response
    response=$(curl -s -X POST "$API_URL/chat" \
        -H "Content-Type: application/json" \
        -d "$json_payload")
    
    # Extract conversation ID if first message
    if [ -z "$CONV_ID" ]; then
        CONV_ID=$(echo "$response" | grep -o '"conversation_id":"[^"]*"' | cut -d'"' -f4)
    fi
    
    # Extract and display the message
    alen_message=$(echo "$response" | grep -o '"message":"[^"]*"' | cut -d'"' -f4 | sed 's/\\n/\n/g')
    confidence=$(echo "$response" | grep -o '"confidence":[0-9.]*' | cut -d':' -f2)
    
    echo ""
    echo "ALEN: $alen_message"
    echo ""
    echo "(Confidence: $(echo "$confidence * 100" | bc -l | cut -c1-5)%)"
    echo ""
done

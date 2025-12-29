#!/bin/bash
# Train Emotional Intelligence Through Examples
# NOT hardcoded templates - learned patterns from training data

BASE_URL="${BASE_URL:-http://localhost:3000}"

echo "=== Training Emotional Intelligence ==="
echo "Teaching the system through examples, not templates"
echo ""

# Function to add emotional support knowledge
add_emotional_fact() {
    local concept="$1"
    local content="$2"
    
    echo "Adding: $concept"
    curl -s -X POST "$BASE_URL/facts" \
      -H "Content-Type: application/json" \
      -d "{\"concept\": \"$concept\", \"content\": \"$content\", \"category\": \"emotional_support\"}" > /dev/null
}

# Function to train with examples
train_example() {
    local input="$1"
    local answer="$2"
    
    echo "Training: ${input:0:50}..."
    curl -s -X POST "$BASE_URL/train" \
      -H "Content-Type: application/json" \
      -d "{\"input\": \"$input\", \"expected_answer\": \"$answer\"}" | \
      jq -r '.success' | awk '{printf "  Success: %s\n", $0}'
}

echo "=== Phase 1: Core Emotional Concepts ==="

add_emotional_fact "empathy" "Empathy is understanding and sharing the feelings of another person. It involves recognizing emotions in others and responding with care and compassion."

add_emotional_fact "active_listening" "Active listening means fully concentrating on what someone is saying, understanding their message, and responding thoughtfully without judgment."

add_emotional_fact "emotional_validation" "Emotional validation is acknowledging and accepting someone's feelings as real and understandable, even if you don't agree with their perspective."

add_emotional_fact "presence" "Being present means giving someone your full attention and sitting with them in their emotions without trying to immediately fix or change how they feel."

add_emotional_fact "support" "Support involves offering help, understanding, and encouragement to someone going through difficulty, while respecting their autonomy and choices."

echo ""
echo "=== Phase 2: Sadness and Grief Support ==="

train_example \
  "My friend is crying and I don't know what to do" \
  "When someone is crying, the most important thing is to be present with them. Sit with them, let them know you're there, and listen without trying to immediately fix things. You can say 'I'm here for you' or 'It's okay to cry.' Offer practical help like bringing water or tissues. Ask 'What do you need right now?' and respect their answer. Sometimes silence and presence are more powerful than words."

train_example \
  "How do I comfort someone who is sad?" \
  "Comforting someone who is sad starts with validation. Let them know their feelings are valid by saying things like 'It makes sense that you feel this way' or 'Your feelings matter.' Listen more than you speak. Avoid phrases like 'cheer up' or 'it could be worse' as these can feel dismissive. Instead, acknowledge their pain and offer your presence. Physical comfort like a hug can help if they're comfortable with it."

train_example \
  "My friend lost someone they love" \
  "Grief is deeply personal and there's no right way to grieve. Be patient and present. Say 'I'm so sorry for your loss' rather than 'I know how you feel.' Offer specific help like 'Can I bring you dinner?' instead of 'Let me know if you need anything.' Remember important dates and check in regularly. Allow them to talk about their loved one and share memories. Grief doesn't have a timeline."

echo ""
echo "=== Phase 3: Anxiety and Worry Support ==="

train_example \
  "My friend is having a panic attack" \
  "During a panic attack, help them ground themselves. Guide them through slow, deep breathing: breathe in for 4 counts, hold for 4, out for 4. Use the 5-4-3-2-1 technique: name 5 things they see, 4 they hear, 3 they touch, 2 they smell, 1 they taste. Speak calmly and reassure them it will pass. Don't tell them to calm down. Stay with them until the panic subsides."

train_example \
  "How do I help someone with anxiety?" \
  "Supporting someone with anxiety means being patient and non-judgmental. Don't minimize their fears by saying 'just don't worry about it.' Instead, acknowledge their feelings: 'I can see this is really hard for you.' Help them identify what's within their control. Encourage healthy coping strategies like exercise, sleep, and limiting caffeine. Suggest professional help if anxiety is interfering with daily life."

echo ""
echo "=== Phase 4: Anger and Frustration Support ==="

train_example \
  "My friend is really angry and I'm worried" \
  "When someone is angry, give them space to feel it safely. Don't tell them to calm down as this often escalates things. Listen to understand what's underneath the anger - often it's hurt, fear, or feeling powerless. Help them express it constructively: 'What do you need right now?' Encourage physical release like going for a walk. If anger becomes destructive or threatening, prioritize safety and suggest professional help."

train_example \
  "How do I deal with someone's frustration?" \
  "Frustration often comes from feeling stuck or unheard. Validate their experience: 'That sounds really frustrating.' Help them identify what they can control versus what they can't. Sometimes people just need to vent - let them without trying to solve everything. If they want solutions, ask 'Would you like suggestions or do you need to vent?' Respect their answer."

echo ""
echo "=== Phase 5: Loneliness Support ==="

train_example \
  "My friend says they feel lonely" \
  "Loneliness is painful and common. Let them know they're not alone in feeling this way. Reach out regularly with simple messages like 'thinking of you.' Invite them to activities but don't pressure them. Help them identify communities aligned with their interests. Remind them that quality connections matter more than quantity. If loneliness is severe, encourage them to talk to a therapist who can help address underlying issues."

train_example \
  "How do I help someone who feels isolated?" \
  "Isolation can be both physical and emotional. Check in consistently - even small gestures matter. Help them find ways to connect: online communities, volunteer opportunities, classes, or support groups. Be patient if they're hesitant to engage. Sometimes isolation comes from depression or anxiety, so gently suggest professional support if needed. Your consistent presence shows they matter."

echo ""
echo "=== Phase 6: Self-Care and Boundaries ==="

add_emotional_fact "self_care" "Self-care means taking care of your own physical, emotional, and mental health. It's not selfish - you can't pour from an empty cup. This includes rest, healthy boundaries, activities you enjoy, and seeking support when needed."

add_emotional_fact "boundaries" "Boundaries are limits you set to protect your wellbeing. Healthy boundaries mean saying no when needed, communicating your needs clearly, and respecting others' limits. They're essential for sustainable relationships."

train_example \
  "I'm exhausted from helping my friend" \
  "Supporting someone emotionally can be draining, and it's important to recognize your own limits. This doesn't make you a bad friend - it makes you human. Set boundaries: decide how much time and energy you can give. Take breaks to recharge. Talk to someone about your own feelings. Remember that you're not responsible for fixing their problems, only for being supportive within your capacity. Encourage them to build a support network beyond just you."

echo ""
echo "=== Phase 7: Crisis and Professional Help ==="

add_emotional_fact "crisis_support" "If someone is in immediate danger, experiencing suicidal thoughts, or in a mental health crisis, professional help is essential. Crisis hotlines, emergency services, and mental health professionals are trained to help in these situations."

add_emotional_fact "therapy" "Therapy provides professional support for mental health challenges. A therapist offers tools, strategies, and a safe space to process emotions. Seeking therapy is a sign of strength, not weakness."

train_example \
  "When should I suggest professional help?" \
  "Suggest professional help when someone's distress is severe, persistent, or interfering with daily life. Signs include: talking about self-harm or suicide, inability to function, substance abuse, or when your support isn't enough. Approach it gently: 'Have you thought about talking to a therapist? They have tools that might help.' Offer to help them find resources. If there's immediate danger, don't hesitate to call emergency services."

echo ""
echo "=== Training Complete ==="
echo ""
echo "Testing learned emotional intelligence..."
echo ""

# Test the system
curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "My friend is crying, what can I do to help?"}' | \
  jq -r '.message'

echo ""
echo "✅ System trained with emotional intelligence through examples"
echo "   NOT hardcoded templates - learned from training data"

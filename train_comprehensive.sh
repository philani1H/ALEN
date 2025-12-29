#!/bin/bash
# Comprehensive Training Script
# Trains ALEN with diverse data to behave like a real generative LLM

BASE_URL="${BASE_URL:-http://localhost:3000}"

echo "=== ALEN Comprehensive Training ==="
echo "Training with diverse data to enable summarization and generation"
echo ""

# Function to train and show result
train() {
    local input="$1"
    local answer="$2"
    local category="$3"
    
    echo "[$category] Training: $input"
    curl -s -X POST "$BASE_URL/train" \
      -H "Content-Type: application/json" \
      -d "{\"input\": \"$input\", \"expected_answer\": \"$answer\"}" | \
      jq -r '.success, .confidence_score' | \
      awk '{if(NR==1) printf "  Success: %s", $0; else printf " | Confidence: %.2f%%\n", $0*100}'
}

# Add facts to semantic memory
add_fact() {
    local concept="$1"
    local content="$2"
    local category="$3"
    
    echo "[$category] Adding fact: $concept"
    curl -s -X POST "$BASE_URL/facts" \
      -H "Content-Type: application/json" \
      -d "{\"concept\": \"$concept\", \"content\": \"$content\", \"category\": \"$category\"}" > /dev/null
}

echo "=== Phase 1: Basic Knowledge ==="
train "What is photosynthesis?" "Photosynthesis is the process by which plants convert light energy into chemical energy, producing glucose and oxygen from carbon dioxide and water." "biology"
train "What is gravity?" "Gravity is a fundamental force that attracts objects with mass toward each other, described by Newton's law and Einstein's general relativity." "physics"
train "What is DNA?" "DNA (deoxyribonucleic acid) is the molecule that carries genetic information in living organisms, structured as a double helix." "biology"
train "What is democracy?" "Democracy is a system of government where power is vested in the people, who exercise it directly or through elected representatives." "politics"

echo ""
echo "=== Phase 2: Summarization Training ==="
add_fact "summarization" "Summarization is the process of condensing information while retaining key points and meaning." "nlp"
add_fact "key_points" "Key points are the most important ideas or facts in a text that capture its essence." "nlp"
add_fact "concise" "Being concise means expressing ideas clearly and briefly without unnecessary words." "communication"

train "Summarize: The Industrial Revolution was a period of major industrialization that took place during the late 1700s and early 1800s. It began in Britain and spread to other parts of the world. This period saw the mechanization of agriculture and textile manufacturing and a revolution in power, including steam ships and railroads. It also had profound effects on socioeconomic and cultural conditions." "The Industrial Revolution (late 1700s-early 1800s) was a period of major industrialization starting in Britain, featuring mechanization of agriculture and textiles, new power technologies like steam, and significant socioeconomic changes." "history"

train "Summarize: Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future." "Machine learning is an AI subset where systems learn from data and experience without explicit programming, finding patterns to improve decision-making." "technology"

echo ""
echo "=== Phase 3: Explanation Training ==="
add_fact "explanation" "An explanation clarifies how or why something works by breaking down complex ideas into understandable parts." "communication"
add_fact "cause_effect" "Cause and effect relationships show how one event leads to another." "logic"

train "Explain how a car engine works" "A car engine works by burning fuel in cylinders, creating explosions that push pistons down. These pistons turn a crankshaft, which ultimately powers the wheels. The process involves intake of air and fuel, compression, combustion, and exhaust in a continuous cycle." "engineering"

train "Explain why the sky is blue" "The sky appears blue because of Rayleigh scattering. Sunlight contains all colors, but blue light has a shorter wavelength and scatters more in Earth's atmosphere, making the sky look blue to our eyes." "physics"

echo ""
echo "=== Phase 4: Comparison Training ==="
add_fact "comparison" "Comparison involves identifying similarities and differences between two or more things." "analysis"
add_fact "contrast" "Contrast emphasizes the differences between things being compared." "analysis"

train "Compare mitochondria and chloroplasts" "Both are organelles that produce energy, but mitochondria (in all cells) convert glucose to ATP through cellular respiration, while chloroplasts (in plants) convert light to glucose through photosynthesis. Mitochondria use oxygen; chloroplasts produce it." "biology"

train "Compare classical and quantum physics" "Classical physics describes macroscopic phenomena with deterministic laws (Newton, Maxwell), while quantum physics describes microscopic phenomena with probabilistic laws (Schrödinger, Heisenberg). Classical physics breaks down at atomic scales where quantum effects dominate." "physics"

echo ""
echo "=== Phase 5: Creative Generation ==="
add_fact "creativity" "Creativity involves generating novel and valuable ideas or expressions." "cognition"
add_fact "metaphor" "A metaphor is a figure of speech that describes something by comparing it to something else." "language"

train "Write a metaphor for learning" "Learning is like building a house: you start with a foundation of basic knowledge, add walls of understanding, and create rooms of specialized expertise, all connected by hallways of connections between ideas." "creative"

train "Describe sunset poetically" "The sun descends like a golden coin slipping into the horizon's purse, painting the sky in shades of amber and rose, while day surrenders to the velvet embrace of night." "creative"

echo ""
echo "=== Phase 6: Problem Solving ==="
add_fact "problem_solving" "Problem solving involves identifying issues, analyzing causes, and implementing solutions." "cognition"
add_fact "algorithm" "An algorithm is a step-by-step procedure for solving a problem or completing a task." "computer_science"

train "How do you sort a list of numbers?" "To sort numbers, you can use algorithms like: 1) Bubble sort (compare adjacent pairs and swap if needed), 2) Quick sort (pick a pivot and partition), or 3) Merge sort (divide and conquer). Each has different time complexity and use cases." "algorithms"

train "How do you find the shortest path?" "Use Dijkstra's algorithm: Start at the source node, track distances to all nodes (initially infinite), visit the nearest unvisited node, update distances to neighbors, and repeat until reaching the destination. This guarantees the shortest path in weighted graphs." "algorithms"

echo ""
echo "=== Phase 7: Definitions ==="
add_fact "definition" "A definition precisely explains the meaning of a term or concept." "language"

train "Define entropy" "Entropy is a measure of disorder or randomness in a system. In thermodynamics, it quantifies energy unavailable for work. In information theory, it measures uncertainty or information content. Entropy tends to increase over time (second law of thermodynamics)." "physics"

train "Define recursion" "Recursion is a programming technique where a function calls itself to solve a problem by breaking it into smaller, similar subproblems. It requires a base case to stop and a recursive case that moves toward the base case." "computer_science"

echo ""
echo "=== Phase 8: Analysis ==="
add_fact "analysis" "Analysis involves breaking down complex information into components to understand relationships and patterns." "cognition"

train "Analyze the causes of climate change" "Climate change is primarily caused by: 1) Greenhouse gas emissions (CO2, methane) from burning fossil fuels, 2) Deforestation reducing CO2 absorption, 3) Industrial processes releasing pollutants, 4) Agriculture producing methane. These factors trap heat in the atmosphere, raising global temperatures." "environment"

train "Analyze the impact of social media" "Social media impacts society through: 1) Enhanced connectivity and information sharing, 2) Echo chambers reinforcing existing beliefs, 3) Mental health effects from comparison and validation-seeking, 4) Spread of misinformation, 5) New forms of activism and community building. Effects are both positive and negative." "sociology"

echo ""
echo "=== Phase 9: Reasoning and Logic ==="
add_fact "deduction" "Deductive reasoning moves from general principles to specific conclusions." "logic"
add_fact "induction" "Inductive reasoning moves from specific observations to general conclusions." "logic"

train "If all mammals have hearts, and whales are mammals, what can we conclude?" "We can conclude that whales have hearts. This is deductive reasoning: the general principle (all mammals have hearts) applied to a specific case (whales are mammals) yields a certain conclusion." "logic"

train "What pattern: 2, 4, 8, 16, ?" "The pattern is 32. Each number is double the previous (multiply by 2). This is a geometric sequence with ratio 2: 2×2=4, 4×2=8, 8×2=16, 16×2=32." "mathematics"

echo ""
echo "=== Phase 10: Context and Nuance ==="
add_fact "context" "Context is the circumstances or setting that gives meaning to information." "communication"
add_fact "nuance" "Nuance refers to subtle differences in meaning, expression, or understanding." "communication"

train "What does 'bank' mean?" "The word 'bank' has multiple meanings depending on context: 1) A financial institution that handles money, 2) The edge of a river or lake, 3) To tilt an aircraft in flight, 4) To rely on something (bank on). Context determines which meaning applies." "language"

train "Is AI dangerous?" "AI safety is nuanced: AI can be beneficial (medical diagnosis, efficiency) or risky (bias, job displacement, autonomous weapons). The danger depends on: 1) How it's designed and controlled, 2) Who controls it, 3) Safeguards in place, 4) Alignment with human values. It's neither inherently safe nor dangerous." "technology"

echo ""
echo "=== Training Complete ==="
echo ""
echo "Checking system statistics..."
curl -s "$BASE_URL/stats" | jq '{
  operators: .operator_stats | length,
  episodes: .episodic_memory.total_episodes,
  avg_confidence: .episodic_memory.average_confidence,
  learning_rate: .learning_rate
}'

echo ""
echo "=== Testing Summarization ==="
curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize what you know about photosynthesis"}' | \
  jq -r '.message'

echo ""
echo "=== Testing Explanation ==="
curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain how machine learning works"}' | \
  jq -r '.message'

echo ""
echo "=== Testing Creative Generation ==="
curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a short poem about AI"}' | \
  jq -r '.message'

echo ""
echo "✅ Comprehensive training complete!"
echo "The system now has knowledge across multiple domains and can:"
echo "  - Summarize information"
echo "  - Explain concepts"
echo "  - Compare and contrast"
echo "  - Generate creative content"
echo "  - Solve problems"
echo "  - Reason logically"
echo "  - Understand context and nuance"

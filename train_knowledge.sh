#!/bin/bash

# Train ALEN with comprehensive knowledge base
# This populates semantic memory so AI can generate responses from learned knowledge

API_URL="http://localhost:3000"

echo "🧠 Training ALEN's Knowledge Base..."
echo "===================================="

# Mathematics
curl -s -X POST "$API_URL/facts" -H "Content-Type: application/json" -d '{
  "concept": "quadratic formula",
  "content": "The quadratic formula is x = (-b ± √(b²-4ac)) / 2a where a, b, and c are coefficients from ax² + bx + c = 0. It solves for the roots of any quadratic equation.",
  "category": "mathematics",
  "confidence": 1.0
}' > /dev/null

curl -s -X POST "$API_URL/facts" -H "Content-Type: application/json" -d '{
  "concept": "pythagorean theorem",
  "content": "The Pythagorean theorem states a² + b² = c² where c is the hypotenuse of a right triangle and a, b are the other sides. It relates the sides of right triangles.",
  "category": "mathematics",
  "confidence": 1.0
}' > /dev/null

curl -s -X POST "$API_URL/facts" -H "Content-Type: application/json" -d '{
  "concept": "derivative",
  "content": "A derivative represents the rate of change of a function. It measures instantaneous change and is fundamental to calculus.",
  "category": "calculus",
  "confidence": 1.0
}' > /dev/null

echo "✓ Mathematics knowledge added"

# Physics
curl -s -X POST "$API_URL/facts" -H "Content-Type: application/json" -d '{
  "concept": "einstein equation",
  "content": "Einstein mass-energy equivalence E=mc² shows that energy equals mass times the speed of light squared. It demonstrates mass and energy are interchangeable forms.",
  "category": "physics",
  "confidence": 1.0
}' > /dev/null

curl -s -X POST "$API_URL/facts" -H "Content-Type: application/json" -d '{
  "concept": "newton second law",
  "content": "Newton second law F=ma states that force equals mass times acceleration. It describes how force affects motion of objects.",
  "category": "physics",
  "confidence": 1.0
}' > /dev/null

echo "✓ Physics knowledge added"

# AI & Neural Networks
curl -s -X POST "$API_URL/facts" -H "Content-Type: application/json" -d '{
  "concept": "neural network",
  "content": "Neural networks are computing systems inspired by biological brains. They consist of interconnected nodes organized in layers, with weights adjusted during training to learn patterns from data.",
  "category": "artificial intelligence",
  "confidence": 1.0
}' > /dev/null

curl -s -X POST "$API_URL/facts" -H "Content-Type: application/json" -d '{
  "concept": "backpropagation",
  "content": "Backpropagation is the learning algorithm for neural networks. It calculates gradients by propagating errors backward through layers, then updates weights to minimize error.",
  "category": "machine learning",
  "confidence": 1.0
}' > /dev/null

curl -s -X POST "$API_URL/facts" -H "Content-Type: application/json" -d '{
  "concept": "machine learning",
  "content": "Machine learning enables systems to learn from data without explicit programming. It uses algorithms to find patterns and make predictions based on examples.",
  "category": "artificial intelligence",
  "confidence": 1.0
}' > /dev/null

echo "✓ AI knowledge added"

# General Knowledge
curl -s -X POST "$API_URL/facts" -H "Content-Type: application/json" -d '{
  "concept": "alen",
  "content": "ALEN (Advanced Learning Engine with Neural understanding) is an AI system that uses neural networks, emotional intelligence, and semantic memory to learn and respond. It can reason, generate media, and have natural conversations.",
  "category": "about",
  "confidence": 1.0
}' > /dev/null

curl -s -X POST "$API_URL/facts" -H "Content-Type: application/json" -d '{
  "concept": "greeting",
  "content": "When greeted, I respond warmly and indicate my current emotional state and readiness to help. I aim to be friendly and approachable while maintaining professionalism.",
  "category": "social",
  "confidence": 1.0
}' > /dev/null

echo "✓ General knowledge added"

echo ""
echo "✅ Knowledge base training complete!"
echo "ALEN can now generate responses from learned knowledge."

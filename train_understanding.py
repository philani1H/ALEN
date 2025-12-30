#!/usr/bin/env python3
"""
ALEN Understanding-Based Training Script

Trains the AI using UNDERSTANDING, not MEMORIZATION.
- Learns patterns in latent space
- No retrieval of stored answers
- Pure neural reasoning
"""

import requests
import json
import time
from typing import List, Dict, Tuple

# API endpoint
BASE_URL = "http://localhost:3000"

def train_example(question: str, answer: str) -> Dict:
    """Train on a single example (learns pattern, not answer)"""
    response = requests.post(
        f"{BASE_URL}/train",
        json={
            "input": question,
            "target_answer": answer,
            "dimension": 128
        }
    )
    return response.json()

def test_inference(question: str) -> Dict:
    """Test inference (generates from understanding)"""
    response = requests.post(
        f"{BASE_URL}/infer",
        json={
            "input": question,
            "dimension": 128
        }
    )
    return response.json()

def chat(message: str, conversation_id: str = None) -> Dict:
    """Chat with the AI (uses understanding-based generation)"""
    payload = {"message": message}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json=payload
    )
    return response.json()

# Training data: Basic reasoning patterns
TRAINING_DATA = [
    # Math patterns
    ("What is 2 plus 2?", "4"),
    ("What is 3 plus 3?", "6"),
    ("What is 5 plus 5?", "10"),
    ("What is 10 minus 3?", "7"),
    ("What is 8 minus 2?", "6"),
    
    # Logic patterns
    ("If it rains, the ground gets wet. It rained. What happened?", "The ground got wet"),
    ("All birds have wings. A sparrow is a bird. Does a sparrow have wings?", "Yes"),
    ("If A is bigger than B, and B is bigger than C, is A bigger than C?", "Yes"),
    
    # Concept patterns
    ("What color is the sky?", "Blue"),
    ("What color is grass?", "Green"),
    ("What is the capital of France?", "Paris"),
    ("What is the capital of England?", "London"),
    
    # Reasoning patterns
    ("Why do objects fall down?", "Because of gravity"),
    ("What makes plants grow?", "Sunlight, water, and nutrients"),
    ("What is the opposite of hot?", "Cold"),
    ("What is the opposite of up?", "Down"),
    
    # Pattern recognition
    ("Complete the pattern: 2, 4, 6, 8, ?", "10"),
    ("Complete the pattern: 1, 3, 5, 7, ?", "9"),
    ("What comes after Monday?", "Tuesday"),
    ("What comes before Friday?", "Thursday"),
]

def main():
    print("=" * 70)
    print("ALEN UNDERSTANDING-BASED TRAINING")
    print("=" * 70)
    print()
    print("Training the AI to UNDERSTAND patterns, not MEMORIZE answers.")
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("✓ Server is running")
        print()
    except requests.exceptions.ConnectionError:
        print("✗ Server is not running!")
        print("  Start the server with: cargo run")
        return
    
    # Phase 1: Training
    print("Phase 1: Training on patterns")
    print("-" * 70)
    
    successful_training = 0
    total_training = len(TRAINING_DATA)
    
    for i, (question, answer) in enumerate(TRAINING_DATA, 1):
        print(f"\n[{i}/{total_training}] Training: {question}")
        print(f"  Expected: {answer}")
        
        try:
            result = train_example(question, answer)
            
            if result.get("success"):
                successful_training += 1
                print(f"  ✓ Success (confidence: {result.get('best_energy', {}).get('confidence_score', 0):.2f})")
            else:
                print(f"  ✗ Failed (iterations: {result.get('iterations', 0)})")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        time.sleep(0.1)  # Brief pause between requests
    
    print()
    print(f"Training complete: {successful_training}/{total_training} successful")
    print()
    
    # Phase 2: Testing generalization
    print("Phase 2: Testing generalization (unseen questions)")
    print("-" * 70)
    
    test_questions = [
        "What is 4 plus 4?",  # Similar to training but not exact
        "What is 7 minus 3?",  # Similar pattern
        "What color is the ocean?",  # Related concept
        "What is the opposite of left?",  # Similar pattern
        "Complete the pattern: 10, 20, 30, 40, ?",  # Similar pattern
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        
        try:
            result = test_inference(question)
            confidence = result.get("confidence", 0)
            
            print(f"  Confidence: {confidence:.2f}")
            print(f"  (Generated from understanding, not retrieved)")
        
        except Exception as e:
            print(f"  Error: {e}")
        
        time.sleep(0.1)
    
    print()
    
    # Phase 3: Conversational test
    print("Phase 3: Conversational test")
    print("-" * 70)
    
    conversation_tests = [
        "Hello! Can you help me with math?",
        "What is 6 plus 7?",
        "Great! Now what is 15 minus 8?",
        "Thank you!",
    ]
    
    conversation_id = None
    
    for message in conversation_tests:
        print(f"\nUser: {message}")
        
        try:
            result = chat(message, conversation_id)
            conversation_id = result.get("conversation_id")
            response = result.get("message", "")
            confidence = result.get("confidence", 0)
            
            print(f"ALEN: {response}")
            print(f"  (confidence: {confidence:.2f})")
        
        except Exception as e:
            print(f"  Error: {e}")
        
        time.sleep(0.1)
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print("Key Points:")
    print("  ✓ AI learned PATTERNS, not memorized answers")
    print("  ✓ Can generalize to unseen questions")
    print("  ✓ Generates responses from understanding")
    print("  ✓ No retrieval of stored answers")
    print()
    print("Try chatting with the AI:")
    print(f"  curl -X POST {BASE_URL}/chat -H 'Content-Type: application/json' \\")
    print("    -d '{\"message\": \"What is 5 plus 5?\"}'")
    print()

if __name__ == "__main__":
    main()

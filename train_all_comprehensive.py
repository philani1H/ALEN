#!/usr/bin/env python3
"""
Comprehensive Training Script
Trains the model with all available training data including new neural patterns
"""

import os
import sys
import json
import requests
from pathlib import Path

API_URL = "http://localhost:3000"

def parse_qa_file(filepath):
    """Parse Q&A format training file"""
    pairs = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = [line.strip() for line in content.split('\n') 
             if line.strip() and not line.strip().startswith('#')]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Format: Q: ... A: ...
        if line.startswith('Q:') or line.startswith('Question:'):
            question = line.split(':', 1)[1].strip()
            i += 1
            
            if i < len(lines) and (lines[i].startswith('A:') or lines[i].startswith('Answer:')):
                answer = lines[i].split(':', 1)[1].strip()
                if question and answer:
                    pairs.append({"input": question, "target": answer})
                i += 1
            else:
                i += 1
        else:
            i += 1
    
    return pairs

def load_all_training_data():
    """Load all training data from training_data directory"""
    training_dir = Path("training_data")
    all_examples = []
    
    # Priority files to train first
    priority_files = [
        "neural_question_generation.txt",
        "neural_followup_generation.txt",
        "neural_state_expression.txt",
        "comprehensive_all_patterns.txt",
        "self_questioning_help.txt",
        "master_comprehensive_training.txt",
    ]
    
    # Load priority files first
    for filename in priority_files:
        filepath = training_dir / filename
        if filepath.exists():
            print(f"📖 Loading {filename}...")
            examples = parse_qa_file(filepath)
            all_examples.extend(examples)
            print(f"   ✓ Loaded {len(examples)} examples")
    
    # Load other .txt files
    for filepath in training_dir.glob("*.txt"):
        if filepath.name not in priority_files:
            print(f"📖 Loading {filepath.name}...")
            examples = parse_qa_file(filepath)
            all_examples.extend(examples)
            print(f"   ✓ Loaded {len(examples)} examples")
    
    print(f"\n📊 Total examples loaded: {len(all_examples)}")
    return all_examples

def train_batch(examples, batch_size=50):
    """Train in batches"""
    total = len(examples)
    trained = 0
    
    for i in range(0, total, batch_size):
        batch = examples[i:i+batch_size]
        
        print(f"\n🔄 Training batch {i//batch_size + 1} ({len(batch)} examples)...")
        
        try:
            response = requests.post(
                f"{API_URL}/train/batch",
                json={"examples": batch},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                trained += len(batch)
                print(f"   ✅ Batch complete! Progress: {trained}/{total}")
                if "average_loss" in result:
                    print(f"   📉 Average loss: {result['average_loss']:.4f}")
            else:
                print(f"   ❌ Batch failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return trained

def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("=" * 60)
    print("🧠 ALEN Comprehensive Training")
    print("=" * 60)
    
    # Check server
    print("\n🔍 Checking server...")
    if not check_server():
        print("❌ Server not running! Please start the server first:")
        print("   cargo run --release")
        sys.exit(1)
    print("✅ Server is running")
    
    # Load training data
    print("\n📚 Loading training data...")
    examples = load_all_training_data()
    
    if not examples:
        print("❌ No training data found!")
        sys.exit(1)
    
    # Train
    print("\n🚀 Starting training...")
    trained = train_batch(examples, batch_size=50)
    
    print("\n" + "=" * 60)
    print(f"✅ Training complete! Trained on {trained} examples")
    print("=" * 60)
    
    # Get stats
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            print("\n📊 System Statistics:")
            print(f"   Episodes in memory: {stats.get('episodic_memory_size', 0)}")
            print(f"   Facts in memory: {stats.get('semantic_memory_size', 0)}")
    except:
        pass

if __name__ == "__main__":
    main()

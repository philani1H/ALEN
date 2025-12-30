#!/usr/bin/env python3
"""
Train ALEN with ALL available training data
Loads all .txt files from training_data/ and trains the system
"""

import requests
import json
import time
import os
from pathlib import Path

BASE_URL = "http://localhost:3000"

def train_pair(input_text, expected_answer, show_progress=True):
    """Train a single input/answer pair"""
    try:
        response = requests.post(
            f"{BASE_URL}/train",
            json={
                "input": input_text,
                "expected_answer": expected_answer
            },
            timeout=30
        )
        result = response.json()
        
        if show_progress:
            success = "✓" if result.get("success") else "✗"
            conf = result.get("confidence_score", 0) * 100
            print(f"  {success} Confidence: {conf:.1f}%")
        
        return result.get("success", False)
    except Exception as e:
        if show_progress:
            print(f"  ✗ Error: {e}")
        return False

def parse_training_file(filepath):
    """Parse a training data file and extract Q&A pairs"""
    pairs = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            i += 1
            continue
        
        # Look for Q: or Question: patterns
        if line.startswith('Q:') or line.startswith('Question:'):
            question = line.split(':', 1)[1].strip()
            i += 1
            
            # Look for A: or Answer: on next line
            if i < len(lines):
                next_line = lines[i].strip()
                if next_line.startswith('A:') or next_line.startswith('Answer:'):
                    answer = next_line.split(':', 1)[1].strip()
                    pairs.append((question, answer))
                    i += 1
                    continue
        
        # Look for Input:/Output: patterns
        if line.startswith('Input:'):
            input_text = line.split(':', 1)[1].strip()
            i += 1
            
            if i < len(lines):
                next_line = lines[i].strip()
                if next_line.startswith('Output:'):
                    output_text = next_line.split(':', 1)[1].strip()
                    pairs.append((input_text, output_text))
                    i += 1
                    continue
        
        # Look for Problem:/Solution: patterns
        if line.startswith('Problem:'):
            problem = line.split(':', 1)[1].strip()
            i += 1
            
            if i < len(lines):
                next_line = lines[i].strip()
                if next_line.startswith('Solution:'):
                    solution = next_line.split(':', 1)[1].strip()
                    pairs.append((problem, solution))
                    i += 1
                    continue
        
        i += 1
    
    return pairs

def train_from_file(filepath, category_name):
    """Train from a single file"""
    print(f"\n{'='*70}")
    print(f"Training from: {filepath.name}")
    print(f"Category: {category_name}")
    print(f"{'='*70}")
    
    pairs = parse_training_file(filepath)
    
    if not pairs:
        print(f"⚠️  No training pairs found in {filepath.name}")
        return 0, 0
    
    print(f"Found {len(pairs)} training pairs")
    
    successes = 0
    for idx, (input_text, expected_answer) in enumerate(pairs, 1):
        print(f"\n[{idx}/{len(pairs)}] {input_text[:60]}...")
        if train_pair(input_text, expected_answer):
            successes += 1
        time.sleep(0.1)  # Small delay to not overwhelm the server
    
    success_rate = (successes / len(pairs) * 100) if pairs else 0
    print(f"\n✓ Completed: {successes}/{len(pairs)} ({success_rate:.1f}% success rate)")
    
    return successes, len(pairs)

def main():
    print("="*70)
    print("  ALEN COMPREHENSIVE TRAINING")
    print("  Training with ALL available data")
    print("="*70)
    
    # Check server health
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding. Please start the server first.")
            return
        print("✓ Server is healthy\n")
    except:
        print("❌ Cannot connect to server. Please start the server first.")
        print("   Run: cargo run --release")
        return
    
    # Find all training data files
    training_dir = Path("training_data")
    if not training_dir.exists():
        print(f"❌ Training data directory not found: {training_dir}")
        return
    
    training_files = list(training_dir.glob("*.txt"))
    if not training_files:
        print(f"❌ No .txt files found in {training_dir}")
        return
    
    print(f"Found {len(training_files)} training files:")
    for f in training_files:
        print(f"  • {f.name}")
    
    # Train from each file
    total_successes = 0
    total_pairs = 0
    
    for filepath in sorted(training_files):
        category = filepath.stem.replace('_', ' ').title()
        successes, pairs = train_from_file(filepath, category)
        total_successes += successes
        total_pairs += pairs
    
    # Final summary
    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)
    print(f"Total training pairs: {total_pairs}")
    print(f"Successful trainings: {total_successes}")
    print(f"Success rate: {(total_successes/total_pairs*100):.1f}%")
    
    # Get final stats
    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=5)
        stats = response.json()
        
        print("\n" + "="*70)
        print("  SYSTEM STATISTICS")
        print("="*70)
        
        episodic = stats.get("episodic_memory", {})
        print(f"Episodic memory episodes: {episodic.get('total_episodes', 0)}")
        print(f"Verified episodes: {episodic.get('verified_episodes', 0)}")
        print(f"Average confidence: {episodic.get('average_confidence', 0)*100:.1f}%")
        
        semantic = stats.get("semantic_memory", {})
        print(f"Semantic facts: {semantic.get('total_facts', 0)}")
        
        print("\nOperators:")
        for op in stats.get("operator_stats", [])[:5]:
            print(f"  • {op['operator_type']}: {op['usage_count']} uses, {op['success_rate']*100:.0f}% success")
        
    except Exception as e:
        print(f"\n⚠️  Could not fetch final stats: {e}")
    
    print("\n✓ Training complete! Test the system:")
    print(f"  curl -X POST {BASE_URL}/chat -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"message\": \"Hello! How are you?\"}}'")
    print()

if __name__ == "__main__":
    main()

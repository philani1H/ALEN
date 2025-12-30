#!/usr/bin/env python3
"""
Comprehensive Model Training Script

This script trains ALEN with all available training data from the training_data directory.
It processes Q&A pairs and feeds them to the model for learning.
"""

import os
import re
import json
import subprocess
import sys
from pathlib import Path

TRAINING_DATA_DIR = "/workspace/training_data"
DATA_DIR = "/workspace/data"

def parse_qa_file(filepath):
    """Parse a Q&A training file into question-answer pairs."""
    pairs = []
    current_q = None
    current_a = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.rstrip()
        
        # Skip comments and empty lines at the start
        if line.startswith('#') or (not line and current_q is None):
            continue
        
        if line.startswith('Q:'):
            # Save previous Q&A pair
            if current_q and current_a:
                pairs.append({
                    'input': current_q,
                    'output': ' '.join(current_a)
                })
            current_q = line[2:].strip()
            current_a = []
        elif line.startswith('A:'):
            current_a.append(line[2:].strip())
        elif current_a is not None and line:  # Continuation of answer
            current_a.append(line.strip())
    
    # Don't forget the last pair
    if current_q and current_a:
        pairs.append({
            'input': current_q,
            'output': ' '.join(current_a)
        })
    
    return pairs

def main():
    print("=" * 60)
    print("ALEN Comprehensive Training")
    print("=" * 60)
    
    # Collect all training data
    all_pairs = []
    txt_files = sorted(Path(TRAINING_DATA_DIR).glob("*.txt"))
    
    print(f"\nFound {len(txt_files)} training files:")
    for filepath in txt_files:
        pairs = parse_qa_file(filepath)
        print(f"  - {filepath.name}: {len(pairs)} Q&A pairs")
        all_pairs.extend(pairs)
    
    # Also load JSON data files
    json_files = sorted(Path(DATA_DIR).glob("*.json"))
    print(f"\nFound {len(json_files)} JSON data files:")
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'input' in item and 'output' in item:
                        all_pairs.append(item)
                    elif isinstance(item, dict) and 'question' in item and 'answer' in item:
                        all_pairs.append({
                            'input': item['question'],
                            'output': item['answer']
                        })
                print(f"  - {filepath.name}: {len(data)} entries")
        except Exception as e:
            print(f"  - {filepath.name}: Error loading - {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Total training examples: {len(all_pairs)}")
    print(f"{'=' * 60}")
    
    # Save combined training data
    output_file = "/workspace/data/all_training_combined.json"
    with open(output_file, 'w') as f:
        json.dump(all_pairs, f, indent=2)
    print(f"\nSaved combined training data to: {output_file}")
    
    # Show some sample data
    print("\nSample training examples:")
    for i, pair in enumerate(all_pairs[:5]):
        print(f"\n{i+1}. Q: {pair['input'][:80]}...")
        print(f"   A: {pair['output'][:80]}...")
    
    print(f"\n{'=' * 60}")
    print("Training data ready!")
    print(f"{'=' * 60}")
    
    return all_pairs

if __name__ == "__main__":
    main()

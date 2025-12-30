#!/usr/bin/env python3
"""
Parse all training data files and convert to understanding-based format.
Extracts Q&A pairs that teach patterns, not memorization.
"""

import os
import re
import json
from typing import List, Tuple, Dict

def parse_qa_file(filepath: str) -> List[Tuple[str, str]]:
    """Parse a training file and extract Q&A pairs."""
    pairs = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip comments and empty lines
    lines = [line.strip() for line in content.split('\n') 
             if line.strip() and not line.strip().startswith('#')]
    
    # Try different formats
    
    # Format 1: "question -> answer" or "input -> output"
    for line in lines:
        if '->' in line or '→' in line:
            parts = re.split(r'\s*->\s*|\s*→\s*', line, maxsplit=1)
            if len(parts) == 2:
                q, a = parts[0].strip(), parts[1].strip()
                if q and a and len(q) > 3 and len(a) > 0:
                    pairs.append((q, a))
    
    # Format 2: "Q: ... A: ..." or "Question: ... Answer: ..."
    qa_pattern = r'(?:Q|Question|input):\s*(.+?)\s*(?:A|Answer|output|response):\s*(.+?)(?=(?:Q|Question|input):|$)'
    matches = re.findall(qa_pattern, content, re.IGNORECASE | re.DOTALL)
    for q, a in matches:
        q, a = q.strip(), a.strip()
        if q and a and len(q) > 3 and len(a) > 0:
            pairs.append((q, a))
    
    # Format 3: Alternating lines (question, answer, question, answer)
    if not pairs and len(lines) >= 2:
        for i in range(0, len(lines) - 1, 2):
            q, a = lines[i].strip(), lines[i+1].strip()
            # Check if it looks like Q&A
            if not q.startswith(('input:', 'reasoning:', 'behavior:')):
                if len(q) > 3 and len(a) > 0 and len(a) < 500:
                    pairs.append((q, a))
    
    return pairs

def parse_all_training_data(data_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """Parse all training data files."""
    all_data = {}
    
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)
            category = filename.replace('.txt', '').replace('_', ' ').title()
            
            print(f"Parsing {filename}...")
            pairs = parse_qa_file(filepath)
            
            if pairs:
                all_data[category] = pairs
                print(f"  Found {len(pairs)} Q&A pairs")
            else:
                print(f"  No Q&A pairs found")
    
    return all_data

def create_training_json(data: Dict[str, List[Tuple[str, str]]], output_file: str):
    """Create a JSON file with all training data."""
    training_data = []
    
    for category, pairs in data.items():
        for question, answer in pairs:
            training_data.append({
                "category": category,
                "question": question,
                "answer": answer
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nCreated {output_file} with {len(training_data)} examples")

def create_python_training_script(data: Dict[str, List[Tuple[str, str]]], output_file: str):
    """Create a Python training script with all data."""
    
    script = '''#!/usr/bin/env python3
"""
Comprehensive ALEN Training Script
Auto-generated from training_data folder

Trains the AI on understanding patterns across multiple domains.
"""

import requests
import json
import time
from typing import List, Tuple

BASE_URL = "http://localhost:3000"

def train_example(question: str, answer: str, category: str = None) -> dict:
    """Train on a single example."""
    try:
        response = requests.post(
            f"{BASE_URL}/train",
            json={
                "input": question,
                "target_answer": answer,
                "dimension": 128
            },
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

# Training data organized by category
TRAINING_DATA = {
'''
    
    # Add all training data
    for category, pairs in data.items():
        script += f'    "{category}": [\n'
        for q, a in pairs[:100]:  # Limit to 100 per category for manageability
            # Escape quotes
            q_escaped = q.replace('"', '\\"').replace('\n', ' ')
            a_escaped = a.replace('"', '\\"').replace('\n', ' ')
            script += f'        ("{q_escaped}", "{a_escaped}"),\n'
        script += '    ],\n'
    
    script += '''
}

def main():
    print("=" * 70)
    print("COMPREHENSIVE ALEN TRAINING")
    print("=" * 70)
    print()
    
    # Check server
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print("✓ Server is running")
    except:
        print("✗ Server is not running!")
        print("  Start with: cargo run --release")
        return
    
    print()
    print(f"Training on {sum(len(pairs) for pairs in TRAINING_DATA.values())} examples")
    print()
    
    total_success = 0
    total_attempts = 0
    
    for category, pairs in TRAINING_DATA.items():
        print(f"\\n{'='*70}")
        print(f"Category: {category}")
        print(f"{'='*70}")
        
        category_success = 0
        
        for i, (question, answer) in enumerate(pairs, 1):
            total_attempts += 1
            
            if i % 10 == 1:  # Print progress every 10 examples
                print(f"  [{i}/{len(pairs)}] Training...")
            
            result = train_example(question, answer, category)
            
            if result.get("success"):
                category_success += 1
                total_success += 1
            
            time.sleep(0.05)  # Brief pause
        
        print(f"  ✓ {category}: {category_success}/{len(pairs)} successful")
    
    print()
    print("=" * 70)
    print(f"TRAINING COMPLETE: {total_success}/{total_attempts} successful")
    print(f"Success rate: {100*total_success/total_attempts:.1f}%")
    print("=" * 70)
    print()
    print("Test the AI:")
    print(f"  curl -X POST {BASE_URL}/chat \\\\")
    print("    -H 'Content-Type: application/json' \\\\")
    print("    -d '{\\"message\\": \\"What is 5 plus 5?\\"}'")
    print()

if __name__ == "__main__":
    main()
'''
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(script)
    
    os.chmod(output_file, 0o755)
    print(f"Created {output_file}")

def main():
    print("=" * 70)
    print("TRAINING DATA PARSER")
    print("=" * 70)
    print()
    
    data_dir = "training_data"
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} directory not found")
        return
    
    # Parse all data
    all_data = parse_all_training_data(data_dir)
    
    if not all_data:
        print("No training data found!")
        return
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_pairs = sum(len(pairs) for pairs in all_data.values())
    print(f"Total categories: {len(all_data)}")
    print(f"Total Q&A pairs: {total_pairs}")
    print()
    
    for category, pairs in sorted(all_data.items(), key=lambda x: -len(x[1])):
        print(f"  {category:40s} {len(pairs):4d} pairs")
    
    print()
    
    # Create outputs
    create_training_json(all_data, "training_data_compiled.json")
    create_python_training_script(all_data, "train_comprehensive_understanding.py")
    
    print()
    print("=" * 70)
    print("READY TO TRAIN")
    print("=" * 70)
    print()
    print("Run training with:")
    print("  python3 train_comprehensive_understanding.py")
    print()

if __name__ == "__main__":
    main()

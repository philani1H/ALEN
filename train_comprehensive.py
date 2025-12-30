#!/usr/bin/env python3
"""
Comprehensive Training Script with Backward Verification
Trains ALEN on ALL data with proof-of-understanding before learning
"""

import requests
import json
import time
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# API endpoint
API_BASE = "http://localhost:3000"

class TrainingStats:
    def __init__(self):
        self.total = 0
        self.verified = 0
        self.failed = 0
        self.by_category = {}
    
    def add(self, category: str, verified: bool):
        self.total += 1
        if verified:
            self.verified += 1
        else:
            self.failed += 1
        
        if category not in self.by_category:
            self.by_category[category] = {"total": 0, "verified": 0}
        self.by_category[category]["total"] += 1
        if verified:
            self.by_category[category]["verified"] += 1
    
    def print_summary(self):
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total Examples: {self.total}")
        print(f"Verified: {self.verified} ({self.verified*100//self.total if self.total > 0 else 0}%)")
        print(f"Failed: {self.failed}")
        print("\nBy Category:")
        for cat, stats in self.by_category.items():
            pct = stats["verified"]*100//stats["total"] if stats["total"] > 0 else 0
            print(f"  {cat:30s}: {stats['verified']:4d}/{stats['total']:4d} ({pct:3d}%)")

def check_server():
    """Check if ALEN server is running"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def parse_training_line(line: str) -> Tuple[str, str]:
    """Parse 'input -> expected_answer' format"""
    if '->' not in line:
        return None, None
    
    parts = line.split('->', 1)
    if len(parts) != 2:
        return None, None
    
    input_text = parts[0].strip()
    expected = parts[1].strip()
    
    return input_text, expected

def train_example(input_text: str, expected: str, context: str) -> bool:
    """Train on a single example with backward verification"""
    try:
        response = requests.post(
            f"{API_BASE}/train",
            json={
                "input": input_text,
                "expected_answer": expected,
                "context": context
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("success", False)
        return False
    except Exception as e:
        print(f"    ⚠️  Error training: {e}")
        return False

def train_file(filepath: Path, category: str, stats: TrainingStats):
    """Train on all examples in a file"""
    print(f"\n📚 Training: {category}")
    print(f"   File: {filepath.name}")
    
    if not filepath.exists():
        print(f"   ⚠️  File not found, skipping")
        return
    
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse line
            input_text, expected = parse_training_line(line)
            if not input_text or not expected:
                continue
            
            # Train with backward verification
            verified = train_example(input_text, expected, category)
            stats.add(category, verified)
            count += 1
            
            # Progress indicator
            if count % 10 == 0:
                print(".", end="", flush=True)
    
    print()
    cat_stats = stats.by_category[category]
    pct = cat_stats["verified"]*100//cat_stats["total"] if cat_stats["total"] > 0 else 0
    print(f"   ✅ Trained: {cat_stats['total']} examples")
    print(f"   ✓  Verified: {cat_stats['verified']} ({pct}%)")
    if cat_stats["total"] - cat_stats["verified"] > 0:
        print(f"   ⚠️  Failed verification: {cat_stats['total'] - cat_stats['verified']}")

def main():
    print("="*60)
    print("ALEN Comprehensive Training with Backward Verification")
    print("="*60)
    print()
    
    # Check server
    print("Checking server status...")
    if not check_server():
        print("❌ Error: ALEN server is not running on port 3000")
        print("Please start the server first: cargo run --release")
        sys.exit(1)
    
    print("✅ Server is running")
    
    # Training data directory
    data_dir = Path("training_data")
    if not data_dir.exists():
        print(f"❌ Error: {data_dir} directory not found")
        sys.exit(1)
    
    stats = TrainingStats()
    
    # Training order (most important first)
    training_files = [
        # Core thinking and reasoning
        ("all_thinking_types.txt", "All Thinking Types"),
        ("advanced_reasoning.txt", "Advanced Reasoning"),
        ("reasoning_patterns.txt", "Reasoning Patterns"),
        
        # Conversations
        ("comprehensive_conversations.txt", "Comprehensive Conversations"),
        ("enhanced_conversations.txt", "Enhanced Conversations"),
        ("conversation_skills.txt", "Conversation Skills"),
        ("conversations.txt", "Basic Conversations"),
        ("advanced_qa.txt", "Advanced Q&A"),
        
        # Emotional and social intelligence
        ("emotional_intelligence.txt", "Emotional Intelligence"),
        ("personality_personalization.txt", "Personality & Personalization"),
        ("manners_etiquette.txt", "Manners & Etiquette"),
        
        # Knowledge domains
        ("mathematics.txt", "Mathematics"),
        ("math_fundamentals.txt", "Math Fundamentals"),
        ("science.txt", "Science"),
        ("general_knowledge.txt", "General Knowledge"),
        ("geography.txt", "Geography"),
        ("programming.txt", "Programming"),
    ]
    
    print("\nStarting comprehensive training...")
    start_time = time.time()
    
    for filename, category in training_files:
        filepath = data_dir / filename
        train_file(filepath, category, stats)
    
    elapsed = time.time() - start_time
    
    # Print summary
    stats.print_summary()
    print(f"\nTime elapsed: {elapsed:.1f} seconds")
    
    # Get system statistics
    print("\n" + "="*60)
    print("SYSTEM STATISTICS")
    print("="*60)
    
    try:
        # Overall stats
        response = requests.get(f"{API_BASE}/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"\nSystem Stats:")
            print(f"  Dimension: {data.get('dimension', 'N/A')}")
            print(f"  Total Inferences: {data.get('total_inferences', 0)}")
        
        # Operator performance
        response = requests.get(f"{API_BASE}/operators")
        if response.status_code == 200:
            data = response.json()
            print(f"\n🧠 Top Operators:")
            operators = data.get('operators', [])
            for op in sorted(operators, key=lambda x: x.get('success_rate', 0), reverse=True)[:5]:
                print(f"  {op['name']:20s}: {op.get('success_rate', 0)*100:.1f}% success, "
                      f"{op.get('avg_confidence', 0)*100:.1f}% confidence")
        
        # Memory stats
        response = requests.get(f"{API_BASE}/memory/episodic/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"\n💾 Episodic Memory:")
            print(f"  Total Episodes: {data.get('total_episodes', 0)}")
            print(f"  Verified: {data.get('verified_episodes', 0)}")
            print(f"  Avg Confidence: {data.get('avg_confidence', 0)*100:.1f}%")
    
    except Exception as e:
        print(f"⚠️  Could not fetch system stats: {e}")
    
    print("\n" + "="*60)
    print("✅ Training complete! The model is ready for conversations.")
    print("="*60)
    print("\nTest it with:")
    print('  curl -X POST http://localhost:3000/chat -H "Content-Type: application/json" '
          '-d \'{"message": "How are you?"}\'')
    print()

if __name__ == "__main__":
    main()

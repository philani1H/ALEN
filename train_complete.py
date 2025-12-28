#!/usr/bin/env python3
"""
Complete ALEN Training Script
Trains ALEN with comprehensive knowledge including:
- Manners and etiquette
- Personality and personalization
- Emotional intelligence
- Conversation skills
- General knowledge (math, science, programming, etc.)
"""

import requests
import json
import time
from pathlib import Path

API_URL = "http://localhost:3000"

def load_training_file(filepath):
    """Load and parse training file"""
    print(f"\n📖 Loading: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    facts = []
    for line in lines:
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        facts.append(line)

    print(f"   ✓ Loaded {len(facts)} facts")
    return facts

def train_batch(facts, category, batch_size=50):
    """Train ALEN with batch of facts"""
    print(f"\n🎓 Training {category}...")
    print(f"   Total facts: {len(facts)}")

    # Split into batches
    total_batches = (len(facts) + batch_size - 1) // batch_size
    trained_count = 0

    for i in range(0, len(facts), batch_size):
        batch = facts[i:i + batch_size]
        batch_num = (i // batch_size) + 1

        try:
            response = requests.post(
                f"{API_URL}/train/batch",
                json={
                    "facts": batch,
                    "confidence": 0.95
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                trained_count += len(batch)
                print(f"   ✓ Batch {batch_num}/{total_batches}: {len(batch)} facts trained")
            else:
                print(f"   ✗ Batch {batch_num} failed: {response.status_code}")

        except Exception as e:
            print(f"   ✗ Error training batch {batch_num}: {str(e)}")

        # Small delay between batches
        time.sleep(0.5)

    print(f"   ✅ Completed {category}: {trained_count}/{len(facts)} facts trained")
    return trained_count

def main():
    print("="*70)
    print("🚀 ALEN Complete Training System")
    print("="*70)

    # Check if server is running
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server is running")
        else:
            print("✗ Server returned unexpected status")
            return
    except Exception as e:
        print(f"✗ Cannot connect to server at {API_URL}")
        print(f"  Please start the server first: cargo run --release")
        return

    # Training data files
    training_files = [
        # Social and Emotional Skills
        ("training_data/manners_etiquette.txt", "Manners & Etiquette"),
        ("training_data/personality_personalization.txt", "Personality & Personalization"),
        ("training_data/emotional_intelligence.txt", "Emotional Intelligence"),
        ("training_data/conversation_skills.txt", "Conversation Skills"),

        # Knowledge Domains
        ("training_data/general_knowledge.txt", "General Knowledge"),
        ("training_data/science.txt", "Science"),
        ("training_data/mathematics.txt", "Mathematics"),
        ("training_data/programming.txt", "Programming"),
        ("training_data/geography.txt", "Geography"),
        ("training_data/conversations.txt", "Human Conversations"),
    ]

    total_facts_trained = 0
    successful_categories = 0

    print("\n" + "="*70)
    print("📚 Training Categories")
    print("="*70)

    for filepath, category in training_files:
        if Path(filepath).exists():
            facts = load_training_file(filepath)
            if facts:
                count = train_batch(facts, category)
                total_facts_trained += count
                if count > 0:
                    successful_categories += 1
        else:
            print(f"\n⚠️  File not found: {filepath}")

    # Summary
    print("\n" + "="*70)
    print("📊 Training Summary")
    print("="*70)
    print(f"Categories trained: {successful_categories}/{len(training_files)}")
    print(f"Total facts trained: {total_facts_trained}")

    # Test trained knowledge
    print("\n" + "="*70)
    print("🧪 Testing Trained Knowledge")
    print("="*70)

    test_queries = [
        ("What is please used for?", "Manners"),
        ("How do I show genuine interest?", "Conversation"),
        ("How do I validate someone's feelings?", "Emotional Intelligence"),
        ("What does remembering names show?", "Personalization"),
        ("What is Python?", "Programming"),
        ("What is photosynthesis?", "Science"),
    ]

    for query, category in test_queries:
        try:
            response = requests.post(
                f"{API_URL}/generate/factual",
                json={
                    "query": query,
                    "max_tokens": 50,
                    "mode": "balanced"
                },
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "")
                verified_pct = result.get("verification_percentage", 0)

                print(f"\n❓ [{category}] {query}")
                print(f"💬 {answer}")
                print(f"✓ Verification: {verified_pct:.0f}%")
            else:
                print(f"\n❓ [{category}] {query}")
                print(f"✗ Query failed: {response.status_code}")

        except Exception as e:
            print(f"\n❓ [{category}] {query}")
            print(f"✗ Error: {str(e)}")

    # Export trained knowledge
    print("\n" + "="*70)
    print("💾 Exporting Trained Knowledge")
    print("="*70)

    try:
        response = requests.post(
            f"{API_URL}/export/semantic",
            json={"output_path": "data/exports/trained_knowledge.json"},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✓ Exported to: {result.get('file_path')}")
            print(f"  Items: {result.get('items_exported', 0)}")
            print(f"  Size: {result.get('file_size_kb', 0):.1f} KB")
        else:
            print(f"✗ Export failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Export error: {str(e)}")

    print("\n" + "="*70)
    print("🎉 Training Complete!")
    print("="*70)
    print("\n💡 Next Steps:")
    print("   1. Chat with ALEN: http://localhost:3000 (Click 'Chat' tab)")
    print("   2. Test personality: Ask 'How should I greet someone?'")
    print("   3. Test manners: Ask 'What is proper table etiquette?'")
    print("   4. Test emotions: Ask 'How do I comfort a sad friend?'")
    print("   5. Test conversation: Ask 'How do I start a conversation?'")
    print("\n🚀 ALEN is now trained with comprehensive social and emotional skills!")

if __name__ == "__main__":
    main()

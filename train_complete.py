#!/usr/bin/env python3
"""
Complete ALEN Training Script
Trains ALEN with comprehensive knowledge including:
- Manners and etiquette
- Personality and personalization
- Emotional intelligence
- Conversation skills
- Math fundamentals and reasoning
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

def parse_fact(fact_text):
    """Parse fact text into concept and content"""
    # Handle different formats:
    # "concept means explanation"
    # "concept is explanation"
    # "concept shows explanation"
    # "concept indicates explanation"

    separators = [' means ', ' is ', ' shows ', ' indicates ', ' demonstrates ',
                  ' expresses ', ' creates ', ' allows ', ' encourages ', ' validates ',
                  ' acknowledges ', ' represents ', ' refers ', ' compares ', ' maintains ']

    for sep in separators:
        if sep in fact_text:
            parts = fact_text.split(sep, 1)
            if len(parts) == 2:
                concept = parts[0].strip()
                content = fact_text  # Full sentence as content
                return {"concept": concept, "content": content}

    # If no separator found, use first 3 words as concept, full as content
    words = fact_text.split()
    if len(words) > 3:
        concept = ' '.join(words[:3])
    else:
        concept = fact_text

    return {"concept": concept, "content": fact_text}

def train_knowledge(facts, category, batch_size=100):
    """Train ALEN with knowledge facts"""
    print(f"\n🎓 Training {category}...")
    print(f"   Total facts: {len(facts)}")

    # Parse all facts
    parsed_facts = [parse_fact(f) for f in facts]

    # Split into batches
    total_batches = (len(parsed_facts) + batch_size - 1) // batch_size
    trained_count = 0

    for i in range(0, len(parsed_facts), batch_size):
        batch = parsed_facts[i:i + batch_size]
        batch_num = (i // batch_size) + 1

        try:
            response = requests.post(
                f"{API_URL}/learn",
                json={
                    "category": category,
                    "facts": batch
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                stored = result.get("facts_stored", 0)
                trained_count += stored
                print(f"   ✓ Batch {batch_num}/{total_batches}: {stored} facts stored")
            else:
                print(f"   ✗ Batch {batch_num} failed: {response.status_code}")
                # print(f"      Response: {response.text[:200]}")

        except Exception as e:
            print(f"   ✗ Error training batch {batch_num}: {str(e)}")

        # Small delay between batches
        time.sleep(0.3)

    print(f"   ✅ Completed {category}: {trained_count}/{len(facts)} facts stored")
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
        ("training_data/manners_etiquette.txt", "manners-etiquette"),
        ("training_data/personality_personalization.txt", "personality"),
        ("training_data/emotional_intelligence.txt", "emotional-intelligence"),
        ("training_data/conversation_skills.txt", "conversation-skills"),

        # Knowledge Domains
        ("training_data/general_knowledge.txt", "general-knowledge"),
        ("training_data/science.txt", "science"),
        ("training_data/mathematics.txt", "mathematics"),
        ("training_data/math_fundamentals.txt", "math-fundamentals"),
        ("training_data/programming.txt", "programming"),
        ("training_data/geography.txt", "geography"),
        ("training_data/conversations.txt", "conversations"),
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
                count = train_knowledge(facts, category)
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
        ("Why does 2 equal 2?", "Math Fundamentals"),
        ("Why does 2 plus 2 equal 4?", "Math Reasoning"),
        ("What is the distributive property?", "Math Properties"),
        ("What is a mathematical formula?", "Formula Construction"),
    ]

    for query, category in test_queries:
        try:
            response = requests.post(
                f"{API_URL}/generate/factual",
                json={
                    "query": query,
                    "max_tokens": 80,
                    "mode": "balanced"
                },
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "")
                verified_pct = result.get("verification_percentage", 0)

                print(f"\n❓ [{category}] {query}")
                print(f"💬 {answer[:150]}")
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
    print("   6. Test math: Ask 'Why does 2 equal 2?'")
    print("\n🚀 ALEN is now trained with comprehensive knowledge!")

if __name__ == "__main__":
    main()

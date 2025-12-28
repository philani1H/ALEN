#!/usr/bin/env python3
"""
ALEN Training Script

This script trains ALEN on multiple datasets and monitors:
- Training success rates
- Mood changes during training
- Confidence scores
- Energy levels

Usage:
    python3 train_alen.py [--domain DOMAIN] [--epochs EPOCHS]

Examples:
    python3 train_alen.py --domain mathematics --epochs 3
    python3 train_alen.py --domain all --epochs 2
"""

import requests
import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import sys

# API Configuration
API_URL = "http://localhost:3000"

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print colored header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def check_server() -> bool:
    """Check if ALEN server is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'healthy':
                print_success("ALEN server is running and healthy")
                return True
    except requests.exceptions.RequestException:
        pass

    print_error("ALEN server is not running!")
    print_info("Start the server with: cargo run --release")
    return False

def get_mood_state() -> Dict:
    """Get current mood and emotion state"""
    try:
        response = requests.get(f"{API_URL}/emotions/state")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}

def print_mood_state(mood_data: Dict):
    """Print formatted mood state"""
    if not mood_data:
        return

    mood = mood_data.get('mood', {})
    emotion = mood_data.get('emotion', {})

    print(f"\n{Colors.BOLD}Current Emotional State:{Colors.END}")
    print(f"  Mood: {Colors.CYAN}{mood.get('current_mood', 'Unknown')}{Colors.END}")
    print(f"  Emotion: {Colors.CYAN}{emotion.get('current_emotion', 'Unknown')}{Colors.END}")
    print(f"  Reward (Dopamine): {mood.get('reward_level', 0):.2f}")
    print(f"  Stress (Cortisol): {mood.get('stress_level', 0):.2f}")
    print(f"  Curiosity: {mood.get('curiosity_level', 0):.2f}")
    print(f"  Energy: {mood.get('energy_level', 0):.2f}")
    print(f"  Perception Bias: {mood.get('perception_bias', 0):.3f}")
    print(f"  Reaction Threshold: {mood.get('reaction_threshold', 0):.3f}")

def load_training_data(filepath: str) -> List[Tuple[str, str]]:
    """Load training data from file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse "input -> answer" format
            if ' -> ' in line:
                input_text, answer = line.split(' -> ', 1)
                data.append((input_text.strip(), answer.strip()))

    return data

def train_single(input_text: str, expected_answer: str, context: List[str] = None) -> Dict:
    """Train ALEN on a single example"""
    payload = {
        "input": input_text,
        "expected_answer": expected_answer,
        "context": context or []
    }

    try:
        response = requests.post(f"{API_URL}/train", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        print_error(f"Request failed: {e}")

    return {"success": False}

def train_batch(examples: List[Tuple[str, str]]) -> Dict:
    """Train ALEN on batch of examples"""
    problems = [
        {"input": inp, "expected_answer": ans}
        for inp, ans in examples
    ]

    payload = {"problems": problems}

    try:
        response = requests.post(f"{API_URL}/train/batch", json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        print_error(f"Batch training failed: {e}")

    return {"total_problems": len(problems), "successes": 0, "failures": len(problems)}

def train_domain(domain: str, epochs: int = 1, batch_size: int = 10):
    """Train ALEN on a specific domain"""
    data_file = Path(f"training_data/{domain}.txt")

    if not data_file.exists():
        print_error(f"Training file not found: {data_file}")
        return

    print_header(f"Training on {domain.upper()}")

    # Load data
    examples = load_training_data(data_file)
    print_info(f"Loaded {len(examples)} training examples")

    # Get initial mood state
    print_info("Initial mood state:")
    initial_mood = get_mood_state()
    print_mood_state(initial_mood)

    # Training statistics
    total_trained = 0
    total_success = 0
    total_failed = 0
    avg_confidence = 0.0
    avg_energy = 0.0

    # Train for multiple epochs
    for epoch in range(1, epochs + 1):
        print(f"\n{Colors.BOLD}Epoch {epoch}/{epochs}{Colors.END}")

        # Train in batches
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]

            print(f"\nTraining batch {i//batch_size + 1}/{(len(examples)-1)//batch_size + 1} "
                  f"({len(batch)} examples)...", end=' ')

            result = train_batch(batch)

            successes = result.get('successes', 0)
            failures = result.get('failures', 0)

            total_trained += len(batch)
            total_success += successes
            total_failed += failures

            if successes == len(batch):
                print_success(f"All {len(batch)} examples learned!")
            elif successes > 0:
                print_warning(f"{successes}/{len(batch)} learned")
            else:
                print_error(f"Failed to learn batch")

            # Small delay to avoid overwhelming the server
            time.sleep(0.1)

    # Get final mood state
    print_info("\nFinal mood state:")
    final_mood = get_mood_state()
    print_mood_state(final_mood)

    # Print summary
    print_header("Training Summary")
    print(f"Domain: {Colors.CYAN}{domain}{Colors.END}")
    print(f"Total Examples: {total_trained}")
    print(f"Successful: {Colors.GREEN}{total_success}{Colors.END}")
    print(f"Failed: {Colors.RED}{total_failed}{Colors.END}")
    print(f"Success Rate: {Colors.GREEN}{(total_success/total_trained*100):.1f}%{Colors.END}")

    # Mood change analysis
    if initial_mood and final_mood:
        initial_reward = initial_mood.get('mood', {}).get('reward_level', 0.5)
        final_reward = final_mood.get('mood', {}).get('reward_level', 0.5)
        reward_change = final_reward - initial_reward

        initial_stress = initial_mood.get('mood', {}).get('stress_level', 0.3)
        final_stress = final_mood.get('mood', {}).get('stress_level', 0.3)
        stress_change = final_stress - initial_stress

        print(f"\n{Colors.BOLD}Mood Changes:{Colors.END}")
        if reward_change > 0:
            print(f"  Reward: {Colors.GREEN}+{reward_change:.3f} (happier!){Colors.END}")
        else:
            print(f"  Reward: {Colors.RED}{reward_change:.3f}{Colors.END}")

        if stress_change < 0:
            print(f"  Stress: {Colors.GREEN}{stress_change:.3f} (less stressed!){Colors.END}")
        else:
            print(f"  Stress: {Colors.YELLOW}+{stress_change:.3f}{Colors.END}")

def test_knowledge(domain: str):
    """Test ALEN's knowledge in a domain"""
    data_file = Path(f"training_data/{domain}.txt")

    if not data_file.exists():
        print_error(f"Test file not found: {data_file}")
        return

    print_header(f"Testing Knowledge: {domain.upper()}")

    examples = load_training_data(data_file)

    # Test random samples
    import random
    test_samples = random.sample(examples, min(10, len(examples)))

    correct = 0
    for input_text, expected in test_samples:
        try:
            response = requests.post(f"{API_URL}/infer",
                                    json={"input": input_text},
                                    timeout=10)
            if response.status_code == 200:
                result = response.json()
                confidence = result.get('confidence', 0)

                print(f"\nQ: {Colors.CYAN}{input_text}{Colors.END}")
                print(f"Expected: {Colors.GREEN}{expected}{Colors.END}")
                print(f"Confidence: {confidence:.2f}")

                if confidence > 0.7:
                    print_success("ALEN knows this!")
                    correct += 1
                else:
                    print_warning("ALEN is uncertain")
        except:
            print_error(f"Failed to test: {input_text}")

    print(f"\n{Colors.BOLD}Test Results: {correct}/{len(test_samples)} high confidence{Colors.END}")

def main():
    parser = argparse.ArgumentParser(description="Train ALEN on various datasets")
    parser.add_argument('--domain', default='all',
                       choices=['all', 'mathematics', 'geography', 'science',
                               'programming', 'general_knowledge'],
                       help='Domain to train on')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for training')
    parser.add_argument('--test', action='store_true',
                       help='Test knowledge after training')

    args = parser.parse_args()

    # Check server
    if not check_server():
        sys.exit(1)

    # Determine domains to train
    if args.domain == 'all':
        domains = ['mathematics', 'geography', 'science', 'programming', 'general_knowledge']
    else:
        domains = [args.domain]

    # Train each domain
    for domain in domains:
        train_domain(domain, args.epochs, args.batch_size)

        if args.test:
            time.sleep(1)
            test_knowledge(domain)

        time.sleep(2)  # Pause between domains

    # Final system stats
    print_header("Final System Statistics")
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"Total Episodes: {stats['episodic_memory']['total_episodes']}")
            print(f"Verified Episodes: {stats['episodic_memory']['verified_episodes']}")
            print(f"Knowledge Facts: {stats['semantic_memory']['total_facts']}")
            print(f"Learning Rate: {stats['learning_rate']}")
            print(f"System Confidence: {stats['control_state']['confidence']:.2f}")
    except:
        print_error("Could not fetch system stats")

    print_success("\nTraining complete! ALEN is now smarter! 🎓")

if __name__ == "__main__":
    main()

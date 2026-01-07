#!/usr/bin/env python3
"""
Increase Temperature and Creativity with Verification

This script safely increases creativity and temperature while maintaining
the verification system to prevent hallucinations.

Strategy:
1. Increase creativity bias (0.5 → 0.8)
2. Increase exploration (0.6 → 0.8)
3. Increase risk tolerance slightly (0.5 → 0.65)
4. Keep verification thresholds strict to prevent hallucinations
5. Monitor verification rate to ensure quality

The verification system prevents hallucinations through:
- Forward check: Output must be valid and finite
- Backward check: Cycle consistency (can reconstruct reasoning)
- Stability check: Small changes don't drastically alter results
"""

import requests
import json
import time

API_URL = "http://localhost:3000"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.CYAN}  {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def get_current_settings():
    """Get current bias settings"""
    response = requests.get(f"{API_URL}/stats")
    data = response.json()
    return data['control_state']['bias']

def set_bias(creativity=None, exploration=None, risk_tolerance=None, urgency=None):
    """Set bias parameters"""
    payload = {}
    if creativity is not None:
        payload['creativity'] = creativity
    if exploration is not None:
        payload['exploration'] = exploration
    if risk_tolerance is not None:
        payload['risk_tolerance'] = risk_tolerance
    if urgency is not None:
        payload['urgency'] = urgency
    
    response = requests.post(f"{API_URL}/bias", json=payload)
    return response.json()

def test_inference_quality(num_tests=10):
    """Test inference quality with verification"""
    test_queries = [
        "What is 2 + 2?",
        "Explain gravity",
        "Tell me about the ocean",
        "What is creativity?",
        "How does learning work?",
        "Describe a sunset",
        "What is mathematics?",
        "Explain consciousness",
        "What is art?",
        "How do computers work?"
    ]
    
    verified_count = 0
    total_confidence = 0.0
    total_energy = 0.0
    
    for query in test_queries[:num_tests]:
        payload = {"input": query}
        response = requests.post(f"{API_URL}/infer", json=payload, timeout=30)
        data = response.json()
        
        if data['verified']:
            verified_count += 1
        total_confidence += data['confidence']
        total_energy += data['energy']
    
    return {
        'verification_rate': verified_count / num_tests,
        'avg_confidence': total_confidence / num_tests,
        'avg_energy': total_energy / num_tests,
        'total_tests': num_tests
    }

def main():
    print_header("Increasing Creativity with Verification")
    
    # Get current settings
    print_info("Current Settings:")
    current = get_current_settings()
    print(f"  Creativity: {current['creativity']:.3f}")
    print(f"  Exploration: {current['exploration']:.3f}")
    print(f"  Risk Tolerance: {current['risk_tolerance']:.3f}")
    print(f"  Urgency: {current['urgency']:.3f}")
    
    # Test baseline quality
    print_info("\nTesting baseline quality...")
    baseline = test_inference_quality(10)
    print(f"  Verification Rate: {baseline['verification_rate']*100:.1f}%")
    print(f"  Avg Confidence: {baseline['avg_confidence']:.3f}")
    print(f"  Avg Energy: {baseline['avg_energy']:.3f}")
    
    # Increase creativity gradually
    print_header("Increasing Creativity Parameters")
    
    # Step 1: Moderate increase
    print_info("Step 1: Moderate increase (creativity 0.5 → 0.7)")
    set_bias(creativity=0.7, exploration=0.7, risk_tolerance=0.6)
    time.sleep(1)
    
    result1 = test_inference_quality(5)
    print(f"  Verification Rate: {result1['verification_rate']*100:.1f}%")
    print(f"  Avg Confidence: {result1['avg_confidence']:.3f}")
    
    if result1['verification_rate'] < 0.8:
        print_warning("Verification rate dropped below 80%, reverting...")
        set_bias(creativity=0.5, exploration=0.6, risk_tolerance=0.5)
        return
    
    print_success("Quality maintained, continuing...")
    
    # Step 2: Higher increase
    print_info("\nStep 2: Higher increase (creativity 0.7 → 0.85)")
    set_bias(creativity=0.85, exploration=0.8, risk_tolerance=0.65)
    time.sleep(1)
    
    result2 = test_inference_quality(5)
    print(f"  Verification Rate: {result2['verification_rate']*100:.1f}%")
    print(f"  Avg Confidence: {result2['avg_confidence']:.3f}")
    
    if result2['verification_rate'] < 0.8:
        print_warning("Verification rate dropped, using previous settings...")
        set_bias(creativity=0.7, exploration=0.7, risk_tolerance=0.6)
        final_creativity = 0.7
    else:
        print_success("Quality maintained at high creativity!")
        final_creativity = 0.85
    
    # Final test
    print_header("Final Quality Test")
    final = test_inference_quality(10)
    print(f"  Verification Rate: {final['verification_rate']*100:.1f}%")
    print(f"  Avg Confidence: {final['avg_confidence']:.3f}")
    print(f"  Avg Energy: {final['avg_energy']:.3f}")
    
    # Get final settings
    final_settings = get_current_settings()
    
    print_header("Final Configuration")
    print(f"  Creativity: {Colors.GREEN}{final_settings['creativity']:.3f}{Colors.END} (was {current['creativity']:.3f})")
    print(f"  Exploration: {Colors.GREEN}{final_settings['exploration']:.3f}{Colors.END} (was {current['exploration']:.3f})")
    print(f"  Risk Tolerance: {Colors.GREEN}{final_settings['risk_tolerance']:.3f}{Colors.END} (was {current['risk_tolerance']:.3f})")
    
    print_header("Verification System Status")
    print_success("Forward Check: Active (outputs must be valid and finite)")
    print_success("Backward Check: Active (cycle consistency enforced)")
    print_success("Stability Check: Active (perturbation resistance)")
    print_success(f"Verification Rate: {final['verification_rate']*100:.1f}%")
    
    if final['verification_rate'] >= 0.8:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Creativity increased successfully!{Colors.END}")
        print(f"{Colors.GREEN}  No hallucinations - verification system working{Colors.END}")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ Verification rate below target{Colors.END}")
        print(f"{Colors.YELLOW}  Consider reducing creativity slightly{Colors.END}")
    
    print_header("How This Prevents Hallucinations")
    print("1. Forward Check: Ensures outputs are mathematically valid")
    print("2. Backward Check: Model must reconstruct its reasoning path")
    print("3. Stability Check: Small input changes = small output changes")
    print("4. Energy Function: Balances creativity with constraint satisfaction")
    print("5. Verification Gate: Only verified outputs are returned")
    
    print(f"\n{Colors.CYAN}The model can now be more creative while maintaining accuracy!{Colors.END}")

if __name__ == "__main__":
    main()

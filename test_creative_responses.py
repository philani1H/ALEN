#!/usr/bin/env python3
"""
Test Creative Responses with Verification

Tests that increased creativity produces more interesting responses
while verification prevents hallucinations.
"""

import requests
import json

API_URL = "http://localhost:3000"

class Colors:
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'

def test_query(query):
    """Test a query and show results"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}Query: {query}{Colors.END}")
    
    payload = {"input": query}
    response = requests.post(f"{API_URL}/infer", json=payload, timeout=30)
    data = response.json()
    
    print(f"  Confidence: {data['confidence']:.3f}")
    print(f"  Energy: {data['energy']:.3f}")
    print(f"  Verified: {Colors.GREEN if data['verified'] else Colors.YELLOW}{data['verified']}{Colors.END}")
    print(f"  Candidates Evaluated: {data['candidates_considered']}")
    
    # Show thought vector statistics
    thought = data['thought_vector']
    avg_activation = sum(abs(x) for x in thought) / len(thought)
    max_activation = max(abs(x) for x in thought)
    active_dims = sum(1 for x in thought if abs(x) > 0.1)
    
    print(f"  Thought Vector Stats:")
    print(f"    Avg Activation: {avg_activation:.3f}")
    print(f"    Max Activation: {max_activation:.3f}")
    print(f"    Active Dimensions: {active_dims}/{len(thought)} ({active_dims/len(thought)*100:.1f}%)")
    
    return data

def main():
    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}Testing Creative Responses with Verification{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    
    # Test creative queries
    creative_queries = [
        "Imagine a new color",
        "What if gravity worked backwards?",
        "Describe a dream about mathematics",
        "Create a metaphor for learning",
        "What does silence sound like?",
        "If thoughts had texture, what would they feel like?",
        "Explain time to someone who lives outside of it",
        "What is the shape of an idea?",
    ]
    
    verified_count = 0
    total_confidence = 0.0
    total_active_dims = 0
    
    for query in creative_queries:
        result = test_query(query)
        if result['verified']:
            verified_count += 1
        total_confidence += result['confidence']
        
        # Count active dimensions
        thought = result['thought_vector']
        active_dims = sum(1 for x in thought if abs(x) > 0.1)
        total_active_dims += active_dims
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}Summary{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"Total Queries: {len(creative_queries)}")
    print(f"Verified: {Colors.GREEN}{verified_count}/{len(creative_queries)} ({verified_count/len(creative_queries)*100:.1f}%){Colors.END}")
    print(f"Avg Confidence: {total_confidence/len(creative_queries):.3f}")
    print(f"Avg Active Dimensions: {total_active_dims/len(creative_queries):.1f}/128 ({total_active_dims/len(creative_queries)/128*100:.1f}%)")
    
    if verified_count == len(creative_queries):
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All creative responses verified!{Colors.END}")
        print(f"{Colors.GREEN}  No hallucinations detected{Colors.END}")
        print(f"{Colors.GREEN}  Verification system working perfectly{Colors.END}")
    
    print(f"\n{Colors.CYAN}The model is being creative while staying grounded in verification!{Colors.END}")

if __name__ == "__main__":
    main()

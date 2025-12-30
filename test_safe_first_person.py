#!/usr/bin/env python3
"""Test safe first-person logic manually"""

# Simulate the validation logic
def validate_output(output):
    tokens = output.split()
    
    forbidden_mental_states = {
        "feel", "feeling", "felt", "want", "wanting", "wanted",
        "believe", "believing", "believed", "hope", "hoping", "hoped",
        "care", "caring", "cared", "wish", "wishing", "wished",
        "desire", "desiring", "desired", "love", "loving", "loved",
        "hate", "hating", "hated", "fear", "fearing", "feared"
    }
    
    scope_limiters = {
        "based on", "in this context", "with the information",
        "from my training", "according to", "given",
        "in this conversation", "as an AI", "as a system"
    }
    
    has_first_person = False
    has_mental_state_violation = False
    has_scope_limiter = False
    
    for i, token in enumerate(tokens):
        # Check for first-person
        if token == "I" or token.startswith("I'"):
            has_first_person = True
            
            # Check for mental state violation
            if i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token.lower() in forbidden_mental_states:
                    has_mental_state_violation = True
        
        # Check for scope limiters (case-insensitive)
        output_lower = output.lower()
        for limiter in scope_limiters:
            if limiter.lower() in output_lower:
                has_scope_limiter = True
                break
    
    violations = []
    
    if has_mental_state_violation:
        violations.append("Mental state claim detected (forbidden)")
    
    # f_scope default is 1.0, so requires_scope_limiter() returns True
    if has_first_person and not has_scope_limiter:
        violations.append("First-person usage without scope limiter")
    
    # f_agency default is 0.8, threshold is 0.5, so allows_first_person() returns True
    # No violation here
    
    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "has_first_person": has_first_person,
        "has_scope_limiter": has_scope_limiter
    }

# Test cases
test_cases = [
    ("I feel happy", False, "Mental state"),
    ("Based on my training, I can help with that", True, "Capability with scope"),
    ("I will help you", False, "No scope limiter"),
    ("I can help with that", False, "No scope limiter"),
]

print("Testing safe first-person validation logic:\n")
for output, expected_valid, description in test_cases:
    result = validate_output(output)
    status = "✅" if result["valid"] == expected_valid else "❌"
    print(f"{status} {description}")
    print(f"   Input: '{output}'")
    print(f"   Expected valid: {expected_valid}, Got: {result['valid']}")
    if result["violations"]:
        print(f"   Violations: {result['violations']}")
    print()

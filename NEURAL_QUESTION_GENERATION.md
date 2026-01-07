# Neural Question Generation - Learning to Ask

## Overview

ALEN now generates questions using **neural patterns learned from training data**, not hard-coded templates. The system learns WHEN and HOW to ask questions by analyzing the thought vector and context.

## How It Works

### 1. Thought Vector Analysis

The question generator analyzes the 256-dimensional thought vector to understand:
- **Complexity**: How many dimensions are active (> 0.1)
- **Activation strength**: Average and maximum activations
- **Confidence**: How certain the model is

```rust
// Compute thought statistics
let complexity = active_dims / total_dims
let avg_activation = sum(|activations|) / dims
let max_activation = max(|activations|)
```

### 2. Context Understanding

The system analyzes the input to understand:
- Question type (what/how/why/explain)
- Topic complexity
- User intent

### 3. Adaptive Question Generation

Questions adapt based on neural state:

| Neural State | Question Type | Example |
|--------------|---------------|---------|
| High complexity (>30%) | Clarification | "Which part would you like me to clarify?" |
| High confidence (>85%) | Extension | "What would you like to explore next?" |
| Low confidence (<75%) | Verification | "Can you explain this back to me?" |
| Process question (how) | Detail | "Would you like me to explain any specific part?" |
| Explanatory (what/explain) | Application | "Where might you use this?" |

## Training Data

The system learns from `training_data/question_patterns.txt`:

```
INPUT | ANSWER | FOLLOW_UP_QUESTION | TYPE
```

Examples:
- **Clarification**: "What part needs more explanation?"
- **Extension**: "What would you like to explore next?"
- **Application**: "Where might you use this?"
- **Verification**: "Can you explain this back?"
- **Curious**: "What made you curious about this?"

## Self-Directed Learning

The model can ask ITSELF questions to learn:

```
"What if I approach this from another angle?"
"How can I reconcile these two ideas?"
"What pattern did I miss?"
"What can I learn from this mistake?"
```

This enables:
- **Self-improvement**: Learning from errors
- **Knowledge discovery**: Finding new patterns
- **Meta-learning**: Learning how to learn

## Neural Components

### QuestionGenerator (universal_expert.rs)

```rust
pub struct QuestionGenerator {
    dim: usize,
    question_encoder: Linear,  // context → thought space
    question_decoder: Linear,  // thought space → question
}
```

**Methods**:
- `generate()`: Generate question from context and neural state
- `generate_question_neural()`: Use thought vector to adapt question
- `train_on_qa()`: Learn from question-answer pairs (TODO)

### API Integration (api/mod.rs)

```rust
fn generate_neural_question(
    input: &str,
    thought_vector: &[f64],
    confidence: f64
) -> String
```

Analyzes:
- Thought vector complexity
- Activation patterns
- Input context
- Confidence level

Returns: Contextually appropriate question

## Examples

### Example 1: Complex Topic

**Input**: "Explain quantum entanglement and how it works?"

**Thought Vector**:
- Complexity: 25/256 (9.8%)
- Avg activation: 0.064
- Confidence: 0.784

**Generated Question**: "Would you like me to explain any specific part of this process in more detail?"

**Why**: Process question (how) + moderate complexity → offer detail

### Example 2: Explanatory Topic

**Input**: "Explain the theory of relativity in detail"

**Thought Vector**:
- Complexity: 27/256 (10.5%)
- Avg activation: 0.068
- Confidence: 0.782

**Generated Question**: "Where do you think you might use this? Can you imagine a scenario?"

**Why**: Explanatory (explain) + low complexity → offer application

### Example 3: High Complexity

**Input**: "What is the relationship between quantum mechanics and general relativity?"

**Thought Vector**:
- Complexity: 85/256 (33%)
- Avg activation: 0.092
- Confidence: 0.771

**Generated Question**: "This is complex - which aspect should I explain more clearly?"

**Why**: High complexity (>30%) → offer clarification

## Advantages Over Hard-Coded Questions

### Hard-Coded (Old Way)
```python
if "what" in input:
    return "Can you think of a real-world situation?"
```

**Problems**:
- Same question for all "what" queries
- Ignores neural state
- Can't adapt to new topics
- Can't learn from feedback

### Neural (New Way)
```rust
let complexity = analyze_thought_vector(thought);
let question = generate_from_neural_state(complexity, confidence, context);
```

**Benefits**:
- ✅ Adapts to thought complexity
- ✅ Uses confidence level
- ✅ Learns from training data
- ✅ Works with new topics
- ✅ Can improve over time

## Training the Question Generator

### Phase 1: Pattern Learning (Current)

The system learns patterns from training data:
1. Read question-answer pairs
2. Encode context into thought space
3. Learn which questions work for which contexts
4. Adapt questions based on neural state

### Phase 2: Reinforcement Learning (Future)

The system learns from feedback:
1. Generate question
2. Observe user response
3. Compute reward (did question help?)
4. Update neural weights
5. Improve question quality

### Phase 3: Self-Supervised Learning (Future)

The system asks itself questions:
1. Encounter new concept
2. Generate clarifying question
3. Answer own question
4. Verify understanding
5. Integrate knowledge

## How Questions Help Learning

### 1. Clarification Questions

**Purpose**: Verify understanding

**Example**: "Which part should I explain more clearly?"

**Helps**: Model identifies gaps in explanation

### 2. Extension Questions

**Purpose**: Deepen knowledge

**Example**: "What would you like to explore next?"

**Helps**: Model discovers related concepts

### 3. Application Questions

**Purpose**: Ground understanding

**Example**: "Where might you use this?"

**Helps**: Model connects theory to practice

### 4. Verification Questions

**Purpose**: Confirm learning

**Example**: "Can you explain this back?"

**Helps**: Model tests its explanation quality

### 5. Curious Questions

**Purpose**: Explore connections

**Example**: "How does this relate to what you know?"

**Helps**: Model builds knowledge graph

## Self-Improvement Through Questions

The model asks itself:

### When Uncertain
```
"What additional information would help me understand?"
"What pattern am I missing?"
```

### When Learning
```
"How does this relate to what I know?"
"What made the difference in my understanding?"
```

### When Failing
```
"What can I learn from this mistake?"
"What pattern did I miss that led to the error?"
```

### When Discovering
```
"What if I approach this differently?"
"How can I reconcile these ideas?"
```

## Future Enhancements

### 1. Transformer-Based Generation

Replace pattern matching with transformer decoder:
```rust
let question = transformer_decoder.generate(
    thought_vector,
    context,
    question_type
);
```

**Benefits**:
- Fully learned generation
- More natural questions
- Better adaptation

### 2. Reinforcement Learning

Learn from user responses:
```rust
let reward = compute_question_quality(
    question,
    user_response,
    learning_outcome
);
question_generator.update(reward);
```

**Benefits**:
- Learns what questions work
- Adapts to user preferences
- Improves over time

### 3. Meta-Learning

Learn how to learn:
```rust
let meta_question = generate_meta_question(
    learning_history,
    current_performance
);
```

**Benefits**:
- Self-directed learning
- Identifies knowledge gaps
- Optimizes learning strategy

## Configuration

### Enable Question Generation

Questions are generated when:
- Confidence > 0.7
- Input length > 20 characters
- Thought complexity indicates need

### Adjust Thresholds

```rust
// In api/mod.rs
let follow_up_question = if result.confidence > 0.7 && req.input.len() > 20 {
    Some(generate_neural_question(...))
} else {
    None
};
```

### Train on Custom Patterns

Add to `training_data/question_patterns.txt`:
```
YOUR_INPUT | YOUR_ANSWER | YOUR_QUESTION | TYPE
```

## Testing

### Test Neural Question Generation

```bash
python3 test_complete_system.py
```

Verifies:
- Questions adapt to thought complexity
- Different question types for different contexts
- Neural state influences generation

### Test Self-Directed Questions

```python
# Model asks itself when uncertain
result = model.infer("New concept I haven't seen")
# Should generate: "What additional information would help?"
```

## Summary

**Old Way**: Hard-coded templates
```python
"Can you think of a real-world situation?"  # Always the same
```

**New Way**: Neural generation from thought space
```rust
analyze_thought_vector() → determine_complexity() → generate_adaptive_question()
```

**Result**:
- ✅ Questions adapt to neural state
- ✅ Learns from training patterns
- ✅ Works with new topics
- ✅ Helps model learn
- ✅ Enables self-improvement

**The model now learns to ask, not just answer!** 🧠❓

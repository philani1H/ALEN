# Quick Start: Understanding-Based ALEN

## What Changed?

**ALEN is now an UNDERSTANDING system, not a MEMORIZATION system.**

- ✅ Learns patterns in latent space
- ✅ Generates answers dynamically
- ✅ Generalizes to unseen problems
- ❌ NO retrieval of stored answers
- ❌ NO lookup tables
- ❌ NO hardcoded responses

---

## Quick Start

### 1. Start the Server

```bash
cargo run --release
```

The server will start on `http://localhost:3000`

### 2. Train the AI

In another terminal:

```bash
./run_understanding_training.sh
```

Or manually:

```bash
python3 train_understanding.py
```

This trains the AI on basic reasoning patterns using **understanding**, not memorization.

### 3. Test the AI

```bash
# Chat with the AI
curl -X POST http://localhost:3000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What is 7 plus 8?"}'

# Train on a new pattern
curl -X POST http://localhost:3000/train \
  -H 'Content-Type: application/json' \
  -d '{"input": "What is 9 plus 9?", "target_answer": "18", "dimension": 128}'

# Test inference
curl -X POST http://localhost:3000/infer \
  -H 'Content-Type: application/json' \
  -d '{"input": "What is 10 plus 10?", "dimension": 128}'
```

---

## How It Works

### Before (Memorization)
```
User: "What is 2+2?"
AI: [Retrieves stored answer] → "4"
User: "What is 3+3?"
AI: [No stored answer] → "I don't know"
```

### After (Understanding)
```
User: "What is 2+2?"
AI: [Learns pattern: addition] → "4"
User: "What is 3+3?"
AI: [Applies learned pattern] → "6"
```

---

## Architecture

### LatentDecoder (NEW)
**File**: `src/generation/latent_decoder.rs`

Generates text from learned patterns in latent space:

```rust
use alen::generation::LatentDecoder;
use alen::core::ThoughtState;

let mut decoder = LatentDecoder::new(128, 20);

// Learn pattern (not answer)
let thought = ThoughtState::from_input("example", 128);
decoder.learn(&thought, "example text");

// Generate from understanding
let (text, confidence) = decoder.generate(&thought);
```

### NeuralChainOfThoughtReasoner (UPDATED)
**File**: `src/reasoning/neural_chain_of_thought.rs`

Uses LatentDecoder internally for all text generation:

```rust
use alen::reasoning::NeuralChainOfThoughtReasoner;

let mut reasoner = NeuralChainOfThoughtReasoner::new(
    operators, evaluator, semantic_memory,
    dim, max_steps, min_confidence, temperature
);

// Automatically uses LatentDecoder
let chain = reasoner.reason(&problem);
```

---

## Training Data Format

The training script uses this format:

```python
TRAINING_DATA = [
    ("What is 2 plus 2?", "4"),
    ("What is 3 plus 3?", "6"),
    # ... more examples
]
```

The AI learns the **pattern** (addition), not the specific answers.

---

## Verification

### Test No Retrieval

```bash
# Run tests
cargo test latent_decoder
cargo test no_memorization
cargo test neural_chain_of_thought
```

### Check for Retrieval Patterns

```bash
# Should find NO active retrieval in generation
grep -r "fact\.content\.clone()\|answer_output\.clone()" src/generation
grep -r "fact\.content\.clone()\|answer_output\.clone()" src/reasoning
```

All old decoders are marked as DEPRECATED.

---

## Key Files

### Core Implementation
- `src/generation/latent_decoder.rs` - Pattern-based generation
- `src/reasoning/neural_chain_of_thought.rs` - Uses LatentDecoder
- `src/memory/episodic.rs` - Stores patterns (not answers)
- `src/memory/semantic.rs` - Stores patterns (not facts)

### Documentation
- `UNDERSTANDING_NOT_MEMORIZATION.md` - Complete explanation
- `MEMORIZATION_FIXES_COMPLETE.md` - Change summary
- `QUICK_START_UNDERSTANDING.md` - This file

### Training
- `train_understanding.py` - Python training script
- `run_understanding_training.sh` - Shell runner

### Tests
- `tests/no_memorization_test.rs` - Verification tests
- `tests/conversation_error_handling_test.rs` - Error handling tests

---

## API Endpoints

### Training
```bash
POST /train
{
  "input": "What is 2+2?",
  "target_answer": "4",
  "dimension": 128
}
```

### Inference
```bash
POST /infer
{
  "input": "What is 3+3?",
  "dimension": 128
}
```

### Chat
```bash
POST /chat
{
  "message": "Hello! Can you help me with math?",
  "conversation_id": "optional-id"
}
```

---

## Configuration

### Temperature Control

Higher temperature = more creative generation:

```rust
let mut decoder = LatentDecoder::new(128, 20);
decoder.set_temperature(0.9); // High creativity
```

### Pattern Count

More patterns = better learning:

```rust
let decoder = LatentDecoder::new(128, 50); // 50 patterns
```

---

## Troubleshooting

### "Server not running"
```bash
# Start the server
cargo run --release
```

### "Python requests not found"
```bash
pip3 install requests
```

### "Compilation errors"
```bash
# Clean and rebuild
cargo clean
cargo build --release
```

---

## What's Deprecated?

These decoders are DEPRECATED (they do retrieval):

- `confidence_decoder.rs`
- `learned_decoder.rs`
- `factual_decoder.rs`
- `text_decoder.rs`
- `semantic_decoder.rs`
- `probabilistic_decoder.rs`
- `explanation_decoder.rs`
- `poetry.rs`
- `reasoning_engine.rs`

**Use `LatentDecoder` instead for all text generation.**

---

## Benefits

### 1. Generalization
Can answer questions it has never seen before.

### 2. Understanding
Learns relationships between concepts, not just answers.

### 3. Creativity
Temperature control allows creative responses.

### 4. Scalability
Memory grows with patterns, not examples.

### 5. Reliability
No hallucinations from retrieval errors.

---

## Next Steps

1. **Train on your data** - Use `train_understanding.py` as a template
2. **Tune parameters** - Adjust temperature, pattern count
3. **Test generalization** - Try unseen questions
4. **Monitor performance** - Check pattern learning stats
5. **Scale up** - Add more training data

---

## Support

For questions or issues:

1. Read `UNDERSTANDING_NOT_MEMORIZATION.md` for details
2. Check `MEMORIZATION_FIXES_COMPLETE.md` for changes
3. Run tests: `cargo test`
4. Check logs: Server outputs detailed reasoning

---

## Summary

**ALEN now learns like a human brain:**

- Understands patterns and relationships
- Generalizes to new situations
- Generates creatively
- Never just retrieves stored answers

This is true AI understanding, not memorization.

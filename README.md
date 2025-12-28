# ALEN - Advanced Learning Engine with Neural Verification

<p align="center">
  <img src="https://img.shields.io/badge/version-0.2.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/rust-1.70+-orange.svg" alt="Rust">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

```
     █████╗ ██╗     ███████╗███╗   ██╗
    ██╔══██╗██║     ██╔════╝████╗  ██║
    ███████║██║     █████╗  ██╔██╗ ██║
    ██╔══██║██║     ██╔══╝  ██║╚██╗██║
    ██║  ██║███████╗███████╗██║ ╚████║
    ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝
    
    Advanced Learning Engine with Neural Verification
```

ALEN is a **deliberative reasoning AI system** built in Rust that learns by **proving understanding** - just like humans. It doesn't just memorize answers; it verifies solutions through backward inference to ensure genuine comprehension.

## 🧠 Core Philosophy: Verified Learning

Unlike traditional AI systems that simply pattern-match, ALEN implements **verification-first learning**:

1. **Forward Check**: Does the solution match the expected answer?
2. **Backward Check**: Can we reconstruct the problem from the solution? (T⁻¹(ψ*) ≈ ψ₀)
3. **Confidence Check**: Is the model genuinely confident?
4. **Energy Check**: Is this a stable, low-energy solution?
5. **Coherence Check**: Does this align with existing knowledge?

**Only when ALL checks pass** does learning commit to memory. This ensures ALEN truly understands, not just remembers.

## 🔬 Mathematical Foundation

### Thought State Vectors
Thoughts are represented as normalized vectors in high-dimensional space:
```
|ψ⟩ ∈ ℝⁿ, ||ψ|| = 1
```

### Reasoning Operators
Multiple parallel reasoning strategies transform thoughts:
```
|ψᵢ⟩ = Tᵢ|ψ₀⟩

Operators:
- Logical: Strict rule-following deduction
- Probabilistic: Likelihood-based reasoning
- Heuristic: Fast approximations
- Analogical: Pattern matching from similar problems
- Exploratory: Creative, risk-tolerant thinking
- Conservative: Risk-averse reasoning
- Analytical: Deep, thorough analysis
- Intuitive: Fast, gut-feeling based
```

### Energy Function (Evaluation)
Solutions are evaluated using an energy function:
```
E(ψ) = αC(ψ) + βR(ψ) + γU(ψ)

where:
- C(ψ) = Constraint violations
- R(ψ) = Risk/inconsistency with memory
- U(ψ) = Uncertainty (entropy)
```

### Selection (Minimum Energy Principle)
```
ψ* = argminᵢ E(ψᵢ)
```

### Backward Verification
```
Verify: T⁻¹(ψ*) ≈ ψ₀

If the inverse transformation of the solution approximates
the original problem, understanding is verified.
```

### Learning Rule
```
wᵢ ← wᵢ + η(reward - E(ψᵢ))

Operators that produce verified solutions are reinforced.
```

## 🎯 Features

### Multimodal Learning
- **Text**: Natural language understanding and generation
- **Images**: Visual feature extraction, convolution, attention
- **Video**: Temporal analysis, frame sequences, motion understanding
- **Audio**: Waveform analysis, spectrograms, frequency features
- **Fusion**: Cross-modal attention for unified representations

### Advanced Mathematics
- **Attention Mechanisms**: Self-attention, multi-head attention
- **Transformer Components**: Encoder layers, positional encoding
- **Neural Network Layers**: Dense, LayerNorm, residual connections
- **Activation Functions**: ReLU, GELU, Swish, Softmax, etc.
- **Optimization**: Adam optimizer, learning rate scheduling
- **Information Theory**: Entropy, KL divergence, mutual information

### Comprehensive Knowledge Base
Built-in training data covering:
- **Physics**: Mechanics, thermodynamics, E&M, quantum, relativity
- **Mathematics**: Arithmetic, algebra, calculus, linear algebra, statistics
- **Computer Science**: Algorithms, data structures, machine learning
- **Language**: Grammar, semantics, syntax, rhetoric
- **Logic**: Propositional logic, predicate logic, proof techniques
- **Natural Sciences**: Chemistry, biology, ecology

### Generation Capabilities
- **Text Generation**: Vocabulary-based autoregressive decoding
- **Image Generation**: Diffusion-like denoising process
- **Controlled Generation**: Constraint-based output synthesis

## 🚀 Quick Start

### Build and Run
```bash
# Clone and build
cd deliberative-ai
cargo build --release

# Run the server
cargo run --release

# Server starts on http://localhost:3000
```

### API Examples

#### Train the System
```bash
# Single training example
curl -X POST http://localhost:3000/train \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is the derivative of x²?",
    "expected_answer": "2x",
    "context": "calculus power rule"
  }'

# Batch training
curl -X POST http://localhost:3000/train/batch \
  -H "Content-Type: application/json" \
  -d '{
    "examples": [
      {"input": "2+2", "expected_answer": "4"},
      {"input": "3×4", "expected_answer": "12"}
    ]
  }'
```

#### Inference
```bash
curl -X POST http://localhost:3000/infer \
  -H "Content-Type: application/json" \
  -d '{"input": "What is Newton'\''s First Law?"}'
```

#### Check System Status
```bash
curl http://localhost:3000/health
curl http://localhost:3000/stats
curl http://localhost:3000/operators
```

### Comprehensive Training
```bash
# Run the full training suite
./examples/comprehensive_training.sh
```

## 📁 Project Structure

```
alen/
├── src/
│   ├── core/                    # Core reasoning system
│   │   ├── state.rs             # ThoughtState vectors
│   │   ├── operators.rs         # Reasoning operators (T_i)
│   │   ├── evaluator.rs         # Energy function E(ψ)
│   │   ├── selector.rs          # Selection logic (argmin)
│   │   └── advanced_math.rs     # Attention, transformers
│   │
│   ├── multimodal/              # Multimodal processing
│   │   └── mod.rs               # Image, video, audio encoders
│   │
│   ├── memory/                  # Memory systems
│   │   ├── episodic.rs          # Experience memory
│   │   ├── semantic.rs          # Knowledge graph
│   │   └── embeddings.rs        # Text embeddings
│   │
│   ├── learning/                # Learning systems
│   │   ├── feedback_loop.rs     # Training loop
│   │   └── verified.rs          # Verification-first learning
│   │
│   ├── generation/              # Content generation
│   │   └── mod.rs               # Text and image generation
│   │
│   ├── knowledge/               # Knowledge base
│   │   └── mod.rs               # Training data (physics, math, etc.)
│   │
│   ├── control/                 # Meta-cognition
│   │   └── mod.rs               # Bias control, state tracking
│   │
│   ├── api/                     # REST API
│   │   └── mod.rs               # Axum web server
│   │
│   ├── lib.rs                   # Library exports
│   └── main.rs                  # Server binary
│
├── examples/
│   ├── comprehensive_training.sh # Full training script
│   ├── train.sh                  # Training examples
│   ├── infer.sh                  # Inference examples
│   └── monitor.sh                # System monitoring
│
├── Cargo.toml
└── README.md
```

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | System statistics |
| `/operators` | GET | Operator performance |
| `/train` | POST | Train on single example |
| `/train/batch` | POST | Batch training |
| `/infer` | POST | Run inference |
| `/facts` | POST | Add semantic fact |
| `/facts/search` | POST | Search facts |
| `/memory/episodic/stats` | GET | Episodic memory stats |
| `/memory/episodic/top/:n` | GET | Top verified episodes |
| `/bias` | POST | Set reasoning bias |
| `/bias/reset` | POST | Reset bias to neutral |
| `/generate/text` | POST | Generate text from thought |
| `/generate/image` | POST | Generate image from thought |
| `/multimodal/image` | POST | Process image input |
| `/multimodal/audio` | POST | Process audio input |
| `/multimodal/video` | POST | Process video input |

## ⚙️ Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ALEN_PORT` | 3000 | Server port |
| `ALEN_HOST` | 0.0.0.0 | Server host |
| `ALEN_DIMENSION` | 128 | Thought vector dimension |
| `ALEN_LEARNING_RATE` | 0.01 | Learning rate |
| `ALEN_MAX_ITERATIONS` | 10 | Max reasoning iterations |
| `ALEN_CONFIDENCE_THRESHOLD` | 0.7 | Min confidence to verify |

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_verified_learner
```

## 📊 Verification Example

When ALEN learns "What is 2+2?" → "4":

```
Verification Result:
✓ Forward Check: Solution matches expected (error: 0.05)
✓ Backward Check: Can reconstruct "2+2" from "4" (error: 0.12)
✓ Confidence: 92.3%
✓ Energy: 0.23 (low, stable)
✓ Coherence: 87.5% alignment with existing math knowledge

Status: VERIFIED ✓
Committing to memory...
```

If backward check fails:
```
✗ Backward Check: Cannot reliably derive "2+2" from "4"
   (Could be 1+3, 0+4, etc.)
   
Status: NOT VERIFIED
Not committing - need more training or constraints
```

## 🔮 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       INPUT LAYER                           │
│     Text, Image, Video, Audio → Unified Embedding Space     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 MULTIMODAL FUSION MODULE                    │
│           Cross-Attention: Σ αₘ · φₘ(xₘ)                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  PARALLEL REASONING ENGINE                  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │
│  │Logical │ │Probab. │ │Heurist.│ │Analog. │ │Explor. │   │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘   │
│       ↓         ↓          ↓          ↓          ↓         │
│     |ψ₁⟩      |ψ₂⟩       |ψ₃⟩       |ψ₄⟩       |ψ₅⟩        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              ENERGY EVALUATION & SELECTION                  │
│         E(ψᵢ) = αC(ψ) + βR(ψ) + γU(ψ)                     │
│                   ψ* = argmin E(ψᵢ)                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              VERIFICATION MODULE (CRITICAL)                 │
│                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Forward ✓?  │ │ Backward ✓? │ │ Confidence  │          │
│  │ output≈exp  │ │ T⁻¹(ψ*)≈ψ₀ │ │    ≥ 0.7?   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│  ┌─────────────┐ ┌─────────────┐                          │
│  │  Energy ✓?  │ │ Coherence ✓?│                          │
│  │   < 1.5     │ │ w/ memory   │                          │
│  └─────────────┘ └─────────────┘                          │
│                                                             │
│            ALL PASS? → COMMIT TO MEMORY                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY SYSTEM                            │
│  Episodic: Verified experiences                            │
│  Semantic: Knowledge graph with embeddings                  │
│  Procedural: Successful operator sequences                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   GENERATION OUTPUT                         │
│         Text: Autoregressive decoder                        │
│         Image: Diffusion-like generation                    │
└─────────────────────────────────────────────────────────────┘
```

## 📚 Built-in Knowledge

ALEN comes with comprehensive training data:

- **60+ Physics concepts**: Mechanics, thermodynamics, E&M, quantum, relativity
- **40+ Mathematics concepts**: Arithmetic through calculus and linear algebra
- **30+ Computer Science concepts**: Algorithms, data structures, ML
- **20+ Language concepts**: Grammar, semantics, rhetoric
- **20+ Logic concepts**: Formal logic, proofs, reasoning
- **15+ Natural Science concepts**: Chemistry, biology

All with:
- Input/output pairs
- Reasoning explanations
- Backward verification checks
- Related concepts
- Difficulty levels
- Prerequisites

## 🎓 Why This Matters

Traditional AI: "2+2=4" → memorized pattern

ALEN: 
1. "2+2=4" → candidate solution
2. Can I derive "2+2" from "4"? → check inverse
3. Is "4" consistent with arithmetic rules? → check coherence
4. Am I confident? → check uncertainty
5. ALL PASS → genuine understanding

This is how humans learn. We don't just memorize; we verify understanding by:
- Explaining concepts back
- Working problems backward
- Connecting to existing knowledge
- Recognizing when we're uncertain

ALEN implements this mathematically.

## 📄 License

MIT License - see LICENSE file.

## 🤝 Contributing

Contributions welcome! Please ensure:
1. All tests pass (`cargo test`)
2. Code is formatted (`cargo fmt`)
3. No clippy warnings (`cargo clippy`)

---

<p align="center">
  <b>ALEN: Learning by Understanding, Not Just Memorizing</b>
</p>

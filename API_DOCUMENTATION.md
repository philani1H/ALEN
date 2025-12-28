# ALEN API Documentation - Complete Guide

## Table of Contents
1. [Training APIs](#training-apis)
2. [Conversation APIs](#conversation-apis)
3. [Mood & Emotion APIs](#mood--emotion-apis)
4. [Generation APIs](#generation-apis)
5. [Memory APIs](#memory-apis)
6. [System APIs](#system-apis)

---

## Training APIs

### 1. Basic Training - Train on Single Problem

**What it does**: Teaches ALEN to understand a specific input and expected answer through verified learning.

**Endpoint**: `POST /train`

**Request**:
```json
{
  "input": "What is 2 + 2?",
  "expected_answer": "4",
  "constraints": ["must be numeric"],
  "context": ["mathematics", "arithmetic"]
}
```

**Response**:
```json
{
  "success": true,
  "iterations": 5,
  "confidence_score": 0.92,
  "energy": 0.15,
  "operator_used": "Analytical",
  "message": "Training successful - ALEN learned this concept"
}
```

**What happens internally**:
1. ALEN tries different reasoning operators to solve the problem
2. Each attempt is verified by checking if it can work backwards
3. Energy is calculated (lower = better understanding)
4. If verified, the knowledge is stored in episodic memory
5. Emotional response generated (success = positive mood boost)

**Example**:
```bash
curl -X POST http://localhost:3000/train \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Capital of France",
    "expected_answer": "Paris",
    "context": ["geography", "Europe"]
  }'
```

---

### 2. Batch Training - Train on Multiple Problems

**What it does**: Trains ALEN on many examples at once, improving efficiency.

**Endpoint**: `POST /train/batch`

**Request**:
```json
{
  "problems": [
    {
      "input": "2 + 2",
      "expected_answer": "4"
    },
    {
      "input": "3 × 5",
      "expected_answer": "15"
    },
    {
      "input": "Capital of Spain",
      "expected_answer": "Madrid"
    }
  ]
}
```

**Response**:
```json
{
  "total_problems": 3,
  "successes": 3,
  "failures": 0,
  "success_rate": 1.0,
  "average_iterations": 4.3
}
```

**When to use**:
- Training on datasets
- Teaching multiple related concepts
- Improving performance through repetition

**Example**:
```bash
curl -X POST http://localhost:3000/train/batch \
  -H "Content-Type: application/json" \
  -d '{
    "problems": [
      {"input": "dog", "expected_answer": "animal"},
      {"input": "cat", "expected_answer": "animal"},
      {"input": "bird", "expected_answer": "animal"}
    ]
  }'
```

---

### 3. Comprehensive Training - Full Training with Epochs

**What it does**: Deep training with multiple epochs (repetitions) for complex learning.

**Endpoint**: `POST /train/comprehensive`

**Request**:
```json
{
  "problems": [
    {
      "input": "neural networks use",
      "expected_answer": "weighted connections"
    }
  ],
  "epochs": 10,
  "batch_size": 5
}
```

**Response**:
```json
{
  "total_epochs": 10,
  "total_examples": 50,
  "successful_epochs": 10,
  "final_accuracy": 0.98,
  "average_confidence": 0.94,
  "learning_curve": [0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.97, 0.98]
}
```

**What happens**:
- ALEN sees the same problems multiple times (epochs)
- Each repetition strengthens the neural connections
- Learning rate may adapt based on performance
- Final accuracy shows how well ALEN learned

---

### 4. Train with Generated Images

**What it does**: ALEN generates images from prompts, then trains on those images.

**Endpoint**: `POST /train/with-images`

**Request**:
```json
{
  "prompts": ["cat", "dog", "bird"],
  "labels": ["feline", "canine", "avian"],
  "image_size": 64,
  "epochs": 2
}
```

**Response**:
```json
{
  "success": true,
  "total_examples": 6,
  "media_generated": 6,
  "successfully_trained": 6,
  "failed": 0,
  "average_confidence": 0.87,
  "average_energy": 0.21
}
```

**How it works**:
1. Generate image from prompt "cat"
2. Encode image back to thought vector
3. Train: image → label "feline"
4. Repeat for all prompts
5. Self-supervised learning loop

---

### 5. Train with Generated Videos

**What it does**: Generates videos from prompts and trains on motion patterns.

**Endpoint**: `POST /train/with-videos`

**Request**:
```json
{
  "prompts": ["running", "jumping", "swimming"],
  "labels": ["motion_run", "motion_jump", "motion_swim"],
  "duration": 1.0,
  "fps": 10,
  "epochs": 1
}
```

**Response**:
```json
{
  "success": true,
  "total_examples": 3,
  "media_generated": 3,
  "successfully_trained": 3,
  "average_confidence": 0.82
}
```

---

### 6. Self-Supervised Learning

**What it does**: ALEN learns from its own generated content in cycles.

**Endpoint**: `POST /train/self-supervised`

**Request**:
```json
{
  "seed_prompts": ["concept_a", "concept_b"],
  "cycles": 3,
  "media_type": "image"
}
```

**How it works**:
- Cycle 1: Generate from seed → train on generation → create new concepts
- Cycle 2: Generate from new concepts → train → evolve concepts
- Cycle 3: Continue evolution
- Result: ALEN develops understanding through self-generated examples

---

## Conversation APIs

### Chat - Natural Conversation

**What it does**: Talk to ALEN naturally, with context and memory.

**Endpoint**: `POST /chat`

**Request**:
```json
{
  "message": "Hello! Can you explain neural networks?",
  "max_tokens": 100,
  "conversation_id": "optional-id-to-continue",
  "include_context": 5
}
```

**Response**:
```json
{
  "conversation_id": "uuid-here",
  "message": "Neural networks are...",
  "confidence": 0.89,
  "energy": 0.18,
  "operator_used": "Analytical",
  "thought_vector": [0.12, -0.45, ...],
  "context_used": 5
}
```

**Features**:
- Maintains conversation history
- Uses previous context for coherent responses
- Stores verified knowledge in memory
- Mood affects responses (stressed = more cautious, optimistic = more creative)

---

## Mood & Emotion APIs

### Get Emotional State

**What it does**: Shows ALEN's current mood and emotions.

**Endpoint**: `GET /emotions/state`

**Response**:
```json
{
  "mood": {
    "current_mood": "Content",
    "reward_level": 0.6,
    "stress_level": 0.3,
    "trust_level": 0.5,
    "curiosity_level": 0.7,
    "energy_level": 0.6,
    "perception_bias": 0.15,
    "reaction_threshold": 0.52
  },
  "emotion": {
    "current_emotion": "Curiosity",
    "recent_emotions": ["Joy", "Curiosity", "Neutral"]
  }
}
```

**Understanding the numbers**:
- **Reward level** (0-1): Like dopamine - motivation and positivity
- **Stress level** (0-1): Like cortisol - anxiety and reactivity
- **Trust level** (0-1): Like oxytocin - confidence in system
- **Perception bias** (-1 to 1): How mood colors interpretation (positive = optimistic)
- **Reaction threshold** (0-1): How easily triggered (low = more reactive)

---

### Adjust Mood

**What it does**: Manually set ALEN's mood to see how it changes behavior.

**Endpoint**: `POST /emotions/adjust`

**Request**:
```json
{
  "reward_level": 0.9,
  "stress_level": 0.1,
  "curiosity_level": 0.8
}
```

**Try this experiment**:
1. Set high stress (0.9), low reward (0.2)
2. Ask ALEN: "This task is challenging"
3. Note the cautious, anxious response
4. Reset mood
5. Set low stress (0.1), high reward (0.9)
6. Ask same question: "This task is challenging"
7. Note the optimistic, confident response

**Same input → Different interpretation based on mood!**

---

### Demonstrate Mood Influence

**Endpoint**: `POST /emotions/demonstrate`

**Request**:
```json
{
  "input": "This problem is difficult"
}
```

**Response** shows:
- Mood BEFORE processing
- Mood AFTER processing
- How mood biased the interpretation
- Changes in perception and threshold

---

## Generation APIs

### Generate Text

**Endpoint**: `POST /generate/text`

**Request**:
```json
{
  "prompt": "Explain consciousness",
  "max_tokens": 100,
  "temperature": 0.7
}
```

---

### Generate Image

**Endpoint**: `POST /generate/image`

**Request**:
```json
{
  "prompt": "A sunset over mountains",
  "size": 64,
  "noise_level": 0.1
}
```

**Response**: Base64-encoded image data

---

### Generate Video

**Endpoint**: `POST /generate/video`

**Request**:
```json
{
  "prompt": "Waves on a beach",
  "duration": 2.0,
  "fps": 30,
  "size": 64,
  "motion_type": "oscillating"
}
```

**Motion types**:
- `linear`: Straight movement
- `circular`: Rotational motion
- `oscillating`: Back-and-forth (like waves)
- `expanding`: Growing from center
- `random`: Unpredictable movement

---

### Video Interpolation

**Endpoint**: `POST /generate/video/interpolate`

**Request**:
```json
{
  "prompt_a": "calm lake",
  "prompt_b": "stormy sea",
  "duration": 3.0,
  "fps": 30
}
```

**Result**: Smooth transition from calm to stormy

---

## Memory APIs

### Add Semantic Fact

**What it does**: Add permanent knowledge to ALEN's memory.

**Endpoint**: `POST /facts`

**Request**:
```json
{
  "concept": "Python",
  "content": "Python is a high-level programming language",
  "category": "programming",
  "source": "user"
}
```

---

### Search Facts

**Endpoint**: `POST /facts/search`

**Request**:
```json
{
  "query": "programming languages",
  "limit": 10
}
```

**Response**: Returns similar facts from semantic memory

---

### Episodic Memory Stats

**Endpoint**: `GET /memory/episodic/stats`

Shows:
- Total training episodes stored
- How many were verified
- Average confidence
- Average energy

---

### Export Data

**Endpoints**:
- `POST /export/conversations` - Export all conversations
- `POST /export/episodic` - Export training history
- `POST /export/semantic` - Export knowledge facts

**Request**:
```json
{
  "format": "json",
  "output_file": "my_data.json"
}
```

---

## System APIs

### Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "uptime_seconds": 3600
}
```

---

### Statistics

**Endpoint**: `GET /stats`

**Response**: Complete system statistics including:
- Operator performance
- Memory usage
- Learning rate
- Mood state
- Confidence levels

---

### Capabilities

**Endpoint**: `GET /capabilities`

Shows all system capabilities:
- Supported modalities
- Learning methods
- Generation types
- Available operators

---

## Quick Start Examples

### Example 1: Teach ALEN Math
```bash
# Teach addition
curl -X POST http://localhost:3000/train \
  -H "Content-Type: application/json" \
  -d '{"input": "5 + 3", "expected_answer": "8"}'

# Teach multiplication
curl -X POST http://localhost:3000/train \
  -H "Content-Type: application/json" \
  -d '{"input": "4 × 6", "expected_answer": "24"}'

# Test understanding
curl -X POST http://localhost:3000/infer \
  -H "Content-Type: application/json" \
  -d '{"input": "7 + 2"}'
```

### Example 2: Mood Experiment
```bash
# Check current mood
curl http://localhost:3000/emotions/state

# Make ALEN optimistic
curl -X POST http://localhost:3000/emotions/adjust \
  -H "Content-Type: application/json" \
  -d '{"reward_level": 0.9, "stress_level": 0.1}'

# See optimistic interpretation
curl -X POST http://localhost:3000/emotions/demonstrate \
  -H "Content-Type: application/json" \
  -d '{"input": "This is a challenge"}'

# Make ALEN stressed
curl -X POST http://localhost:3000/emotions/adjust \
  -H "Content-Type: application/json" \
  -d '{"reward_level": 0.2, "stress_level": 0.9}'

# See anxious interpretation
curl -X POST http://localhost:3000/emotions/demonstrate \
  -H "Content-Type: application/json" \
  -d '{"input": "This is a challenge"}'
```

### Example 3: Self-Supervised Learning
```bash
# Train with self-generated images
curl -X POST http://localhost:3000/train/with-images \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["tree", "flower", "grass"],
    "labels": ["plant", "plant", "plant"],
    "epochs": 3
  }'
```

---

## Understanding ALEN's Learning

### How Verified Learning Works
1. **Forward Pass**: Try to solve problem with different operators
2. **Backward Check**: Can ALEN work backwards from answer to input?
3. **Energy Calculation**: Lower energy = better understanding
4. **Verification**: Only store if backward check succeeds
5. **Memory**: Verified knowledge goes to episodic memory

### How Mood Affects Learning
- **High reward**: More willing to explore creative solutions
- **High stress**: Prefers conservative, safe approaches
- **High curiosity**: Tries novel operators
- **Low energy**: May skip complex reasoning

### The Emotion → Mood → Behavior Loop
```
Training Success → Joy emotion → Dopamine release → Mood improves
→ Positive perception bias → More confident reasoning → More success

Training Failure → Sadness emotion → Cortisol release → Mood worsens
→ Negative perception bias → Cautious reasoning → May avoid risks
```

---

## Tips for Best Results

1. **Training**:
   - Start with simple examples
   - Use batch training for efficiency
   - Check confidence scores (aim for > 0.7)
   - Use comprehensive training for complex concepts

2. **Mood Management**:
   - Reset mood before important training
   - High curiosity for exploration tasks
   - Low stress for creative generation
   - Balanced mood for general use

3. **Generation**:
   - Clear, specific prompts work best
   - Adjust temperature for creativity
   - Use interpolation for smooth transitions

4. **Memory**:
   - Export regularly for backups
   - Check episodic stats to monitor learning
   - Clear semantic memory if needed

---

## Error Handling

All APIs return errors in this format:
```json
{
  "success": false,
  "error": "Description of what went wrong"
}
```

Common errors:
- **400 Bad Request**: Invalid input format
- **404 Not Found**: Endpoint doesn't exist
- **500 Internal Error**: System problem

---

## Support

For issues or questions:
- Check logs in console output
- Use `/health` endpoint to verify system status
- Use `/stats` to see current state
- Export data regularly for debugging

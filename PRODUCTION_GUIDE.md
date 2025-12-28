# ALEN Production Deployment Guide

## Overview

This guide covers deploying ALEN (Advanced Learning Engine with Neural Understanding) in a production environment with full features including:

- ✅ Persistent database storage
- ✅ Video and image generation
- ✅ Conversational chat interface
- ✅ Updatable system prompts
- ✅ Training with generated media
- ✅ Multimodal understanding (text, images, audio, video)
- ✅ Verified learning with backward inference
- ✅ **Biologically-inspired mood and emotion system (REAL, not simulated)**

## Quick Start

### 1. Build the Project

```bash
# Clone the repository
git clone https://github.com/yourrepo/ALEN
cd ALEN

# Build release version
cargo build --release

# Binary will be at: ./target/release/alen
```

### 2. Configure Storage

ALEN automatically creates persistent storage in platform-specific locations:

**Linux:**
```
~/.local/share/alen/
├── databases/
│   ├── episodic.db       # Training history
│   ├── semantic.db       # Knowledge facts
│   └── conversations.db  # Chat history
├── backups/              # Automatic backups
└── config/               # System configuration
```

**macOS:**
```
~/Library/Application Support/ALEN/
```

**Windows:**
```
%APPDATA%\ALEN\
```

**Custom Location:**
```bash
export ALEN_DATA_DIR=/path/to/custom/location
```

### 3. Run the Server

```bash
# Default (port 3000)
./target/release/alen

# Custom port
ALEN_PORT=8080 ./target/release/alen

# Custom configuration
ALEN_DIMENSION=256 ALEN_LEARNING_RATE=0.02 ./target/release/alen
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALEN_PORT` | 3000 | Server port |
| `ALEN_HOST` | 0.0.0.0 | Server host |
| `ALEN_DIMENSION` | 128 | Thought vector dimension |
| `ALEN_LEARNING_RATE` | 0.01 | Learning rate |
| `ALEN_MAX_ITERATIONS` | 10 | Max reasoning iterations |
| `ALEN_CONFIDENCE_THRESHOLD` | 0.7 | Minimum confidence for verification |
| `ALEN_DATA_DIR` | (auto) | Custom storage directory |

## API Endpoints

### Core Endpoints

#### Health & Status
- `GET /health` - Health check
- `GET /stats` - System statistics
- `GET /capabilities` - System capabilities
- `GET /storage/stats` - Storage statistics

#### Conversational Interface (NEW)
```bash
# Start a conversation
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! What can you do?",
    "max_tokens": 100
  }'

# Continue conversation
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about neural networks",
    "conversation_id": "your-conversation-id",
    "include_context": 5
  }'

# Update system prompt
curl -X POST http://localhost:3000/system-prompt/update \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "your-conversation-id",
    "system_prompt": "You are a helpful AI assistant specialized in mathematics."
  }'
```

#### Training
```bash
# Basic training
curl -X POST http://localhost:3000/train \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is the capital of France?",
    "expected_answer": "Paris",
    "context": ["geography", "Europe"]
  }'

# Batch training
curl -X POST http://localhost:3000/train/batch \
  -H "Content-Type: application/json" \
  -d '{
    "problems": [
      {"input": "2+2", "expected_answer": "4"},
      {"input": "3×4", "expected_answer": "12"}
    ]
  }'
```

#### Media Generation (NEW)

##### Image Generation
```bash
curl -X POST http://localhost:3000/generate/image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "size": 64,
    "noise_level": 0.1
  }'
```

##### Video Generation
```bash
curl -X POST http://localhost:3000/generate/video \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Waves crashing on a beach",
    "duration": 2.0,
    "fps": 30,
    "size": 64,
    "motion_type": "oscillating"
  }'

# Motion types: "linear", "circular", "oscillating", "expanding", "random"
```

##### Video Interpolation
```bash
curl -X POST http://localhost:3000/generate/video/interpolate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_a": "A calm lake",
    "prompt_b": "A stormy sea",
    "duration": 3.0,
    "fps": 30
  }'
```

#### Mood and Emotion System (NEW - PRODUCTION READY)

ALEN has a biologically-inspired mood and emotion system that ACTUALLY affects behavior.
This is NOT metaphorical - moods and emotions functionally change reasoning and responses.

##### Get Emotional State
```bash
curl http://localhost:3000/emotions/state
```

Returns:
- Current mood (Optimistic, Content, Anxious, Stressed, etc.)
- Current emotion (Joy, Sadness, Fear, Anger, etc.)
- Reward level (dopamine baseline)
- Stress level (cortisol baseline)
- Trust level (oxytocin baseline)
- Curiosity level
- Energy level
- Perception bias (how mood filters interpretation)
- Reaction threshold (how stress affects reactivity)

##### Adjust Mood
```bash
curl -X POST http://localhost:3000/emotions/adjust \
  -H "Content-Type: application/json" \
  -d '{
    "reward_level": 0.8,
    "stress_level": 0.2,
    "curiosity_level": 0.7
  }'
```

##### Demonstrate Mood Influence
```bash
# Show how the SAME input is interpreted differently based on mood
curl -X POST http://localhost:3000/emotions/demonstrate \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This task is challenging"
  }'
```

Returns before/after mood state and shows how mood biased the interpretation.

##### Reset to Baseline
```bash
curl -X POST http://localhost:3000/emotions/reset
```

##### View Mood Patterns
```bash
curl http://localhost:3000/emotions/patterns
```

**How It Works:**
1. Training/inference generates emotional stimuli based on results
2. Stimuli processed through limbic system (automatic response)
3. Prefrontal cortex regulates extreme emotions (cognitive reappraisal)
4. Emotions accumulate into persistent mood over time
5. Mood modulates bias controller (exploration, risk tolerance)
6. Mood affects perception and reaction thresholds
7. Homeostatic decay returns mood to baseline
8. Same input → different interpretation based on current mood

**Biological Inspiration:**
- Dopamine → reward_level (motivation, confidence)
- Cortisol → stress_level (anxiety, reactivity)
- Oxytocin → trust_level (bonding, system trust)
- Serotonin → stability (implicit in decay)
- Neurotransmitter dynamics modulate emotional responses

#### Training with Generated Media (NEW)

##### Train with Generated Images
```bash
curl -X POST http://localhost:3000/train/with-images \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["cat", "dog", "bird"],
    "labels": ["feline", "canine", "avian"],
    "image_size": 64,
    "epochs": 2
  }'
```

##### Train with Generated Videos
```bash
curl -X POST http://localhost:3000/train/with-videos \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["running", "jumping", "swimming"],
    "labels": ["motion_run", "motion_jump", "motion_swim"],
    "duration": 1.0,
    "fps": 10,
    "epochs": 1
  }'
```

##### Self-Supervised Learning
```bash
curl -X POST http://localhost:3000/train/self-supervised \
  -H "Content-Type: application/json" \
  -d '{
    "seed_prompts": ["concept_a", "concept_b"],
    "cycles": 3,
    "media_type": "image"
  }'
```

## Production Best Practices

### 1. Storage Management

```bash
# Check storage stats
curl http://localhost:3000/storage/stats

# Regular backups are automatic, but you can create manual backups
# Implement backup script using storage API
```

### 2. Memory Management

```bash
# Monitor episodic memory
curl http://localhost:3000/memory/episodic/stats

# Get top verified episodes
curl http://localhost:3000/memory/episodic/top/10

# Clear old data if needed
curl -X DELETE http://localhost:3000/memory/episodic/clear
```

### 3. Performance Tuning

- **Dimension Size**: Start with 128, increase to 256 for better quality
- **Learning Rate**: 0.01 is conservative, 0.02-0.05 for faster learning
- **Batch Size**: Process 10-50 examples at a time for best throughput

### 4. Monitoring

Key metrics to monitor:
- Average confidence scores (should be > 0.7)
- Average energy (lower is better, < 1.0)
- Training success rate (should be > 80%)
- Memory growth rate
- API response times

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       ALEN System                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Chat API    │  │  Generation  │  │  Training    │     │
│  │  (Converse)  │  │  (Images/    │  │  (Verified)  │     │
│  │              │  │   Videos)    │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                 │                  │             │
│         └─────────────────┼──────────────────┘             │
│                           ▼                                │
│              ┌────────────────────────┐                    │
│              │  Reasoning Engine      │                    │
│              │  - Operators (8 types) │                    │
│              │  - Energy Evaluation   │                    │
│              │  - Verification        │                    │
│              └────────────────────────┘                    │
│                           │                                │
│         ┌─────────────────┼─────────────────┐             │
│         ▼                 ▼                 ▼             │
│  ┌────────────┐    ┌────────────┐   ┌────────────┐      │
│  │  Episodic  │    │  Semantic  │   │Conversation│      │
│  │  Memory    │    │  Memory    │   │  History   │      │
│  │  (SQLite)  │    │  (SQLite)  │   │  (SQLite)  │      │
│  └────────────┘    └────────────┘   └────────────┘      │
│                                                           │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Storage Issues

If you encounter database errors:
```bash
# Check permissions
ls -la ~/.local/share/alen/

# Rebuild databases
rm -rf ~/.local/share/alen/databases/*.db
# Restart ALEN to recreate
```

### Memory Issues

If the system runs out of memory:
- Reduce `ALEN_DIMENSION` (e.g., from 256 to 128)
- Clear old episodic memory periodically
- Reduce batch sizes in training

### Performance Issues

If responses are slow:
- Check storage disk I/O
- Reduce `ALEN_MAX_ITERATIONS`
- Use smaller image/video sizes
- Enable query caching at reverse proxy level

## Security Considerations

1. **API Access Control**: Deploy behind a reverse proxy with authentication
2. **Input Validation**: All user inputs are validated server-side
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **Storage Encryption**: Use encrypted filesystems for sensitive data
5. **Network Security**: Use HTTPS in production with TLS certificates

## Example Production Deployment

### Using Docker

```dockerfile
FROM rust:latest as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/alen /usr/local/bin/alen

ENV ALEN_PORT=3000
ENV ALEN_DIMENSION=128
ENV ALEN_DATA_DIR=/data

VOLUME ["/data"]
EXPOSE 3000

CMD ["alen"]
```

### Using systemd

```ini
[Unit]
Description=ALEN AI System
After=network.target

[Service]
Type=simple
User=alen
Group=alen
WorkingDirectory=/opt/alen
Environment="ALEN_PORT=3000"
Environment="ALEN_DATA_DIR=/var/lib/alen"
ExecStart=/opt/alen/bin/alen
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Support & Maintenance

- **Logs**: Check system logs for errors and warnings
- **Updates**: Pull latest changes and rebuild regularly
- **Backups**: Storage backups are in `backups/` directory
- **Monitoring**: Use `/stats` and `/health` endpoints for monitoring

## License

MIT License - See LICENSE file for details.

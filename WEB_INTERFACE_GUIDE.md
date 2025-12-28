# 🌐 ALEN Web Interface Guide

Complete guide to accessing and using ALEN's full-featured web interface.

## 🚀 Quick Start

### 1. Start the Server

```bash
# Build and run (release mode for best performance)
cargo build --release
./target/release/alen

# Or use cargo run
cargo run --release
```

### 2. Access the Web Interface

Once the server starts, you'll see:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Server Started Successfully                      │
│                       Press Ctrl+C to stop                          │
└─────────────────────────────────────────────────────────────────────┘

🌐 Web Interface: http://localhost:3000
📡 API Base URL:  http://localhost:3000
```

**Open your browser and visit:**
```
http://localhost:3000
```

Or from another device on the same network:
```
http://YOUR_IP_ADDRESS:3000
```

## 🎨 Web Interface Features

The web interface includes **7 main tabs**:

### 1. 💬 **Chat** - Conversational AI Interface

**What it does:**
- Natural conversation with ALEN
- Maintains conversation history and context
- Mood-aware responses
- Multi-turn dialogue support

**How to use:**
1. Click the **"Chat"** tab
2. Type your message in the text area
3. Adjust settings:
   - **Max Tokens**: Response length (10-500)
   - **Context Messages**: How many previous messages to include (1-20)
4. Click **"Send Message"**
5. View the response with:
   - Conversation ID
   - Confidence score
   - Energy level
   - Thought vector
   - Operator used

**Features:**
- ✅ Continuous conversation (remembers context)
- ✅ Load conversation history
- ✅ Start new conversation
- ✅ Mood influences responses
- ✅ Adjustable context window

**Example conversation:**
```
You: Hello! Can you explain how neural networks work?
ALEN: [Responds with knowledge-grounded explanation]

You: Can you give me an example?
ALEN: [Provides example using conversation context]
```

### 2. 📚 **Training** - Teach ALEN New Knowledge

**What it does:**
- Train ALEN with facts and knowledge
- Batch training from text
- Comprehensive training across multiple domains

**How to use:**
1. Click **"Training"** tab
2. Enter knowledge in the text area (one fact per line)
3. Set confidence level (0.0-1.0)
4. Click **"Train"**

**Example training:**
```
Python is a high-level programming language
Machine learning is a subset of artificial intelligence
Neural networks are inspired by biological neurons
```

**Features:**
- ✅ Single fact training
- ✅ Batch training (multiple facts at once)
- ✅ Comprehensive training with predefined datasets
- ✅ Configurable confidence levels
- ✅ Training with emojis (30+ emoji concepts included!)

### 3. 🎭 **Mood & Emotions** - Emotional State Control

**What it does:**
- View current emotional state
- Adjust mood parameters
- See mood influence on responses
- Reset emotional state

**How to use:**
1. Click **"Mood"** tab
2. Click **"Load Current Mood"** to see state
3. Use sliders to adjust:
   - **Dopamine** (0.0-1.0): Reward/pleasure
   - **Serotonin** (0.0-1.0): Contentment/stability
   - **Cortisol** (0.0-1.0): Stress response
   - **Norepinephrine** (0.0-1.0): Alertness
4. Click **"Apply Mood Changes"**

**Mood Display:**
- Current Mood (e.g., "Content", "Curious", "Stressed")
- Reward Level %
- Stress Level %
- Curiosity Level %
- Energy Level %
- Perception Bias
- Current Emotion

**Features:**
- ✅ Real-time mood visualization
- ✅ 4 neurotransmitter controls
- ✅ Mood accumulation from conversations
- ✅ Demonstrate mood influence
- ✅ Reset to neutral state

### 4. 🎨 **Generation** - Create Content

**What it does:**
- Generate text, images, video, poetry
- Factual generation (no hallucinations)
- Universal explanations for different audiences
- Multi-modal content creation

**Generators available:**

#### **Text Generation**
- Natural language text from prompts
- Configurable max tokens and temperature

#### **Image Generation**
- Generate 64x64 pixel images from descriptions
- Visual concepts from text prompts

#### **Video Generation**
- Create video sequences (8-32 frames)
- Temporal motion patterns
- Configurable motion types

#### **Poetry Generation**
- Mood-aware poems
- Multiple styles: Haiku, Free Verse, Sonnet, Limerick
- Themes: Nature, Love, Tech, Existential, Joy

#### **Factual Answers** (Anti-Hallucination)
- Verified responses from knowledge base
- Cosine similarity verification
- Per-token confidence scores
- **ZERO HALLUCINATIONS** - only answers from trained knowledge

#### **Universal Explanations**
- Same knowledge, different audiences:
  - **Child**: Simple, concrete examples
  - **General**: Balanced approach
  - **Elder**: Practical, respectful
  - **Mathematician**: Formal, symbolic
  - **Expert**: Technical, precise
- Vocabulary simplification

**Features:**
- ✅ Multi-modal generation
- ✅ Knowledge-grounded responses
- ✅ Audience-aware explanations
- ✅ Mood-influenced poetry
- ✅ Verification metadata

### 5. 🧠 **Advanced Reasoning** - Complex Problem Solving

**What it does:**
- Mathematical reasoning
- Logical inference
- Chain-of-thought problem solving

**How to use:**
1. Click **"Advanced"** tab
2. **Math Reasoning**: Enter math problems (e.g., "2+2", "solve x^2=16")
3. **Chain of Thought**: Multi-step reasoning problems
4. View detailed reasoning steps

**Features:**
- ✅ Symbolic math solving
- ✅ AST-based math embedding
- ✅ Step-by-step reasoning chains
- ✅ Backward inference verification

### 6. 📦 **Memory** - View and Search Knowledge

**What it does:**
- View episodic memory (experiences)
- Search semantic memory (facts)
- Add new facts
- Clear memory

**How to use:**
1. Click **"Memory"** tab
2. **Semantic Memory**: Search facts by query
3. **Episodic Memory**: View top episodes
4. **Add Facts**: Store new knowledge directly

**Features:**
- ✅ Semantic search with similarity scoring
- ✅ Top episodes by energy
- ✅ Memory statistics
- ✅ Clear memory options

### 7. 💾 **Export** - Save Your Data

**What it does:**
- Export conversations to JSON
- Export episodic memory
- Export semantic memory
- List all exports

**How to use:**
1. Click **"Export"** tab
2. Select export type
3. Click export button
4. Download JSON file

**Export Types:**
- **Conversations**: All chat history
- **Episodic Memory**: All experiences
- **Semantic Memory**: All facts
- **List Exports**: View previous exports

**Features:**
- ✅ Human-readable JSON format
- ✅ Timestamped exports
- ✅ File size information
- ✅ Items count

## 🔌 API Endpoints

All features are accessible via REST API:

### Chat & Conversation
```
POST /chat                          - Send chat message
POST /conversation/get              - Get conversation history
GET  /conversation/list             - List all conversations
POST /conversation/clear            - Clear conversation
POST /system-prompt/update          - Update system prompt
GET  /system-prompt/get-default     - Get default prompt
```

### Training
```
POST /train                         - Train single fact
POST /train/batch                   - Batch training
POST /train/comprehensive           - Comprehensive training
POST /learn                         - Learn knowledge
POST /query                         - Query knowledge
```

### Emotions & Mood
```
GET  /emotions/state                - Get emotional state
POST /emotions/adjust               - Adjust mood
POST /emotions/demonstrate          - Demonstrate mood influence
POST /emotions/reset                - Reset to neutral
GET  /emotions/patterns             - Get mood patterns
```

### Generation
```
POST /generate/text                 - Generate text
POST /generate/image                - Generate image
POST /generate/video                - Generate video
POST /generate/poem                 - Generate poetry
POST /generate/factual              - Factual answer (verified)
POST /explain                       - Universal explanation
```

### Memory
```
POST /facts                         - Add fact
POST /facts/search                  - Search facts
GET  /memory/episodic/stats         - Episodic stats
GET  /memory/episodic/top/:limit    - Top episodes
DELETE /memory/semantic/clear       - Clear semantic memory
DELETE /memory/episodic/clear       - Clear episodic memory
```

### Export
```
POST /export/conversations          - Export conversations
POST /export/episodic               - Export episodic memory
POST /export/semantic               - Export semantic memory
GET  /export/list                   - List exports
```

### System
```
GET  /health                        - Health check
GET  /stats                         - System statistics
GET  /capabilities                  - AI capabilities
GET  /storage/stats                 - Storage statistics
```

## 🎯 Complete Workflow Example

### Example: Train → Chat → Export

#### 1. Train ALEN with Emoji Knowledge
```javascript
// API call
POST /train/batch
{
  "facts": [
    "😊 happy face means joy and positive emotion",
    "🚀 rocket means progress or launching something",
    "💡 light bulb means idea or inspiration",
    "🔥 fire emoji means something is hot or trending"
  ],
  "confidence": 0.95
}
```

Or use the **Training** tab:
- Paste the emoji facts
- Set confidence to 0.95
- Click "Train"

#### 2. Chat About Emojis
```javascript
// API call
POST /chat
{
  "message": "What does 🚀💡 mean?",
  "max_tokens": 100,
  "include_context": 5
}
```

Or use the **Chat** tab:
- Type: "What does 🚀💡 mean?"
- Click "Send Message"
- ALEN responds with knowledge-grounded answer about rocket and light bulb

#### 3. Export Conversation
```javascript
// API call
POST /export/conversations
```

Or use the **Export** tab:
- Click "Export Conversations"
- Download JSON file with full chat history

## ⚙️ Configuration

### Environment Variables
```bash
# Port (default: 3000)
export ALEN_PORT=3000

# Host (default: 0.0.0.0 - all interfaces)
export ALEN_HOST=0.0.0.0

# Vector dimension (default: 128)
export ALEN_DIMENSION=128

# Data directory (default: ./data)
export ALEN_DATA_DIR=./data
```

### Custom Port Example
```bash
export ALEN_PORT=8080
./target/release/alen

# Access at http://localhost:8080
```

## 🎨 Multi-Step Reasoning Example

Using the integrated ChainOfThought + ReasoningEngine:

```javascript
// Via API (internal processing)
POST /chat
{
  "message": "celebrate a big achievement and show strength",
  "max_tokens": 150
}

// ALEN internally:
// Step 1: Decompose → ["celebrate a big achievement", "show strength"]
// Step 2: Each step gets knowledge-anchored latent
// Step 3: Verify each step against emoji knowledge:
//         - "celebrate" → 🎉 (verified)
//         - "strength" → 💪 (verified)
// Step 4: Combine verified steps → Final response
```

## 🚀 Advanced Features

### 1. **Knowledge-Anchored Image Generation**
```javascript
POST /generate/image
{
  "prompt": "🚀⭐ rocket to stars",
  "include_metadata": true
}

// Returns:
// - 64x64 image data
// - Verification: true/false
// - Confidence: 0.95
// - Supporting facts: ["rocket to stars means aiming high or big success"]
```

### 2. **Temporal Video Generation**
```javascript
POST /generate/video
{
  "prompt": "planets orbiting in solar system",
  "num_frames": 8,
  "fps": 24
}

// Uses latent propagation: h_{t+1} = h_t + Δh_temporal
// All frames knowledge-verified
```

### 3. **Vocabulary Simplification**
```javascript
POST /explain
{
  "query": "explain photosynthesis",
  "audience": "child",
  "max_sentences": 5
}

// Child: "how plants make food using light"
// Expert: "biochemical process converting light energy..."
```

## 📊 System Architecture

```
Browser (Web Interface)
    ↓ HTTP Requests
Web Server (Axum)
    ↓ API Calls
AppState
    ├─ ReasoningEngine (Multi-step + Knowledge Anchoring)
    ├─ SemanticMemory (Trained Knowledge)
    ├─ EpisodicMemory (Experiences)
    ├─ ConversationManager (Chat History)
    ├─ EmotionSystem (Mood State)
    └─ ContentGenerator (Multi-modal)
```

## 🎉 Complete Feature List

✅ **Chat Interface** - Natural conversation with context
✅ **Multi-Step Reasoning** - ChainOfThought + Knowledge Verification
✅ **Emoji Knowledge** - 30+ emoji concepts trained
✅ **Anti-Hallucination** - Verified factual responses only
✅ **Multi-Audience** - Child, General, Expert, Mathematician, Elder
✅ **Mood-Aware** - Emotional state influences responses
✅ **Multi-Modal Generation** - Text, Images, Video, Poetry
✅ **Knowledge Grounding** - Every output traceable to training
✅ **Temporal Consistency** - Video with latent propagation
✅ **Creativity Control** - α parameter (0=factual, 1=creative)
✅ **Export/Import** - Save conversations and memories
✅ **Real-time Stats** - System health and performance

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Change port
export ALEN_PORT=8080
./target/release/alen
```

### Can't Access from Other Devices
```bash
# Ensure binding to all interfaces (default)
export ALEN_HOST=0.0.0.0

# Find your IP
ip addr show  # Linux
ipconfig      # Windows
ifconfig      # Mac

# Access from other device
http://YOUR_IP:3000
```

### Web Interface Not Loading
```bash
# Check web directory exists
ls -la web/

# Should contain:
# - index.html
# - (other assets if any)
```

## 📝 Summary

**To use the full web interface:**

1. **Build & Run**: `cargo run --release`
2. **Open Browser**: `http://localhost:3000`
3. **Start Chatting**: Click "Chat" tab → Type message → Send
4. **Train Knowledge**: Click "Training" tab → Add facts → Train
5. **Adjust Mood**: Click "Mood" tab → Adjust sliders → Apply
6. **Generate Content**: Click "Generation" tab → Choose type → Generate
7. **Export Data**: Click "Export" tab → Select type → Export

**Everything is accessible through the web interface at `http://localhost:3000`!**

🚀 **ALEN is now fully accessible via web browser with all features including chat, training, generation, mood control, and export!**

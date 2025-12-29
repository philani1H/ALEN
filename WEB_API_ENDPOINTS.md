# Web API Endpoints - Complete Reference

## Server Configuration

**Default URL**: `http://localhost:3000`  
**Host**: `0.0.0.0` (configurable via `ALEN_HOST`)  
**Port**: `3000` (configurable via `ALEN_PORT`)

## Frontend Configuration

**Location**: `/workspaces/ALEN/web/index.html`  
**API Base URL**: `http://localhost:3000` (line 903)

## Available Endpoints

### Health & Status

#### GET /health
Health check endpoint
```javascript
const result = await fetch('http://localhost:3000/health');
```

#### GET /stats
System statistics
```javascript
const result = await fetch('http://localhost:3000/stats');
```

#### GET /capabilities
System capabilities
```javascript
const result = await fetch('http://localhost:3000/capabilities');
```

#### GET /storage/stats
Storage statistics
```javascript
const result = await fetch('http://localhost:3000/storage/stats');
```

### Conversation & Chat

#### POST /chat
Main chat endpoint
```javascript
const result = await fetch('http://localhost:3000/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        message: "Hello",
        conversation_id: "optional-id"
    })
});
```

#### POST /conversation/get
Get conversation history
```javascript
const result = await fetch('http://localhost:3000/conversation/get', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        conversation_id: "conversation-id"
    })
});
```

#### GET /conversation/list
List all conversations
```javascript
const result = await fetch('http://localhost:3000/conversation/list');
```

#### POST /conversation/clear
Clear conversation
```javascript
const result = await fetch('http://localhost:3000/conversation/clear', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        conversation_id: "conversation-id"
    })
});
```

#### POST /feedback
Submit feedback
```javascript
const result = await fetch('http://localhost:3000/feedback', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        conversation_id: "id",
        message_index: 0,
        rating: 5,
        comment: "Great response!"
    })
});
```

### Training

#### POST /train
Train on single problem
```javascript
const result = await fetch('http://localhost:3000/train', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        problem: "What is 2+2?",
        solution: "4",
        explanation: "Addition of two and two"
    })
});
```

#### POST /train/batch
Train on multiple problems
```javascript
const result = await fetch('http://localhost:3000/train/batch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        problems: [
            {problem: "2+2", solution: "4"},
            {problem: "3+3", solution: "6"}
        ]
    })
});
```

#### POST /train/comprehensive
Comprehensive training
```javascript
const result = await fetch('http://localhost:3000/train/comprehensive', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        problems: [...],
        epochs: 10
    })
});
```

#### POST /train/with-images
Train with generated images
```javascript
const result = await fetch('http://localhost:3000/train/with-images', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        problems: [...],
        image_prompts: [...]
    })
});
```

#### POST /train/with-videos
Train with generated videos
```javascript
const result = await fetch('http://localhost:3000/train/with-videos', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        problems: [...],
        video_prompts: [...]
    })
});
```

#### POST /train/self-supervised
Self-supervised learning
```javascript
const result = await fetch('http://localhost:3000/train/self-supervised', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        data: [...]
    })
});
```

### Knowledge

#### POST /learn
Learn knowledge
```javascript
const result = await fetch('http://localhost:3000/learn', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        knowledge: "The sky is blue"
    })
});
```

#### POST /query
Query knowledge
```javascript
const result = await fetch('http://localhost:3000/query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        query: "What color is the sky?"
    })
});
```

### Emotions & Mood

#### GET /emotions/state
Get emotional state
```javascript
const result = await fetch('http://localhost:3000/emotions/state');
```

#### POST /emotions/adjust
Adjust mood
```javascript
const result = await fetch('http://localhost:3000/emotions/adjust', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        mood: "happy",
        intensity: 0.8
    })
});
```

#### POST /emotions/demonstrate
Demonstrate mood influence
```javascript
const result = await fetch('http://localhost:3000/emotions/demonstrate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        input: "Tell me a story"
    })
});
```

#### POST /emotions/reset
Reset mood
```javascript
const result = await fetch('http://localhost:3000/emotions/reset', {
    method: 'POST'
});
```

#### GET /emotions/patterns
Get mood patterns
```javascript
const result = await fetch('http://localhost:3000/emotions/patterns');
```

### Generation

#### POST /generate/text
Generate text
```javascript
const result = await fetch('http://localhost:3000/generate/text', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        prompt: "Write a story about..."
    })
});
```

#### POST /generate/image
Generate image
```javascript
const result = await fetch('http://localhost:3000/generate/image', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        prompt: "A beautiful sunset"
    })
});
```

#### POST /generate/video
Generate video
```javascript
const result = await fetch('http://localhost:3000/generate/video', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        prompt: "A cat playing"
    })
});
```

#### POST /generate/poem
Generate poem
```javascript
const result = await fetch('http://localhost:3000/generate/poem', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        theme: "nature"
    })
});
```

#### POST /generate/factual
Generate factual answer (no hallucinations)
```javascript
const result = await fetch('http://localhost:3000/generate/factual', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        question: "What is the capital of France?"
    })
});
```

### Memory

#### POST /facts
Add semantic fact
```javascript
const result = await fetch('http://localhost:3000/facts', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        fact: "Paris is the capital of France",
        confidence: 1.0
    })
});
```

#### POST /facts/search
Search facts
```javascript
const result = await fetch('http://localhost:3000/facts/search', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        query: "capital of France",
        limit: 10
    })
});
```

#### GET /memory/episodic/stats
Get episodic memory statistics
```javascript
const result = await fetch('http://localhost:3000/memory/episodic/stats');
```

#### GET /memory/episodic/top/:limit
Get top episodes
```javascript
const result = await fetch('http://localhost:3000/memory/episodic/top/10');
```

#### DELETE /memory/episodic/clear
Clear episodic memory
```javascript
const result = await fetch('http://localhost:3000/memory/episodic/clear', {
    method: 'DELETE'
});
```

#### DELETE /memory/semantic/clear
Clear semantic memory
```javascript
const result = await fetch('http://localhost:3000/memory/semantic/clear', {
    method: 'DELETE'
});
```

### Export

#### POST /export/conversations
Export conversations
```javascript
const result = await fetch('http://localhost:3000/export/conversations', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        format: "json"
    })
});
```

#### POST /export/episodic
Export episodic memory
```javascript
const result = await fetch('http://localhost:3000/export/episodic', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        format: "json"
    })
});
```

#### POST /export/semantic
Export semantic memory
```javascript
const result = await fetch('http://localhost:3000/export/semantic', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        format: "json"
    })
});
```

#### GET /export/list
List exports
```javascript
const result = await fetch('http://localhost:3000/export/list');
```

### Inference

#### POST /infer
Perform reasoning
```javascript
const result = await fetch('http://localhost:3000/infer', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        input: "What is 2+2?"
    })
});
```

### Multimodal

#### POST /multimodal/image
Process image
```javascript
const result = await fetch('http://localhost:3000/multimodal/image', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        image_data: "base64..."
    })
});
```

#### POST /multimodal/audio
Process audio
```javascript
const result = await fetch('http://localhost:3000/multimodal/audio', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        audio_data: "base64..."
    })
});
```

#### POST /multimodal/video
Process video
```javascript
const result = await fetch('http://localhost:3000/multimodal/video', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        video_data: "base64..."
    })
});
```

#### POST /multimodal/fuse
Fuse modalities
```javascript
const result = await fetch('http://localhost:3000/multimodal/fuse', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        modalities: {
            text: "...",
            image: "...",
            audio: "..."
        }
    })
});
```

### Operators

#### GET /operators
Get operator performance
```javascript
const result = await fetch('http://localhost:3000/operators');
```

### Control

#### POST /bias
Set bias parameters
```javascript
const result = await fetch('http://localhost:3000/bias', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        bias_type: "exploration",
        value: 0.5
    })
});
```

#### POST /bias/reset
Reset bias
```javascript
const result = await fetch('http://localhost:3000/bias/reset', {
    method: 'POST'
});
```

#### POST /learning/reset
Reset learning rate
```javascript
const result = await fetch('http://localhost:3000/learning/reset', {
    method: 'POST'
});
```

### System Prompts

#### POST /system-prompt/update
Update system prompt
```javascript
const result = await fetch('http://localhost:3000/system-prompt/update', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        conversation_id: "id",
        system_prompt: "You are a helpful assistant"
    })
});
```

#### POST /system-prompt/set-default
Set default system prompt
```javascript
const result = await fetch('http://localhost:3000/system-prompt/set-default', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        system_prompt: "You are a helpful assistant"
    })
});
```

#### GET /system-prompt/get-default
Get default system prompt
```javascript
const result = await fetch('http://localhost:3000/system-prompt/get-default');
```

### Verification

#### POST /verify/statement
Verify statement
```javascript
const result = await fetch('http://localhost:3000/verify/statement', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        statement: "The Earth is flat"
    })
});
```

### Explanation

#### POST /explain
Universal explanation (multi-audience)
```javascript
const result = await fetch('http://localhost:3000/explain', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        concept: "quantum mechanics",
        audience: "beginner"
    })
});
```

## CORS Configuration

The server is configured with CORS to allow:
- All origins
- All methods
- All headers

## Static Files

The web interface is served from `/web` directory at the root path `/`.

## Error Handling

All endpoints return JSON responses with the following structure:

### Success Response
```json
{
    "success": true,
    "data": {...}
}
```

### Error Response
```json
{
    "success": false,
    "error": "Error message"
}
```

## Testing Endpoints

You can test endpoints using:

### curl
```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### Browser Console
```javascript
fetch('http://localhost:3000/health')
  .then(r => r.json())
  .then(console.log);
```

### Web Interface
Open `http://localhost:3000` in your browser to use the full web interface.

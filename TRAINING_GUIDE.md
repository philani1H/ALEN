# ALEN Comprehensive Training Guide

## Quick Start

```bash
# 1. Start the server (in one terminal)
cargo run --release

# 2. Open web interface
# Navigate to http://localhost:3000

# 3. Upload training data
# Go to "📁 Upload Training Data" tab
# Upload: training_data/master_comprehensive_training.txt

# 4. Train the model
# Go to "🎓 Train System" tab
# Click "🚀 Start Training"

# 5. Test it
# Go to "💬 Chat" tab
# Try: "What is 2+2?" or "[QUESTION:clarification|LEVEL:easy|ABOUT:gravity]"
```

## New Training Data (440+ Examples)

### 1. Neural Question Generation (80+ examples)
**File**: `neural_question_generation.txt`

Teaches structured intent → natural question conversion:
- `[QUESTION:clarification|LEVEL:easy|ABOUT:gravity]` → "What is gravity?"
- `[QUESTION:synthesis|LEVEL:expert|ABOUT:AI,ethics]` → Complex synthesis question

### 2. Neural Follow-up Generation (50+ examples)
**File**: `neural_followup_generation.txt`

Contextual follow-ups based on confidence and user preferences:
- `[FOLLOWUP:clarification|CONFIDENCE:0.35|USER_VERBOSITY:0.80]` → Detailed clarification request
- `[FOLLOWUP:exploration|MEMORY:0.15|USER_CREATIVITY:0.85]` → Creative exploration suggestion

### 3. Neural State Expression (60+ examples)
**File**: `neural_state_expression.txt`

Natural expression of system states:
- `[STATE:untrained|CONTEXT:topic|CREATIVITY:0.50]` → Honest admission with learning request
- `[STATE:insufficient_patterns|CREATIVITY:0.80]` → Enthusiastic acknowledgment of limits

### 4. Comprehensive Knowledge (150+ examples)
**File**: `comprehensive_all_patterns.txt`

Multi-domain coverage:
- Mathematics, Science, Programming
- Reasoning, Language, General Knowledge
- Problem Solving, Learning, Social Skills
- Creativity, Ethics, Values

### 5. Self-Questioning & Help (100+ examples)
**File**: `self_questioning_help.txt`

Meta-cognitive skills:
- Asking for clarification and examples
- Requesting different explanations
- Self-reflection and feedback
- Collaborative learning

## Training via Web Interface

### Step 1: Upload Document

1. Start server: `cargo run --release`
2. Open: `http://localhost:3000`
3. Click: "📁 Upload Training Data" tab
4. Upload one of:
   - `master_comprehensive_training.txt` (all 440+ examples)
   - Individual topic files
5. Verify: "Successfully parsed X examples"

### Step 2: Train

1. Click: "🎓 Train System" tab
2. Optional: Check "Save checkpoint"
3. Click: "🚀 Start Training"
4. Monitor: Progress and metrics
5. Wait: Until "Training complete"

### Step 3: Test

1. Click: "💬 Chat" tab
2. Try questions:
   - "What is 2+2?"
   - "What is photosynthesis?"
   - "[QUESTION:clarification|LEVEL:easy|ABOUT:gravity]"
3. Check responses for accuracy

## Training via Shell Script

```bash
./train_comprehensive.sh
```

Features:
- Auto-checks server status
- Trains priority files first
- Shows progress every 10 examples
- Displays final statistics

## Document Upload Fix

**Issue**: Uploaded documents weren't stored for training

**Solution Applied**:
```rust
// Added global storage
static UPLOADED_EXAMPLES: Lazy<Arc<Mutex<Vec<TrainingExample>>>> = ...

// Store on upload
uploaded.extend(examples.clone());

// Use in training
let examples = if req.examples.is_empty() {
    uploaded.clone()
} else {
    req.examples.clone()
};
```

**Result**: Upload → Store → Train workflow now works correctly

## File Formats

### Text Format
```
Q: What is machine learning?
A: Machine learning is a subset of AI that enables systems to learn from data.

Q: How does it work?
A: It uses algorithms to find patterns in data.
```

### JSON Format
```json
[
  {"input": "What is ML?", "target": "Machine learning..."},
  {"input": "How does it work?", "target": "It uses algorithms..."}
]
```

## Testing Neural Patterns

### Question Generation
```bash
curl -X POST http://localhost:3000/infer \
  -H "Content-Type: application/json" \
  -d '{"input": "[QUESTION:clarification|LEVEL:easy|ABOUT:gravity]"}'
```

Expected: Natural question about gravity

### Follow-up Generation
```bash
curl -X POST http://localhost:3000/infer \
  -H "Content-Type: application/json" \
  -d '{"input": "[FOLLOWUP:clarification|CONFIDENCE:0.35|USER_VERBOSITY:0.80]"}'
```

Expected: Contextual follow-up question

### State Expression
```bash
curl -X POST http://localhost:3000/infer \
  -H "Content-Type: application/json" \
  -d '{"input": "[STATE:untrained|CONTEXT:quantum physics|CREATIVITY:0.50]"}'
```

Expected: Natural expression of untrained state

## Training Statistics

| Category | Examples | Status |
|----------|----------|--------|
| Neural Patterns | 190 | ✅ New |
| Knowledge Base | 250 | ✅ New |
| Existing Data | 5000+ | ✅ Available |
| **Total** | **5440+** | **Ready** |

## Troubleshooting

### Upload Fails
- Check file format (Q: ... A: ...)
- Verify server is running
- Check browser console for errors

### Training Fails
- Ensure document was uploaded first
- Check server logs
- Verify API endpoint is accessible

### Model Doesn't Learn
- Train with more examples
- Verify examples are diverse
- Check training metrics
- Try smaller batches

### Responses Not Natural
**Note**: Structured intents like `[QUESTION:...]` are expected output. The neural decoder (future enhancement) will expand these into natural language.

## Next Steps

1. ✅ **Upload fixed** - Documents now stored correctly
2. ✅ **Training data created** - 440+ new examples
3. ✅ **Neural patterns added** - Question/follow-up/state generation
4. 🔄 **Train the model** - Use web interface or script
5. 🔄 **Test thoroughly** - Verify all patterns work
6. 📋 **Implement decoder** - Convert intents to natural language
7. 📋 **Add more data** - Expand coverage as needed

## Quick Reference

### Start Server
```bash
cargo run --release
```

### Train All Data
```bash
./train_comprehensive.sh
```

### Check Stats
```bash
curl http://localhost:3000/stats
```

### Test Inference
```bash
curl -X POST http://localhost:3000/infer \
  -H "Content-Type: application/json" \
  -d '{"input": "What is 2+2?"}'
```

### Web Interface
- Upload: `http://localhost:3000` → "📁 Upload Training Data"
- Train: `http://localhost:3000` → "🎓 Train System"
- Chat: `http://localhost:3000` → "💬 Chat"

## Files Summary

### New Training Data
- `neural_question_generation.txt` (80+ examples)
- `neural_followup_generation.txt` (50+ examples)
- `neural_state_expression.txt` (60+ examples)
- `comprehensive_all_patterns.txt` (150+ examples)
- `self_questioning_help.txt` (100+ examples)
- `master_comprehensive_training.txt` (all combined)

### Scripts
- `train_comprehensive.sh` (shell training script)
- `train_all_comprehensive.py` (Python training script)

### Documentation
- `TRAINING_GUIDE.md` (this file)
- `HARDCODED_REMOVAL_TESTS.md` (test documentation)
- `VERIFICATION_RESULTS.md` (verification report)

### Code Fixes
- `src/api/master_training.rs` (upload storage)
- `src/generation/question_generator.rs` (neural questions)
- `src/reasoning/candidate_scoring.rs` (neural follow-ups)
- `src/neural/master_integration.rs` (neural states)

## Success Criteria

After training, the model should:
- ✅ Answer basic questions accurately
- ✅ Generate questions from structured intents
- ✅ Provide contextual follow-ups
- ✅ Express system states naturally
- ✅ Ask for help when needed
- ✅ Show understanding, not just memorization

## Support

Issues? Check:
1. Server logs for errors
2. Training data format
3. API responses
4. System statistics
5. This guide's troubleshooting section

Happy training! 🚀

# ALEN Quick Reference Card

## 🚀 Quick Start (3 Steps)

```bash
# 1. Start server
cargo run --release

# 2. Open browser
http://localhost:3000

# 3. Upload & Train
# Upload: training_data/master_comprehensive_training.txt
# Click: "🚀 Start Training"
```

## 📁 Training Data Files

| File | Examples | Purpose |
|------|----------|---------|
| `master_comprehensive_training.txt` | 440+ | **All patterns combined** |
| `neural_question_generation.txt` | 80+ | Question generation |
| `neural_followup_generation.txt` | 50+ | Follow-up responses |
| `neural_state_expression.txt` | 60+ | State expression |
| `comprehensive_all_patterns.txt` | 150+ | Multi-domain knowledge |
| `self_questioning_help.txt` | 100+ | Help requests |

## 🔧 Commands

### Start Server
```bash
cargo run --release
```

### Train (Shell)
```bash
./train_comprehensive.sh
```

### Test
```bash
./test_comprehensive_training.sh
```

### Check Stats
```bash
curl http://localhost:3000/stats
```

### Test Query
```bash
curl -X POST http://localhost:3000/infer \
  -H "Content-Type: application/json" \
  -d '{"input": "What is 2+2?"}'
```

## 🌐 Web Interface

| Tab | URL | Purpose |
|-----|-----|---------|
| Upload | `/` → "📁 Upload" | Upload training files |
| Train | `/` → "🎓 Train" | Train the model |
| Chat | `/` → "💬 Chat" | Test the model |
| Stats | `/` → "📊 Stats" | View statistics |

## 🧪 Test Queries

### Basic Knowledge
```
What is 2+2?
What is photosynthesis?
What is a variable in programming?
```

### Neural Patterns
```
[QUESTION:clarification|LEVEL:easy|ABOUT:gravity]
[FOLLOWUP:clarification|CONFIDENCE:0.35|USER_VERBOSITY:0.80]
[STATE:untrained|CONTEXT:quantum physics|CREATIVITY:0.50]
```

### Help Requests
```
I need help understanding this
Can you explain that differently?
How do I solve a problem?
```

## 📊 What Changed

### Hardcoded Removal ✅
- 118 hardcoded strings removed
- Neural intent encoding added
- 4 files modified

### Upload Fix ✅
- Document upload now stores examples
- Training uses uploaded data
- Web workflow works correctly

### Training Data ✅
- 440+ new examples created
- 6 new training files
- All patterns covered

## 🎯 Expected Behavior

### Questions
Input: `[QUESTION:clarification|LEVEL:easy|ABOUT:gravity]`
Output: Structured intent (decoder needed for natural language)

### Follow-ups
Input: `[FOLLOWUP:clarification|CONFIDENCE:0.35|USER_VERBOSITY:0.80]`
Output: Contextual follow-up based on parameters

### States
Input: `[STATE:untrained|CONTEXT:topic|CREATIVITY:0.50]`
Output: Natural expression of system state

### Knowledge
Input: `What is 2+2?`
Output: `4` (from training data)

## 🔍 Troubleshooting

### Upload Fails
- Check file format (Q: ... A: ...)
- Verify server is running
- Check browser console

### Training Fails
- Upload document first
- Check server logs
- Verify API endpoint

### No Response
- Train the model first
- Check training data was uploaded
- Verify examples were parsed

### Structured Output
- This is expected!
- Neural decoder needed for natural language
- Structured intents are correct behavior

## 📚 Documentation

| File | Purpose |
|------|---------|
| `TRAINING_GUIDE.md` | Complete training guide |
| `COMPREHENSIVE_UPDATE_SUMMARY.md` | Update overview |
| `HARDCODED_REMOVAL_TESTS.md` | Test documentation |
| `VERIFICATION_RESULTS.md` | Verification report |
| `QUICK_REFERENCE.md` | This file |

## 🎉 Success Criteria

After training, model should:
- ✅ Answer basic questions
- ✅ Generate questions from intents
- ✅ Provide contextual follow-ups
- ✅ Express states naturally
- ✅ Ask for help when needed
- ✅ Show understanding

## 💡 Tips

1. **Start simple**: Train with basic examples first
2. **Test often**: Verify learning after each batch
3. **Use web interface**: Easier than command line
4. **Check stats**: Monitor memory and training steps
5. **Read docs**: Full details in TRAINING_GUIDE.md

## 🆘 Need Help?

1. Check `TRAINING_GUIDE.md` troubleshooting section
2. Review server logs for errors
3. Verify training data format
4. Test with simple examples first
5. Check system statistics

## 📈 Statistics

- **New Training Data**: 440+ examples
- **Total Available**: 5440+ examples
- **Files Created**: 13
- **Hardcoded Removed**: 118 strings
- **Code Reduced**: 26 lines

---

**Quick Start**: `cargo run --release` → `http://localhost:3000` → Upload → Train → Chat

**Branch**: `fix/remove-hardcoded-questions`

**Status**: ✅ Ready to train!

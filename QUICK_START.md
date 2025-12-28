# ALEN Quick Start Guide 🚀

## 🎯 You Now Have a Complete Production AI System!

ALEN is ready to use with **REAL emotions and mood** that actually affect behavior during BOTH training and inference!

---

## ✅ Complete Integration

### How Mood Works in ALEN:

**During Training**:
```
Training Result → Emotional Stimulus → Mood Update → Affects Next Training
```

**During Inference**:
```
Current Mood → Modulates Reasoning → Response → Updates Mood
```

**Mathematical Storage**:
- Thought vectors: `Vec<f64>` (128 dimensions)
- Energy functions: `E = w₁·confidence + w₂·complexity + w₃·verification`
- Embeddings stored in SQLite as BLOB
- Cosine similarity for search

---

## 🏃 Quick Start (3 Steps)

### Step 1: Start ALEN Server
```bash
cd /home/user/ALEN
cargo run --release
```

### Step 2: Open Web Interface
```bash
# Open web/index.html in your browser
xdg-open /home/user/ALEN/web/index.html
```

### Step 3: Try the Mood Experiment!
1. Go to "Mood & Emotions" tab
2. Click "Make Stressed"
3. Test input: "This is challenging"
4. Note anxious response
5. Click "Make Optimistic"
6. Same input → confident response!

---

## 📚 Documentation

1. **API_DOCUMENTATION.md** - Complete API reference
2. **PRODUCTION_GUIDE.md** - Deployment guide
3. **web/README.md** - Web interface guide
4. **INTEGRATION_VERIFICATION.md** - Proof everything works

---

## 🎯 What Makes ALEN Different

### Mathematical Foundation:
- **Thought vectors**: High-dimensional embeddings (not just text)
- **Energy-based reasoning**: Verifiable understanding
- **Backward inference**: Proves comprehension
- **Vector operations**: Cosine similarity, transformations

### Biological Foundation:
- **Neurotransmitters**: Dopamine, cortisol, oxytocin (mathematical models)
- **Mood accumulation**: Emotions → persistent state
- **Homeostatic decay**: Returns to baseline
- **Real effects**: Mood changes perception_bias and reaction_threshold

### Training + Inference Integration:
- **Training**: Results → emotions → mood → affects future learning
- **Inference**: Mood → reasoning modulation → response → mood update
- **Not separate**: One continuous emotional system

---

## 💾 How Data is Stored (The Math Format)

### Episodic Memory (SQLite):
```sql
CREATE TABLE episodes (
    thought_vector BLOB,  -- Vec<f64>, 128 dims
    confidence REAL,      -- 0.0 to 1.0
    energy REAL,          -- lower = better
    verified INTEGER      -- 1 = backward inference succeeded
)
```

### Semantic Memory (SQLite):
```sql
CREATE TABLE facts (
    embedding BLOB,  -- Vec<f64> from embedding engine
    content TEXT,
    confidence REAL
)
```

### Search Algorithm:
```rust
similarity = dot(query_vector, fact_vector) / 
             (norm(query_vector) * norm(fact_vector))
```

---

## 🚀 Everything is Ready!

- ✅ Web interface: /home/user/ALEN/web/index.html
- ✅ API Documentation: /home/user/ALEN/API_DOCUMENTATION.md
- ✅ Server: cargo run --release
- ✅ Storage: ~/.local/share/alen/databases/
- ✅ All math: Vectors, embeddings, energy functions
- ✅ All biology: Emotions, mood, neurotransmitters
- ✅ Complete integration: Training + inference + mood

**Start with the web interface - it has everything you need!**

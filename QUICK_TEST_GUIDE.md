# ALEN Web Interface - Quick Test Guide

## 🚀 Quick Start

**1. Access the Web Interface**
```
https://3000--019b63e4-04b9-74e1-a056-24f16abdba59.eu-central-1-01.gitpod.dev
```

**2. Server is Running**
- Backend: Rust/Axum on port 3000
- Status: ✅ Healthy and ready

## 🧪 5-Minute Test Plan

### Test 1: Math Formula (30 seconds)
1. Click **Chat** tab
2. Click **"Math Formula"** example button
3. Click **"Send Message"**
4. ✅ **Verify**: Formula renders with proper LaTeX formatting

**Expected**: Beautiful mathematical formula with proper symbols

---

### Test 2: Image Generation (1 minute)
1. Click **"Generate Media"** tab
2. Scroll to **"Generate Image"** section
3. Enter prompt: `sunset over mountains 🌅`
4. Click **"Generate Image"**
5. ✅ **Verify**: Image displays in styled canvas

**Expected**: Image appears in professional container with shadows

---

### Test 3: Video Player (1 minute)
1. Stay in **"Generate Media"** tab
2. Scroll to **"Generate Video"** section
3. Enter prompt: `ocean waves 🌊`
4. Select motion: **"circular"**
5. Click **"Generate Video"**
6. Click **▶️ Play** button
7. ✅ **Verify**: Video plays with frame counter

**Expected**: Animated video with working controls

---

### Test 4: Emoji Support (30 seconds)
1. Go back to **Chat** tab
2. Click **"With Emojis"** example button
3. Click **"Send Message"**
4. ✅ **Verify**: Response includes emojis naturally

**Expected**: Response with 🧠 💡 and other relevant emojis

---

### Test 5: Reasoning Chain (1 minute)
1. In **Chat** tab, type: `How does machine learning work?`
2. Click **"Send Message"**
3. ✅ **Verify**: See reasoning steps displayed

**Expected**:
```
🧠 Reasoning Process:
Step 1: Analyzed input using [operator] operator
Step 2: Processed with confidence: XX.X%
Step 3: Generated response in current mood: [mood]
```

---

### Test 6: Professional Formatting (1 minute)
1. In **Chat** tab, type: `Explain the formula E=mc^2`
2. Click **"Send Message"**
3. ✅ **Verify**: 
   - Formula rendered
   - Professional chat bubbles
   - Metadata shown (confidence, mood, emotion)

**Expected**: Beautiful formatted response with all elements

---

## ✅ Quick Checklist

After 5 minutes, you should have verified:

- [ ] LaTeX formulas render correctly
- [ ] Images display in canvas
- [ ] Video player works with controls
- [ ] Emojis appear naturally
- [ ] Reasoning chains are visible
- [ ] Professional chat formatting
- [ ] Metadata displays (confidence, mood, emotion)
- [ ] Smooth animations
- [ ] Responsive design

## 🎯 Key Features to Notice

### Visual Elements
- **Purple gradient** for user messages (right side)
- **Gray background** for ALEN responses (left side)
- **Smooth fade-in** animations
- **Professional shadows** on media
- **Clean typography** throughout

### Functional Elements
- **MathJax** rendering formulas
- **Canvas API** displaying images/videos
- **Highlight.js** for code blocks
- **Marked.js** for markdown
- **Emoji support** with proper sizing

### Information Display
- **Confidence percentage** (📊)
- **Current mood** (😊)
- **Current emotion** (💭)
- **Operator used** (⚙️)
- **Reasoning steps** (🧠)

## 🔍 Troubleshooting

### Formula not rendering?
- Wait 2-3 seconds for MathJax to load
- Refresh the page if needed

### Image not showing?
- Check that generation completed (look for success message)
- Try a smaller size (64 or 128)

### Video not playing?
- Ensure generation finished
- Click Play button
- Check frame counter updates

### Emojis not visible?
- Check browser supports UTF-8
- Try different emojis

## 📱 Mobile Testing

If testing on mobile:
- Interface is responsive
- Touch controls work
- Scrolling is smooth
- Buttons are touch-friendly

## 🎨 Style Verification

Look for these design elements:
- ✅ Rounded corners on containers
- ✅ Box shadows on media
- ✅ Gradient backgrounds
- ✅ Smooth transitions
- ✅ Professional color scheme (purple/gray)

## 📊 Performance Check

The interface should:
- ✅ Load quickly
- ✅ Respond immediately to clicks
- ✅ Render formulas within 2 seconds
- ✅ Display images/videos smoothly
- ✅ Animate without lag

## 🎓 Example Questions to Try

### Mathematics
```
- What is the Pythagorean theorem?
- Explain derivatives
- What is Euler's identity?
```

### Physics
```
- What is Newton's second law?
- Explain the wave equation
- What is E=mc^2?
```

### Creative
```
- Generate an image of a neural network 🧠
- Create a video of flowing water 🌊
- Explain AI with emojis 🤖💡
```

### Professional
```
- How do you write a professional email? 📧
- What makes a good presentation? 📊
- Explain active listening 👂
```

## ✨ Success Criteria

**You'll know it's working when**:
1. Formulas look like textbook quality
2. Images appear in styled containers
3. Videos play smoothly with controls
4. Emojis integrate naturally
5. Reasoning steps are clear
6. Everything looks professional

## 🚀 Ready to Test!

Open the URL and start with Test 1. Each test takes 30 seconds to 1 minute.

**Total time**: ~5 minutes for complete verification

**URL**: [https://3000--019b63e4-04b9-74e1-a056-24f16abdba59.eu-central-1-01.gitpod.dev](https://3000--019b63e4-04b9-74e1-a056-24f16abdba59.eu-central-1-01.gitpod.dev)

---

## 📞 Support

If you encounter issues:
1. Check browser console for errors
2. Verify server is running (check health endpoint)
3. Refresh the page
4. Try a different browser

**Server Health Check**:
```bash
curl http://localhost:3000/health
```

Expected: `{"service":"deliberative-ai","status":"healthy","version":"0.1.0"}`

---

**Happy Testing! 🎉**

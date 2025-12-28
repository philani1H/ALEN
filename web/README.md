# ALEN Web Interface

## Quick Start

1. **Start the ALEN server**:
```bash
cd /home/user/ALEN
cargo run --release
```

2. **Open the web interface**:
   - Open `web/index.html` in your browser
   - Or visit: `file:///home/user/ALEN/web/index.html`
   - **Note**: Server must be running on `localhost:3000`

3. **Start using ALEN**!

## Features

### 📚 Training Tab
- **Single Problem Training**: Teach ALEN one concept at a time
- **Batch Training**: Train on multiple examples efficiently
- **Image-Based Training**: ALEN generates and learns from images
- **Example buttons**: Quick-load common training scenarios

### 💬 Chat Tab
- Natural conversation with ALEN
- Context-aware responses
- Conversation history
- Adjustable response length
- New conversation management

### 🎭 Mood & Emotions Tab
- **Real-time mood display**: See ALEN's emotional state
- **Mood adjustment**: Change reward, stress, curiosity, energy levels
- **Preset moods**: Quick-switch to Optimistic, Stressed, or Neutral
- **Mood experiments**: Test how mood affects interpretation
- **Mood history**: View patterns over time

**Try this experiment**:
1. Click "Make Stressed" (high stress, low reward)
2. Test input: "This is a challenge"
3. See anxious, cautious response
4. Click "Make Optimistic" (high reward, low stress)
5. Test same input: "This is a challenge"
6. See confident, positive response!

### 🎨 Generate Media Tab
- Generate images from text prompts
- Create videos with different motion types:
  - Linear, Circular, Oscillating, Expanding, Random
- Adjustable quality and duration
- Base64 encoded output

### 💾 Memory Tab
- View training statistics
- Add knowledge facts
- Search semantic memory
- Export all data (conversations, training, knowledge)

## Understanding the Interface

### Color Coding
- **Purple gradient**: Primary actions
- **Green**: Success messages
- **Red**: Error messages
- **Blue info boxes**: Explanations and tips

### Status Indicator
- **Green dot**: Connected to ALEN
- **Red dot**: Connection failed - check if server is running

### Dashboard
- System statistics
- Quick action buttons
- Status overview

## Common Tasks

### Teach ALEN Math
1. Go to **Train ALEN** tab
2. Input: `5 + 3`
3. Answer: `8`
4. Click **Train ALEN**
5. Check result for confidence score

### Have a Conversation
1. Go to **Chat** tab
2. Type your message
3. Click **Send Message**
4. Responses include thought vectors and confidence

### Change ALEN's Mood
1. Go to **Mood & Emotions** tab
2. Click **Check Current Mood** to see current state
3. Adjust sliders or use preset buttons
4. Click **Apply Custom Mood**
5. Use **Mood Experiment** to test effects

### Generate an Image
1. Go to **Generate Media** tab
2. Enter a prompt: "sunset over mountains"
3. Adjust size and noise level
4. Click **Generate Image**
5. View base64 data in result

## Tips

1. **Server Connection**: Always start ALEN server before using web interface
2. **Mood Effects**: Mood actually changes responses - try the experiment!
3. **Training**: Higher confidence scores (>0.7) mean better learning
4. **Context**: More context messages = better conversation continuity
5. **Export Data**: Regularly export to save your work

## Troubleshooting

### "Connection Failed"
- Make sure ALEN server is running on port 3000
- Check console for errors
- Try refreshing the page

### No Response from APIs
- Check browser console (F12) for errors
- Verify API endpoint in browser DevTools
- Make sure request format is correct

### Mood Not Changing Behavior
- Make extreme changes (stress: 0.9, reward: 0.1)
- Use the "Mood Experiment" feature
- Check "before" and "after" states

## Browser Compatibility

Works with:
- Chrome/Edge (recommended)
- Firefox
- Safari
- Any modern browser with JavaScript enabled

## Security Note

This interface communicates with `localhost:3000` only. If deploying to production:
1. Update `API_URL` in the HTML
2. Configure CORS on the server
3. Add authentication if needed
4. Use HTTPS in production

## Advanced Usage

### Custom API Calls
Check the browser console for all API requests. You can:
- Copy request format
- Modify parameters
- Use curl commands from browser

### Training Data Format
Batch training format: `input -> answer` (one per line)

Example:
```
2+2 -> 4
cat -> animal
H2O -> water
```

### Exporting Data
Exports are saved to server's storage directory:
- `~/.local/share/alen/exports/` (Linux)
- `~/Library/Application Support/ALEN/exports/` (macOS)

## Next Steps

1. Read `API_DOCUMENTATION.md` for complete API reference
2. Check `PRODUCTION_GUIDE.md` for deployment info
3. Experiment with mood and training
4. Try self-supervised learning

## Support

For issues:
- Check server logs
- Verify `/health` endpoint works
- Report bugs on GitHub

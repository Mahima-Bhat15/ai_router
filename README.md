# AI Router — Hackathon Setup Guide

## What this is
A unified AI chat interface that:
1. **Detects intent** from your prompt (code, explanation, research, creative, math...)
2. **Selects the best model** from a routing table (Gemini, Groq/LLaMA, DeepSeek, etc.)
3. **Re-engineers your prompt** optimized for the chosen model
4. **Calls the API** and returns the response with full metadata
5. **Auto-fails over** to another model if one is rate-limited

All using free-tier APIs. No credit card needed.

---

## Prerequisites
- **Python 3.10+** (check: `python3 --version`)
- **A code editor** — VS Code recommended (free: https://code.visualstudio.com)
- **A terminal** — VS Code's built-in terminal works great

---

## Step-by-Step Setup (15 minutes)

### Step 1: Get your free API keys (5 min)

Open these links in your browser and create free accounts:

1. **Gemini**: https://aistudio.google.com/apikey
   - Sign in with Google → Click "Create API Key" → Copy it

2. **Groq**: https://console.groq.com/keys
   - Sign up with email → Go to API Keys → Create → Copy it

3. **OpenRouter**: https://openrouter.ai/settings/keys
   - Sign up → Go to Keys → Create Key → Copy it

Save all 3 keys somewhere — you'll need them in Step 3.

### Step 2: Open the project (1 min)

```bash
# Open VS Code, then open a terminal (Ctrl+` or Cmd+`)
# Navigate to where you downloaded/unzipped this project:
cd ai-router
```

### Step 3: Configure your API keys (1 min)

```bash
cd backend
cp .env.example .env
```

Now open `.env` in your editor and paste your API keys:
```
GEMINI_API_KEY=AIzaSy...your_actual_key
GROQ_API_KEY=gsk_...your_actual_key
OPENROUTER_API_KEY=sk-or-...your_actual_key
```

### Step 4: Install Python dependencies (2 min)

```bash
# Make sure you're in the backend/ directory
cd backend

# Create a virtual environment (keeps things clean)
python3 -m venv venv

# Activate it:
# On Mac/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt
```

### Step 5: Start the server (1 min)

```bash
# Still in the backend/ directory with venv activated:
python main.py
```

You should see:
```
🚀 AI Router starting up...
  ✓ Gemini API connected
  ✓ Groq API connected
  ✓ OpenRouter API connected

  Available providers: {'gemini', 'groq', 'openrouter'}

INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 6: Open the app (0 min)

Open your browser and go to: **http://localhost:8000**

That's it! You should see the AI Router chat interface.

---

## How to Use It

1. **Type any prompt** in the chat input
2. Watch the system:
   - Detect intent (shown as a colored badge: "code generation", "explanation", etc.)
   - Select the best model (shown next to the badge: "→ Gemini 2.5 Flash")
   - Show latency (e.g., "1240ms")
3. Click **"view optimized prompt"** to see how your prompt was rewritten
4. Use the **model override dropdown** to force a specific provider

### Example prompts to try:
- "Write a Python script to parse a CSV and find duplicate rows" → routes to Gemini Pro (code)
- "Explain how TCP/IP works" → routes to Gemini Flash (explanation)
- "Solve: integral of x²·sin(x) dx" → routes to DeepSeek R1 (math)
- "Compare React vs Vue" → routes to Gemini Pro (research)
- "Write a haiku about debugging" → routes to Gemini Flash (creative)

---

## API Endpoints (for the demo)

Test these directly in your browser or with curl:

```bash
# Health check — see which providers are connected
curl http://localhost:8000/health

# Classify intent only
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function to sort a list"}'

# Full chat pipeline
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing in simple terms"}'

# With model override
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "override_provider": "groq"}'
```

Interactive API docs: **http://localhost:8000/docs** (Swagger UI)

---

## Troubleshooting

**"Could not reach the backend"**
→ Make sure `python main.py` is running in your terminal

**"No API providers configured"**
→ Check your `.env` file has real API keys (not the placeholder text)

**"Gemini rate limited" / 429 error**
→ You hit the free tier limit. The system will auto-failover to Groq or OpenRouter. 
→ Gemini free tier: 250 requests/day for Flash, 100/day for Pro

**Import errors**
→ Make sure your virtual environment is activated: `source venv/bin/activate`
→ Make sure you ran: `pip install -r requirements.txt`

**Port 8000 already in use**
→ Kill the existing process: `lsof -ti:8000 | xargs kill`
→ Or use a different port: `uvicorn main:app --port 8001`

---

## What to Show in the Hackathon Demo

1. **The flow**: Type a prompt → show intent badge + model badge + latency
2. **Prompt optimization**: Click "view optimized prompt" to show before/after
3. **Model override**: Switch to a different provider and show different response
4. **Auto-failover**: If one provider hits a limit, it auto-routes to another
5. **API docs**: Show http://localhost:8000/docs for the interactive Swagger UI
6. **The architecture**: Walk through the flowchart from your whiteboard

---

## Next Steps (after hackathon)

- [ ] Add HuggingFace leaderboard scraper for dynamic rankings
- [ ] Add streaming responses (SSE)
- [ ] Add auth + BYOK (Bring Your Own Key) for premium models
- [ ] Add conversation history / context
- [ ] Deploy: Vercel (frontend) + Railway (backend)
- [ ] Add quota dashboard with visual bars

"""
AI Router — Backend Server (v0.3)
Providers: Gemini (FREE), Claude (Anthropic), Groq (FREE)
Split API: /api/analyze (fast) → animation → /api/generate (LLM call)
"""

import os, time, pathlib
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")

gemini_model = None
claude_client = None
groq_client = None


def init_clients():
    global gemini_model, claude_client, groq_client
    if GEMINI_KEY and GEMINI_KEY != "your_gemini_key_here":
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_KEY)
            gemini_model = genai.GenerativeModel("gemini-2.5-flash")
            print("  ✓ Gemini API connected (free tier)")
        except Exception as e:
            print(f"  ✗ Gemini: {e}")
    else:
        print("  - Gemini: no API key")

    if ANTHROPIC_KEY and ANTHROPIC_KEY != "your_anthropic_key_here":
        try:
            import anthropic
            claude_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            print("  ✓ Claude API connected")
        except Exception as e:
            print(f"  ✗ Claude: {e}")
    else:
        print("  - Claude: no API key")

    if GROQ_KEY and GROQ_KEY != "your_groq_key_here":
        try:
            from groq import Groq
            groq_client = Groq(api_key=GROQ_KEY)
            print("  ✓ Groq API connected (free tier)")
        except Exception as e:
            print(f"  ✗ Groq: {e}")
    else:
        print("  - Groq: no API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 AI Router starting up...")
    init_clients()
    available = get_available_providers()
    print(f"\n  Available: {available or 'NONE — add API keys to .env!'}\n")
    from services.leaderboard_scraper import get_benchmarks
    await get_benchmarks(force_refresh=True)
    yield

app = FastAPI(title="AI Router", version="0.3.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

from services.intent_classifier import classify_intent
from services.model_selector import select_model, mark_limited, mark_available, get_limited_providers
from services.prompt_engineer import reprompt
from services.leaderboard_scraper import get_benchmarks, get_full_leaderboard

def get_available_providers() -> set[str]:
    p = set()
    if gemini_model: p.add("gemini")
    if claude_client: p.add("claude")
    if groq_client: p.add("groq")
    return p


# --- STEP 1: Analyze (fast — no LLM call) ---
class AnalyzeRequest(BaseModel):
    prompt: str
    override_provider: str | None = None

class AnalyzeResponse(BaseModel):
    intent: str
    intent_confidence: float
    selected_model_id: str
    selected_provider: str
    selected_display_name: str
    selection_reason: str
    selection_method: str
    intent_score: float
    score_category: str
    benchmark_name: str
    rank: int
    total_candidates: int
    all_rankings: list[Any]
    data_source: str
    source_url: str
    reprompted_prompt: str

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """Step 1: Classify intent + select model + reprompt. No LLM call — instant."""
    available = get_available_providers()
    if not available:
        raise HTTPException(503, "No providers configured")

    intent_result = classify_intent(req.prompt)
    model_choice = select_model(intent=intent_result.intent,
        override_provider=req.override_provider, available_providers=available)
    optimized = reprompt(req.prompt, model_choice.provider, intent_result.intent)

    # Get benchmark name from rankings
    benchmark_name = ""
    if model_choice.all_rankings:
        benchmark_name = model_choice.all_rankings[0].get("benchmark_name", "")

    return AnalyzeResponse(
        intent=intent_result.intent,
        intent_confidence=intent_result.confidence,
        selected_model_id=model_choice.model_id,
        selected_provider=model_choice.provider,
        selected_display_name=model_choice.display_name,
        selection_reason=model_choice.reason,
        selection_method=model_choice.selection_method,
        intent_score=model_choice.intent_score,
        score_category=model_choice.score_category,
        benchmark_name=benchmark_name,
        rank=model_choice.rank,
        total_candidates=model_choice.total_candidates,
        all_rankings=model_choice.all_rankings,
        data_source=model_choice.data_source,
        source_url="https://vellum.ai/llm-leaderboard",
        reprompted_prompt=optimized,
    )


# --- STEP 2: Generate (calls the LLM) ---
class GenerateRequest(BaseModel):
    reprompted_prompt: str
    provider: str
    model_id: str
    original_prompt: str  # for failover reprompting
    intent: str

class GenerateResponse(BaseModel):
    response: str
    provider: str
    model_id: str
    model_display_name: str
    latency_ms: int
    failover_from: str | None = None

async def call_gemini(prompt: str, model_id: str) -> str:
    if not gemini_model: raise HTTPException(503, "Gemini not configured")
    try:
        import google.generativeai as genai
        return genai.GenerativeModel(model_id).generate_content(prompt).text
    except Exception as e:
        if any(k in str(e).lower() for k in ["429","quota","rate"]):
            mark_limited("gemini"); raise HTTPException(429, f"Gemini rate limited")
        raise HTTPException(500, f"Gemini error: {e}")

async def call_claude(prompt: str, model_id: str) -> str:
    if not claude_client: raise HTTPException(503, "Claude not configured")
    try:
        r = claude_client.messages.create(model=model_id, max_tokens=4096,
            messages=[{"role":"user","content":prompt}])
        return r.content[0].text
    except Exception as e:
        if any(k in str(e).lower() for k in ["429","rate"]):
            mark_limited("claude"); raise HTTPException(429, f"Claude rate limited")
        raise HTTPException(500, f"Claude error: {e}")

async def call_groq(prompt: str, model_id: str) -> str:
    if not groq_client: raise HTTPException(503, "Groq not configured")
    try:
        r = groq_client.chat.completions.create(model=model_id,
            messages=[{"role":"user","content":prompt}], temperature=0.7, max_tokens=4096)
        return r.choices[0].message.content
    except Exception as e:
        if any(k in str(e).lower() for k in ["429","rate"]):
            mark_limited("groq"); raise HTTPException(429, f"Groq rate limited")
        raise HTTPException(500, f"Groq error: {e}")

CALLERS = {"gemini": call_gemini, "claude": call_claude, "groq": call_groq}
DISPLAY_NAMES = {"claude-haiku-4-5-20251001":"Claude Haiku 4.5","gemini-2.5-pro":"Gemini 2.5 Pro",
    "gemini-2.5-flash":"Gemini 2.5 Flash","gemini-2.5-flash-lite":"Gemini 2.5 Flash-Lite",
    "llama-3.3-70b-versatile":"LLaMA 3.3 70B (Groq)"}

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Step 2: Actually call the LLM. Called after the animation finishes."""
    start = time.time()
    available = get_available_providers()

    caller = CALLERS.get(req.provider)
    failover_from = None
    provider = req.provider
    model_id = req.model_id
    prompt = req.reprompted_prompt

    try:
        response_text = await caller(prompt, model_id)
    except HTTPException as e:
        if e.status_code == 429:
            # Failover
            remaining = available - {req.provider} - get_limited_providers()
            if remaining:
                fb = select_model(intent=req.intent, available_providers=remaining)
                failover_from = req.provider
                provider = fb.provider
                model_id = fb.model_id
                prompt = reprompt(req.original_prompt, fb.provider, req.intent)
                response_text = await CALLERS[fb.provider](prompt, fb.model_id)
            else:
                raise HTTPException(429, "All providers rate-limited")
        else:
            raise

    return GenerateResponse(
        response=response_text,
        provider=provider,
        model_id=model_id,
        model_display_name=DISPLAY_NAMES.get(model_id, model_id),
        latency_ms=int((time.time()-start)*1000),
        failover_from=failover_from,
    )


# --- Other endpoints ---
@app.get("/health")
async def health():
    a = get_available_providers()
    return {"status":"ok","providers":{"gemini":"gemini" in a,"claude":"claude" in a,"groq":"groq" in a},
            "limited":list(get_limited_providers())}

@app.get("/api/leaderboard")
async def leaderboard():
    return get_full_leaderboard(await get_benchmarks())

@app.post("/api/leaderboard/refresh")
async def refresh_lb():
    lb = get_full_leaderboard(await get_benchmarks(force_refresh=True))
    return {"status":"refreshed","source":lb["source"]}

@app.post("/api/quota/reset/{provider}")
async def reset_quota(provider: str):
    mark_available(provider)
    return {"ok": True}

FRONTEND_DIR = pathlib.Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

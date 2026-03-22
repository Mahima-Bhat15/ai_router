"""
AI Router v0.6 — Per-user API keys (stored in Supabase, passed per-request)
No global key storage — each request includes the user's own provider keys.
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

# ─── Per-request provider callers (keys passed in, no global state) ───────────

async def call_gemini(prompt: str, model_id: str, api_key: str) -> str:
    if not api_key: raise HTTPException(503, "Gemini API key not provided")
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_id).generate_content(prompt).text
    except Exception as e:
        if any(k in str(e).lower() for k in ["429","quota","rate"]):
            raise HTTPException(429, f"Gemini rate limited")
        raise HTTPException(500, f"Gemini: {e}")

async def call_claude(prompt: str, model_id: str, api_key: str) -> str:
    if not api_key: raise HTTPException(503, "Claude API key not provided")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        r = client.messages.create(model=model_id, max_tokens=4096,
            messages=[{"role":"user","content":prompt}])
        return r.content[0].text
    except Exception as e:
        if any(k in str(e).lower() for k in ["429","rate"]):
            raise HTTPException(429, "Claude rate limited")
        raise HTTPException(500, f"Claude: {e}")

async def call_groq(prompt: str, model_id: str, api_key: str) -> str:
    if not api_key: raise HTTPException(503, "Groq API key not provided")
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        r = client.chat.completions.create(model=model_id,
            messages=[{"role":"user","content":prompt}], temperature=0.7, max_tokens=4096)
        return r.choices[0].message.content
    except Exception as e:
        if any(k in str(e).lower() for k in ["429","rate"]):
            raise HTTPException(429, "Groq rate limited")
        raise HTTPException(500, f"Groq: {e}")

async def call_openrouter(prompt: str, model_id: str, api_key: str) -> str:
    if not api_key: raise HTTPException(503, "OpenRouter API key not provided")
    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        r = client.chat.completions.create(model=model_id,
            messages=[{"role":"user","content":prompt}], temperature=0.7, max_tokens=4096)
        return r.choices[0].message.content
    except Exception as e:
        if any(k in str(e).lower() for k in ["429","rate"]):
            raise HTTPException(429, "OpenRouter rate limited")
        raise HTTPException(500, f"OpenRouter: {e}")

CALLERS = {"gemini": call_gemini, "claude": call_claude, "groq": call_groq, "openrouter": call_openrouter}

def get_available_providers(api_keys: dict) -> set[str]:
    """Determine which providers the user has keys for."""
    p = set()
    if api_keys.get("gemini"): p.add("gemini")
    if api_keys.get("claude"): p.add("claude")
    if api_keys.get("groq"): p.add("groq")
    if api_keys.get("openrouter"): p.add("openrouter")
    return p

def get_provider_key(api_keys: dict, provider: str) -> str:
    return (api_keys or {}).get(provider, "")

# ─── App setup ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 AI Router v0.6 starting (per-user keys)...")
    from services.leaderboard_scraper import get_benchmarks
    await get_benchmarks(force_refresh=True)
    print("  ✓ Leaderboard loaded\n")
    yield

app = FastAPI(title="AI Router", version="0.6.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

from services.intent_classifier import classify_intent
from services.model_selector import select_model, mark_limited, mark_available, get_limited_providers
from services.prompt_engineer import reprompt, debug_prompt_plan
from services.leaderboard_scraper import get_benchmarks, get_full_leaderboard

DISPLAY_NAMES = {
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-opus-4-20250514": "Claude Opus 4",
    "gemini-2.5-pro": "Gemini 2.5 Pro", "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.0-flash": "Gemini 2.0 Flash", "gemini-2.0-flash-lite": "Gemini 2.0 Flash Lite",
    "gemini-1.5-flash": "Gemini 1.5 Flash", "gemini-1.5-pro": "Gemini 1.5 Pro",
    "llama-3.3-70b-versatile": "LLaMA 3.3 70B", "llama-3.1-8b-instant": "LLaMA 3.1 8B",
    "mixtral-8x7b-32768": "Mixtral 8x7B", "gemma2-9b-it": "Gemma 2 9B",
    "llama3-70b-8192": "LLaMA 3 70B",
    "deepseek/deepseek-r1:free": "DeepSeek R1",
    "mistralai/mistral-small-3.1-24b-instruct:free": "Mistral Small 3.1",
    "mistralai/mistral-7b-instruct:free": "Mistral 7B",
    "qwen/qwen3-32b:free": "Qwen 3 32B", "qwen/qwen-2.5-72b-instruct:free": "Qwen 2.5 72B",
    "meta-llama/llama-3.1-8b-instruct:free": "LLaMA 3.1 8B",
    "meta-llama/llama-3.2-3b-instruct:free": "LLaMA 3.2 3B",
    "google/gemma-3-27b-it:free": "Gemma 3 27B",
}

MODEL_TO_PROVIDER = {
    "claude-haiku-4-5-20251001":"claude","claude-sonnet-4-20250514":"claude","claude-opus-4-20250514":"claude",
    "gemini-2.5-pro":"gemini","gemini-2.5-flash":"gemini","gemini-2.0-flash":"gemini",
    "gemini-2.0-flash-lite":"gemini","gemini-1.5-flash":"gemini","gemini-1.5-pro":"gemini",
    "llama-3.3-70b-versatile":"groq","llama-3.1-8b-instant":"groq","mixtral-8x7b-32768":"groq",
    "gemma2-9b-it":"groq","llama3-70b-8192":"groq",
    "deepseek/deepseek-r1:free":"openrouter","mistralai/mistral-small-3.1-24b-instruct:free":"openrouter",
    "mistralai/mistral-7b-instruct:free":"openrouter","qwen/qwen3-32b:free":"openrouter",
    "qwen/qwen-2.5-72b-instruct:free":"openrouter","meta-llama/llama-3.1-8b-instruct:free":"openrouter",
    "meta-llama/llama-3.2-3b-instruct:free":"openrouter","google/gemma-3-27b-it:free":"openrouter",
}

# ─── API Keys model (sent per-request from frontend) ─────────────────────────
class UserApiKeys(BaseModel):
    gemini: str = ""
    claude: str = ""
    groq: str = ""
    openrouter: str = ""

# ─── Analyze ──────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    prompt: str
    api_keys: UserApiKeys = UserApiKeys()
    override_provider: str | None = None
    override_model_id: str | None = None

class AnalyzeResponse(BaseModel):
    intent: str
    intent_confidence: float
    selected_model_id: str
    selected_provider: str
    selected_display_name: str
    selected_vendor: str
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
    prompt_standard: str

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    keys = req.api_keys.model_dump()
    available = get_available_providers(keys)
    if not available:
        raise HTTPException(503, "No API keys provided — add at least one in the keys panel")

    intent_result = classify_intent(req.prompt)

    # Direct model override
    if req.override_model_id and req.override_model_id in MODEL_TO_PROVIDER:
        provider = MODEL_TO_PROVIDER[req.override_model_id]
        if provider in available:
            display_name = DISPLAY_NAMES.get(req.override_model_id, req.override_model_id)
            optimized = reprompt(req.prompt, provider, intent_result.intent,
                model_id=req.override_model_id, model_display_name=display_name)
            plan = debug_prompt_plan(req.prompt, provider, intent_result.intent,
                model_id=req.override_model_id, model_display_name=display_name)
            return AnalyzeResponse(
                intent=intent_result.intent, intent_confidence=intent_result.confidence,
                selected_model_id=req.override_model_id, selected_provider=provider,
                selected_display_name=display_name, selected_vendor=provider,
                selection_reason=f"User selected {display_name} directly",
                selection_method="manual_override",
                intent_score=0.0, score_category="manual",
                benchmark_name="N/A (manual)", rank=0, total_candidates=0,
                all_rankings=[], data_source="manual", source_url="",
                reprompted_prompt=optimized,
                prompt_standard=plan.get("prompt_standard", "unknown"),
            )

    # Fallback reason if override provider missing
    fallback_reason = ""
    if req.override_model_id and req.override_model_id in MODEL_TO_PROVIDER:
        mp = MODEL_TO_PROVIDER[req.override_model_id]
        if mp not in available:
            fallback_reason = f" (fallback — {DISPLAY_NAMES.get(req.override_model_id, req.override_model_id)} unavailable)"

    model_choice = select_model(intent=intent_result.intent,
        override_provider=req.override_provider, available_providers=available)
    optimized = reprompt(req.prompt, model_choice.provider, intent_result.intent,
        model_id=model_choice.model_id, model_display_name=model_choice.display_name)
    plan = debug_prompt_plan(req.prompt, model_choice.provider, intent_result.intent,
        model_id=model_choice.model_id, model_display_name=model_choice.display_name)
    return AnalyzeResponse(
        intent=intent_result.intent, intent_confidence=intent_result.confidence,
        selected_model_id=model_choice.model_id, selected_provider=model_choice.provider,
        selected_display_name=model_choice.display_name, selected_vendor=model_choice.vendor,
        selection_reason=model_choice.reason + fallback_reason,
        selection_method=model_choice.selection_method,
        intent_score=model_choice.intent_score, score_category=model_choice.score_category,
        benchmark_name=model_choice.benchmark_name, rank=model_choice.rank,
        total_candidates=model_choice.total_candidates, all_rankings=model_choice.all_rankings,
        data_source=model_choice.data_source, source_url="https://vellum.ai/llm-leaderboard",
        reprompted_prompt=optimized,
        prompt_standard=plan.get("prompt_standard", "unknown"),
    )

# ─── Generate ─────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    reprompted_prompt: str
    provider: str
    model_id: str
    original_prompt: str
    intent: str
    api_keys: UserApiKeys = UserApiKeys()

class GenerateResponse(BaseModel):
    response: str
    provider: str
    model_id: str
    model_display_name: str
    vendor: str
    latency_ms: int
    failover_from: str | None = None
    reprompted_prompt: str | None = None

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    start = time.time()
    keys = req.api_keys.model_dump()
    available = get_available_providers(keys)
    provider = req.provider; model_id = req.model_id; prompt = req.reprompted_prompt
    failover_from = None

    try:
        key = get_provider_key(keys, provider)
        if not key or provider not in available:
            raise HTTPException(503, f"{provider} key not available")
        response_text = await CALLERS[provider](prompt, model_id, key)
    except HTTPException as e:
        # Failover to next best provider
        remaining = available - {req.provider} - get_limited_providers()
        if remaining:
            fb = select_model(intent=req.intent, available_providers=remaining)
            failover_from = req.provider; provider = fb.provider; model_id = fb.model_id
            prompt = reprompt(req.original_prompt, fb.provider, req.intent,
                model_id=fb.model_id, model_display_name=fb.display_name)
            try:
                response_text = await CALLERS[fb.provider](prompt, fb.model_id, get_provider_key(keys, fb.provider))
            except HTTPException:
                remaining2 = remaining - {fb.provider}
                if remaining2:
                    fb2 = select_model(intent=req.intent, available_providers=remaining2)
                    provider = fb2.provider; model_id = fb2.model_id
                    prompt = reprompt(req.original_prompt, fb2.provider, req.intent,
                        model_id=fb2.model_id, model_display_name=fb2.display_name)
                    response_text = await CALLERS[fb2.provider](prompt, fb2.model_id, get_provider_key(keys, fb2.provider))
                else:
                    raise HTTPException(503, "All fallback providers failed")
        else:
            raise HTTPException(503, f"No fallback providers. Error: {e.detail}")

    from services.leaderboard_scraper import get_cached_benchmarks
    vendor = ""
    for m in get_cached_benchmarks():
        if m.model_id == model_id: vendor = m.vendor; break

    return GenerateResponse(
        response=response_text, provider=provider, model_id=model_id,
        model_display_name=DISPLAY_NAMES.get(model_id, model_id), vendor=vendor,
        latency_ms=int((time.time()-start)*1000), failover_from=failover_from,
        reprompted_prompt=prompt if failover_from else None)

# ─── Other endpoints ──────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.6", "note": "API keys are per-user (stored in Supabase)"}

@app.get("/api/leaderboard")
async def leaderboard():
    return get_full_leaderboard(await get_benchmarks())

@app.post("/api/leaderboard/refresh")
async def refresh_lb():
    lb = get_full_leaderboard(await get_benchmarks(force_refresh=True))
    return {"status":"refreshed","source":lb["source"]}

@app.get("/api/supabase-config")
async def supabase_config():
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_ANON_KEY", "")
    if not url or not key:
        raise HTTPException(503, "Supabase not configured — add SUPABASE_URL and SUPABASE_ANON_KEY to .env")
    return {"url": url, "anon_key": key}

# ─── Frontend serving ─────────────────────────────────────────────────────────
FRONTEND_DIR = pathlib.Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    @app.get("/")
    async def serve_landing():
        landing = FRONTEND_DIR / "tap.html"
        if landing.exists(): return FileResponse(str(landing))
        return FileResponse(str(FRONTEND_DIR / "index.html"))
    @app.get("/chat")
    async def serve_chat():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

if __name__ == "__main__":
    import uvicorn; uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
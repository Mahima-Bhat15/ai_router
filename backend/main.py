"""
AI Router v0.5 — 7 models, 4 providers, advanced prompt engineering
Providers: Gemini (FREE), Claude (Anthropic), Groq (FREE), OpenRouter (FREE open-source)
API keys are provided at runtime via the frontend (no hardcoded keys).
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

# --- Runtime key store (populated via /api/keys endpoint or .env fallback) ---
_api_keys: dict[str, str] = {}
_cleared_keys: set[str] = set()   # providers explicitly cleared via UI

gemini_model = None
claude_client = None
groq_client = None
openrouter_client = None

def _get_key(provider: str) -> str:
    """Get API key for a provider — runtime store first, then .env fallback.
    If a key was explicitly cleared via the UI, skip the .env fallback."""
    env_map = {
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    # Explicitly cleared — return empty regardless of .env
    if provider in _cleared_keys:
        return ""
    key = _api_keys.get(provider, "")
    if not key:
        key = os.getenv(env_map.get(provider, ""), "")
    placeholder_vals = {"your_gemini_key_here", "your_anthropic_key_here",
                        "your_groq_key_here", "your_openrouter_key_here"}
    if key in placeholder_vals:
        return ""
    return key

def init_clients():
    """Initialize SDK clients for any providers that have keys."""
    global gemini_model, claude_client, groq_client, openrouter_client

    k = _get_key("gemini")
    if k:
        try:
            import google.generativeai as genai
            genai.configure(api_key=k)
            gemini_model = genai.GenerativeModel("gemini-2.5-flash")
            print("  ✓ Gemini (free)")
        except Exception as e:
            gemini_model = None
            print(f"  ✗ Gemini: {e}")
    else:
        gemini_model = None
        print("  - Gemini: no key")

    k = _get_key("claude")
    if k:
        try:
            import anthropic
            claude_client = anthropic.Anthropic(api_key=k)
            print("  ✓ Claude")
        except Exception as e:
            claude_client = None
            print(f"  ✗ Claude: {e}")
    else:
        claude_client = None
        print("  - Claude: no key")

    k = _get_key("groq")
    if k:
        try:
            from groq import Groq
            groq_client = Groq(api_key=k)
            print("  ✓ Groq (free)")
        except Exception as e:
            groq_client = None
            print(f"  ✗ Groq: {e}")
    else:
        groq_client = None
        print("  - Groq: no key")

    k = _get_key("openrouter")
    if k:
        try:
            from openai import OpenAI
            openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=k)
            print("  ✓ OpenRouter (free open-source models)")
        except Exception as e:
            openrouter_client = None
            print(f"  ✗ OpenRouter: {e}")
    else:
        openrouter_client = None
        print("  - OpenRouter: no key")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 AI Router v0.5 starting...")
    init_clients()
    a = get_available_providers()
    print(f"\n  Providers: {a or 'NONE — add keys via the UI'}\n")
    from services.leaderboard_scraper import get_benchmarks
    await get_benchmarks(force_refresh=True)
    yield

app = FastAPI(title="AI Router", version="0.5.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

from services.intent_classifier import classify_intent
from services.model_selector import select_model, mark_limited, mark_available, get_limited_providers
from services.prompt_engineer import reprompt, debug_prompt_plan
from services.leaderboard_scraper import get_benchmarks, get_full_leaderboard

def get_available_providers() -> set[str]:
    p = set()
    if gemini_model: p.add("gemini")
    if claude_client: p.add("claude")
    if groq_client: p.add("groq")
    if openrouter_client: p.add("openrouter")
    return p

# ─── API Keys Management ─────────────────────────────────────────────────────
class KeySaveRequest(BaseModel):
    provider: str
    key: str

class KeysStatusResponse(BaseModel):
    gemini: bool
    claude: bool
    groq: bool
    openrouter: bool

@app.get("/api/keys/status", response_model=KeysStatusResponse)
async def keys_status():
    """Return which providers currently have a key configured."""
    return KeysStatusResponse(
        gemini=bool(_get_key("gemini")),
        claude=bool(_get_key("claude")),
        groq=bool(_get_key("groq")),
        openrouter=bool(_get_key("openrouter")),
    )

@app.post("/api/keys/save")
async def keys_save(req: KeySaveRequest):
    """Save an API key for a provider and reinitialize its client."""
    valid = ("gemini", "claude", "groq", "openrouter")
    if req.provider not in valid:
        raise HTTPException(400, f"Invalid provider. Must be one of: {valid}")
    _api_keys[req.provider] = req.key.strip()
    _cleared_keys.discard(req.provider)  # un-clear if was previously cleared
    print(f"\n🔑 Key updated for {req.provider}, reinitializing...")
    init_clients()
    return {"ok": True, "provider": req.provider, "has_key": True}

@app.post("/api/keys/clear/{provider}")
async def keys_clear(provider: str):
    """Remove the runtime key for a provider and reinitialize."""
    valid = ("gemini", "claude", "groq", "openrouter")
    if provider not in valid:
        raise HTTPException(400, f"Invalid provider. Must be one of: {valid}")
    _api_keys.pop(provider, None)
    _cleared_keys.add(provider)  # block .env fallback until re-saved
    print(f"\n🔑 Key cleared for {provider}, reinitializing...")
    init_clients()
    return {"ok": True, "provider": provider, "has_key": False}


# ─── Provider callers ─────────────────────────────────────────────────────────
async def call_gemini(prompt: str, model_id: str) -> str:
    if not gemini_model: raise HTTPException(503, "Gemini not configured — add your API key")
    try:
        import google.generativeai as genai
        return genai.GenerativeModel(model_id).generate_content(prompt).text
    except Exception as e:
        if any(k in str(e).lower() for k in ["429","quota","rate"]):
            mark_limited("gemini"); raise HTTPException(429, f"Gemini rate limited")
        raise HTTPException(500, f"Gemini: {e}")

async def call_claude(prompt: str, model_id: str) -> str:
    if not claude_client: raise HTTPException(503, "Claude not configured — add your API key")
    try:
        r = claude_client.messages.create(model=model_id, max_tokens=4096,
            messages=[{"role":"user","content":prompt}])
        return r.content[0].text
    except Exception as e:
        if any(k in str(e).lower() for k in ["429","rate"]):
            mark_limited("claude"); raise HTTPException(429, "Claude rate limited")
        raise HTTPException(500, f"Claude: {e}")

async def call_groq(prompt: str, model_id: str) -> str:
    if not groq_client: raise HTTPException(503, "Groq not configured — add your API key")
    try:
        r = groq_client.chat.completions.create(model=model_id,
            messages=[{"role":"user","content":prompt}], temperature=0.7, max_tokens=4096)
        return r.choices[0].message.content
    except Exception as e:
        if any(k in str(e).lower() for k in ["429","rate"]):
            mark_limited("groq"); raise HTTPException(429, "Groq rate limited")
        raise HTTPException(500, f"Groq: {e}")

async def call_openrouter(prompt: str, model_id: str) -> str:
    if not openrouter_client: raise HTTPException(503, "OpenRouter not configured — add your API key")
    try:
        r = openrouter_client.chat.completions.create(model=model_id,
            messages=[{"role":"user","content":prompt}], temperature=0.7, max_tokens=4096)
        return r.choices[0].message.content
    except Exception as e:
        if any(k in str(e).lower() for k in ["429","rate"]):
            mark_limited("openrouter"); raise HTTPException(429, "OpenRouter rate limited")
        raise HTTPException(500, f"OpenRouter: {e}")

CALLERS = {"gemini": call_gemini, "claude": call_claude, "groq": call_groq, "openrouter": call_openrouter}

DISPLAY_NAMES = {
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-opus-4-20250514": "Claude Opus 4",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-2.0-flash-lite": "Gemini 2.0 Flash Lite",
    "gemini-1.5-flash": "Gemini 1.5 Flash",
    "gemini-1.5-pro": "Gemini 1.5 Pro",
    "llama-3.3-70b-versatile": "LLaMA 3.3 70B",
    "llama-3.1-8b-instant": "LLaMA 3.1 8B",
    "mixtral-8x7b-32768": "Mixtral 8x7B",
    "gemma2-9b-it": "Gemma 2 9B",
    "llama3-70b-8192": "LLaMA 3 70B",
    "deepseek/deepseek-r1:free": "DeepSeek R1",
    "mistralai/mistral-small-3.1-24b-instruct:free": "Mistral Small 3.1",
    "mistralai/mistral-7b-instruct:free": "Mistral 7B",
    "qwen/qwen3-32b:free": "Qwen 3 32B",
    "qwen/qwen-2.5-72b-instruct:free": "Qwen 2.5 72B",
    "meta-llama/llama-3.1-8b-instruct:free": "LLaMA 3.1 8B",
    "meta-llama/llama-3.2-3b-instruct:free": "LLaMA 3.2 3B",
    "google/gemma-3-27b-it:free": "Gemma 3 27B",
}

# Map model_id → provider for direct override
MODEL_TO_PROVIDER = {
    # Claude
    "claude-haiku-4-5-20251001": "claude",
    "claude-sonnet-4-20250514": "claude",
    "claude-opus-4-20250514": "claude",
    # Gemini
    "gemini-2.5-pro": "gemini", "gemini-2.5-flash": "gemini",
    "gemini-2.0-flash": "gemini", "gemini-2.0-flash-lite": "gemini",
    "gemini-1.5-flash": "gemini", "gemini-1.5-pro": "gemini",
    # Groq
    "llama-3.3-70b-versatile": "groq", "llama-3.1-8b-instant": "groq",
    "mixtral-8x7b-32768": "groq", "gemma2-9b-it": "groq",
    "llama3-70b-8192": "groq",
    # OpenRouter
    "deepseek/deepseek-r1:free": "openrouter",
    "mistralai/mistral-small-3.1-24b-instruct:free": "openrouter",
    "mistralai/mistral-7b-instruct:free": "openrouter",
    "qwen/qwen3-32b:free": "openrouter",
    "qwen/qwen-2.5-72b-instruct:free": "openrouter",
    "meta-llama/llama-3.1-8b-instruct:free": "openrouter",
    "meta-llama/llama-3.2-3b-instruct:free": "openrouter",
    "google/gemma-3-27b-it:free": "openrouter",
}

# ─── Step 1: Analyze ──────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    prompt: str
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
    available = get_available_providers()
    if not available:
        raise HTTPException(503, "No providers configured — add at least one API key via the keys panel")

    intent_result = classify_intent(req.prompt)

    # Direct model override — user picked a specific model in the picker
    if req.override_model_id and req.override_model_id in MODEL_TO_PROVIDER:
        provider = MODEL_TO_PROVIDER[req.override_model_id]
        if provider in available:
            # Provider available — use the selected model directly
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
                benchmark_name="N/A (manual pick)", rank=0, total_candidates=0,
                all_rankings=[], data_source="manual", source_url="",
                reprompted_prompt=optimized,
                prompt_standard=plan.get("prompt_standard", "unknown"),
            )
        # else: provider key missing — fall through to auto selection below

    # Determine if this is a fallback from a failed model override
    fallback_reason = ""
    if req.override_model_id and req.override_model_id in MODEL_TO_PROVIDER:
        missing_prov = MODEL_TO_PROVIDER[req.override_model_id]
        if missing_prov not in available:
            orig_name = DISPLAY_NAMES.get(req.override_model_id, req.override_model_id)
            fallback_reason = f" (fallback — {orig_name} unavailable, {missing_prov} key missing)"

    # Normal flow: leaderboard-driven selection (optionally scoped to a provider)
    model_choice = select_model(intent=intent_result.intent,
        override_provider=req.override_provider, available_providers=available)

    optimized = reprompt(req.prompt, model_choice.provider, intent_result.intent,
        model_id=model_choice.model_id, model_display_name=model_choice.display_name)

    plan = debug_prompt_plan(req.prompt, model_choice.provider, intent_result.intent,
        model_id=model_choice.model_id, model_display_name=model_choice.display_name)

    reason = model_choice.reason + fallback_reason

    return AnalyzeResponse(
        intent=intent_result.intent, intent_confidence=intent_result.confidence,
        selected_model_id=model_choice.model_id, selected_provider=model_choice.provider,
        selected_display_name=model_choice.display_name, selected_vendor=model_choice.vendor,
        selection_reason=reason, selection_method=model_choice.selection_method,
        intent_score=model_choice.intent_score, score_category=model_choice.score_category,
        benchmark_name=model_choice.benchmark_name, rank=model_choice.rank,
        total_candidates=model_choice.total_candidates, all_rankings=model_choice.all_rankings,
        data_source=model_choice.data_source, source_url="https://vellum.ai/llm-leaderboard",
        reprompted_prompt=optimized,
        prompt_standard=plan.get("prompt_standard", "unknown"),
    )

# ─── Step 2: Generate ─────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    reprompted_prompt: str
    provider: str
    model_id: str
    original_prompt: str
    intent: str

class GenerateResponse(BaseModel):
    response: str
    provider: str
    model_id: str
    model_display_name: str
    vendor: str
    latency_ms: int
    failover_from: str | None = None

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    start = time.time()
    available = get_available_providers()
    caller = CALLERS.get(req.provider)
    failover_from = None; provider = req.provider; model_id = req.model_id; prompt = req.reprompted_prompt

    try:
        if not caller or req.provider not in available:
            raise HTTPException(503, f"{req.provider} not available")
        response_text = await caller(prompt, model_id)
    except HTTPException as e:
        # Any caller error (429 rate-limit, 500 model error, 503 unavailable, etc.)
        # → attempt failover to the next best available model
        if any(k in str(e.detail).lower() for k in ["429","rate"]):
            mark_limited(req.provider)
        remaining = available - {req.provider} - get_limited_providers()
        if remaining:
            fb = select_model(intent=req.intent, available_providers=remaining)
            failover_from = req.provider; provider = fb.provider; model_id = fb.model_id
            prompt = reprompt(req.original_prompt, fb.provider, req.intent,
                model_id=fb.model_id, model_display_name=fb.display_name)
            try:
                response_text = await CALLERS[fb.provider](prompt, fb.model_id)
            except HTTPException:
                # Second failover attempt with remaining providers
                remaining2 = remaining - {fb.provider}
                if remaining2:
                    fb2 = select_model(intent=req.intent, available_providers=remaining2)
                    provider = fb2.provider; model_id = fb2.model_id
                    prompt = reprompt(req.original_prompt, fb2.provider, req.intent,
                        model_id=fb2.model_id, model_display_name=fb2.display_name)
                    response_text = await CALLERS[fb2.provider](prompt, fb2.model_id)
                else:
                    raise HTTPException(503, f"All fallback providers failed")
        else:
            raise HTTPException(503, f"No fallback providers available. Original error: {e.detail}")

    from services.leaderboard_scraper import get_cached_benchmarks
    vendor = ""
    for m in get_cached_benchmarks():
        if m.model_id == model_id: vendor = m.vendor; break

    return GenerateResponse(
        response=response_text, provider=provider, model_id=model_id,
        model_display_name=DISPLAY_NAMES.get(model_id, model_id), vendor=vendor,
        latency_ms=int((time.time()-start)*1000), failover_from=failover_from)

# ─── Other endpoints ──────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    a = get_available_providers()
    return {"status":"ok","providers":{"gemini":"gemini" in a,"claude":"claude" in a,
        "groq":"groq" in a,"openrouter":"openrouter" in a},"limited":list(get_limited_providers())}

@app.get("/api/leaderboard")
async def leaderboard():
    return get_full_leaderboard(await get_benchmarks())

@app.post("/api/leaderboard/refresh")
async def refresh_lb():
    lb = get_full_leaderboard(await get_benchmarks(force_refresh=True))
    return {"status":"refreshed","source":lb["source"]}

@app.post("/api/quota/reset/{provider}")
async def reset_quota(provider: str):
    mark_available(provider); return {"ok": True}

FRONTEND_DIR = pathlib.Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    @app.get("/")
    async def serve_frontend(): return FileResponse(str(FRONTEND_DIR / "index.html"))

if __name__ == "__main__":
    import uvicorn; uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
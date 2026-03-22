"""
AI Router — Backend Server
Providers: Gemini (Google, FREE), Claude (Anthropic), Groq (LLaMA, FREE)

Run with: python3 main.py
"""

import os
import time
import pathlib
from contextlib import asynccontextmanager

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
            print(f"  ✗ Gemini init failed: {e}")
    else:
        print("  - Gemini: no API key configured")

    if ANTHROPIC_KEY and ANTHROPIC_KEY != "your_anthropic_key_here":
        try:
            import anthropic
            claude_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            print("  ✓ Claude API connected")
        except Exception as e:
            print(f"  ✗ Claude init failed: {e}")
    else:
        print("  - Claude: no API key configured")

    if GROQ_KEY and GROQ_KEY != "your_groq_key_here":
        try:
            from groq import Groq
            groq_client = Groq(api_key=GROQ_KEY)
            print("  ✓ Groq API connected (free tier)")
        except Exception as e:
            print(f"  ✗ Groq init failed: {e}")
    else:
        print("  - Groq: no API key configured")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 AI Router starting up...")
    init_clients()
    available = get_available_providers()
    print(f"\n  Available providers: {available or 'NONE — add API keys to .env!'}\n")
    yield
    print("AI Router shutting down.")


app = FastAPI(
    title="AI Router",
    description="Intent-based LLM routing: Gemini + Claude + Groq",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from services.intent_classifier import classify_intent
from services.model_selector import (
    select_model, mark_limited, mark_available, get_limited_providers,
)
from services.prompt_engineer import reprompt


class ChatRequest(BaseModel):
    prompt: str
    override_provider: str | None = None  # "gemini", "claude", "groq"

class ChatResponse(BaseModel):
    response: str
    intent: str
    intent_confidence: float
    model_used: str
    provider: str
    model_display_name: str
    selection_reason: str
    original_prompt: str
    reprompted_prompt: str
    latency_ms: int

class IntentRequest(BaseModel):
    prompt: str

class IntentResponse(BaseModel):
    intent: str
    confidence: float

class HealthResponse(BaseModel):
    status: str
    providers: dict[str, bool]
    limited_providers: list[str]


def get_available_providers() -> set[str]:
    providers = set()
    if gemini_model:
        providers.add("gemini")
    if claude_client:
        providers.add("claude")
    if groq_client:
        providers.add("groq")
    return providers


# --- Provider call functions ---
async def call_gemini(prompt: str, model_id: str) -> str:
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Gemini not configured")
    try:
        import google.generativeai as genai
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "quota" in error_msg or "rate" in error_msg:
            mark_limited("gemini")
            raise HTTPException(status_code=429, detail=f"Gemini rate limited: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")


async def call_claude(prompt: str, model_id: str) -> str:
    if not claude_client:
        raise HTTPException(status_code=503, detail="Claude not configured")
    try:
        response = claude_client.messages.create(
            model=model_id,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "rate" in error_msg:
            mark_limited("claude")
            raise HTTPException(status_code=429, detail=f"Claude rate limited: {e}")
        raise HTTPException(status_code=500, detail=f"Claude error: {e}")


async def call_groq(prompt: str, model_id: str) -> str:
    if not groq_client:
        raise HTTPException(status_code=503, detail="Groq not configured")
    try:
        response = groq_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "rate" in error_msg:
            mark_limited("groq")
            raise HTTPException(status_code=429, detail=f"Groq rate limited: {e}")
        raise HTTPException(status_code=500, detail=f"Groq error: {e}")


PROVIDER_CALLERS = {
    "gemini": call_gemini,
    "claude": call_claude,
    "groq": call_groq,
}


# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    available = get_available_providers()
    return HealthResponse(
        status="ok",
        providers={
            "gemini": "gemini" in available,
            "claude": "claude" in available,
            "groq": "groq" in available,
        },
        limited_providers=list(get_limited_providers()),
    )


@app.post("/api/classify", response_model=IntentResponse)
async def classify(req: IntentRequest):
    result = classify_intent(req.prompt)
    return IntentResponse(intent=result.intent, confidence=result.confidence)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    available = get_available_providers()

    if not available:
        raise HTTPException(
            status_code=503,
            detail="No API providers configured. Add API keys to your .env file.",
        )

    # 1. Classify intent
    intent_result = classify_intent(req.prompt)

    # 2. Select model
    model_choice = select_model(
        intent=intent_result.intent,
        override_provider=req.override_provider,
        available_providers=available,
    )

    # 3. Re-prompt
    optimized_prompt = reprompt(
        original_prompt=req.prompt,
        provider=model_choice.provider,
        intent=intent_result.intent,
    )

    # 4. Call provider with auto-failover
    response_text = None
    final_model = model_choice
    tried_providers = set()

    while response_text is None:
        tried_providers.add(final_model.provider)
        caller = PROVIDER_CALLERS.get(final_model.provider)
        if not caller:
            break

        try:
            response_text = await caller(optimized_prompt, final_model.model_id)
        except HTTPException as e:
            if e.status_code == 429:
                remaining = available - tried_providers - get_limited_providers()
                if remaining:
                    final_model = select_model(
                        intent=intent_result.intent,
                        available_providers=remaining,
                    )
                    optimized_prompt = reprompt(
                        original_prompt=req.prompt,
                        provider=final_model.provider,
                        intent=intent_result.intent,
                    )
                else:
                    raise HTTPException(
                        status_code=429,
                        detail="All providers rate-limited. Try again in a few minutes.",
                    )
            else:
                raise

    if response_text is None:
        raise HTTPException(status_code=500, detail="Failed to get response from any provider.")

    latency = int((time.time() - start) * 1000)

    return ChatResponse(
        response=response_text,
        intent=intent_result.intent,
        intent_confidence=intent_result.confidence,
        model_used=final_model.model_id,
        provider=final_model.provider,
        model_display_name=final_model.display_name,
        selection_reason=final_model.reason,
        original_prompt=req.prompt,
        reprompted_prompt=optimized_prompt,
        latency_ms=latency,
    )


@app.post("/api/quota/reset/{provider}")
async def reset_quota(provider: str):
    mark_available(provider)
    return {"status": "ok", "message": f"{provider} marked as available"}


# --- Serve frontend ---
FRONTEND_DIR = pathlib.Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

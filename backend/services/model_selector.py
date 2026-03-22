"""
Model Selector — picks the best available model based on intent.
Providers: Gemini (Google), Claude (Anthropic), Groq (LLaMA)

Strengths:
  Claude → code, creative writing, nuanced reasoning
  Gemini → explanation, summarization, math, general tasks (FREE)
  Groq   → fast inference, code, conversation (FREE — LLaMA 3.3 70B)
"""

from dataclasses import dataclass

@dataclass
class ModelChoice:
    model_id: str
    provider: str         # "gemini" | "claude" | "groq"
    display_name: str
    reason: str

ROUTING_TABLE: dict[str, list[tuple[str, str, str]]] = {
    "code_generation": [
        ("claude-haiku-4-5-20251001", "claude", "Claude Haiku 4.5"),
        ("gemini-2.5-pro", "gemini", "Gemini 2.5 Pro"),
        ("llama-3.3-70b-versatile", "groq", "LLaMA 3.3 70B"),
    ],
    "code_review": [
        ("claude-haiku-4-5-20251001", "claude", "Claude Haiku 4.5"),
        ("llama-3.3-70b-versatile", "groq", "LLaMA 3.3 70B"),
        ("gemini-2.5-flash", "gemini", "Gemini 2.5 Flash"),
    ],
    "explanation": [
        ("gemini-2.5-flash", "gemini", "Gemini 2.5 Flash"),
        ("claude-haiku-4-5-20251001", "claude", "Claude Haiku 4.5"),
        ("llama-3.3-70b-versatile", "groq", "LLaMA 3.3 70B"),
    ],
    "creative_writing": [
        ("claude-haiku-4-5-20251001", "claude", "Claude Haiku 4.5"),
        ("gemini-2.5-flash", "gemini", "Gemini 2.5 Flash"),
        ("llama-3.3-70b-versatile", "groq", "LLaMA 3.3 70B"),
    ],
    "research": [
        ("gemini-2.5-pro", "gemini", "Gemini 2.5 Pro"),
        ("claude-haiku-4-5-20251001", "claude", "Claude Haiku 4.5"),
        ("llama-3.3-70b-versatile", "groq", "LLaMA 3.3 70B"),
    ],
    "math_reasoning": [
        ("gemini-2.5-pro", "gemini", "Gemini 2.5 Pro"),
        ("claude-haiku-4-5-20251001", "claude", "Claude Haiku 4.5"),
        ("llama-3.3-70b-versatile", "groq", "LLaMA 3.3 70B"),
    ],
    "summarization": [
        ("llama-3.3-70b-versatile", "groq", "LLaMA 3.3 70B"),
        ("gemini-2.5-flash", "gemini", "Gemini 2.5 Flash"),
        ("claude-haiku-4-5-20251001", "claude", "Claude Haiku 4.5"),
    ],
    "conversation": [
        ("llama-3.3-70b-versatile", "groq", "LLaMA 3.3 70B"),
        ("gemini-2.5-flash", "gemini", "Gemini 2.5 Flash"),
        ("claude-haiku-4-5-20251001", "claude", "Claude Haiku 4.5"),
    ],
}

_limited_providers: set[str] = set()

def mark_limited(provider: str):
    _limited_providers.add(provider)

def mark_available(provider: str):
    _limited_providers.discard(provider)

def get_limited_providers() -> set[str]:
    return _limited_providers.copy()

def select_model(
    intent: str,
    override_provider: str | None = None,
    available_providers: set[str] | None = None,
) -> ModelChoice:
    if available_providers is None:
        available_providers = {"gemini", "claude", "groq"}

    candidates = ROUTING_TABLE.get(intent, ROUTING_TABLE["conversation"])

    if override_provider:
        for model_id, provider, display_name in candidates:
            if provider == override_provider and provider in available_providers:
                return ModelChoice(model_id=model_id, provider=provider,
                    display_name=display_name, reason=f"User selected {provider}")
        for intent_models in ROUTING_TABLE.values():
            for model_id, provider, display_name in intent_models:
                if provider == override_provider and provider in available_providers:
                    return ModelChoice(model_id=model_id, provider=provider,
                        display_name=display_name, reason=f"User selected {provider}")

    for model_id, provider, display_name in candidates:
        if provider in available_providers and provider not in _limited_providers:
            return ModelChoice(model_id=model_id, provider=provider,
                display_name=display_name, reason=f"Top ranked for {intent}")

    for model_id, provider, display_name in candidates:
        if provider in available_providers:
            return ModelChoice(model_id=model_id, provider=provider,
                display_name=display_name, reason="Fallback (other providers rate-limited)")

    return ModelChoice(model_id="gemini-2.5-flash", provider="gemini",
        display_name="Gemini 2.5 Flash", reason="Default fallback")

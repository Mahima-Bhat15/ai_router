"""
Model Selector — uses leaderboard benchmarks. 7 models, 4 providers.
Providers: Claude (Anthropic), Gemini (Google), Groq (LLaMA), OpenRouter (DeepSeek/Mistral/Qwen)
"""
from dataclasses import dataclass, field
from services.leaderboard_scraper import get_rankings_for_intent, get_source

@dataclass
class ModelChoice:
    model_id: str
    provider: str
    display_name: str
    vendor: str
    reason: str
    intent_score: float = 0.0
    score_category: str = ""
    benchmark_name: str = ""
    rank: int = 0
    total_candidates: int = 0
    all_rankings: list = field(default_factory=list)
    selection_method: str = "leaderboard"
    data_source: str = ""

_limited_providers: set[str] = set()
def mark_limited(p: str): _limited_providers.add(p)
def mark_available(p: str): _limited_providers.discard(p)
def get_limited_providers() -> set[str]: return _limited_providers.copy()

def select_model(intent: str, override_provider: str | None = None,
                 available_providers: set[str] | None = None) -> ModelChoice:
    if available_providers is None:
        available_providers = {"gemini", "claude", "groq", "openrouter"}

    rankings = get_rankings_for_intent(intent)
    data_source = get_source()
    benchmark_name = rankings[0].get("benchmark_name", "") if rankings else ""

    if override_provider:
        for e in rankings:
            if e["provider"] == override_provider and override_provider in available_providers:
                return ModelChoice(
                    model_id=e["model_id"], provider=e["provider"],
                    display_name=e["display_name"], vendor=e["vendor"],
                    reason=f"User override → {override_provider}",
                    intent_score=e["intent_score"], score_category=e["score_category"],
                    benchmark_name=benchmark_name, rank=e["rank"],
                    total_candidates=len(rankings), all_rankings=rankings,
                    selection_method="user_override", data_source=data_source)

    for e in rankings:
        p = e["provider"]
        if p in available_providers and p not in _limited_providers:
            return ModelChoice(
                model_id=e["model_id"], provider=p,
                display_name=e["display_name"], vendor=e["vendor"],
                reason=f"Rank #{e['rank']} for {intent.replace('_',' ')} ({e['score_category']}: {e['intent_score']})",
                intent_score=e["intent_score"], score_category=e["score_category"],
                benchmark_name=benchmark_name, rank=e["rank"],
                total_candidates=len(rankings), all_rankings=rankings,
                selection_method="leaderboard", data_source=data_source)

    for e in rankings:
        if e["provider"] in available_providers:
            return ModelChoice(
                model_id=e["model_id"], provider=e["provider"],
                display_name=e["display_name"], vendor=e.get("vendor",""),
                reason="Fallback (others rate-limited)",
                intent_score=e["intent_score"], score_category=e["score_category"],
                benchmark_name=benchmark_name, rank=e["rank"],
                total_candidates=len(rankings), all_rankings=rankings,
                selection_method="fallback", data_source=data_source)

    return ModelChoice(model_id="gemini-2.5-flash", provider="gemini",
        display_name="Gemini 2.5 Flash", vendor="google",
        reason="Default fallback", data_source=data_source)

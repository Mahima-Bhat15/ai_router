"""
Model Selector — uses leaderboard benchmark scores to pick the best model.
Now provides full transparency: WHY this model, with scores and rankings.
"""

from dataclasses import dataclass, field
from services.leaderboard_scraper import get_rankings_for_intent, get_source

@dataclass
class ModelChoice:
    model_id: str
    provider: str
    display_name: str
    reason: str
    # Transparency fields
    intent_score: float = 0.0
    score_category: str = ""
    rank: int = 0
    total_candidates: int = 0
    all_rankings: list = field(default_factory=list)  # full ranked list for UI
    selection_method: str = "leaderboard"  # "leaderboard" | "user_override" | "fallback"
    data_source: str = ""

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

    # Get leaderboard rankings for this intent
    rankings = get_rankings_for_intent(intent)
    data_source = get_source()

    # User override — skip rankings, just pick from that provider
    if override_provider:
        for entry in rankings:
            if entry["provider"] == override_provider and override_provider in available_providers:
                return ModelChoice(
                    model_id=entry["model_id"],
                    provider=entry["provider"],
                    display_name=entry["display_name"],
                    reason=f"User override → {override_provider}",
                    intent_score=entry["intent_score"],
                    score_category=entry["score_category"],
                    rank=entry["rank"],
                    total_candidates=len(rankings),
                    all_rankings=rankings,
                    selection_method="user_override",
                    data_source=data_source,
                )

    # Auto-selection: pick highest-ranked available, non-limited model
    for entry in rankings:
        provider = entry["provider"]
        if provider in available_providers and provider not in _limited_providers:
            return ModelChoice(
                model_id=entry["model_id"],
                provider=provider,
                display_name=entry["display_name"],
                reason=f"Rank #{entry['rank']} for {intent.replace('_', ' ')} "
                       f"({entry['score_category']}: {entry['intent_score']})",
                intent_score=entry["intent_score"],
                score_category=entry["score_category"],
                rank=entry["rank"],
                total_candidates=len(rankings),
                all_rankings=rankings,
                selection_method="leaderboard",
                data_source=data_source,
            )

    # Everything limited — pick any available
    for entry in rankings:
        if entry["provider"] in available_providers:
            return ModelChoice(
                model_id=entry["model_id"],
                provider=entry["provider"],
                display_name=entry["display_name"],
                reason="Fallback (other providers rate-limited)",
                intent_score=entry["intent_score"],
                score_category=entry["score_category"],
                rank=entry["rank"],
                total_candidates=len(rankings),
                all_rankings=rankings,
                selection_method="fallback",
                data_source=data_source,
            )

    return ModelChoice(
        model_id="gemini-2.5-flash",
        provider="gemini",
        display_name="Gemini 2.5 Flash",
        reason="Default fallback — no models available",
        selection_method="fallback",
        data_source=data_source,
    )

"""
Intent Classifier — Rule-based v1
Classifies user prompts into intent categories to drive model selection.
"""

import re
from dataclasses import dataclass

@dataclass
class IntentResult:
    intent: str
    confidence: float
    keywords_matched: list[str]

# Intent patterns: each intent maps to a list of (pattern, weight) tuples
INTENT_PATTERNS: dict[str, list[tuple[str, float]]] = {
    "code_generation": [
        (r"\b(write|create|build|make|implement|code|develop|generate)\b.*\b(script|function|program|app|api|class|module|component|code|bot)\b", 2.0),
        (r"\b(python|javascript|typescript|java|rust|go|c\+\+|react|html|css|sql|bash)\b", 1.5),
        (r"\b(code|coding|program|programming|script|scripting)\b", 1.0),
        (r"\b(parse|scrape|automate|deploy|compile|render)\b", 0.8),
    ],
    "code_review": [
        (r"\b(review|debug|fix|optimize|refactor|improve)\b.*\b(code|script|function|bug|error)\b", 2.0),
        (r"\b(what'?s wrong|find the bug|fix this|error in)\b", 1.5),
        (r"\b(bug|error|issue|broken|crash|fail|exception)\b", 1.0),
    ],
    "explanation": [
        (r"\b(explain|what is|what are|how does|how do|define|describe|tell me about)\b", 2.0),
        (r"\b(mean|meaning|concept|difference between|work)\b", 1.0),
        (r"\b(why|understand|clarify|elaborate)\b", 0.8),
    ],
    "creative_writing": [
        (r"\b(write|create|compose|draft)\b.*\b(story|poem|essay|blog|article|song|letter|email|post)\b", 2.0),
        (r"\b(creative|fiction|narrative|prose|verse|haiku|limerick)\b", 1.5),
        (r"\b(rewrite|rephrase|tone|style|voice)\b", 0.8),
    ],
    "research": [
        (r"\b(compare|contrast|analyze|research|investigate|evaluate)\b", 1.5),
        (r"\b(pros and cons|advantages|disadvantages|differences|similarities)\b", 2.0),
        (r"\b(which is better|recommend|best|should i use|versus|vs)\b", 1.0),
        (r"\b(latest|recent|trend|state of the art|survey)\b", 0.8),
    ],
    "math_reasoning": [
        (r"\b(solve|calculate|compute|prove|derive|integrate|differentiate)\b", 2.0),
        (r"\b(equation|formula|theorem|proof|matrix|vector|probability)\b", 1.5),
        (r"\b(math|algebra|calculus|statistics|geometry|logic)\b", 1.0),
        (r"\d+\s*[\+\-\*\/\^]\s*\d+", 0.8),
    ],
    "summarization": [
        (r"\b(summarize|summary|tl;?dr|key points|brief|condense|digest)\b", 2.0),
        (r"\b(shorten|reduce|overview|highlights|takeaways)\b", 1.0),
    ],
}

FALLBACK_INTENT = "conversation"


def classify_intent(prompt: str) -> IntentResult:
    """Classify a user prompt into an intent category."""
    prompt_lower = prompt.lower().strip()
    scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}

    for intent, patterns in INTENT_PATTERNS.items():
        total_score = 0.0
        intent_matches = []
        for pattern, weight in patterns:
            if re.search(pattern, prompt_lower):
                total_score += weight
                intent_matches.append(pattern[:40])
        if total_score > 0:
            scores[intent] = total_score
            matched[intent] = intent_matches

    if not scores:
        return IntentResult(
            intent=FALLBACK_INTENT,
            confidence=0.3,
            keywords_matched=[],
        )

    best_intent = max(scores, key=scores.get)
    max_possible = sum(w for _, w in INTENT_PATTERNS[best_intent])
    confidence = min(scores[best_intent] / max_possible, 1.0)

    # Boost confidence if score is significantly ahead of runner-up
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] > sorted_scores[1] * 1.5:
        confidence = min(confidence + 0.1, 1.0)

    return IntentResult(
        intent=best_intent,
        confidence=round(confidence, 2),
        keywords_matched=matched.get(best_intent, []),
    )

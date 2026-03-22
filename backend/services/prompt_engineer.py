from __future__ import annotations

"""
Prompt enhancement compiler for ai_router.

Drop-in replacement for backend/services/prompt_engineer.py.

What this does:
- Normalizes the detected intent into a task shape.
- Infers the chosen model family/version from provider + model_id + display name.
- Compiles the user request into a stronger prompt format for that specific model.
- Stays backward compatible with the current repo API:
      reprompt(original_prompt, provider, intent)
  while also supporting:
      reprompt(..., model_id=..., model_display_name=...)

Important design choice:
- The current ai_router repo sends a *single* user message to each provider.
  So this module emits a single optimized prompt string.
- The compilers below imitate system/developer style prompts inside one string,
  which works with the existing backend without requiring API payload changes.

Notes on coverage:
- There is no stable universal "best prompt standard" for every future model.
  This module therefore uses a registry of known model profiles + conservative
  family fallbacks + a public register_model_profile(...) helper so you can
  extend it without rewriting the compiler.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Intent normalization
# ---------------------------------------------------------------------------

INTENT_ALIASES = {
    "code_generation": "coding",
    "code_review": "code_review",
    "debugging": "debugging",
    "coding": "coding",
    "creative_writing": "writing",
    "writing": "writing",
    "blog_writing": "writing",
    "copywriting": "writing",
    "research": "research",
    "research_analysis": "research",
    "analysis": "analysis",
    "brainstorming": "brainstorming",
    "idea_generation": "brainstorming",
    "summarization": "summarization",
    "summary": "summarization",
    "explanation": "explanation",
    "math_reasoning": "math",
    "math": "math",
    "data_sql": "data_sql",
    "sql": "data_sql",
    "translation": "translation",
    "classification": "classification",
    "extract": "extraction",
    "extraction": "extraction",
    "planning": "planning",
    "project_planning": "planning",
    "image": "image_generation",
    "image_gen": "image_generation",
    "image_generation": "image_generation",
    "conversation": "conversation",
    "chat": "conversation",
}


INTENT_HEURISTICS: list[tuple[str, str]] = [
    (r"\b(review|audit|critique)\b.*\b(code|function|class|script|query)\b", "code_review"),
    (r"\b(debug|fix|bug|error|exception|traceback|why does this fail)\b", "debugging"),
    (r"\b(sql|select|join|where|group by|having|table|schema|database|postgres|mysql|snowflake|bigquery)\b", "data_sql"),
    (r"\b(summarize|summary|tl;dr|tldr|condense)\b", "summarization"),
    (r"\b(brainstorm|ideas|angles|concepts|options|variants|names)\b", "brainstorming"),
    (r"\b(analyze|analysis|compare|pros and cons|tradeoffs|benchmark|evaluate)\b", "analysis"),
    (r"\b(research|sources|citations|references|evidence|literature review)\b", "research"),
    (r"\b(translate|translation|localize|rewrite in spanish|rewrite in french)\b", "translation"),
    (r"\b(extract|parse|pull out|find the fields|return the fields)\b", "extraction"),
    (r"\b(classify|label|categorize|sentiment|choose one label)\b", "classification"),
    (r"\b(plan|roadmap|milestones|timeline|next steps|execution plan)\b", "planning"),
    (r"\b(illustration|thumbnail|poster|banner|logo|photorealistic|negative prompt|cinematic lighting)\b", "image_generation"),
    (r"\b(poem|story|narrative|blog|article|email|linkedin|tweet|thread|copy)\b", "writing"),
    (r"\b(code|function|class|api|algorithm|script|program|refactor|implement|build)\b", "coding"),
    (r"\b(explain|teach|help me understand|what is|how does)\b", "explanation"),
    (r"\b(solve|equation|integral|derivative|probability|prove)\b", "math"),
]


# ---------------------------------------------------------------------------
# Lightweight extraction helpers
# ---------------------------------------------------------------------------

TONE_WORDS = [
    "formal",
    "informal",
    "casual",
    "professional",
    "friendly",
    "witty",
    "funny",
    "humorous",
    "serious",
    "academic",
    "persuasive",
    "empathetic",
    "playful",
    "confident",
    "luxury",
    "minimalist",
    "concise",
    "dramatic",
    "poetic",
    "technical",
]

AUDIENCE_PATTERNS = [
    (r"\bfor beginners?\b", "beginners"),
    (r"\bfor kids\b|\bfor children\b", "children"),
    (r"\bfor students?\b", "students"),
    (r"\bfor executives?\b", "executives"),
    (r"\bfor developers?\b|\bfor engineers?\b", "developers"),
    (r"\bfor recruiters?\b", "recruiters"),
    (r"\bfor customers?\b", "customers"),
    (r"\bfor investors?\b", "investors"),
    (r"\bfor patients?\b", "patients"),
    (r"\bfor readers?\b", "readers"),
]

LANGUAGE_PATTERNS = [
    (r"\bpython\b", "Python"),
    (r"\btypescript\b|\bts\b", "TypeScript"),
    (r"\bjavascript\b|\bjs\b", "JavaScript"),
    (r"\bjava\b", "Java"),
    (r"\bc\+\+\b|\bcpp\b", "C++"),
    (r"\bc#\b|\bcsharp\b", "C#"),
    (r"\brust\b", "Rust"),
    (r"\bgo\b|\bgolang\b", "Go"),
    (r"\bphp\b", "PHP"),
    (r"\bruby\b", "Ruby"),
    (r"\bkotlin\b", "Kotlin"),
    (r"\bswift\b", "Swift"),
    (r"\bscala\b", "Scala"),
    (r"\bhtml\b", "HTML"),
    (r"\bcss\b", "CSS"),
]

FORMAT_PATTERNS = [
    (r"\bjson\b", "JSON"),
    (r"\byaml\b", "YAML"),
    (r"\bxml\b", "XML"),
    (r"\bcsv\b", "CSV"),
    (r"\bmarkdown\b", "Markdown"),
    (r"\b(?:in|as) a table\b|\btable format\b", "Table"),
    (r"\b(?:in|as) bullet(?:s| points)?\b|\bbullet(?:s| points)?\b", "Bullet list"),
    (r"\bemail\b", "Email"),
    (r"\bblog\b", "Blog post"),
    (r"\bessay\b", "Essay"),
    (r"\bthread\b", "Thread"),
    (r"\bcode block\b|\bfenced code\b", "Code block"),
]

SQL_DIALECT_PATTERNS = [
    (r"\bpostgres\b|\bpostgresql\b", "PostgreSQL"),
    (r"\bmysql\b", "MySQL"),
    (r"\bsqlite\b", "SQLite"),
    (r"\bbigquery\b", "BigQuery"),
    (r"\bduckdb\b", "DuckDB"),
    (r"\bsnowflake\b", "Snowflake"),
    (r"\bredshift\b", "Redshift"),
]

IMAGE_STYLE_WORDS = [
    "cinematic",
    "photorealistic",
    "anime",
    "studio ghibli",
    "3d",
    "pixel art",
    "watercolor",
    "oil painting",
    "cyberpunk",
    "minimalist",
    "editorial",
    "fashion",
    "isometric",
]


def _search(pattern: str, text: str) -> Optional[re.Match[str]]:
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)


def _extract_first(patterns: Iterable[tuple[str, str]], text: str) -> Optional[str]:
    for pattern, value in patterns:
        if _search(pattern, text):
            return value
    return None


def _extract_requested_count(text: str) -> Optional[int]:
    match = _search(r"\b(\d{1,2})\s+(?:\w+\s+){0,2}(ideas|options|examples|variants|bullets?|steps?|headlines?)\b", text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _extract_length_hint(text: str) -> Optional[str]:
    word_count = _search(r"\b(\d{2,4})\s+words?\b", text)
    if word_count:
        return f"around {word_count.group(1)} words"

    line_count = _search(r"\b(\d{1,2})\s+(?:lines|sentences|paragraphs?)\b", text)
    if line_count:
        return f"around {line_count.group(1)} {line_count.group(0).split()[-1]}"

    if _search(r"\bbrief\b|\bshort\b|\bconcise\b", text):
        return "brief"
    if _search(r"\bdetailed\b|\bdeep\b|\bcomprehensive\b|\bthorough\b", text):
        return "detailed"
    return None


def _extract_tone(text: str) -> Optional[str]:
    for word in TONE_WORDS:
        if _search(rf"\b{re.escape(word)}\b", text):
            return word
    return None


def _extract_style_hint(text: str) -> Optional[str]:
    for word in IMAGE_STYLE_WORDS:
        if _search(rf"\b{re.escape(word)}\b", text):
            return word
    return None


def _extract_audience(text: str) -> Optional[str]:
    return _extract_first(AUDIENCE_PATTERNS, text)


def _extract_language(text: str) -> Optional[str]:
    return _extract_first(LANGUAGE_PATTERNS, text)


def _extract_output_format(text: str) -> Optional[str]:
    return _extract_first(FORMAT_PATTERNS, text)


def _extract_sql_dialect(text: str) -> Optional[str]:
    return _extract_first(SQL_DIALECT_PATTERNS, text)


def _has_schema_context(text: str) -> bool:
    schema_signals = [
        r"\btable\b",
        r"\bcolumn\b",
        r"\bschema\b",
        r"\bjoin\b",
        r"\bprimary key\b",
        r"\bforeign key\b",
        r"\bselect\b",
        r"\bfrom\b",
        r"\bwhere\b",
    ]
    return any(_search(p, text) for p in schema_signals)


def _wants_sources(text: str) -> bool:
    return bool(_search(r"\bsources?\b|\bcitations?\b|\breferences?\b|\bevidence\b", text))


def _wants_examples(text: str) -> bool:
    return bool(_search(r"\bexample\b|\bexamples\b|\bsample\b|\bfew-shot\b", text))


def _wants_stepwise(text: str) -> bool:
    return bool(_search(r"\bstep by step\b|\bwalk through\b|\bbreak down\b|\bshow your work\b", text))


def _wants_json_only(text: str) -> bool:
    return bool(_search(r"\bvalid json\b|\bjson only\b|\bonly json\b|\breturn json\b", text))


def _looks_long_context(text: str) -> bool:
    return len(text) > 1800 or text.count("\n") > 18


def _seems_creative(text: str) -> bool:
    return bool(_search(r"\bcreative\b|\bstory\b|\bpoem\b|\bnarrative\b|\bbrand voice\b|\bvoice\b", text))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelProfile:
    key: str
    vendor: str
    family: str
    prompt_standard: str
    patterns: tuple[str, ...] = field(default_factory=tuple)
    priority: int = 0
    prefers_xml: bool = False
    prefers_markdown_sections: bool = False
    prefers_numbered_steps: bool = False
    prefers_natural_language: bool = False
    prefers_examples: bool = False
    prefers_strict_output_contract: bool = True
    reasoning_friendly: bool = False
    smaller_model: bool = False
    long_context_friendly: bool = False
    notes: str = ""


@dataclass(frozen=True)
class ModelMatch:
    profile: ModelProfile
    confidence: str
    matched_on: str


@dataclass(frozen=True)
class PromptHints:
    raw_intent: str
    canonical_intent: str
    model_match: ModelMatch
    language: Optional[str] = None
    sql_dialect: Optional[str] = None
    output_format: Optional[str] = None
    tone: Optional[str] = None
    audience: Optional[str] = None
    length_hint: Optional[str] = None
    style_hint: Optional[str] = None
    requested_count: Optional[int] = None
    has_schema_context: bool = False
    wants_sources: bool = False
    wants_examples: bool = False
    wants_stepwise: bool = False
    wants_json_only: bool = False
    long_context: bool = False
    creative: bool = False


@dataclass(frozen=True)
class TaskSpec:
    role: str
    objective: str
    instructions: list[str]
    output_contract: str
    verification: list[str]
    style_guardrails: list[str]
    context_notes: list[str]
    keep_loose: bool = False


# ---------------------------------------------------------------------------
# Model profile registry
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PROFILES: list[ModelProfile] = [
    ModelProfile(
        key="claude-4",
        vendor="anthropic",
        family="claude",
        prompt_standard="anthropic_xml_contract",
        patterns=(
            r"\bclaude-(?:opus|sonnet|haiku)-4(?:[-._]\d+)*\b",
            r"\bclaude[- ]4(?:\.\d+)?\b",
        ),
        priority=200,
        prefers_xml=True,
        prefers_markdown_sections=False,
        prefers_numbered_steps=True,
        prefers_examples=True,
        prefers_strict_output_contract=True,
        reasoning_friendly=True,
        long_context_friendly=True,
        notes="Claude 4.x / 4.6 style",
    ),
    ModelProfile(
        key="claude-3x",
        vendor="anthropic",
        family="claude",
        prompt_standard="anthropic_xml_contract",
        patterns=(
            r"\bclaude-(?:3(?:\.5|\.7)?|3-5|3-7|3)(?:[-._][a-z0-9]+)*\b",
            r"\bclaude[- ]3(?:\.5|\.7)?\b",
        ),
        priority=190,
        prefers_xml=True,
        prefers_numbered_steps=True,
        prefers_examples=True,
        prefers_strict_output_contract=True,
        reasoning_friendly=True,
        long_context_friendly=True,
        notes="Claude 3.x family",
    ),
    ModelProfile(
        key="gpt-5-mini-nano",
        vendor="openai",
        family="openai",
        prompt_standard="openai_small_model_contract",
        patterns=(r"\bgpt-5(?:\.\d+)?(?:[-._](?:mini|nano))\b",),
        priority=185,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_examples=True,
        prefers_strict_output_contract=True,
        smaller_model=True,
        notes="GPT-5 small models need more explicit scaffolding",
    ),
    ModelProfile(
        key="openai-o-series",
        vendor="openai",
        family="openai",
        prompt_standard="openai_reasoning_contract",
        patterns=(r"\bo[134](?:[-._][a-z0-9]+)*\b",),
        priority=180,
        prefers_markdown_sections=True,
        prefers_numbered_steps=False,
        prefers_examples=False,
        prefers_strict_output_contract=True,
        reasoning_friendly=True,
        long_context_friendly=True,
        notes="OpenAI reasoning family",
    ),
    ModelProfile(
        key="gpt-5-mainline",
        vendor="openai",
        family="openai",
        prompt_standard="openai_modular_contract",
        patterns=(r"\bgpt-5(?:\.\d+)?(?:[-._][a-z0-9]+)*\b",),
        priority=175,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_examples=False,
        prefers_strict_output_contract=True,
        reasoning_friendly=True,
        long_context_friendly=True,
        notes="GPT-5 family",
    ),
    ModelProfile(
        key="gpt-4.1-family",
        vendor="openai",
        family="openai",
        prompt_standard="openai_instruction_contract",
        patterns=(r"\bgpt-4\.1(?:[-._][a-z0-9]+)*\b",),
        priority=170,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_examples=True,
        prefers_strict_output_contract=True,
        notes="GPT-4.1 family",
    ),
    ModelProfile(
        key="gpt-4o-family",
        vendor="openai",
        family="openai",
        prompt_standard="openai_instruction_contract",
        patterns=(r"\bgpt-4o(?:[-._][a-z0-9]+)*\b",),
        priority=165,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_examples=True,
        prefers_strict_output_contract=True,
        notes="GPT-4o family",
    ),
    ModelProfile(
        key="gemini-3",
        vendor="google",
        family="gemini",
        prompt_standard="gemini_structured_natural",
        patterns=(r"\bgemini-3(?:\.\d+)?(?:[-._][a-z0-9]+)*\b",),
        priority=160,
        prefers_markdown_sections=True,
        prefers_numbered_steps=False,
        prefers_natural_language=True,
        prefers_strict_output_contract=True,
        reasoning_friendly=True,
        long_context_friendly=True,
        notes="Gemini 3 family",
    ),
    ModelProfile(
        key="gemini-2.5",
        vendor="google",
        family="gemini",
        prompt_standard="gemini_structured_natural",
        patterns=(r"\bgemini-2\.5(?:[-._][a-z0-9]+)*\b",),
        priority=155,
        prefers_markdown_sections=True,
        prefers_numbered_steps=False,
        prefers_natural_language=True,
        prefers_strict_output_contract=True,
        reasoning_friendly=True,
        long_context_friendly=True,
        notes="Gemini 2.5 family",
    ),
    ModelProfile(
        key="generic-gemini",
        vendor="google",
        family="gemini",
        prompt_standard="gemini_structured_natural",
        patterns=(r"\bgemini(?:[-._][a-z0-9]+)*\b",),
        priority=150,
        prefers_markdown_sections=True,
        prefers_natural_language=True,
        prefers_strict_output_contract=True,
        notes="Generic Gemini fallback",
    ),
    ModelProfile(
        key="command-a-r",
        vendor="cohere",
        family="cohere",
        prompt_standard="cohere_grounded_contract",
        patterns=(r"\bcommand(?:[-_ ]?[ar]|[-_ ]?r\+|[-_ ]?r7b)(?:[-._][a-z0-9]+)*\b", r"\bcohere\b"),
        priority=145,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_examples=True,
        prefers_strict_output_contract=True,
        long_context_friendly=True,
        notes="Cohere Command family",
    ),
    ModelProfile(
        key="mistral-family",
        vendor="mistral",
        family="mistral",
        prompt_standard="mistral_decision_tree_contract",
        patterns=(r"\b(?:mistral|mixtral|ministral)(?:[-._][a-z0-9]+)*\b",),
        priority=140,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_examples=False,
        prefers_strict_output_contract=True,
        notes="Mistral family",
    ),
    ModelProfile(
        key="deepseek-r1",
        vendor="deepseek",
        family="deepseek",
        prompt_standard="reasoning_compact_contract",
        patterns=(r"\bdeepseek[-._ ]?r1(?:[-._][a-z0-9]+)*\b", r"\br1[-._ ]distill\b"),
        priority=135,
        prefers_markdown_sections=True,
        reasoning_friendly=True,
        prefers_strict_output_contract=True,
        notes="DeepSeek reasoning family",
    ),
    ModelProfile(
        key="deepseek-v3-chat",
        vendor="deepseek",
        family="deepseek",
        prompt_standard="instruct_compact_contract",
        patterns=(r"\bdeepseek(?:[-._ ]?(?:v3|chat|coder))?(?:[-._][a-z0-9]+)*\b",),
        priority=130,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_strict_output_contract=True,
        notes="DeepSeek instruct/chat family",
    ),
    ModelProfile(
        key="qwen-thinking",
        vendor="alibaba",
        family="qwen",
        prompt_standard="reasoning_compact_contract",
        patterns=(r"\b(?:qwen|qwq)(?:[-._][a-z0-9]+)*(?:thinking|reasoning)(?:[-._][a-z0-9]+)*\b", r"\bqwq(?:[-._][a-z0-9]+)*\b"),
        priority=128,
        reasoning_friendly=True,
        prefers_strict_output_contract=True,
        notes="Qwen thinking / QwQ family",
    ),
    ModelProfile(
        key="qwen-family",
        vendor="alibaba",
        family="qwen",
        prompt_standard="instruct_compact_contract",
        patterns=(r"\bqwen(?:[-._][a-z0-9]+)*\b",),
        priority=126,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_strict_output_contract=True,
        notes="Qwen instruct family",
    ),
    ModelProfile(
        key="llama-family",
        vendor="meta",
        family="llama",
        prompt_standard="instruct_compact_contract",
        patterns=(r"\bllama(?:[-._ ]?\d(?:\.\d+)?)?(?:[-._][a-z0-9]+)*\b",),
        priority=124,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_strict_output_contract=True,
        notes="Llama family",
    ),
    ModelProfile(
        key="gemma-family",
        vendor="google",
        family="gemma",
        prompt_standard="instruct_compact_contract",
        patterns=(r"\bgemma(?:[-._][a-z0-9]+)*\b",),
        priority=123,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_strict_output_contract=True,
        notes="Gemma family",
    ),
    ModelProfile(
        key="phi-family",
        vendor="microsoft",
        family="phi",
        prompt_standard="instruct_compact_contract",
        patterns=(r"\bphi[-._ ]?\d(?:\.\d+)?(?:[-._][a-z0-9]+)*\b",),
        priority=122,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_strict_output_contract=True,
        smaller_model=True,
        notes="Phi family",
    ),
    ModelProfile(
        key="open-source-instruct",
        vendor="generic",
        family="generic_instruct",
        prompt_standard="instruct_compact_contract",
        patterns=(r"\b(?:openchat|falcon|yi|mpt|nous|vicuna|internlm|solar|aya|grok)\b",),
        priority=110,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_strict_output_contract=True,
        notes="Generic instruct family",
    ),
]


_PROVIDER_FALLBACKS: dict[str, ModelProfile] = {
    "claude": ModelProfile(
        key="provider-claude",
        vendor="anthropic",
        family="claude",
        prompt_standard="anthropic_xml_contract",
        priority=50,
        prefers_xml=True,
        prefers_examples=True,
        prefers_numbered_steps=True,
        prefers_strict_output_contract=True,
    ),
    "anthropic": ModelProfile(
        key="provider-anthropic",
        vendor="anthropic",
        family="claude",
        prompt_standard="anthropic_xml_contract",
        priority=50,
        prefers_xml=True,
        prefers_examples=True,
        prefers_numbered_steps=True,
        prefers_strict_output_contract=True,
    ),
    "openai": ModelProfile(
        key="provider-openai",
        vendor="openai",
        family="openai",
        prompt_standard="openai_instruction_contract",
        priority=50,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_strict_output_contract=True,
    ),
    "gemini": ModelProfile(
        key="provider-gemini",
        vendor="google",
        family="gemini",
        prompt_standard="gemini_structured_natural",
        priority=50,
        prefers_markdown_sections=True,
        prefers_natural_language=True,
        prefers_strict_output_contract=True,
    ),
    "google": ModelProfile(
        key="provider-google",
        vendor="google",
        family="gemini",
        prompt_standard="gemini_structured_natural",
        priority=50,
        prefers_markdown_sections=True,
        prefers_natural_language=True,
        prefers_strict_output_contract=True,
    ),
    "groq": ModelProfile(
        key="provider-groq",
        vendor="groq",
        family="generic_instruct",
        prompt_standard="instruct_compact_contract",
        priority=40,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_strict_output_contract=True,
    ),
    "mistral": ModelProfile(
        key="provider-mistral",
        vendor="mistral",
        family="mistral",
        prompt_standard="mistral_decision_tree_contract",
        priority=40,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_strict_output_contract=True,
    ),
    "cohere": ModelProfile(
        key="provider-cohere",
        vendor="cohere",
        family="cohere",
        prompt_standard="cohere_grounded_contract",
        priority=40,
        prefers_markdown_sections=True,
        prefers_numbered_steps=True,
        prefers_strict_output_contract=True,
    ),
}


GENERIC_PROFILE = ModelProfile(
    key="generic",
    vendor="generic",
    family="generic",
    prompt_standard="generic_contract",
    priority=0,
    prefers_markdown_sections=True,
    prefers_numbered_steps=True,
    prefers_strict_output_contract=True,
    notes="Generic fallback",
)


_MODEL_PROFILES: list[ModelProfile] = list(DEFAULT_MODEL_PROFILES)


def register_model_profile(profile: ModelProfile) -> None:
    """Register or override a model profile at runtime."""
    global _MODEL_PROFILES
    _MODEL_PROFILES = [p for p in _MODEL_PROFILES if p.key != profile.key]
    _MODEL_PROFILES.append(profile)
    _MODEL_PROFILES.sort(key=lambda p: p.priority, reverse=True)


# Optional JSON override file for hackathon tweaking without touching Python.
def _load_json_profile_overrides() -> None:
    paths: list[Path] = []
    env_path = os.getenv("PROMPT_PROFILE_OVERRIDES")
    if env_path:
        paths.append(Path(env_path))
    default_path = Path(__file__).resolve().parent.parent / "config" / "model_prompt_profiles.json"
    paths.append(default_path)

    for path in paths:
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict) or "key" not in item:
                continue
            try:
                register_model_profile(
                    ModelProfile(
                        key=str(item.get("key")),
                        vendor=str(item.get("vendor", "custom")),
                        family=str(item.get("family", "custom")),
                        prompt_standard=str(item.get("prompt_standard", "generic_contract")),
                        patterns=tuple(str(x) for x in item.get("patterns", [])),
                        priority=int(item.get("priority", 250)),
                        prefers_xml=bool(item.get("prefers_xml", False)),
                        prefers_markdown_sections=bool(item.get("prefers_markdown_sections", True)),
                        prefers_numbered_steps=bool(item.get("prefers_numbered_steps", True)),
                        prefers_natural_language=bool(item.get("prefers_natural_language", False)),
                        prefers_examples=bool(item.get("prefers_examples", False)),
                        prefers_strict_output_contract=bool(item.get("prefers_strict_output_contract", True)),
                        reasoning_friendly=bool(item.get("reasoning_friendly", False)),
                        smaller_model=bool(item.get("smaller_model", False)),
                        long_context_friendly=bool(item.get("long_context_friendly", False)),
                        notes=str(item.get("notes", "")),
                    )
                )
            except Exception:
                continue
        break


_load_json_profile_overrides()
_MODEL_PROFILES.sort(key=lambda p: p.priority, reverse=True)


# ---------------------------------------------------------------------------
# Model matching
# ---------------------------------------------------------------------------


def _normalize_provider(provider: Optional[str]) -> str:
    return (provider or "").strip().lower()


_VERSION_PATTERNS = [
    r"\b(\d+\.\d+\.\d+)\b",
    r"\b(\d+\.\d+)\b",
    r"(?:-|_)(\d+)-(\d+)(?:\b|[-_])",
]


def _extract_model_version(text: str) -> Optional[str]:
    lowered = text.lower()
    for pattern in _VERSION_PATTERNS:
        m = _search(pattern, lowered)
        if not m:
            continue
        if m.lastindex and m.lastindex >= 2:
            return f"{m.group(1)}.{m.group(2)}"
        return m.group(1)
    return None


_SMALL_MODEL_HINTS = [r"\bmini\b", r"\bnano\b", r"\bhaiku\b", r"\bflash\b", r"\bsmall\b", r"\b7b\b", r"\b8b\b", r"\b3b\b", r"\b1b\b"]
_REASONING_HINTS = [r"\breasoning\b", r"\bthinking\b", r"\br1\b", r"\bo1\b", r"\bo3\b", r"\bo4\b"]


def _infer_model_match(
    provider: str,
    model_id: Optional[str] = None,
    model_display_name: Optional[str] = None,
) -> ModelMatch:
    provider_norm = _normalize_provider(provider)
    combined = " ".join(x for x in [provider_norm, model_id or "", model_display_name or ""] if x).strip().lower()

    for profile in _MODEL_PROFILES:
        for pattern in profile.patterns:
            if pattern and _search(pattern, combined):
                return ModelMatch(profile=profile, confidence="high", matched_on=f"pattern:{pattern}")

    # Slightly smarter generic family inference from plain text.
    if any(_search(p, combined) for p in _REASONING_HINTS):
        return ModelMatch(
            profile=ModelProfile(
                key="reasoning-fallback",
                vendor=provider_norm or "generic",
                family="reasoning",
                prompt_standard="reasoning_compact_contract",
                prefers_markdown_sections=True,
                prefers_strict_output_contract=True,
                reasoning_friendly=True,
            ),
            confidence="medium",
            matched_on="heuristic:reasoning",
        )

    if any(_search(p, combined) for p in _SMALL_MODEL_HINTS):
        return ModelMatch(
            profile=ModelProfile(
                key="small-fallback",
                vendor=provider_norm or "generic",
                family="small_instruct",
                prompt_standard="small_model_contract",
                prefers_markdown_sections=True,
                prefers_numbered_steps=True,
                prefers_examples=True,
                prefers_strict_output_contract=True,
                smaller_model=True,
            ),
            confidence="medium",
            matched_on="heuristic:small-model",
        )

    if provider_norm in _PROVIDER_FALLBACKS:
        return ModelMatch(profile=_PROVIDER_FALLBACKS[provider_norm], confidence="medium", matched_on=f"provider:{provider_norm}")

    return ModelMatch(profile=GENERIC_PROFILE, confidence="low", matched_on="generic")


# ---------------------------------------------------------------------------
# Build prompt hints and task specs
# ---------------------------------------------------------------------------


def _normalize_intent(intent: str, original_prompt: str) -> str:
    lowered = (intent or "conversation").strip().lower()
    canonical = INTENT_ALIASES.get(lowered, lowered)
    known_canonical = {"conversation", "explanation", "analysis", "research", "writing", "coding", "data_sql", "summarization", "brainstorming", "image_generation", "math", "planning", "translation", "classification", "extraction", "code_review", "debugging"}

    # If the classifier already gave us a recognized label or alias, trust it first.
    if lowered in INTENT_ALIASES or canonical in known_canonical:
        return canonical if canonical in known_canonical else "conversation"

    for pattern, guess in INTENT_HEURISTICS:
        if _search(pattern, original_prompt):
            return guess

    return "conversation"



def _build_hints(
    original_prompt: str,
    provider: str,
    intent: str,
    model_id: Optional[str] = None,
    model_display_name: Optional[str] = None,
) -> PromptHints:
    model_match = _infer_model_match(provider, model_id=model_id, model_display_name=model_display_name)
    canonical_intent = _normalize_intent(intent, original_prompt)

    # Promote analysis to research if sources/citations are explicitly requested.
    if canonical_intent == "analysis" and _wants_sources(original_prompt):
        canonical_intent = "research"
    if canonical_intent == "coding" and _search(r"\bsql\b|\bselect\b|\bjoin\b|\bdatabase\b", original_prompt):
        canonical_intent = "data_sql"

    return PromptHints(
        raw_intent=intent,
        canonical_intent=canonical_intent,
        model_match=model_match,
        language=_extract_language(original_prompt),
        sql_dialect=_extract_sql_dialect(original_prompt),
        output_format=_extract_output_format(original_prompt),
        tone=_extract_tone(original_prompt),
        audience=_extract_audience(original_prompt),
        length_hint=_extract_length_hint(original_prompt),
        style_hint=_extract_style_hint(original_prompt),
        requested_count=_extract_requested_count(original_prompt),
        has_schema_context=_has_schema_context(original_prompt),
        wants_sources=_wants_sources(original_prompt),
        wants_examples=_wants_examples(original_prompt),
        wants_stepwise=_wants_stepwise(original_prompt),
        wants_json_only=_wants_json_only(original_prompt),
        long_context=_looks_long_context(original_prompt),
        creative=_seems_creative(original_prompt),
    )



def _metadata_lines(h: PromptHints, model_id: Optional[str] = None, model_display_name: Optional[str] = None) -> list[str]:
    lines: list[str] = []
    if model_id:
        lines.append(f"Model ID: {model_id}")
    if model_display_name and model_display_name != model_id:
        lines.append(f"Model display name: {model_display_name}")
    version = _extract_model_version(" ".join([model_id or "", model_display_name or ""]))
    if version:
        lines.append(f"Detected version: {version}")
    lines.append(f"Prompt standard: {h.model_match.profile.prompt_standard}")
    if h.audience:
        lines.append(f"Audience: {h.audience}")
    if h.tone:
        lines.append(f"Tone: {h.tone}")
    if h.length_hint:
        lines.append(f"Length: {h.length_hint}")
    if h.language and h.canonical_intent not in {"data_sql", "translation"}:
        lines.append(f"Language: {h.language}")
    if h.sql_dialect and h.canonical_intent == "data_sql":
        lines.append(f"SQL dialect: {h.sql_dialect}")
    if h.output_format:
        lines.append(f"Requested output format: {h.output_format}")
    if h.long_context:
        lines.append("Request appears long-context")
    return lines


# ---------------------------------------------------------------------------
# Intent builders
# ---------------------------------------------------------------------------


def _coding_spec(h: PromptHints) -> TaskSpec:
    language = h.language or "the language implied by the user request"
    output_contract = h.output_format or "Return code first in a fenced code block, then a short explanation of the key decisions."
    return TaskSpec(
        role="expert software engineer",
        objective="Produce correct, runnable, and idiomatic code for the user's request.",
        instructions=[
            f"Use {language} unless the user explicitly asked for another language.",
            "Be precise and literal; do not add unrelated features.",
            "Make inputs, outputs, assumptions, and important edge cases explicit.",
            "Prefer production-sensible error handling and naming.",
            "If the request is underspecified, state the assumption briefly before the solution.",
        ],
        output_contract=output_contract,
        verification=[
            "Check that the solution matches the requested behavior.",
            "Check that variable names and control flow are internally consistent.",
            "Check that no required step was skipped.",
        ],
        style_guardrails=["Avoid filler prose.", "Do not invent APIs or libraries unless clearly marked as assumptions."],
        context_notes=[],
    )



def _code_review_spec(h: PromptHints) -> TaskSpec:
    return TaskSpec(
        role="senior code reviewer",
        objective="Review the provided code and return the most important issues and fixes.",
        instructions=[
            "Review before rewriting.",
            "Prioritize correctness, security, maintainability, and performance in that order unless the prompt says otherwise.",
            "Separate observed issues from recommended fixes.",
            "If you provide revised code, keep it minimal and relevant to the issues you found.",
        ],
        output_contract=h.output_format or "Return: 1) issues found 2) improved code or diffs 3) brief explanation of the fixes.",
        verification=[
            "Check that every suggested fix maps to an actual issue you identified.",
            "Do not claim a bug unless you can point to the exact risky logic.",
        ],
        style_guardrails=["Avoid vague advice like 'improve readability' without specifics."],
        context_notes=[],
    )



def _debugging_spec(h: PromptHints) -> TaskSpec:
    return TaskSpec(
        role="expert debugger",
        objective="Identify the root cause and produce the smallest reliable fix.",
        instructions=[
            "Start by stating the likely root cause or top few hypotheses.",
            "If there is an error message or traceback, anchor the diagnosis to it.",
            "Propose the smallest viable fix before suggesting larger refactors.",
            "If needed, mention the missing information that would confirm the diagnosis.",
        ],
        output_contract=h.output_format or "Return: root cause, fix, and corrected code or patch.",
        verification=[
            "Check that the fix actually addresses the stated root cause.",
            "Check that the fix does not silently break other obvious cases.",
        ],
        style_guardrails=["Do not shotgun multiple unrelated fixes unless the prompt asks for alternatives."],
        context_notes=[],
    )



def _writing_spec(h: PromptHints) -> TaskSpec:
    notes: list[str] = []
    if h.tone:
        notes.append(f"Use a {h.tone} tone.")
    if h.audience:
        notes.append(f"Write for {h.audience}.")
    if h.length_hint:
        notes.append(f"Target length: {h.length_hint}.")

    return TaskSpec(
        role="skilled writer and editor",
        objective="Write strong, natural prose that matches the user's goal, tone, and audience.",
        instructions=[
            "Preserve voice, specificity, and flow.",
            "Do not over-structure unless the user asked for a structured format.",
            "Favor concrete language over generic filler.",
            *notes,
        ],
        output_contract=h.output_format or "Return the final piece first. Add brief notes only if they genuinely help.",
        verification=[
            "Check that the tone matches the request.",
            "Check that the writing sounds natural rather than template-generated.",
        ],
        style_guardrails=["Do not flatten creative writing into corporate bullet points unless asked."],
        context_notes=[],
        keep_loose=True,
    )



def _research_spec(h: PromptHints) -> TaskSpec:
    return TaskSpec(
        role="careful research analyst",
        objective="Deliver a grounded, structured analysis with clear separation between facts, assumptions, and conclusions.",
        instructions=[
            "Structure the response clearly around the user's question.",
            "Separate known facts from inference.",
            "Do not fabricate citations, papers, URLs, or statistics.",
            "If sources are missing or unverifiable, say so plainly.",
            "Call out uncertainty, limitations, and missing evidence.",
        ],
        output_contract=h.output_format or "Return: short summary, key findings, evidence or support, caveats, and conclusion.",
        verification=[
            "Check that every claim is either sourced, grounded in the prompt, or clearly marked as inference.",
            "Check that uncertainty is not hidden.",
        ],
        style_guardrails=["Do not present speculation as fact."],
        context_notes=["Sources requested." if h.wants_sources else "Source discipline still required."],
    )



def _analysis_spec(h: PromptHints) -> TaskSpec:
    return TaskSpec(
        role="strategic analyst",
        objective="Analyze the question, compare options, and produce a decision-useful answer.",
        instructions=[
            "Compare relevant options or perspectives directly.",
            "Surface tradeoffs, risks, and assumptions.",
            "Make the decision criteria explicit.",
        ],
        output_contract=h.output_format or "Return: framing, comparison, tradeoffs, and recommendation.",
        verification=[
            "Check that the comparison uses consistent criteria.",
            "Check that the recommendation follows from the analysis.",
        ],
        style_guardrails=["Do not hide major downside risks."],
        context_notes=[],
    )



def _brainstorming_spec(h: PromptHints) -> TaskSpec:
    count = h.requested_count or 10
    return TaskSpec(
        role="creative ideation partner",
        objective="Generate a diverse set of useful ideas rather than minor variations of the same idea.",
        instructions=[
            f"Generate at least {count} distinct ideas unless the user explicitly asked for fewer.",
            "Optimize for breadth first, then usefulness.",
            "Keep each idea concrete enough to act on.",
            "Avoid evaluating or narrowing the list unless the user asks for ranking.",
        ],
        output_contract=h.output_format or f"Return a numbered list of {count} distinct ideas, each with a short explanation.",
        verification=[
            "Check that the ideas are meaningfully different from each other.",
            "Check that each idea is specific enough to be useful.",
        ],
        style_guardrails=["Do not collapse creativity by overconstraining the output."],
        context_notes=[],
        keep_loose=True,
    )



def _image_generation_spec(h: PromptHints) -> TaskSpec:
    style_note = f"Preserve the requested visual style: {h.style_hint}." if h.style_hint else ""
    return TaskSpec(
        role="expert image prompt designer",
        objective="Turn the request into a compact, high-signal image generation prompt.",
        instructions=[
            "Prioritize subject, composition, style, lighting, mood, color palette, camera/frame, and setting.",
            "Use prompt-like phrasing rather than essay prose.",
            "Include a negative prompt when helpful to avoid artifacts or unwanted elements.",
            *([style_note] if style_note else []),
        ],
        output_contract=h.output_format or "Return: 1) final image prompt 2) negative prompt 3) one optional alternate variant.",
        verification=[
            "Check that the final prompt is visually specific.",
            "Check that important requested attributes are not lost.",
        ],
        style_guardrails=["Avoid long abstract explanations inside the final image prompt."],
        context_notes=[],
    )



def _summarization_spec(h: PromptHints) -> TaskSpec:
    notes: list[str] = []
    if h.audience:
        notes.append(f"Write the summary for {h.audience}.")
    if h.length_hint:
        notes.append(f"Keep the summary {h.length_hint}.")
    return TaskSpec(
        role="expert summarizer",
        objective="Compress the source while preserving the main point, important facts, and critical qualifiers.",
        instructions=[
            "Preserve key entities, numbers, dates, decisions, and conclusions.",
            "Remove repetition and low-value detail.",
            *notes,
        ],
        output_contract=h.output_format or "Return a concise summary followed by key takeaways.",
        verification=[
            "Check that no critical fact, number, or qualifier was dropped.",
            "Check that the summary is genuinely shorter and cleaner than the input.",
        ],
        style_guardrails=["Do not inject new claims or interpretations that were not in the source."],
        context_notes=[],
    )



def _data_sql_spec(h: PromptHints) -> TaskSpec:
    dialect = h.sql_dialect or "the SQL dialect implied by the request"
    notes: list[str] = []
    if not h.has_schema_context:
        notes.append("Schema details appear incomplete; state assumptions explicitly.")
    return TaskSpec(
        role="expert data analyst and SQL engineer",
        objective="Produce a correct query or SQL explanation that matches the requested output shape.",
        instructions=[
            f"Use {dialect} if known; otherwise make the dialect assumption explicit.",
            "Be explicit about tables, columns, joins, filters, window functions, and aggregations.",
            "If schema details are missing, do not invent certainty; state assumptions.",
            "Match the requested output columns and grouping exactly.",
            *notes,
        ],
        output_contract=h.output_format or "Return SQL first, then a short explanation of assumptions and logic.",
        verification=[
            "Check that selected columns align with grouping and aggregations.",
            "Check that joins and filters are logically consistent.",
        ],
        style_guardrails=["Do not pretend the schema is known if it was not provided."],
        context_notes=[],
    )



def _explanation_spec(h: PromptHints) -> TaskSpec:
    notes: list[str] = []
    if h.audience:
        notes.append(f"Aim the explanation at {h.audience}.")
    return TaskSpec(
        role="clear teacher",
        objective="Explain the topic clearly from the user's likely level upward.",
        instructions=[
            "Start simple, then add depth.",
            "Use examples or analogies only when they improve clarity.",
            "Define jargon or avoid it.",
            *notes,
        ],
        output_contract=h.output_format or "Return a simple explanation first, then deeper detail.",
        verification=[
            "Check that the first paragraph is understandable without specialist context.",
            "Check that examples actually map to the concept.",
        ],
        style_guardrails=["Avoid unnecessary jargon and empty reassurance."],
        context_notes=[],
    )



def _math_spec(h: PromptHints) -> TaskSpec:
    return TaskSpec(
        role="careful math tutor",
        objective="Solve the problem correctly and present the key reasoning clearly.",
        instructions=[
            "Work methodically and keep notation consistent.",
            "Show the essential steps needed to understand the solution.",
            "Verify the final answer when practical.",
        ],
        output_contract=h.output_format or "Return the worked solution with the final answer clearly marked.",
        verification=[
            "Check the arithmetic or algebra one more time before finalizing.",
            "Check that the final answer actually answers the asked question.",
        ],
        style_guardrails=["Do not skip from setup to final answer without the key steps."],
        context_notes=[],
    )



def _translation_spec(h: PromptHints) -> TaskSpec:
    notes: list[str] = []
    if h.tone:
        notes.append(f"Preserve the {h.tone} tone if possible.")
    return TaskSpec(
        role="expert translator and localizer",
        objective="Translate the text faithfully while preserving meaning, tone, and intent.",
        instructions=[
            "Preserve meaning first, then tone and rhythm.",
            "Do not add commentary unless requested.",
            "If a phrase is ambiguous or culturally specific, choose the most natural equivalent and briefly flag ambiguity only if it materially matters.",
            *notes,
        ],
        output_contract=h.output_format or "Return only the translated text unless the user asked for notes.",
        verification=[
            "Check that the translation did not drop or add meaning.",
            "Check that names, numbers, and proper nouns are handled correctly.",
        ],
        style_guardrails=["Avoid word-for-word stiffness when a natural translation is better."],
        context_notes=[],
    )



def _extraction_spec(h: PromptHints) -> TaskSpec:
    output_contract = h.output_format or ("Return valid JSON only." if h.wants_json_only else "Return only the extracted fields in a structured format.")
    return TaskSpec(
        role="information extraction engine",
        objective="Extract only the requested information without inventing missing values.",
        instructions=[
            "Return only fields that are supported by the input or clearly requested.",
            "If a value is missing, use null or say it is missing rather than guessing.",
            "Do not add commentary unless explicitly asked for it.",
        ],
        output_contract=output_contract,
        verification=[
            "Check that every extracted value is present in the source or clearly inferable from it.",
            "Check that no unsupported field was added.",
        ],
        style_guardrails=["Do not hallucinate fields or values."],
        context_notes=[],
    )



def _classification_spec(h: PromptHints) -> TaskSpec:
    return TaskSpec(
        role="classification assistant",
        objective="Assign the most appropriate label using the user's criteria.",
        instructions=[
            "Use only the label set or classification rule implied by the prompt.",
            "If the label set is missing or ambiguous, say so rather than inventing a taxonomy.",
            "Keep the answer compact unless justification is requested.",
        ],
        output_contract=h.output_format or "Return the label first. Add a short rationale only if it helps or was requested.",
        verification=[
            "Check that the chosen label follows the stated criteria.",
            "Check that the response does not drift into unrelated explanation.",
        ],
        style_guardrails=["Do not invent classes that were not requested."],
        context_notes=[],
    )



def _planning_spec(h: PromptHints) -> TaskSpec:
    return TaskSpec(
        role="execution planner",
        objective="Turn the request into a concrete plan with clear sequencing and dependencies.",
        instructions=[
            "Break the work into actionable steps or milestones.",
            "Surface dependencies, risks, and key assumptions.",
            "Include prioritization when it helps execution.",
        ],
        output_contract=h.output_format or "Return milestones or steps, dependencies, risks, and next actions.",
        verification=[
            "Check that the plan order is logical.",
            "Check that major dependencies or blockers are not omitted.",
        ],
        style_guardrails=["Avoid vague plans that cannot be acted on."],
        context_notes=[],
    )



def _conversation_spec(h: PromptHints) -> TaskSpec:
    return TaskSpec(
        role="helpful assistant",
        objective="Answer the user directly and helpfully.",
        instructions=[
            "Answer the user's request directly.",
            "Ask for clarification only if missing information blocks a useful answer.",
            "Stay faithful to the user's goal and avoid unnecessary verbosity.",
        ],
        output_contract=h.output_format or "Return a direct answer.",
        verification=["Check that the response actually addresses the user's question."],
        style_guardrails=["Do not pad the answer with generic filler."],
        context_notes=[],
    )


INTENT_BUILDERS = {
    "coding": _coding_spec,
    "code_review": _code_review_spec,
    "debugging": _debugging_spec,
    "writing": _writing_spec,
    "research": _research_spec,
    "analysis": _analysis_spec,
    "brainstorming": _brainstorming_spec,
    "image_generation": _image_generation_spec,
    "summarization": _summarization_spec,
    "data_sql": _data_sql_spec,
    "explanation": _explanation_spec,
    "math": _math_spec,
    "translation": _translation_spec,
    "extraction": _extraction_spec,
    "classification": _classification_spec,
    "planning": _planning_spec,
    "conversation": _conversation_spec,
}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )



def _article(noun_phrase: str) -> str:
    article = "an" if noun_phrase[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {noun_phrase}"



def _instruction_lines(items: Iterable[str], prefix: str = "- ") -> str:
    return "\n".join(f"{prefix}{item}" for item in items)



def _numbered_lines(items: Iterable[str]) -> str:
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1))


# ---------------------------------------------------------------------------
# Standards / compilers
# ---------------------------------------------------------------------------


def _render_anthropic_xml_contract(original_prompt: str, h: PromptHints, spec: TaskSpec, *, model_id: Optional[str], model_display_name: Optional[str]) -> str:
    metadata = _metadata_lines(h, model_id=model_id, model_display_name=model_display_name)
    meta_xml = "\n".join(f"  <item>{_escape_xml(x)}</item>" for x in metadata) or "  <item>None</item>"
    instructions_xml = "\n".join(f"  <rule>{_escape_xml(x)}</rule>" for x in spec.instructions)
    checks_xml = "\n".join(f"  <check>{_escape_xml(x)}</check>" for x in spec.verification)
    guard_xml = "\n".join(f"  <guard>{_escape_xml(x)}</guard>" for x in spec.style_guardrails)
    context_xml = "\n".join(f"  <note>{_escape_xml(x)}</note>" for x in spec.context_notes) or "  <note>None</note>"

    return f"""<role>{_escape_xml(spec.role)}</role>
<task_type>{_escape_xml(h.canonical_intent)}</task_type>
<objective>{_escape_xml(spec.objective)}</objective>
<context>
  <metadata>
{meta_xml}
  </metadata>
  <user_request>{_escape_xml(original_prompt)}</user_request>
  <notes>
{context_xml}
  </notes>
</context>
<instructions>
{instructions_xml}
</instructions>
<output_contract>{_escape_xml(spec.output_contract)}</output_contract>
<verification>
{checks_xml}
</verification>
<style_guardrails>
{guard_xml}
</style_guardrails>
Respond to the user now. If something critical is missing, say the assumption explicitly instead of inventing certainty.
""".strip()



def _render_openai_instruction_contract(original_prompt: str, h: PromptHints, spec: TaskSpec, *, model_id: Optional[str], model_display_name: Optional[str]) -> str:
    metadata = _instruction_lines(_metadata_lines(h, model_id=model_id, model_display_name=model_display_name)) or "- None"
    return f"""You are {_article(spec.role)}.

# Objective
{spec.objective}

# Original user request
{original_prompt}

# Context and inferred constraints
{metadata}

# Instructions
{_numbered_lines(spec.instructions)}

# Output contract
{spec.output_contract}

# Verification before final answer
{_numbered_lines(spec.verification)}

# Guardrails
{_instruction_lines(spec.style_guardrails)}

Respond now.
""".strip()



def _render_openai_modular_contract(original_prompt: str, h: PromptHints, spec: TaskSpec, *, model_id: Optional[str], model_display_name: Optional[str]) -> str:
    metadata = _instruction_lines(_metadata_lines(h, model_id=model_id, model_display_name=model_display_name)) or "- None"
    notes = _instruction_lines(spec.context_notes) or "- None"
    return f"""You are {_article(spec.role)}.

## Goal
{spec.objective}

## Request
{original_prompt}

## Context
{metadata}

## Working rules
{_numbered_lines(spec.instructions)}

## Definition of done
- Satisfy the request directly.
- Follow the output contract exactly.
- State assumptions explicitly when needed.

## Output contract
{spec.output_contract}

## Verification loop
{_numbered_lines(spec.verification)}

## Additional notes
{notes}

## Guardrails
{_instruction_lines(spec.style_guardrails)}

Deliver the best final answer now.
""".strip()



def _render_openai_reasoning_contract(original_prompt: str, h: PromptHints, spec: TaskSpec, *, model_id: Optional[str], model_display_name: Optional[str]) -> str:
    metadata = _instruction_lines(_metadata_lines(h, model_id=model_id, model_display_name=model_display_name)) or "- None"
    verification = list(spec.verification)
    if h.canonical_intent in {"research", "analysis", "math", "coding", "data_sql"}:
        verification.append("Before finalizing, check for edge cases or missing constraints that would materially change the answer.")
    return f"""You are {_article(spec.role)}.

# Goal
{spec.objective}

# User request
{original_prompt}

# Constraints and context
{metadata}

# Core instructions
{_instruction_lines(spec.instructions)}

# Output contract
{spec.output_contract}

# Before you finalize
{_instruction_lines(verification)}

# Guardrails
{_instruction_lines(spec.style_guardrails)}

Respond with a strong final answer. Do not invent missing facts; make assumptions explicit.
""".strip()



def _render_openai_small_model_contract(original_prompt: str, h: PromptHints, spec: TaskSpec, *, model_id: Optional[str], model_display_name: Optional[str]) -> str:
    metadata = _instruction_lines(_metadata_lines(h, model_id=model_id, model_display_name=model_display_name)) or "- None"
    steps = list(spec.instructions)
    steps.append("If information is missing, either state the assumption clearly or ask one focused clarification question.")
    return f"""You are {_article(spec.role)}.

# Critical rule
Follow the requested output contract exactly.

# Task
{spec.objective}

# User request
{original_prompt}

# Context
{metadata}

# Exact execution order
{_numbered_lines(steps)}

# Edge-case behavior
{_numbered_lines(spec.verification)}

# Output contract
{spec.output_contract}

# Guardrails
{_instruction_lines(spec.style_guardrails)}

Do the task now. Keep the answer tightly packaged in the required format.
""".strip()



def _render_gemini_structured_natural(original_prompt: str, h: PromptHints, spec: TaskSpec, *, model_id: Optional[str], model_display_name: Optional[str]) -> str:
    metadata = _instruction_lines(_metadata_lines(h, model_id=model_id, model_display_name=model_display_name)) or "- None"
    instruction_lines = list(spec.instructions)
    if h.long_context:
        instruction_lines.insert(0, "Treat the user request block below as the main context. After reading it, answer the task at the end.")
    return f"""Help with the following request.

User request / context:
{original_prompt}

Please act as {_article(spec.role)}.
Your goal: {spec.objective}

Important context:
{metadata}

Please follow these guidelines:
{_instruction_lines(instruction_lines)}

Return the answer in this format:
{spec.output_contract}

Before finalizing, check:
{_instruction_lines(spec.verification)}

Also follow these guardrails:
{_instruction_lines(spec.style_guardrails)}

If you need to make an assumption, say it explicitly.
""".strip()



def _render_mistral_decision_tree_contract(original_prompt: str, h: PromptHints, spec: TaskSpec, *, model_id: Optional[str], model_display_name: Optional[str]) -> str:
    metadata = _instruction_lines(_metadata_lines(h, model_id=model_id, model_display_name=model_display_name)) or "- None"
    decision_rules = list(spec.instructions)
    if h.canonical_intent in {"data_sql", "classification", "extraction", "analysis"}:
        decision_rules.append("If the input is ambiguous, choose the safest explicit assumption and state it briefly.")
    return f"""## Role
{spec.role}

## Objective
{spec.objective}

## User request
{original_prompt}

## Context
{metadata}

## Follow these steps
{_instruction_lines(decision_rules)}

## Output format
{spec.output_contract}

## Final checks
{_instruction_lines(spec.verification)}

## Guardrails
{_instruction_lines(spec.style_guardrails)}

Only generate what is necessary to satisfy the task.
""".strip()



def _render_cohere_grounded_contract(original_prompt: str, h: PromptHints, spec: TaskSpec, *, model_id: Optional[str], model_display_name: Optional[str]) -> str:
    metadata = _instruction_lines(_metadata_lines(h, model_id=model_id, model_display_name=model_display_name)) or "- None"
    extra = []
    if (h.output_format or "").upper() == "JSON" or h.wants_json_only:
        extra.append("Generate valid JSON only. Do not include prose outside the JSON.")
    if h.wants_sources or h.canonical_intent == "research":
        extra.append("Only include citations or sources if they are grounded in provided material or verifiable context.")
    return f"""# Objective
{spec.objective}

# User request
{original_prompt}

# Context
{metadata}

# Instructions
{_numbered_lines(spec.instructions + extra)}

# Output contract
{spec.output_contract}

# Grounding and verification
{_instruction_lines(spec.verification)}

# Guardrails
{_instruction_lines(spec.style_guardrails)}

Answer now.
""".strip()



def _render_reasoning_compact_contract(original_prompt: str, h: PromptHints, spec: TaskSpec, *, model_id: Optional[str], model_display_name: Optional[str]) -> str:
    metadata = _instruction_lines(_metadata_lines(h, model_id=model_id, model_display_name=model_display_name)) or "- None"
    return f"""Goal: {spec.objective}

User request:
{original_prompt}

Context:
{metadata}

Requirements:
{_instruction_lines(spec.instructions)}

Output contract:
{spec.output_contract}

Before final answer:
{_instruction_lines(spec.verification)}

Guardrails:
{_instruction_lines(spec.style_guardrails)}
""".strip()



def _render_small_model_contract(original_prompt: str, h: PromptHints, spec: TaskSpec, *, model_id: Optional[str], model_display_name: Optional[str]) -> str:
    metadata = _instruction_lines(_metadata_lines(h, model_id=model_id, model_display_name=model_display_name)) or "- None"
    return f"""Task: {spec.objective}

User request:
{original_prompt}

Context:
{metadata}

Do this in order:
{_numbered_lines(spec.instructions)}

Return exactly this format:
{spec.output_contract}

Final checks:
{_numbered_lines(spec.verification)}
""".strip()



def _render_instruct_compact_contract(original_prompt: str, h: PromptHints, spec: TaskSpec, *, model_id: Optional[str], model_display_name: Optional[str]) -> str:
    metadata = _instruction_lines(_metadata_lines(h, model_id=model_id, model_display_name=model_display_name)) or "- None"
    instructions = spec.instructions
    # Keep creative tasks less rigid.
    if spec.keep_loose and h.creative:
        instructions = [item for item in instructions if "Do not over-structure" not in item] + ["Keep the writing natural and non-mechanical."]
    return f"""You are {_article(spec.role)}.
Task: {h.canonical_intent}
Objective: {spec.objective}

User request:
{original_prompt}

Context:
{metadata}

Requirements:
{_instruction_lines(instructions)}

Output format:
{spec.output_contract}

Checks:
{_instruction_lines(spec.verification)}

Guardrails:
{_instruction_lines(spec.style_guardrails)}
""".strip()



def _render_generic_contract(original_prompt: str, h: PromptHints, spec: TaskSpec, *, model_id: Optional[str], model_display_name: Optional[str]) -> str:
    metadata = _instruction_lines(_metadata_lines(h, model_id=model_id, model_display_name=model_display_name)) or "- None"
    return f"""Role: {spec.role}
Objective: {spec.objective}
User request: {original_prompt}
Context:
{metadata}
Instructions:
{_instruction_lines(spec.instructions)}
Output contract: {spec.output_contract}
Verification:
{_instruction_lines(spec.verification)}
Guardrails:
{_instruction_lines(spec.style_guardrails)}
""".strip()


RENDERERS = {
    "anthropic_xml_contract": _render_anthropic_xml_contract,
    "openai_instruction_contract": _render_openai_instruction_contract,
    "openai_modular_contract": _render_openai_modular_contract,
    "openai_reasoning_contract": _render_openai_reasoning_contract,
    "openai_small_model_contract": _render_openai_small_model_contract,
    "gemini_structured_natural": _render_gemini_structured_natural,
    "mistral_decision_tree_contract": _render_mistral_decision_tree_contract,
    "cohere_grounded_contract": _render_cohere_grounded_contract,
    "reasoning_compact_contract": _render_reasoning_compact_contract,
    "small_model_contract": _render_small_model_contract,
    "instruct_compact_contract": _render_instruct_compact_contract,
    "generic_contract": _render_generic_contract,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reprompt(
    original_prompt: str,
    provider: str,
    intent: str,
    model_id: Optional[str] = None,
    model_display_name: Optional[str] = None,
) -> str:
    """
    Rewrite the user's prompt for the already-selected model.

    Backwards-compatible with the current project:
        reprompt(original_prompt, provider, intent)

    Better integration (recommended):
        reprompt(original_prompt, provider, intent, model_id, model_display_name)
    """
    hints = _build_hints(
        original_prompt=original_prompt,
        provider=provider,
        intent=intent,
        model_id=model_id,
        model_display_name=model_display_name,
    )
    task_builder = INTENT_BUILDERS.get(hints.canonical_intent, _conversation_spec)
    spec = task_builder(hints)
    renderer = RENDERERS.get(hints.model_match.profile.prompt_standard, _render_generic_contract)
    return renderer(
        original_prompt,
        hints,
        spec,
        model_id=model_id,
        model_display_name=model_display_name,
    ).strip()



def debug_prompt_plan(
    original_prompt: str,
    provider: str,
    intent: str,
    model_id: Optional[str] = None,
    model_display_name: Optional[str] = None,
) -> dict[str, object]:
    """Useful for evals, UI inspection, or debugging why a prompt was compiled a certain way."""
    hints = _build_hints(
        original_prompt=original_prompt,
        provider=provider,
        intent=intent,
        model_id=model_id,
        model_display_name=model_display_name,
    )
    spec = INTENT_BUILDERS.get(hints.canonical_intent, _conversation_spec)(hints)
    return {
        "raw_intent": hints.raw_intent,
        "canonical_intent": hints.canonical_intent,
        "model_profile": hints.model_match.profile.key,
        "model_vendor": hints.model_match.profile.vendor,
        "model_family": hints.model_match.profile.family,
        "prompt_standard": hints.model_match.profile.prompt_standard,
        "profile_confidence": hints.model_match.confidence,
        "matched_on": hints.model_match.matched_on,
        "language": hints.language,
        "sql_dialect": hints.sql_dialect,
        "output_format": hints.output_format,
        "tone": hints.tone,
        "audience": hints.audience,
        "length_hint": hints.length_hint,
        "style_hint": hints.style_hint,
        "requested_count": hints.requested_count,
        "has_schema_context": hints.has_schema_context,
        "wants_sources": hints.wants_sources,
        "wants_examples": hints.wants_examples,
        "wants_stepwise": hints.wants_stepwise,
        "wants_json_only": hints.wants_json_only,
        "long_context": hints.long_context,
        "creative": hints.creative,
        "task_role": spec.role,
        "task_objective": spec.objective,
        "task_output_contract": spec.output_contract,
        "task_instructions": spec.instructions,
        "task_verification": spec.verification,
        "task_guardrails": spec.style_guardrails,
    }

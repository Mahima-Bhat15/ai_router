"""
Prompt Re-engineering — rewrites prompts optimized for target model + intent.
Providers: Claude, Gemini, Groq (LLaMA), OpenRouter (DeepSeek, Gemma, etc.)
"""

TEMPLATES: dict[str, dict[str, str]] = {
    "claude": {
        "code_generation": "<task>\n{prompt}\n</task>\n\n<requirements>\n- Write clean, well-commented, production-quality code\n- Include comprehensive error handling\n- Add type hints (Python) or TypeScript types (JS/TS)\n- Follow the language's idiomatic style\n</requirements>\n\n<format>\nProvide the complete code in a fenced code block, then explain your key design decisions briefly.\n</format>",
        "code_review": "<task>Review and improve this code</task>\n\n{prompt}\n\n<format>\n1. Bugs and errors found\n2. Performance concerns\n3. Improved version with explanations\n</format>",
        "explanation": "<task>Explain the following clearly and thoroughly</task>\n\n{prompt}\n\n<format>\nStart with a simple analogy, then explain in detail with examples.\n</format>",
        "creative_writing": "{prompt}\n\nWrite with vivid detail, strong voice, originality, and engaging structure. Show, don't tell.",
        "research": "<task>Provide balanced research and analysis</task>\n\n{prompt}\n\n<format>\nCover multiple perspectives with evidence. End with your assessment.\n</format>",
        "math_reasoning": "<task>Solve step by step with clear reasoning</task>\n\n{prompt}\n\n<format>\nShow every step of your work. Verify the final answer.\n</format>",
        "summarization": "<task>Summarize concisely</task>\n\n{prompt}\n\n<format>Key points followed by a one-line takeaway.</format>",
        "conversation": "{prompt}",
    },
    "gemini": {
        "code_generation": "Write code for the following task. Be thorough and production-quality.\n\nTask: {prompt}\n\nRequirements:\n- Clean, well-commented code with error handling\n- Type hints and standard conventions\n- Brief explanation of your approach after the code",
        "code_review": "Review the following code carefully. Identify bugs, performance issues, and improvements.\n\n{prompt}\n\nProvide:\n1. Bugs/errors found\n2. Performance concerns\n3. Improved code with explanations",
        "explanation": "Explain the following clearly and thoroughly. Use examples.\n\n{prompt}\n\nStructure from simple to complex. Include a real-world analogy.",
        "creative_writing": "{prompt}\n\nWrite with vivid detail, strong voice, and engaging structure.",
        "research": "Analyze thoroughly with balanced perspectives.\n\n{prompt}\n\nInclude key arguments on each side, evidence, and your assessment.",
        "math_reasoning": "Solve step by step. Show all work clearly.\n\n{prompt}\n\nVerify your answer at the end.",
        "summarization": "Provide a clear, concise summary:\n\n{prompt}\n\nKey points and critical details only.",
        "conversation": "{prompt}",
    },
    "groq": {
        "code_generation": "You are an expert programmer. Write clean, production-quality code.\n\nTask: {prompt}\n\nRequirements: error handling, type hints, comments for non-obvious logic.\nFormat: code block first, then brief explanation.",
        "code_review": "You are a senior code reviewer. Analyze this:\n\n{prompt}\n\nList: (1) Bugs (2) Performance issues (3) Improved code",
        "explanation": "Explain this clearly:\n\n{prompt}\n\nUse a simple analogy first, then go deeper.",
        "creative_writing": "You are a talented writer. Create compelling content:\n\n{prompt}\n\nFocus on strong narrative, vivid language, emotional resonance.",
        "research": "Provide a balanced analysis of:\n\n{prompt}\n\nCover multiple perspectives with key arguments.",
        "math_reasoning": "Solve step by step, showing all work:\n\n{prompt}\n\nDouble-check your final answer.",
        "summarization": "Summarize concisely:\n\n{prompt}\n\n3-5 key points, then a one-sentence takeaway.",
        "conversation": "{prompt}",
    },
    "openrouter": {
        "code_generation": "Write production-quality code for the following task.\n\nTask: {prompt}\n\nRequirements:\n- Clean code with error handling and type hints\n- Comments for non-obvious logic\n- Explanation of approach after the code",
        "code_review": "Review this code for bugs, performance, and best practices:\n\n{prompt}\n\nProvide issues found and improved code.",
        "explanation": "Explain clearly with examples:\n\n{prompt}\n\nStart simple, then go deeper.",
        "creative_writing": "{prompt}\n\nWrite with originality, strong voice, and engaging structure.",
        "research": "Research and analyze:\n\n{prompt}\n\nProvide balanced analysis with multiple perspectives.",
        "math_reasoning": "Solve step by step:\n\n{prompt}\n\nShow all work. Verify the answer.",
        "summarization": "Summarize concisely:\n\n{prompt}",
        "conversation": "{prompt}",
    },
}


def reprompt(original_prompt: str, provider: str, intent: str) -> str:
    provider_templates = TEMPLATES.get(provider, TEMPLATES["gemini"])
    template = provider_templates.get(intent, provider_templates.get("conversation", "{prompt}"))
    return template.replace("{prompt}", original_prompt)

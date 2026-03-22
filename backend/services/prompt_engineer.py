"""
Prompt Re-engineering — rewrites user prompts optimized for the target model + intent.
Providers: Claude, Gemini, Groq (LLaMA)
"""

TEMPLATES: dict[str, dict[str, str]] = {
    "claude": {
        "code_generation": """<task>
{prompt}
</task>

<requirements>
- Write clean, well-commented, production-quality code
- Include comprehensive error handling
- Add type hints (Python) or TypeScript types (JS/TS)
- Follow the language's idiomatic style
</requirements>

<format>
Provide the complete code in a fenced code block, then explain your key design decisions briefly.
</format>""",

        "code_review": """<task>Review and improve this code</task>

{prompt}

<format>
1. Bugs and errors found
2. Performance concerns
3. Improved version of the code with explanations
</format>""",

        "explanation": """<task>Explain the following clearly and thoroughly</task>

{prompt}

<format>
Start with a simple analogy, then explain the concept in detail with examples.
</format>""",

        "creative_writing": """{prompt}

Write with vivid detail, strong voice, originality, and engaging structure. Show, don't tell.""",

        "research": """<task>Provide balanced research and analysis</task>

{prompt}

<format>
Cover multiple perspectives with evidence. End with your assessment of the strongest position.
</format>""",

        "math_reasoning": """<task>Solve step by step with clear reasoning</task>

{prompt}

<format>
Show every step of your work. Verify the final answer.
</format>""",

        "summarization": """<task>Summarize concisely</task>

{prompt}

<format>Key points followed by a one-line takeaway.</format>""",

        "conversation": """{prompt}""",
    },

    "gemini": {
        "code_generation": """Write code for the following task. Be thorough and production-quality.

Task: {prompt}

Requirements:
- Clean, well-commented code with error handling
- Type hints (Python) or TypeScript types (JS/TS)
- Follow the language's standard conventions
- Include a brief explanation of your approach after the code""",

        "code_review": """Review the following code carefully. Identify bugs, performance issues, and improvements.

{prompt}

Provide:
1. Any bugs or errors found
2. Performance concerns
3. Specific improvement suggestions with corrected code""",

        "explanation": """Explain the following topic clearly and thoroughly. Use examples where helpful.

{prompt}

Structure your explanation from simple to complex. Include a real-world analogy if applicable.""",

        "creative_writing": """{prompt}

Write with vivid detail, strong voice, and engaging structure. Show, don't tell.""",

        "research": """Analyze the following topic thoroughly with balanced perspectives.

{prompt}

Include:
- Key arguments on each side
- Data or evidence where relevant
- Your assessment of the strongest position""",

        "math_reasoning": """Solve the following step by step. Show all work clearly.

{prompt}

Walk through each step of your reasoning. Verify your answer at the end.""",

        "summarization": """Provide a clear, concise summary of the following:

{prompt}

Include the key points, main arguments, and critical details. Keep it brief but comprehensive.""",

        "conversation": """{prompt}""",
    },

    "groq": {
        "code_generation": """You are an expert programmer. Write clean, production-quality code.

Task: {prompt}

Requirements:
- Include error handling and type hints
- Add comments for non-obvious logic
- Follow best practices for the language

Format: code block first, then a brief explanation of key decisions.""",

        "code_review": """You are a senior code reviewer. Analyze this carefully:

{prompt}

List: (1) Bugs found (2) Performance issues (3) Suggested improvements with code fixes.""",

        "explanation": """Explain this clearly for someone learning the topic:

{prompt}

Use a simple analogy first, then go deeper into the technical details.""",

        "creative_writing": """You are a talented writer. Create compelling content for:

{prompt}

Focus on strong narrative, vivid language, and emotional resonance.""",

        "research": """Provide a balanced, well-researched analysis of:

{prompt}

Cover multiple perspectives, cite key arguments, and provide your assessment.""",

        "math_reasoning": """Solve step by step, showing all work:

{prompt}

Number each step clearly. Double-check your final answer.""",

        "summarization": """Summarize concisely:

{prompt}

Hit the key points in 3-5 bullet points, then a one-sentence takeaway.""",

        "conversation": """{prompt}""",
    },
}


def reprompt(original_prompt: str, provider: str, intent: str) -> str:
    provider_templates = TEMPLATES.get(provider, TEMPLATES["gemini"])
    template = provider_templates.get(intent, provider_templates.get("conversation", "{prompt}"))
    return template.replace("{prompt}", original_prompt)

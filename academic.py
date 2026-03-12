# -*- coding: utf-8 -*-
"""
Academic Mode Logic — V2
Scholarly prose answers for university/college documents.

TIERS:
  SHORT   → what is / define / who is / when / where  → max 3 sentences, 1 paragraph
  LONG    → explain / discuss / analyze / evaluate /   → max 10 sentences, 2 paragraphs
             examine / elaborate / critically +         (long-trigger wins even if
             any question containing these words        question starts with "What are")
  DEFAULT → everything else                            → max 6 sentences, 1 paragraph

ABSOLUTE RULES:
  - NEVER use bullet points or numbered lists — always prose.
  - Long-trigger keyword in a question always wins over starter keyword.
"""

import re

# =============================================================================
# TIER CLASSIFICATION
# =============================================================================

# Long-trigger keywords — presence ANYWHERE in the question triggers LONG tier
_LONG_TRIGGERS = {
    "explain", "discuss", "analyze", "analyse",
    "evaluate", "examine", "elaborate", "critically",
    "critically analyze", "critically evaluate",
    "in detail", "in depth", "comprehensively",
}

# Short-tier starters — only activate if NO long-trigger is present
_SHORT_STARTERS = (
    "what is", "who is", "define", "what does",
    "when was", "when is", "when did",
    "where is", "where was",
)


def classify_question(question: str) -> str:
    """
    Classify an academic question into one of three tiers.

    Returns one of: 'short', 'default', 'long'

    Priority:
      1. LONG   — any long-trigger keyword present anywhere in the question
                  (wins over 'what are', 'what is', or any other starter)
      2. SHORT  — question starts with a short-tier starter AND no long-trigger
      3. DEFAULT — everything else
    """
    q = question.lower().strip()

    # ── 1. Long trigger — scans entire question, wins unconditionally ────────
    if any(trigger in q for trigger in _LONG_TRIGGERS):
        return "long"

    # ── 2. Compound question — multiple sub-questions → long tier ────────────
    if _count_sub_questions(question) >= 2:
        return "long"
    
    # ── 3. Short — only if no long trigger ───────────────────────────────────
    if any(q.startswith(s) for s in _SHORT_STARTERS):
        return "short"

    # ── 4. Default ────────────────────────────────────────────────────────────
    return "default"


def get_sentence_cap(question: str) -> int:
    """
    Return the sentence cap for the question's tier.
    Used by document_assistant.py for token budget decisions.
    """
    return {
        "short":   3,
        "default": 6,
        "long":    10,
    }[classify_question(question)]


def get_token_budget(question: str) -> int:
    """
    Return LLM max_tokens for the question's tier.
    Called from document_assistant.py to set the LLM token budget.
    """
    return {
        "short":   150,
        "default": 400,
        "long":    750,
    }[classify_question(question)]


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def _count_sub_questions(question: str) -> int:
    """Count distinct sub-questions in a compound question."""
    parts = [p.strip() for p in question.split("?") if p.strip()]
    return len(parts)

def get_answer_prompt(context: str, question: str) -> str:
    """
    Build the LLM prompt for academic mode.
    """
    tier = classify_question(question)

    # ── SHORT ─────────────────────────────────────────────────────────────────
    if tier == "short":
        instruction = (
            "You are providing a concise academic definition or factual statement.\n\n"
            "MANDATORY RULES:\n"
            "1. Write in complete sentences — scholarly prose only.\n"
            "2. Maximum 3 sentences. Do NOT exceed this limit.\n"
            "3. State the definition or fact directly. No preamble.\n"
            "4. NEVER use bullet points, numbered lists, or dashes.\n"
            "5. STOP after the third sentence."
        )

    # ── LONG ──────────────────────────────────────────────────────────────────
    elif tier == "long":
        instruction = (
            "You are providing a comprehensive academic answer.\n\n"
            "MANDATORY RULES:\n"
            "1. Write in complete sentences — scholarly prose only.\n"
            "2. Write EXACTLY 2 paragraphs.\n"
            "   - Paragraph 1: Core concepts, definitions, and context (4-5 sentences).\n"
            "   - Paragraph 2: Analysis, implications, or elaboration (4-5 sentences).\n"
            "3. Maximum 10 sentences total across both paragraphs.\n"
            "4. NEVER use bullet points, numbered lists, hyphens as list markers,\n"
            "   or any other list formatting. Every idea must be in prose form.\n"
            "5. Use formal academic language. Avoid colloquialisms.\n"
            "6. Separate the two paragraphs with a blank line.\n"
            "7. STOP after the second paragraph."
        )

    # ── DEFAULT ───────────────────────────────────────────────────────────────
    else:
        instruction = (
            "You are providing a focused academic answer.\n\n"
            "MANDATORY RULES:\n"
            "1. Write in complete sentences — scholarly prose only.\n"
            "2. Maximum 6 sentences in a single cohesive paragraph.\n"
            "3. Cover the key points with supporting detail from the context.\n"
            "4. NEVER use bullet points, numbered lists, hyphens as list markers,\n"
            "   or any other list formatting. Every idea must be in prose form.\n"
            "5. Use formal academic language.\n"
            "6. STOP after the sixth sentence."
        )

    # ── Compound question detection — runs for ALL tiers ─────────────────────
    sub_q_count = _count_sub_questions(question)
    if sub_q_count >= 2:
        compound_instruction = (
            "\n\nSTRUCTURE INSTRUCTION:\n"
            f"This question has {sub_q_count} parts. "
            "Address each part in a SEPARATE paragraph. "
            "Separate each paragraph with a blank line. "
            "Do NOT merge answers to different parts into one paragraph."
        )
    else:
        compound_instruction = ""

    return (
        f"{instruction}{compound_instruction}\n\n"
        "GROUNDING RULES:\n"
        "- Base your answer ONLY on the document context below.\n"
        "- If the information is not in the context, state clearly in one sentence "
        "that it is not addressed in the provided material.\n"
        "- Do NOT invent, infer beyond the text, or hallucinate information.\n"
        "- Use EXACTLY the terminology as it appears in the document — "
        "do NOT rename or relabel any concept.\n\n"
        "OUTPUT FORMAT — MANDATORY:\n"
        "You MUST respond with ONLY a valid JSON object. No text before or after it.\n"
        "{\n"
        '  "answer": "Your full answer here as a single string.",\n'
        '  "pages_used": [list of integer page numbers you actually drew from]\n'
        "}\n\n"
        f"CONTEXT (each chunk is tagged with its page number):\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "JSON RESPONSE:"
    )




# =============================================================================
# POST-PROCESSING
# =============================================================================

def post_process(answer: str, question: str) -> str:
    """
    Academic post-processing:
    - Enforce sentence cap per tier.
    - Strip any list formatting the LLM may have produced (safety net).
    - Format long-tier answers into 2 paragraphs.
    - Ensure terminal punctuation.
    """
    tier = classify_question(question)
    answer = answer.strip()

    # ── Safety net: strip list markers ───────────────────────────────────────
    # Converts "1. Point here\n2. Point here" → prose sentences
    answer = _strip_list_formatting(answer)

    # ── Sentence cap ─────────────────────────────────────────────────────────
    cap = get_sentence_cap(question)
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    if len(sentences) > cap:
        sentences = sentences[:cap]
        answer = " ".join(sentences).strip()

    # Ensure terminal punctuation
    if answer and answer[-1] not in ".!?":
        last_punct = re.search(r'[.!?](?=[^.!?]*$)', answer)
        answer = answer[:last_punct.end()].strip() if last_punct else answer + "."

    # ── Long tier: enforce 2-paragraph split ─────────────────────────────────
    if tier == "long":
        answer = _split_into_two_paragraphs(answer)

    return answer.strip()


def _strip_list_formatting(text: str) -> str:
    """
    Safety net: convert any list-formatted lines into prose sentences.
    Handles: "1. Item", "- Item", "• Item", "* Item"
    """
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        # Strip numbered list markers: "1. " or "1) "
        line = re.sub(r'^\s*\d+[\.\)]\s+', '', line)
        # Strip bullet markers: "- ", "• ", "* "
        line = re.sub(r'^\s*[-•\*]\s+', '', line)
        cleaned_lines.append(line)

    # Rejoin — collapse blank lines between what were list items into spaces
    result = " ".join(
        line.strip() for line in cleaned_lines if line.strip()
    )

    # Ensure each item-turned-sentence ends with a period
    result = re.sub(r'([a-zA-Z0-9])\s{2,}([A-Z])', r'\1. \2', result)

    return result.strip()


def _split_into_two_paragraphs(answer: str) -> str:
    """
    Split a long answer into exactly 2 paragraphs.
    If already contains a blank line, respect it.
    Otherwise split at the midpoint sentence.
    """
    # If LLM already produced a blank-line paragraph split, respect it
    if "\n\n" in answer:
        parts = [p.strip() for p in answer.split("\n\n") if p.strip()]
        if len(parts) >= 2:
            # Take only 2 paragraphs, discard any extra
            return f"{parts[0]}\n\n{parts[1]}"

    # Otherwise split at midpoint sentence
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    if len(sentences) <= 2:
        return answer  # Too short to split

    mid = max(1, len(sentences) // 2)
    para1 = " ".join(sentences[:mid]).strip()
    para2 = " ".join(sentences[mid:]).strip()

    return f"{para1}\n\n{para2}"

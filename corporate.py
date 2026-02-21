# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 08:39:28 2026

@author: Sashanka
"""

# -*- coding: utf-8 -*-
"""
Corporate Mode Logic
Handles sentence limits, prompt style, and post-processing for
Corporate (Business/Training - Crisp) mode.
"""
import re
from typing import List


def get_sentence_limit(question: str) -> int:
    """
    Determine sentence limit for corporate mode based on question type.
    [P3] Enumeration questions raised to 8 to allow full lists.
    """
    q = question.lower().strip()

    if any(q.startswith(p) for p in ["what is", "define", "who is"]):
        return 3

    if any(k in q for k in [
        "what are", "list", "types", "kinds", "categories",
        "styles", "phases", "stages", "steps", "eras", "periods"
    ]):
        return 8  # [P3] was 5 — allow full enumeration

    if any(q.startswith(p) for p in ["how does", "how do", "why", "explain", "describe"]):
        return 5

    if any(k in q for k in ["compare", "contrast", "difference", "vs", "versus", "similar"]):
        return 6

    return 4


def post_process(answer: str, question: str, sentence_limit: int) -> str:
    """
    Apply corporate-mode post-processing:
    - Truncate to sentence_limit
    - Ensure proper punctuation
    """
    sentences = re.split(r'(?<=[.!?])\s+', answer)

    if len(sentences) > sentence_limit:
        answer = " ".join(sentences[:sentence_limit]).strip()

    if not answer.endswith(('.', '!', '?')):
        match = re.search(r'[.!?](?!.*[.!?])', answer, re.S)
        answer = answer[:match.end()].strip() if match else answer + "."

    return answer.strip()


def get_answer_prompt(
    context: str,
    question: str,
    sentence_limit: int,
    is_simple: bool,
    is_list_q: bool
) -> str:
    """
    Build the LLM prompt for corporate mode answers.
    [P6] Enforces exact document terminology throughout.
    """
    # ✅ AFTER — replace the entire is_list_q block with this:
    if is_list_q:
        instruction = (
            "You are answering a LIST or ENUMERATION question.\n\n"
            "MANDATORY RULES:\n"
            "1. List ONLY the TOP-LEVEL items that DIRECTLY answer the question.\n"
            "   Do NOT include sub-items, descriptions, or characteristics of each item.\n"
            "2. Use EXACTLY the same names and labels that appear in the document context.\n"
            "   Do NOT rename, relabel, or reorder styles/categories/eras.\n"
            "3. Present each item on a new line or as a numbered list.\n"
            "4. STOP immediately after the last list item — NO trailing sentences,\n"
            "   NO summary, NO concluding remarks, NO context sentences.\n"
            "5. NEVER omit top-level items that appear in the context.\n\n"
            "EXAMPLE FORMAT:\n"
            "The major eras are:\n"
            "1. Early Rock and Roll (1950s)\n"
            "2. British Invasion (1960s)\n"
            "3. Classic Rock (1970s)\n"
            "(STOP HERE — do not add any sentence after the last item)"
        )

    elif is_simple:
        instruction = (
            "CRITICAL: This is a SIMPLE FACTUAL question. "
            "Answer in EXACTLY 1-2 sentences MAXIMUM.\n\n"
            "MANDATORY RULES:\n"
            "1. For 'name/list' questions: State ONLY the items, NO explanations.\n"
            "2. For 'who/what/when': Answer directly in ONE sentence.\n"
            "3. NEVER add background context or extra information.\n"
            "4. STOP after stating the fact."
        )
    else:
        instruction = (
            "This is an EXPLANATORY question. Answer in 2-4 sentences.\n\n"
            "Provide specific details from the context.\n"
            "Be comprehensive but concise.\n"
            "IMPORTANT: Use EXACTLY the same terminology as the document context. "
            "Do NOT rename or relabel any styles, roles, or categories."  # [P6]
        )

    limit_instruction = ""
    if sentence_limit > 0 and not is_simple and not is_list_q:
        limit_instruction = f"\nLimit your answer to {sentence_limit} sentences."

    return (
        f"{instruction}{limit_instruction}\n\n"
        "IMPORTANT:\n"
        "- Base your answer ONLY on the document context below.\n"
        "- If the answer isn't in the context, say so in ONE sentence.\n"
        "- Do NOT make up information not present in the context.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

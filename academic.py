# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 08:44:42 2026

@author: Sashanka
"""

# -*- coding: utf-8 -*-
"""
Academic Mode Logic
Handles sentence caps, paragraph formatting, and prompt style for
Academic (University/College - Detailed) mode.
"""
import re
from typing import List


def get_sentence_cap(question: str) -> int:
    """Sentence limit for academic mode."""
    q = question.lower()

    if q.startswith(("what is", "define")):
        return 3

    if any(q.startswith(k) for k in ["explain", "discuss", "analyze", "evaluate"]):
        return 10

    if any(w in q for w in ["compare", "contrast", "difference"]):
        return 7

    return 5


def post_process(answer: str, question: str) -> str:
    """
    Apply academic-mode post-processing:
    - Sentence cap
    - Proper punctuation
    - Paragraph formatting for long answers
    """
    cap = get_sentence_cap(question)
    sentences = re.split(r'(?<=[.!?])\s+', answer)

    if len(sentences) > cap:
        answer = " ".join(sentences[:cap]).strip()

    if not answer.endswith(('.', '!', '?')):
        match = re.search(r'[.!?](?!.*[.!?])', answer, re.S)
        answer = answer[:match.end()].strip() if match else answer + "."

    # Paragraph formatting for long answers
    words = answer.split()
    if len(words) > 100:
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        if len(sentences) > 4:
            para_count = 3 if len(words) > 180 else 2
            para_size = len(sentences) // para_count
            paragraphs = []
            for i in range(para_count):
                start_idx = i * para_size
                end_idx = (
                    start_idx + para_size
                    if i < para_count - 1
                    else len(sentences)
                )
                para = " ".join(sentences[start_idx:end_idx]).strip()
                if para:
                    paragraphs.append(para)
            answer = "\n\n".join(paragraphs)

    return answer.strip()


def get_answer_prompt(
    context: str,
    question: str,
    is_simple: bool,
    is_list_q: bool
) -> str:
    """
    Build the LLM prompt for academic mode answers.
    [P6] Enforces exact document terminology throughout.
    """
    if is_list_q:
        instruction = (
            "You are answering a LIST or ENUMERATION question.\n\n"
            "MANDATORY RULES:\n"
            "1. List EVERY item found in the context — do NOT truncate the list.\n"
            "2. Use EXACTLY the terminology from the document. Do NOT paraphrase labels.\n"
            "3. Use a numbered or bulleted list format.\n"
            "4. You may briefly explain each item in 1 sentence after its name.\n"
            "5. NEVER omit items that appear in the context."
        )
    elif is_simple:
        instruction = (
            "CRITICAL: Simple factual question. "
            "Answer in 1-2 sentences MAXIMUM.\n"
            "State only the fact. Do NOT elaborate."
        )
    else:
        instruction = (
            "Provide a thorough academic answer.\n\n"
            "- Use evidence and detail from the context.\n"
            "- Structure your answer logically.\n"
            "- Be precise and scholarly.\n"
            "IMPORTANT: Use EXACTLY the same terminology as the document context. "
            "Do NOT rename or relabel any concepts, styles, or categories."  # [P6]
        )

    return (
        f"{instruction}\n\n"
        "IMPORTANT:\n"
        "- Base your answer ONLY on the document context below.\n"
        "- If the answer isn't in the context, say so clearly.\n"
        "- Do NOT invent or hallucinate information.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

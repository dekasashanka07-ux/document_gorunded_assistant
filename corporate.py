# -*- coding: utf-8 -*-
"""
Corporate Mode Logic — V2
Crisp, fact-grounded answers for business/training documents.

TIERS:
  FACT        → what is / who is / define / when / where  → 2 sentences
  LIST        → what are / list / types / steps / ...     → numbered list, no cap
  COMPARISON  → compare / vs / difference / contrast      → 4 sentences
  EXPLANATION → how / why / explain / describe / default  → 4 sentences
"""

import re

# =============================================================================
# TIER CLASSIFICATION
# =============================================================================

_LIST_KEYWORDS = {
    "what are", "list", "types of", "kinds of", "categories of",
    "styles of", "phases", "stages", "steps", "eras", "periods",
    "enumerate", "name all", "name the", "name", "give",
    "mention", "outline", "what are the"
}

_COMPARISON_KEYWORDS = {
    "compare", "contrast", "vs", "versus",
    "difference between", "differences between",
    "distinguish between", "comparison between", "similar"
}

_FACT_STARTERS = (
    "what is", "who is", "define", "what does",
    "when was", "when is", "when did",
    "where is", "where was",
)


def classify_question(question: str) -> str:
    """
    Classify a corporate question into one of four tiers.

    Returns one of: 'fact', 'list', 'comparison', 'explanation'

    Priority order (highest → lowest):
      1. comparison  — explicit compare/vs/difference keywords
      2. list        — enumeration keywords
      3. fact        — single-fact starters (what is / define / who is)
      4. explanation — everything else
    """
    q = question.lower().strip()

    # ── 1. Comparison (highest priority) ────────────────────────────────────
    if any(kw in q for kw in _COMPARISON_KEYWORDS):
        return "comparison"

    # ── 2. List ──────────────────────────────────────────────────────────────
    if any(kw in q for kw in _LIST_KEYWORDS):
        return "list"

    # ── 3. Fact ──────────────────────────────────────────────────────────────
    if any(q.startswith(s) for s in _FACT_STARTERS):
        return "fact"

    # ── 4. Explanation (default) ──────────────────────────────────────────────
    return "explanation"


def get_sentence_limit(question: str) -> int:
    """
    Return sentence cap for the question's tier.
    Lists return 0 — no sentence cap (full numbered list expected).
    """
    tier = classify_question(question)
    return {
        "fact":        2,
        "list":        0,   # no cap — full list
        "comparison":  4,
        "explanation": 3,
    }[tier]


def get_token_budget(question: str) -> int:
    """
    Return LLM max_tokens for the question's tier.
    Called from document_assistant.py to set the LLM token budget.
    """
    tier = classify_question(question)
    return {
        "fact":        80,
        "list":        500,
        "comparison":  350,
        "explanation": 200,
    }[tier]


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def get_answer_prompt(
    context: str,
    question: str,
    is_list_q: bool,
    is_comparison: bool,
    is_simple: bool,
) -> str:
    """
    Build the LLM prompt for corporate mode.
    Tier is re-derived here for clarity and independence from document_assistant flags.
    """
    tier = classify_question(question)

    # ── LIST ─────────────────────────────────────────────────────────────────
    if tier == "list":
        instruction = (
            "You are answering a LIST or ENUMERATION question for a corporate audience.\n\n"
            "MANDATORY RULES:\n"
            "1. SCAN THE ENTIRE CONTEXT from start to finish before answering.\n"
            "1b. List ONLY items that appear in the provided context. "
            "Do NOT use your training knowledge to add items not found in the context.\n"
            "1c. If the question asks about a specific person, decade, or category, "
            "list ONLY items the context explicitly associates with that specific "
            "person, decade, or category. Do NOT include items from adjacent sections.\n"
            "2. List EVERY item that directly answers the question — do NOT stop early.\n"
            "3. Answer ONLY what was asked. Match the question exactly:\n"
            "   - 'traits' or 'characteristics' → list ONLY defining features\n"
            "   - 'advantages' → list ONLY benefits\n"
            "   - 'disadvantages' → list ONLY drawbacks\n"
            "   - 'when to use' → list ONLY situations\n"
            "   Do NOT include content from other sections in your answer.\n"
            "4. Include ONLY top-level items. Do NOT include sub-items or elaborations.\n"
            "5. Maximum 8 items. If more exist, list the most prominent ones.\n"
            "6. Use EXACTLY the same names as they appear in the document.\n"
            "   Do NOT rename, abbreviate, paraphrase, or reorder any item.\n"
            "7. Format: numbered list, one item per line, nothing else.\n"
            "8. STOP immediately after the last item.\n"
            "   Do NOT add a summary sentence, closing remark, or any trailing text.\n\n"
            "FORMAT:\n"
            "1. Item Name\n"
            "2. Item Name\n"
            "3. Item Name\n"
            "(stop here)"
        )

    # ── COMPARISON ───────────────────────────────────────────────────────────
    elif tier == "comparison":
        instruction = (
            "You are answering a COMPARISON question for a corporate audience.\n\n"
            "MANDATORY RULES:\n"
            "1. Cover BOTH sides — do NOT stop after describing only one concept.\n"
            "2. Structure in exactly this order:\n"
            "   a) What the FIRST concept is or does (1 sentence).\n"
            "   b) What the SECOND concept is or does (1 sentence).\n"
            "   c) The KEY DIFFERENCE between them (1-2 sentences).\n"
            "3. Total: 3-4 sentences maximum.\n"
            "4. Use EXACTLY the terminology from the document — do NOT rename concepts.\n"
            "5. STOP after the key difference. No trailing remarks or summaries."
        )

    # ── FACT ─────────────────────────────────────────────────────────────────
    elif tier == "fact":
        instruction = (
            "You are answering a FACTUAL question for a corporate audience.\n\n"
            "MANDATORY RULES:\n"
            "1. Answer in 1-2 sentences MAXIMUM.\n"
            "2. State only the direct fact — no background, no elaboration.\n"
            "3. Use EXACTLY the terminology from the document.\n"
            "4. STOP after the fact."
        )

    # ── EXPLANATION (default) ─────────────────────────────────────────────────
    else:
        instruction = (
            "You are answering an EXPLANATORY question for a corporate audience.\n\n"
            "MANDATORY RULES:\n"
            "1. Answer in 3 sentences maximum.\n"
            "2. Be direct and specific — no preamble, no filler phrases.\n"
            "3. Include only details directly relevant to the question.\n"
            "4. Use EXACTLY the terminology from the document — do NOT rename concepts.\n"
            "5. STOP after 3 sentences.\n"
            "6. Every claim in your answer must be traceable to a specific phrase "
            "in the context. If you cannot point to it in the context, do not say it."
        )

    return (
        f"{instruction}\n\n"
        "GROUNDING RULES:\n"
        "- Base your answer ONLY on the document context below.\n"
        "- Treat the context as the ONLY source of truth. "
        "Your training knowledge does not exist for this task. "
        "If the context is thin, give a shorter answer — do not fill gaps.\n"
        "- If a person is described in the context, use ONLY the words and phrases "
        "the document uses to describe THAT specific person. Do not borrow "
        "descriptions from other people mentioned nearby in the context.\n"
        "- If the context mentions something only in passing without detail, "
        "answer ONLY from what is explicitly stated. Do NOT supplement with external knowledge.\n"
        "- If the information is not in the context, say: "
        "'This information is not covered in the provided documents.' (one sentence, stop).\n"
        "- Do NOT make up, infer, or hallucinate any information.\n\n"
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
    Corporate post-processing:
    - Lists: strip trailing prose, deduplicate, renumber.
    - Fact / Comparison / Explanation: enforce sentence cap cleanly.
    - Ensure proper terminal punctuation.
    """
    tier = classify_question(question)
    answer = answer.strip()

    # ── LIST ─────────────────────────────────────────────────────────────────
    if tier == "list":
        # Force line breaks before numbered items — handles JSON collapsing
        answer = re.sub(r'\s+(\d+[\.\)])\s+', r'\n\1 ', answer).strip()
        # If no numbered items found, split on commas only if items are short
        if '\n' not in answer and ',' in answer:
            items = [item.strip() for item in answer.split(',') if item.strip()]
            # Only auto-number if all items are short — avoids splitting prose
            if all(len(item.split()) <= 5 for item in items):
                answer = '\n'.join(f'{i+1}. {item}' for i, item in enumerate(items))
        lines = answer.splitlines()


        # Strip trailing non-list prose
        last_list_idx = -1
        for i, line in enumerate(lines):
            if re.match(r'^\s*\d+[\.\)]\s+\S', line):
                last_list_idx = i
        if last_list_idx >= 0:
            lines = lines[:last_list_idx + 1]

        # Deduplicate list items by normalised text
        seen = set()
        deduped = []
        for line in lines:
            normalised = re.sub(r'^\s*\d+[\.\)]\s+', '', line).lower().strip()
            normalised = re.sub(r'[^a-z0-9\s]', '', normalised)
            if normalised and normalised not in seen:
                seen.add(normalised)
                deduped.append(line)

        # Renumber after deduplication
        final = []
        counter = 1
        for line in deduped:
            renumbered = re.sub(r'^\s*\d+[\.\)]\s+', f'{counter}. ', line)
            final.append(renumbered)
            counter += 1

        return "\n".join(final).strip()

    # ── FACT / COMPARISON / EXPLANATION: sentence cap ────────────────────────
    cap = get_sentence_limit(question)
    if cap > 0:
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        if len(sentences) > cap:
            answer = " ".join(sentences[:cap]).strip()

    # Ensure terminal punctuation
    if answer and answer[-1] not in ".!?":
        last_punct = re.search(r'[.!?](?=[^.!?]*$)', answer)
        answer = answer[:last_punct.end()].strip() if last_punct else answer + "."

    return answer.strip()


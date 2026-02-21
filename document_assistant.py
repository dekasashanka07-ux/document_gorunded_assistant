# -*- coding: utf-8 -*-
"""
Enhanced Generic Document Assistant — Production Ready V5
Multi-chunk, document-grounded QA with STRICT hallucination prevention.
Hybrid retrieval + reranking + semantic chunking.

CHANGELOG V5:
  [P1] Profanity/offensive input sanitized — original text never echoed back
  [P2] Dedicated wide-net list retrieval path (reranker top_n=20, max_tokens=500)
  [P3] Sentence limit raised to 8 and max_tokens=500 for enumeration questions
  [P4] Post-processor strips internal section refs (e.g. "Section A.2")
  [P5] Sentence deduplication pass reduces verbosity/repetition
  [P6] Prompts enforce exact document terminology (delegated to corporate/academic)
"""
# =============================================================================
# IMPORTS
# =============================================================================
import os
import re
from typing import List, Optional, Dict, Tuple, Callable
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.readers.file import PyMuPDFReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    SentenceTransformerRerank
)

import corporate
import academic

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    cache_folder="./embeddings_cache"
)

# =============================================================================
# P1 — PROFANITY / OFFENSIVE INPUT GUARD
# =============================================================================
_BLOCKED_PATTERNS: List[str] = [
    r'\bf+u+c+k+\b', r'\bs+h+i+t+\b', r'\bb+i+t+c+h+\b',
    r'\ba+s+s+h+o+l+e+\b', r'\bc+u+n+t+\b', r'\bd+i+c+k+\b',
    r'\bp+i+s+s+\b', r'\bc+r+a+p+\b', r'\bb+a+s+t+a+r+d+\b',
    r'\bwtf\b', r'\bstfu\b',
]
_BLOCKED_RE = re.compile('|'.join(_BLOCKED_PATTERNS), flags=re.IGNORECASE)

_SAFE_FALLBACK = (
    "I'm sorry, I can't process that input. "
    "Please rephrase your question professionally."
)


def _sanitize_input(text: str) -> Optional[str]:
    """
    Returns None if input is clean.
    Returns a safe generic message if offensive content is detected.
    The original offensive text is NEVER echoed back. [P1]
    """
    if _BLOCKED_RE.search(text):
        return _SAFE_FALLBACK
    return None


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class AnswerResult:
    """Structured answer with metadata."""
    answer: str
    sources: List[str]


# =============================================================================
# DOCUMENT ASSISTANT CLASS
# =============================================================================
class DocumentAssistant:
    """
    Enhanced document-grounded QA assistant.
    Delegates mode-specific logic to corporate.py / academic.py.
    """

    def __init__(
        self,
        documents: List[Document],
        mode: str = "corporate",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        self.mode = mode
        self.documents = documents
        self.index = None
        self.nodes = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.reranker = None
        self.reranker_list = None  # [P2] high-recall variant for list questions
        self._build_index(progress_callback)

    # =========================================================================
    # INDEX BUILDING
    # =========================================================================
    def _build_index(self, progress_callback: Optional[Callable] = None):
        """Build vector index, BM25 index, and both rerankers."""
        total_steps = 5

        if progress_callback:
            progress_callback(1, total_steps, "Step 1/5: Splitting documents into chunks...")

        text_splitter = SentenceSplitter(chunk_size=400, chunk_overlap=100)
        nodes = text_splitter.get_nodes_from_documents(self.documents)

        if progress_callback:
            progress_callback(2, total_steps, "Step 2/5: Processing chunks...")

        chunk_config = self._get_chunk_config()
        self.nodes = self._smart_truncate_nodes(nodes, chunk_config['max_chars'])

        if progress_callback:
            progress_callback(
                3, total_steps,
                f"Step 3/5: Building vector index ({len(self.nodes)} chunks)..."
            )

        self.index = VectorStoreIndex(self.nodes)

        if progress_callback:
            progress_callback(4, total_steps, "Step 4/5: Initializing hybrid retrievers...")

        top_k = 15
        similarity_cutoff = chunk_config['similarity_cutoff']

        self.vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
            ]
        )

        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.nodes,
            similarity_top_k=top_k
        )

        if progress_callback:
            progress_callback(5, total_steps, "Step 5/5: Loading reranking models...")

        # Standard reranker
        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2",
            top_n=10 if self.mode == "corporate" else 8
        )

        # [P2] Wide-net reranker for list/enumeration questions
        self.reranker_list = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2",
            top_n=20
        )

    def _get_chunk_config(self) -> Dict:
        if self.mode == "corporate":
            return {'max_chars': 1200, 'top_k': 15, 'similarity_cutoff': 0.15}
        return {'max_chars': 900, 'top_k': 15, 'similarity_cutoff': 0.18}

    def _smart_truncate_nodes(self, nodes: List, max_chars: int) -> List:
        """Truncate nodes at sentence boundaries."""
        processed_nodes = []
        for node in nodes:
            if len(node.text) <= max_chars:
                processed_nodes.append(node)
                continue
            sentences = re.split(r'(?<=[.!?])\s+', node.text)
            truncated = ""
            for sent in sentences:
                if len(truncated) + len(sent) + 1 <= max_chars:
                    truncated += sent + " "
                else:
                    break
            if len(truncated.strip()) > 50:
                node.text = truncated.strip()
                processed_nodes.append(node)
        return processed_nodes

    # =========================================================================
    # SUMMARY GENERATION
    # =========================================================================
    def generate_summary(
        self,
        groq_api_key: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> str:
        """Generate a concise document summary."""
        try:
            total_steps = 3

            if progress_callback:
                progress_callback(1, total_steps, "Initializing language model...")

            llm = Groq(
                model="llama-3.1-8b-instant",
                api_key=groq_api_key,
                temperature=0.2,
                max_tokens=400
            )

            if progress_callback:
                progress_callback(2, total_steps, "Analyzing document content...")

            filtered_docs = self._filter_summary_docs()
            if not filtered_docs:
                return "Summary unavailable: Document contains primarily structural content."

            context = self._build_summary_context(filtered_docs, max_chars=8000)

            if progress_callback:
                progress_callback(3, total_steps, "Generating summary...")

            prompt = self._get_summary_prompt(context)
            response = llm.complete(prompt)
            summary = str(response).strip()

            if len(summary.split()) < 30:
                return "Summary unavailable: Insufficient substantive content."

            return summary

        except Exception as e:
            return f"Summary generation failed: {str(e)}"

    def _filter_summary_docs(self) -> List[Document]:
        if self.mode == "corporate":
            skip_patterns = ["table of contents", "objectives", "copyright"]
        else:
            skip_patterns = [
                "learning outcomes", "unit objectives", "check your progress",
                "terminal questions", "glossary", "key words"
            ]
        filtered = []
        for doc in self.documents:
            text_lower = doc.text.lower()
            junk_matches = sum(1 for p in skip_patterns if p in text_lower)
            if junk_matches < 2:
                filtered.append(doc)
        return filtered

    def _build_summary_context(self, docs: List[Document], max_chars: int) -> str:
        context = ""
        for doc in docs[:10]:
            chunk = doc.text[:1000].strip()
            if len(context) + len(chunk) > max_chars:
                break
            context += "\n\n" + chunk
        return context.strip()

    def _get_summary_prompt(self, context: str) -> str:
        base = (
            "Create a single-paragraph summary (100-120 words max).\n"
            "Focus ONLY on the main topic and key themes.\n"
            "Use natural, flowing prose — no lists or segmentation.\n"
            "Do NOT start with 'This document discusses' or 'The document...'."
        )
        specific = (
            "\nBusiness focus: What the document covers and why it matters."
            if self.mode == "corporate"
            else "\nAcademic focus: Core arguments and contributions."
        )
        return f"{base}{specific}\n\nDOCUMENT:\n{context}\n\nSUMMARY:"

    # =========================================================================
    # QUESTION ANSWERING — MAIN ENTRY POINT
    # =========================================================================
    def ask_question(
        self,
        question: str,
        groq_api_key: str,
        return_metadata: bool = False
    ):
        """Answer a question grounded in the loaded documents."""

        # ── [P1] Sanitize input first — offensive text never echoed ──────────
        blocked = _sanitize_input(question)
        if blocked:
            result = AnswerResult(answer=blocked, sources=[])
            return result if return_metadata else result.answer

        # ── Basic validation ──────────────────────────────────────────────────
        if not question or len(question.strip()) < 3:
            error = "Please ask a more specific question."
            return AnswerResult(error, []) if return_metadata else error

        if not self.index:
            error = "Error: Index not initialized."
            return AnswerResult(error, []) if return_metadata else error

        try:
            is_list_q = self._is_list_question(question)
            is_simple  = self._is_simple_question(question)

            sentence_limit = 0
            if self.mode == "corporate":
                sentence_limit = corporate.get_sentence_limit(question)

            expanded_queries = self._expand_query(question)

            # ── [P2] Use wide-net path for list questions ─────────────────────
            if is_list_q:
                retrieved_nodes = self._hybrid_retrieve_list(expanded_queries)
            else:
                retrieved_nodes = self._hybrid_retrieve(expanded_queries)

            if not retrieved_nodes:
                result = AnswerResult(
                    answer="This information is not covered in the provided documents.",
                    sources=[]
                )
                return result if return_metadata else result.answer

            # Extra sweep if list result set is still thin
            if is_list_q and len(retrieved_nodes) < 8:
                additional_nodes = self._get_additional_context(question, retrieved_nodes)
                if additional_nodes:
                    retrieved_nodes.extend(additional_nodes)

            context, source_map = self._build_context_with_sources(
                retrieved_nodes, question, wide=is_list_q
            )

            # ── TEMPORARY DIAGNOSTIC — remove before release ──────────────────────
            print(f"\n{'='*60}")
            print(f"[DEBUG] Question   : {question}")
            print(f"[DEBUG] is_list_q  : {is_list_q} | is_simple: {is_simple}")
            print(f"[DEBUG] Nodes fetched: {len(retrieved_nodes)}")
            print(f"[DEBUG] Context preview:\n{context[:600]}")
            print(f"{'='*60}\n")
            # ── END DIAGNOSTIC ────────────────────────────────────────────────────
 
            # ── [P2/P3] Token budget ──────────────────────────────────────────
            if is_list_q:
                max_tokens = 500   # was 300 — allows full enumeration
            elif is_simple:
                max_tokens = 80
            else:
                max_tokens = 300

            llm = Groq(
                model="llama-3.1-8b-instant",
                api_key=groq_api_key,
                temperature=0.0,
                max_tokens=max_tokens
            )

            # ── Build prompt via mode module ──────────────────────────────────
            if self.mode == "corporate":
                prompt = corporate.get_answer_prompt(
                    context, question, sentence_limit, is_simple, is_list_q
                )
            else:
                prompt = academic.get_answer_prompt(
                    context, question, is_simple, is_list_q
                )

            response = llm.complete(prompt)
            answer = str(response).strip()

            if not answer or len(answer.strip()) < 10:
                error = "Unable to generate answer. Please rephrase."
                return AnswerResult(error, []) if return_metadata else error

            answer = self._post_process_answer(answer, question)
            is_negative = self._is_negative_response(answer)

            result = AnswerResult(answer=answer, sources=[] if is_negative else [])
            return result if return_metadata else result.answer

        except Exception:
            error = "Error processing question. Please try rephrasing."
            return AnswerResult(error, []) if return_metadata else error

    # =========================================================================
    # RETRIEVAL HELPERS
    # =========================================================================
    def _is_list_question(self, question: str) -> bool:
        q = question.lower()
        indicators = [
            "what are", "list", "types of", "kinds of", "styles of",
            "phases", "stages", "steps", "eras", "periods", "categories",
            "enumerate", "name all", "name the"
        ]
        return any(i in q for i in indicators)

    def _is_simple_question(self, question: str) -> bool:
        q = question.lower().strip()
        triggers = [
            'name', 'list', 'who sang', 'who wrote', 'who is', 'what is',
            'what are', 'which', 'what year', 'when did', 'when was',
            'how many', 'name a', 'name an', 'give me', 'tell me the'
        ]
        return any(t in q for t in triggers)

    def _expand_query(self, question: str) -> List[str]:
        queries = [question]
        q = question.lower().strip()

        if q.startswith("what is "):
            term = question[8:].strip()
            queries += [f"define {term}", f"explain {term}", f"{term} meaning"]

        if q.startswith("what are "):
            term = question[9:].strip()
            queries += [f"list of {term}", f"types of {term}"]

        if q.startswith("how to "):
            queries += [
                question.replace("how to", "steps for"),
                question.replace("how to", "process of")
            ]

        if q.startswith("why "):
            queries.append(question.replace("why", "reasons for"))

        filler = {"the", "a", "an", "is", "are", "was", "were"}
        key_terms = " ".join(w for w in q.split() if w not in filler)
        if key_terms != q:
            queries.append(key_terms)

        return queries[:4]

    def _collect_nodes(self, queries: List[str]) -> Dict:
        """Shared deduplicating node collection for both retrieval paths."""
        all_nodes: Dict = {}
        for query in queries:
            for node in (
                self.vector_retriever.retrieve(query)
                + self.bm25_retriever.retrieve(query)
            ):
                nid = node.node_id
                if nid not in all_nodes or node.score > all_nodes[nid].score:
                    all_nodes[nid] = node
        return all_nodes

    def _hybrid_retrieve(self, queries: List[str]) -> List:
        """Standard hybrid retrieval with default reranker."""
        all_nodes = self._collect_nodes(queries)
        if not all_nodes:
            return []
        return self.reranker.postprocess_nodes(
            list(all_nodes.values()), query_str=queries[0]
        )

    def _hybrid_retrieve_list(self, queries: List[str]) -> List:
        """
        [P2] Wide-net hybrid retrieval for list/enumeration questions.
        Adds synonym queries and uses the high-recall reranker (top_n=20).
        """
        extra: List[str] = []
        for q in queries:
            extra.append(q.replace("what are", "overview of"))
            extra.append(q.replace("list", "summary of"))
            extra.append(q.replace("eras", "periods history"))
            extra.append(q.replace("styles", "types approaches"))

        all_queries = (queries + extra)[:8]
        all_nodes = self._collect_nodes(all_queries)
        if not all_nodes:
            return []

        return self.reranker_list.postprocess_nodes(
            list(all_nodes.values()), query_str=queries[0]
        )

    def _get_additional_context(self, question: str, existing_nodes: List) -> List:
        """Extra sweep when list result set is thin."""
        existing_ids = {n.node_id for n in existing_nodes}
        broader = (
            question
            .replace("what are", "about")
            .replace("list", "information")
            .replace("eras", "history periods")
        )
        additional = self.vector_retriever.retrieve(broader)
        return [n for n in additional if n.node_id not in existing_ids][:5]

    # =========================================================================
    # CONTEXT BUILDING
    # =========================================================================
    def _build_context_with_sources(
        self,
        nodes: List,
        question: str,
        wide: bool = False
    ) -> Tuple[str, Dict[str, List[str]]]:
        """
        Assemble context string and source map from retrieved nodes.
        [P2] 'wide=True' increases the character budget for list questions.
        """
        question_words = len(question.split())

        if question_words < 5:
            base_chars = 3000
        elif question_words < 12:
            base_chars = 5000
        else:
            base_chars = 7000

        if wide:                        # [P2/P3]
            base_chars = max(base_chars, 8000)

        max_context_chars = (
            int(base_chars * 1.5) if self.mode == "academic" else base_chars
        )

        context_parts: List[str] = []
        source_map: Dict[str, List[str]] = {}
        total_chars = 0

        for node in nodes:
            txt = node.node.text.strip()
            if total_chars + len(txt) > max_context_chars:
                break
            context_parts.append(txt)
            total_chars += len(txt)

            meta = node.node.metadata
            if meta:
                filename = meta.get("filename", "")
                page     = meta.get("page", "")
                if filename:
                    source_str = filename
                    if page and str(page).isdigit():
                        source_str += f" (p.{page})"
                    source_map.setdefault(source_str, []).append(txt)

        return "\n\n---\n\n".join(context_parts), source_map

    # =========================================================================
    # POST-PROCESSING
    # =========================================================================
    def _post_process_answer(self, answer: str, question: str) -> str:
        """
        Full post-processing pipeline:
          1. [P4] Strip internal section references
          2. [P5] Deduplicate near-identical sentences
          3. Apply mode-specific length rules
        """
        # ── Step 1 [P4]: Strip internal section references ────────────────────
        answer = self._strip_internal_refs(answer)

        # ── Step 2 [P5]: Sentence deduplication ──────────────────────────────
        answer = self._deduplicate_sentences(answer)

        # ── Step 3: Mode-specific length / brevity rules ──────────────────────
        is_simple = self._is_simple_question(question)
        is_list_q = self._is_list_question(question)

        # Simple (non-list) questions — extreme brevity
        if is_simple and not is_list_q:
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            q_lower = question.lower().strip()
            if any(t in q_lower for t in ['name', 'list']):
                answer = sentences[0].strip() if sentences else answer
            else:
                answer = (
                    " ".join(sentences[:2]).strip()
                    if len(sentences) > 2
                    else answer
                )
            if not answer.endswith(('.', '!', '?')):
                answer += "."
            return answer.strip()

        # List questions — preserve full enumeration, only clean up trailing noise
        if is_list_q:
            if not answer.endswith(('.', '!', '?')):
                match = re.search(r'[.!?](?!.*[.!?])', answer, re.S)
                answer = answer[:match.end()].strip() if match else answer + "."
            return answer.strip()

        # Corporate / Academic standard length rules
        if self.mode == "corporate":
            answer = corporate.post_process(
                answer, question, corporate.get_sentence_limit(question)
            )
        else:
            answer = academic.post_process(answer, question)

        return answer.strip()

    @staticmethod
    def _strip_internal_refs(text: str) -> str:
        """
        [P4] Remove internal document references that leak into answers.
        Covers: Section A.2, Chapter 4, Unit 2, Appendix B, Page 12,
                (p.12), Figure 3.1, Table 2.3, 'see above/below/page N'.
        """
        patterns = [
            r'\b[Ss]ection\s+[A-Z0-9]+(?:\.[A-Z0-9]+)*\b',  # Section A.2
            r'\b[Cc]hapter\s+\d+\b',                          # Chapter 4
            r'\b[Uu]nit\s+\d+\b',                             # Unit 2
            r'\b[Aa]ppendix\s+[A-Z]\b',                       # Appendix B
            r'\b[Pp]age\s+\d+\b',                             # Page 12
            r'\(p\.\s*\d+\)',                                   # (p.12)
            r'\b[Ff]igure\s+\d+(?:\.\d+)?\b',                 # Figure 3.1
            r'\b[Tt]able\s+\d+(?:\.\d+)?\b',                  # Table 2.3
            r'\bsee\s+(?:above|below|page\s+\d+)\b',          # see above / see page 5
        ]
        cleaned = text
        for pat in patterns:
            cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE)

        # Collapse multiple spaces / orphaned commas left behind
        cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
        cleaned = re.sub(r',\s*,', ',', cleaned)
        cleaned = re.sub(r'\(\s*\)', '', cleaned)

        return cleaned.strip()

    @staticmethod
    def _deduplicate_sentences(text: str) -> str:
        """
        [P5] Remove near-duplicate sentences to reduce verbosity/repetition.

        Strategy:
          - Split on sentence boundaries (handles both prose and numbered lists).
          - Normalise each sentence (lowercase, strip punctuation/whitespace).
          - Keep a sentence only if its normalised form has not been seen before
            AND its normalised form is not a substring of an already-kept sentence
            (catches paraphrased repetition).
        """
        # Preserve numbered-list lines as atomic units
        lines = text.splitlines(keepends=True)

        # Rebuild into segments: list items stay whole, prose gets split
        segments: List[str] = []
        prose_buffer = ""

        for line in lines:
            stripped = line.strip()
            # Detect numbered or bulleted list items
            if re.match(r'^(\d+[\.\)]\s+|[-•]\s+)', stripped):
                if prose_buffer.strip():
                    # Flush prose buffer first
                    segments.extend(re.split(r'(?<=[.!?])\s+', prose_buffer.strip()))
                    prose_buffer = ""
                segments.append(stripped)
            else:
                prose_buffer += " " + line

        if prose_buffer.strip():
            segments.extend(re.split(r'(?<=[.!?])\s+', prose_buffer.strip()))

        def normalise(s: str) -> str:
            return re.sub(r'[^a-z0-9\s]', '', s.lower()).strip()

        seen_norms: List[str] = []
        unique_segments: List[str] = []

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            norm = normalise(seg)

            # Skip if exact duplicate
            if norm in seen_norms:
                continue

            # Skip if this sentence is substantially contained in an existing one
            # (catches "X is Y" vs "X is Y and also Z" — keep the longer one)
            is_subset = any(
                norm in existing and len(norm) > 20
                for existing in seen_norms
            )
            if is_subset:
                continue

            # Replace an existing shorter sentence with a longer, more complete one
            replaced = False
            for i, existing in enumerate(seen_norms):
                if existing in norm and len(existing) > 20:
                    seen_norms[i] = norm
                    unique_segments[i] = seg
                    replaced = True
                    break

            if not replaced:
                seen_norms.append(norm)
                unique_segments.append(seg)

        # Re-join: list items get their own line, prose gets a space
        result_parts: List[str] = []
        for seg in unique_segments:
            if re.match(r'^\d+[\.\)]\s+|^[-•]\s+', seg):
                result_parts.append("\n" + seg)
            else:
                result_parts.append(seg)

        return " ".join(result_parts).strip()

    def _is_negative_response(self, answer: str) -> bool:
        """Detect 'not covered in documents' responses."""
        negatives = [
            "not covered in the documents",
            "not addressed in the documents",
            "this information is not available",
            "not found in the documents",
            "doesn't provide information",
            "this topic is not covered"
        ]
        return any(p in answer.lower() for p in negatives)


# =============================================================================
# DOCUMENT LOADER
# =============================================================================
def load_documents(
    file_paths: List[str],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> List[Document]:
    """
    Load PDF, TXT, or DOCX files into LlamaIndex Document objects
    with rich metadata (filename, page, source, file_type).

    Args:
        file_paths: List of absolute file paths to load.
        progress_callback: Optional callback(current, total, message).

    Returns:
        List of Document objects ready for indexing.
    """
    documents: List[Document] = []
    total_files = len(file_paths)

    for idx, path in enumerate(file_paths, 1):
        if progress_callback:
            progress_callback(
                idx, total_files, f"Loading {os.path.basename(path)}..."
            )

        filename = os.path.basename(path)

        try:
            # ── PDF ───────────────────────────────────────────────────────────
            if path.lower().endswith('.pdf'):
                reader = PyMuPDFReader()
                docs = reader.load(file_path=path)

                for i, doc in enumerate(docs):
                    doc.metadata = {
                        "filename": filename,
                        "page":     i + 1,
                        "source":   path,
                        "file_type": "pdf"
                    }
                    documents.append(doc)

            # ── TXT ───────────────────────────────────────────────────────────
            elif path.lower().endswith('.txt'):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()

                documents.append(Document(
                    text=text,
                    metadata={
                        "filename":  filename,
                        "source":    path,
                        "file_type": "txt"
                    }
                ))

            # ── DOCX ──────────────────────────────────────────────────────────
            elif path.lower().endswith('.docx'):
                try:
                    # Use python-docx if available (clean text extraction)
                    from docx import Document as DocxDocument
                    docx_doc = DocxDocument(path)
                    text = "\n".join(
                        para.text for para in docx_doc.paragraphs
                        if para.text.strip()
                    )
                except ImportError:
                    # Fallback: raw binary read (may include artefacts)
                    with open(path, 'rb') as f:
                        raw = f.read()
                    text = raw.decode('utf-8', errors='ignore')

                documents.append(Document(
                    text=text,
                    metadata={
                        "filename":  filename,
                        "source":    path,
                        "file_type": "docx"
                    }
                ))

        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            continue

    return documents


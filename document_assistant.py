# -*- coding: utf-8 -*-
"""
Generic Document Assistant – Streamlit-compatible
Multi-chunk, document-grounded QA (STRICT + LOW HALLUCINATION)
Hybrid retrieval + semantic chunking for consistency
Mode-aware summary & prompts
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import re
from typing import List, Optional

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.readers.file import PyMuPDFReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# =============================================================================
# CORPORATE MODE REASONING CONTRACT WITH LOCALITY CONSTRAINT
# =============================================================================
CORPORATE_REASONING_CONTRACT = """
You are answering questions about an internal corporate or explanatory document.

ALLOWED AGGREGATION:
You may combine information from multiple chunks ONLY if they belong to:
- The SAME paragraph (across chunk boundaries)
- The SAME numbered or bulleted list
- Directly continuing text under the SAME heading

FORBIDDEN AGGREGATION:
- Do NOT combine information from DIFFERENT headings, even if topics seem related
- Do NOT combine information from separate sections, policies, or independent discussions
- Do NOT merge conceptually similar statements that appear in different parts of the document

GROUNDING RULES:
- Every statement must be explicitly supported in the text
- Do NOT expand the author's meaning
- Do NOT infer intentions, motivations, or benefits
- Do NOT add common knowledge
- Do NOT generalize beyond what is written
- Use semantically equivalent wording if phrasing differs
- Prefer the shortest complete answer

If a specific detail is not stated in the document,
respond exactly: Not covered in the documents.

Your job is to state what the document says, not to explain it or connect distant ideas.
"""


# =============================================================================
# CLASS-BASED ASSISTANT
# =============================================================================
class DocumentAssistant:
    def __init__(self, documents: List[Document], mode: str = "corporate"):
        self.mode = mode
        self.documents = documents
        self.index = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self._build_index()

    def _build_index(self):
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=Settings.embed_model
        )

        nodes = splitter.get_nodes_from_documents(self.documents)

        max_chunk_chars = 1200 if self.mode == "corporate" else 800
        for node in nodes:
            if len(node.text) > max_chunk_chars:
                node.text = node.text[:max_chunk_chars]

        self.index = VectorStoreIndex(nodes)

        top_k = 15 if self.mode == "corporate" else 5
        similarity_cutoff = 0.15

        self.vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]
        )

        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=top_k
        )

    # =============================================================================
    # SUMMARY (UNCHANGED)
    # =============================================================================
    def generate_summary(self, groq_api_key: str) -> str:
        llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.0, max_tokens=300)

        if self.mode == "corporate":
            skip_patterns = ["table of contents", "objectives", "front matter", "copyright", "disclaimer"]
        else:
            skip_patterns = [
                "learning outcomes", "understand the", "identify the", "describe the",
                "unit objectives", "block introduction", "after studying this unit",
                "check your progress", "reflection and action", "terminal questions",
                "further reading", "suggested readings", "key words", "glossary",
                "contents", "introduction", "conclusion", "course coordinator",
            ]

        filtered_docs = []
        for doc in self.documents:
            text_lower = doc.text.lower()
            if not any(p in text_lower for p in skip_patterns):
                filtered_docs.append(doc)

        if not filtered_docs:
            return "Summary unavailable due to document structure (mostly front-matter)."

        max_context_chars = 8000
        context = ""
        for d in filtered_docs:
            chunk = d.text[:1000]
            if len(context) + len(chunk) > max_context_chars:
                break
            context += "\n\n" + chunk

        prompt = f"""
Provide a concise summary of the document in about 120 words.
Focus on core content only, in one natural paragraph.
Ignore tables of contents, objectives, front matter, glossaries, exercises.

TEXT:
{context}

SUMMARY (~120 words):
"""
        response = llm.complete(prompt)
        return str(response).strip()

    # =============================================================================
    # QUESTION ANSWERING (CORPORATE MODE FIXED)
    # =============================================================================
    def ask_question(self, question: str, groq_api_key: str) -> str:
        if not self.index:
            return "Index not initialized."

        llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.0, max_tokens=180)

        vector_query = f"Represent this question for retrieving relevant passages: {question}"
        vector_nodes = self.vector_retriever.retrieve(vector_query)
        bm25_nodes = self.bm25_retriever.retrieve(question)

        all_nodes = {}
        for node in vector_nodes + bm25_nodes:
            node_id = node.node_id
            if node_id not in all_nodes or node.score > all_nodes[node_id].score:
                all_nodes[node_id] = node

        retrieved = list(all_nodes.values())
        retrieved.sort(key=lambda n: (0 if n in vector_nodes else 1, -n.score))

        if not retrieved:
            return "Not covered in the documents."

        # ===== HEADING-AWARE FILTERING =====
        # Only apply if chunks appear to come from different headings
        if self.mode == "corporate" and len(retrieved) > 1:
            # Simple heuristic: look for common heading patterns
            def guess_heading(text):
                lines = text.strip().split('\n')
                for line in lines[:3]:  # Check first few lines
                    line = line.strip()
                    if line and (line.isupper() or 
                                line.endswith(':') or
                                re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', line) or
                                len(line) < 100 and not line.endswith('.')):
                        return line[:50]  # Return heading candidate
                return None
            
            headings = []
            for n in retrieved:
                h = guess_heading(n.node.text)
                headings.append(h)
            
            # If we detect multiple distinct headings, keep only chunks from first heading
            if len(set(headings)) > 1:
                first_heading = headings[0] if headings else None
                if first_heading:
                    filtered = []
                    for i, n in enumerate(retrieved):
                        h = headings[i] if i < len(headings) else None
                        if h == first_heading or h is None:
                            filtered.append(n)
                    if filtered:  # Only replace if we have something left
                        retrieved = filtered
        # ===== END HEADING-AWARE FILTERING =====

        context_parts = []
        total_chars = 0
        for n in retrieved:
            txt = n.node.text.strip()
            if total_chars + len(txt) > 5500:
                break
            context_parts.append(txt)
            total_chars += len(txt)

        context = "\n\n".join(context_parts)

        # ---------------- CORPORATE MODE PROMPT ----------------
        if self.mode == "corporate":
            prompt = f"""
{CORPORATE_REASONING_CONTRACT}

CONTEXT:
{context}

QUESTION: {question}

Write a concise natural answer to the question.
Do not organize into sections or bullet points unless asked.
Keep the answer concise but complete. Typically 1-4 sentences.
Avoid introductory phrases like "The document states".

ANSWER:
"""
        else:
            prompt = f"""
Answer in natural paragraphs with proper sentence structure, using ONLY the provided context.
Academic style: clear, explanatory, no fluff.
If not directly covered, say exactly: Not covered in the documents.
Do NOT add external knowledge.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""

        response = llm.complete(prompt)
        answer = str(response).strip()
        return answer


# =============================================================================
# DOCUMENT LOADER
# =============================================================================
def load_documents(file_paths: List[str]) -> List[Document]:
    documents = []
    for path in file_paths:
        if path.lower().endswith('.pdf'):
            reader = PyMuPDFReader()
            docs = reader.load(file_path=path)
            documents.extend(docs)
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            documents.append(Document(text=text))

    return documents
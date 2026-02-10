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
from typing import List

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
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

        self.index = VectorStoreIndex(nodes)

        top_k = 15 if self.mode == "corporate" else 5
        similarity_cutoff = 0.15

        self.vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]
        )

        self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
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
            if not any(p in doc.text.lower() for p in skip_patterns):
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
        return str(llm.complete(prompt)).strip()

    # -------------------------------------------------------------------------
    # QUESTION ANSWERING
    # -------------------------------------------------------------------------
    def ask_question(self, question: str, groq_api_key: str) -> str:
        if not self.index:
            return "Index not initialized."

        llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.0, max_tokens=256)

        vector_nodes = self.vector_retriever.retrieve(question)

        # If question is definitional/category style → rely more on keyword search
        broad_query = len(question.split()) <= 6 or question.lower().startswith(("what are", "list", "name"))
        bm25_k = 12 if broad_query else 5
        bm25_nodes = self.bm25_retriever.retrieve(question)[:bm25_k]


        all_nodes = {}
        for node in vector_nodes + bm25_nodes:
            if node.node_id not in all_nodes or node.score > all_nodes[node.node_id].score:
                all_nodes[node.node_id] = node

        retrieved = sorted(all_nodes.values(), key=lambda n: n.score, reverse=True)
        if not retrieved:
            return "Not covered in the documents."

        context_parts = []
        total_chars = 0
        for n in retrieved:
            txt = n.node.text.strip()
            if total_chars + len(txt) > 5500:
                break
            context_parts.append(txt)
            total_chars += len(txt)

        context = "\n\n---\n\n".join(context_parts)

        # ---------------- GROUNDING RULES ----------------
        GROUNDING_RULES = """
You are a document-grounded reader, not a general assistant.

Rules:
1) Base every statement on the context, but wording may differ.
2) You may combine statements only to complete a single fact, process, or outcome described by the document.
    Do not combine separate statements to invent a broader category or role.
3) If the document describes a process or consequence, you may state it directly as the answer.
4) Do not introduce new facts, examples, reasons, or roles not stated in the context.
5) Do not guess missing details.

Aggregation rule:
Only present a combined list if the document explicitly groups the items under a shared heading, list, or category. 
Otherwise answer only the directly stated fact instead of constructing a general category summary.

Each section separated by --- is independent context. 
Do not combine information across sections unless one section alone is incomplete but clearly part of the same statement.
   
Refusal:
Respond "Not covered in the documents." only when the answer cannot be determined from the context.   
"""

        if self.mode == "corporate":
            prompt = f"""
{GROUNDING_RULES}
Answer concisely and professionally using only the context.
Prefer wording close to the document.
Use bullets if useful.

CONTEXT:
{context}

QUESTION: {question}
ANSWER:
"""
        else:
            prompt = f"""
{GROUNDING_RULES}
Write clear natural paragraphs using only the context.
Do not infer beyond the text.

CONTEXT:
{context}

QUESTION: {question}
ANSWER:
"""

        answer = str(llm.complete(prompt)).strip()
        if not answer:
            return "Not covered in the documents."
        return answer

# =============================================================================
# DOCUMENT LOADER
# =============================================================================

def load_documents(file_paths: List[str]) -> List[Document]:
    documents = []
    for path in file_paths:
        if path.lower().endswith('.pdf'):
            reader = PyMuPDFReader()
            documents.extend(reader.load(file_path=path))
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                documents.append(Document(text=f.read()))
    return documents

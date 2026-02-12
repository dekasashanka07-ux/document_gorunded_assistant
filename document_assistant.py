# -*- coding: utf-8 -*-
"""
Generic Document Assistant – Streamlit-compatible
Multi-chunk, document-grounded QA (STRICT + LOW HALLUCINATION)

Compliance mode is specialized for clause-level verification.
Other modes unchanged.
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

    # ---------------------------------------------------------------------
    # INDEX BUILD
    # ---------------------------------------------------------------------
    def _build_index(self):

        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=Settings.embed_model
        )

        nodes = splitter.get_nodes_from_documents(self.documents)

        # Chunk size rules
        max_chunk_chars = 800 if self.mode == "academic" else 1200

        # --- SAFE TRIM (prevents broken numbered clauses) ---
        for node in nodes:
            if len(node.text) > max_chunk_chars:
                trimmed = node.text[:max_chunk_chars]
                if "." in trimmed:
                    trimmed = trimmed.rsplit(".", 1)[0] + "."
                node.text = trimmed

        self.index = VectorStoreIndex(nodes)

        top_k = 5 if self.mode == "academic" else 15

        self.vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.15)]
        )

        self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

    # ---------------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------------
    def generate_summary(self, groq_api_key: str) -> str:

        llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.0, max_tokens=300)

        if self.mode in ["corporate", "compliance"]:
            skip_patterns = ["table of contents", "objectives", "front matter", "copyright", "disclaimer"]
        else:
            skip_patterns = [
                "learning outcomes","understand the","identify the","describe the",
                "unit objectives","block introduction","after studying this unit",
                "check your progress","reflection and action","terminal questions",
                "further reading","suggested readings","key words","glossary",
                "contents","introduction","conclusion","course coordinator"
            ]

        filtered_docs = []
        for doc in self.documents:
            if not any(p in doc.text.lower() for p in skip_patterns):
                filtered_docs.append(doc)

        if not filtered_docs:
            return "Summary unavailable due to document structure (mostly front-matter)."

        context = ""
        for d in filtered_docs:
            chunk = d.text[:1000]
            if len(context) + len(chunk) > 8000:
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

    # ---------------------------------------------------------------------
    # QUESTION ANSWERING
    # ---------------------------------------------------------------------
    def ask_question(self, question: str, groq_api_key: str) -> str:

        if not self.index:
            return "Index not initialized."

        llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.0, max_tokens=256)

        vector_nodes = self.vector_retriever.retrieve(question)
        bm25_nodes = self.bm25_retriever.retrieve(question)

        # =========================
        # COMPLIANCE MODE
        # =========================
        if self.mode == "compliance":

            # semantic first to find meaning, bm25 to anchor wording
            retrieved = vector_nodes if vector_nodes else bm25_nodes
            if not retrieved:
                return "Not covered in the documents."

            # single clause only
            # take small local neighborhood instead of single fragment
            context_parts = []
            for n in retrieved[:3]:
                context_parts.append(n.node.text.strip())

            context = "\n\n".join(context_parts)

            prompt = f"""
You are answering questions about a legal, compliance, or policy document.

RULES:
- Answer ONLY if the context explicitly states the answer.
- Otherwise reply exactly: Not covered in the documents.
- Use the document wording.
- Do not infer or summarize.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""
            return str(llm.complete(prompt)).strip()

        # =========================
        # OTHER MODES (unchanged)
        # =========================
        all_nodes = {}
        for node in vector_nodes + bm25_nodes:
            node_id = node.node_id
            if node_id not in all_nodes or node.score > all_nodes[node_id].score:
                all_nodes[node_id] = node

        retrieved = list(all_nodes.values())
        retrieved.sort(key=lambda n: n.score, reverse=True)

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
        context = "\n\n".join(context_parts)

        if self.mode == "corporate":
            prompt = f"""
Answer using ONLY the provided context.
Up to 3 sentences. No intro phrases.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""
        else:  # academic
            prompt = f"""
Answer in natural paragraphs with proper sentence structure, using ONLY the provided context.
If not directly covered, say exactly: Not covered in the documents.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""

        return str(llm.complete(prompt)).strip()

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

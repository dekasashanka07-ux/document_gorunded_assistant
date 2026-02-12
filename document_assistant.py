# -*- coding: utf-8 -*-
"""
Generic Document Assistant – Streamlit-compatible
Multi-chunk, document-grounded QA
Compliance mode specialized for clause-level verification
"""

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

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


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

        if self.mode == "academic":
            max_chunk_chars = 800
        else:
            max_chunk_chars = 1200

        # --- SAFE TRIM (important for compliance numbering integrity) ---
        for node in nodes:
            if len(node.text) > max_chunk_chars:
                trimmed = node.text[:max_chunk_chars]
                if "." in trimmed:
                    trimmed = trimmed.rsplit(".", 1)[0] + "."
                node.text = trimmed

        self.index = VectorStoreIndex(nodes)

        if self.mode == "academic":
            top_k = 5
        else:
            top_k = 15

        similarity_cutoff = 0.15

        self.vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]
        )

        self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

    # ---------------------------------------------------------------------

    def ask_question(self, question: str, groq_api_key: str) -> str:

        if not self.index:
            return "Index not initialized."

        llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.0, max_tokens=256)

        vector_nodes = self.vector_retriever.retrieve(question)
        bm25_nodes = self.bm25_retriever.retrieve(question)

        # =========================
        # COMPLIANCE MODE RETRIEVAL
        # =========================
        if self.mode == "compliance":
            # Legal queries rely on keyword precision more than semantics
            retrieved = bm25_nodes if bm25_nodes else vector_nodes

            if not retrieved:
                return "Not covered in the documents."

            # Use ONLY the top clause to avoid cross-section synthesis
            context = retrieved[0].node.text.strip()

            prompt = f"""
You are answering questions about a legal or policy document.

RULES:
- Answer ONLY if the context explicitly states the answer.
- If not explicitly stated, reply exactly:
Not covered in the documents.
- Use wording from the document.
- Do not infer or summarize.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""

            response = llm.complete(prompt)
            return str(response).strip()

        # =========================
        # ORIGINAL MODES UNCHANGED
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
Answer in clear explanatory paragraphs using ONLY the context.
If not covered: Not covered in the documents.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""

        response = llm.complete(prompt)
        return str(response).strip()

# ---------------------------------------------------------------------

def load_documents(file_paths: List[str]) -> List[Document]:
    documents = []
    for path in file_paths:
        if path.lower().endswith('.pdf'):
            reader = PyMuPDFReader()
            docs = reader.load(file_path=path)
            documents.extend(docs)
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                documents.append(Document(text=f.read()))
    return documents

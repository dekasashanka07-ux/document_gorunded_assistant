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
# CORPORATE MODE CONTRACT
# =============================================================================
CORPORATE_REASONING_CONTRACT = """
You are answering questions about a document.

Every statement must be supported by the text.
Do not add outside knowledge.
If the answer is not present, respond exactly: Not covered in the documents.
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

        max_context_chars = 8000
        context = ""
        for d in self.documents:
            chunk = d.text[:1000]
            if len(context) + len(chunk) > max_context_chars:
                break
            context += "\n\n" + chunk

        prompt = f"""
Provide a concise summary of the document in about 120 words.
Focus on core content only, in one natural paragraph.

TEXT:
{context}

SUMMARY (~120 words):
"""
        response = llm.complete(prompt)
        return str(response).strip()

    # =============================================================================
    # QUESTION ANSWERING
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

        # -------- PASSAGE LABELLING (CRITICAL CHANGE) --------
        context_parts = []
        total_chars = 0
        for i, n in enumerate(retrieved, 1):
            txt = n.node.text.strip()
            if total_chars + len(txt) > 5500:
                break
            context_parts.append(f"[PASSAGE {i}]\n{txt}")
            total_chars += len(txt)

        context = "\n\n".join(context_parts)

        # -------- NEW PROMPT BEHAVIOR --------
        if self.mode == "corporate":
            prompt = f"""
{CORPORATE_REASONING_CONTRACT}

CONTEXT:
{context}

QUESTION: {question}

First determine which single passage best answers the question.
Then answer using only that passage.

ANSWER:
"""
        else:
            prompt = f"""
Answer in natural paragraphs with proper sentence structure, using ONLY the provided context.
If not directly covered, say exactly: Not covered in the documents.

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

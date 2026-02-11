# -*- coding: utf-8 -*-
"""
Generic Document Assistant – Stable Local-Region QA
Hybrid retrieval + semantic chunking
Prevents cross-section blending without over-refusal
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
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
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# =============================================================================
# ASSISTANT
# =============================================================================
class DocumentAssistant:
    def __init__(self, documents: List[Document], mode: str = "corporate"):
        self.mode = mode
        self.documents = documents
        self.index = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self._build_index()

    # -------------------------------------------------------------------------
    # INDEX
    # -------------------------------------------------------------------------
    def _build_index(self):
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=Settings.embed_model
        )

        nodes = splitter.get_nodes_from_documents(self.documents)

        # keep larger coherent chunks
        max_chunk_chars = 1200
        for node in nodes:
            if len(node.text) > max_chunk_chars:
                node.text = node.text[:max_chunk_chars]

        self.index = VectorStoreIndex(nodes)

        self.vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=15,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.15)]
        )

        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=15
        )

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    def generate_summary(self, groq_api_key: str) -> str:
        llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.0, max_tokens=300)

        context = "\n\n".join(d.text[:800] for d in self.documents[:8])

        prompt = f"""
Provide a concise 120 word summary of the document.

TEXT:
{context}

SUMMARY:
"""
        return str(llm.complete(prompt)).strip()

    # -------------------------------------------------------------------------
    # QUESTION ANSWERING (LOCAL REGION GROUPING)
    # -------------------------------------------------------------------------
    def ask_question(self, question: str, groq_api_key: str) -> str:
        if not self.index:
            return "Index not initialized."

        llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.0, max_tokens=180)

        # ---- Retrieve ----
        vector_query = f"Represent this question for retrieving relevant passages: {question}"
        vector_nodes = self.vector_retriever.retrieve(vector_query)
        bm25_nodes = self.bm25_retriever.retrieve(question)

        all_nodes = {}
        for node in vector_nodes + bm25_nodes:
            if node.node_id not in all_nodes or node.score > all_nodes[node.node_id].score:
                all_nodes[node.node_id] = node

        retrieved = list(all_nodes.values())
        if not retrieved:
            return "Not covered in the documents."

        # ---- Sort by document order ----
        retrieved.sort(key=lambda n: (
            n.node.metadata.get("page_label", 0),
            n.node.start_char_idx or 0
        ))

        # ---- Build local clusters ----
        clusters = []
        current = [retrieved[0]]

        for prev, cur in zip(retrieved, retrieved[1:]):
            prev_end = prev.node.end_char_idx or 0
            cur_start = cur.node.start_char_idx or 0

            # nearby chunks = same region
            if abs(cur_start - prev_end) < 1200:
                current.append(cur)
            else:
                clusters.append(current)
                current = [cur]

        clusters.append(current)

        # ---- Pick best region ----
        best_cluster = max(clusters, key=lambda c: sum(n.score for n in c))

        # ---- Build context ----
        context_parts = []
        total = 0
        for n in best_cluster:
            text = n.node.text.strip()
            if total + len(text) > 5500:
                break
            context_parts.append(text)
            total += len(text)

        context = "\n\n".join(context_parts)

        # ---- Simple grounded prompt ----
        prompt = f"""
Answer using only the provided context.
If the answer is not present, respond exactly: Not covered in the documents.

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
            docs = reader.load(file_path=path)
            documents.extend(docs)
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                documents.append(Document(text=f.read()))
    return documents

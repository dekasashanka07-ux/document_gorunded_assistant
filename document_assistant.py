# -*- coding: utf-8 -*-
"""
Document Assistant – Verified Grounded QA
Hybrid retrieval + local region selection + fail-closed answering
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
# SETTINGS
# =============================================================================
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# =============================================================================
# ASSISTANT
# =============================================================================
class DocumentAssistant:
    def __init__(self, documents: List[Document]):
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

        # prevent overly long fragments
        for node in nodes:
            if len(node.text) > 1200:
                node.text = node.text[:1200]

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
    # QA (VERIFIED ANSWERING)
    # -------------------------------------------------------------------------
    def ask_question(self, question: str, groq_api_key: str) -> str:
        llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.0, max_tokens=220)

        # ---- retrieval ----
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

        # ---- order by document position ----
        retrieved.sort(key=lambda n: (
            n.node.metadata.get("page_label", 0),
            n.node.start_char_idx or 0
        ))

        # ---- cluster into local regions ----
        clusters = []
        current = [retrieved[0]]

        for prev, cur in zip(retrieved, retrieved[1:]):
            prev_end = prev.node.end_char_idx or 0
            cur_start = cur.node.start_char_idx or 0

            if abs(cur_start - prev_end) < 1200:
                current.append(cur)
            else:
                clusters.append(current)
                current = [cur]

        clusters.append(current)

        # ---- best region ----
        best_cluster = max(clusters, key=lambda c: sum(n.score for n in c))

        context_parts = []
        total = 0
        for n in best_cluster:
            text = n.node.text.strip()
            if total + len(text) > 5500:
                break
            context_parts.append(text)
            total += len(text)

        context = "\n\n".join(context_parts)

        # ---- verified answering prompt ----
        prompt = f"""
You are verifying whether a question can be answered strictly from the provided document.

STEP 1:
Decide if the answer is explicitly present in the context.
Respond with exactly one token: ANSWERABLE or NOT_ANSWERABLE.

STEP 2:
If ANSWERABLE, provide the answer using only the text.
If NOT_ANSWERABLE, output exactly: Not covered in the documents.

Never use outside knowledge.

FORMAT:
VERDICT: <ANSWERABLE or NOT_ANSWERABLE>
ANSWER: <final answer or Not covered in the documents>

CONTEXT:
{context}

QUESTION: {question}
"""

        raw = str(llm.complete(prompt)).strip()

        # ---- parse output safely ----
        verdict_match = re.search(r"VERDICT:\s*(ANSWERABLE|NOT_ANSWERABLE)", raw)
        answer_match = re.search(r"ANSWER:\s*(.*)", raw, re.DOTALL)

        if not verdict_match:
            return "Not covered in the documents."

        verdict = verdict_match.group(1)

        if verdict == "NOT_ANSWERABLE":
            return "Not covered in the documents."

        if answer_match:
            return answer_match.group(1).strip()

        return "Not covered in the documents."


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

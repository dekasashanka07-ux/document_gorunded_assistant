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
        # Semantic chunking (meaningful splits based on embeddings)
        splitter = SemanticSplitterNodeParser(
            buffer_size=1, # 1 sentence buffer for context
            breakpoint_percentile_threshold=95, # Aggressive splits only when needed
            embed_model=Settings.embed_model
        )
        nodes = splitter.get_nodes_from_documents(self.documents)
        
        # Mode tweaks: larger chunks for corporate/compliance, smaller for academic
        if self.mode == "academic":
            max_chunk_chars = 800
        else:
            max_chunk_chars = 1200  # corporate & compliance need more context
        
        for node in nodes:
            if len(node.text) > max_chunk_chars:
                node.text = node.text[:max_chunk_chars]
        
        self.index = VectorStoreIndex(nodes)
        
        # ============= MODE-SPECIFIC RETRIEVAL PARAMETERS =============
        if self.mode == "academic":
            top_k = 5
            similarity_cutoff = 0.15
        elif self.mode == "compliance":
            top_k = 25  # Higher recall for scattered policy rules
            similarity_cutoff = 0.12  # More forgiving matching for policy wording
        else:  # corporate
            top_k = 15
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

    def generate_summary(self, groq_api_key: str) -> str:
        llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.0, max_tokens=300)
        
        # Mode-aware junk filtering
        if self.mode == "corporate" or self.mode == "compliance":
            # Lighter: Keep headings like "Principles", "Privacy Policy"
            skip_patterns = ["table of contents", "objectives", "front matter", "copyright", "disclaimer"]
        else:  # academic
            # Heavier for academic
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
        
        # Truncate context to avoid Groq token limit
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

    def _guess_heading(self, text: str) -> Optional[str]:
        """Simple heuristic to guess section heading from text."""
        lines = text.strip().split('\n')
        for line in lines[:3]:
            line = line.strip()
            if line and (line.isupper() or 
                        line.endswith(':') or
                        re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', line) or
                        (len(line) < 100 and not line.endswith('.'))):
                return line[:50]
        return None

    def ask_question(self, question: str, groq_api_key: str) -> str:
        if not self.index:
            return "Index not initialized."
        
        llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.0, max_tokens=256)
        
        # Hybrid retrieval: Combine vector + BM25, simple RRF-style dedup & rank
        vector_nodes = self.vector_retriever.retrieve(question)
        bm25_nodes = self.bm25_retriever.retrieve(question)
        
        # Combine & dedup (prefer higher score / vector first)
        all_nodes = {}
        for node in vector_nodes + bm25_nodes:
            node_id = node.node_id
            if node_id not in all_nodes or node.score > all_nodes[node_id].score:
                all_nodes[node_id] = node
        
        retrieved = list(all_nodes.values())
        retrieved.sort(key=lambda n: n.score, reverse=True)
        
        if not retrieved:
            return "Not covered in the documents."
        
        # ============= COMPLIANCE MODE: SECTION FILTERING =============
        if self.mode == "compliance" and len(retrieved) > 1:
            # Try to keep chunks from the same major section
            primary_heading = self._guess_heading(retrieved[0].node.text)
            if primary_heading:
                filtered = []
                for n in retrieved:
                    h = self._guess_heading(n.node.text)
                    if h == primary_heading or h is None:
                        filtered.append(n)
                if filtered:  # Only replace if we have something left
                    retrieved = filtered
        # ==============================================================
        
        # Format context
        context_parts = []
        total_chars = 0
        for n in retrieved:
            txt = n.node.text.strip()
            if total_chars + len(txt) > 5500:
                break
            context_parts.append(txt)
            total_chars += len(txt)
        
        context = "\n\n".join(context_parts)
        
        # ============= MODE-SPECIFIC PROMPTS =============
        
        if self.mode == "corporate":
            prompt = f"""
Answer using ONLY the provided context.
If the context clearly lists items (such as phases, steps, traits, or components),
reproduce those items as a numbered or bulleted list, preserving the same items and count.
Otherwise, answer in up to 3 sentences with no bullet points, no lists, no markdown.
No introductory phrases.
Just state the information directly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""
        
        elif self.mode == "compliance":
            prompt = f"""
You are answering questions about a legal, compliance, or policy document.

STRICT RULES:
- Answer using ONLY the provided context.
- Do NOT combine information from different sections, headings, or topics.
- If the answer is stated as a list, reproduce it VERBATIM as a numbered or bulleted list with one item per line.
- If the answer is a prohibition, restriction, obligation, or exception, use the EXACT language from the document.
- Do NOT paraphrase rules, policies, or compliance requirements.
- Do NOT add interpretations, examples, or "in other words".
- Do NOT use introductory phrases.
- Start your answer with a capital letter.
- If the exact answer is not in the provided context, say exactly:
  "Not covered in the documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""
        
        else:  # academic
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
        
        # Academic mode only: sentence cap + complete sentences + paragraph breaks
        if self.mode == "academic":
            cap = self._academic_sentence_cap(question)
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            if len(sentences) > cap:
                answer = " ".join(sentences[:cap]).strip()
            # Enforce complete ending (trim incomplete last sentence)
            m = re.search(r'[.!?](?!.*[.!?])', answer, re.S)
            if m:
                answer = answer[:m.end()].strip()
            if not answer.endswith(('.', '!', '?')):
                answer += "."
            # Break medium/long answers into paragraphs
            words = answer.split()
            if len(words) > 100:
                sentences = re.split(r'(?<=[.!?])\s+', answer)
                if len(sentences) > 3:
                    para_count = 3 if len(words) > 150 else 2
                    para_len = len(sentences) // para_count
                    paras = []
                    start = 0
                    for i in range(para_count):
                        end = start + para_len if i < para_count - 1 else len(sentences)
                        para_sentences = sentences[start:end]
                        para_text = " ".join(para_sentences).strip()
                        if para_text:
                            paras.append(para_text)
                        start = end
                    answer = "\n\n".join(paras)
        
        return answer

    def _academic_sentence_cap(self, question: str) -> int:
        q = question.lower().strip()
        if q.startswith(("what is", "define", "meaning of")):
            return 3
        if q.startswith(("explain", "discuss", "comment", "examine", "assess", "evaluate", "analyze", "critically assess", "critique", "review", "elaborate")):
            return 9
        return 5

# =============================================================================
# DOCUMENT LOADER (with PyMuPDF for clean PDF text)
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
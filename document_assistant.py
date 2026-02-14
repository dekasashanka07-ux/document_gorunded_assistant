# -*- coding: utf-8 -*-
"""
Enhanced Generic Document Assistant – Production Ready V2
Multi-chunk, document-grounded QA with STRICT hallucination prevention
Hybrid retrieval + reranking + semantic chunking + source attribution
NEW: Dynamic sentence limits, confidence scoring, progress tracking
"""
# =============================================================================
# IMPORTS
# =============================================================================
import os
import re
from typing import List, Optional, Dict, Tuple, Callable
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.readers.file import PyMuPDFReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    SentenceTransformerRerank
)

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./embeddings_cache"
)

# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class AnswerResult:
    """Structured answer with metadata"""
    answer: str
    sources: List[str]
    confidence: str
    retrieved_chunks: int
    sentence_limit: int


# =============================================================================
# DOCUMENT ASSISTANT CLASS
# =============================================================================
class DocumentAssistant:
    """
    Enhanced document-grounded QA assistant
    """
    
    def __init__(
        self, 
        documents: List[Document], 
        mode: str = "corporate",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        """Initialize assistant with documents"""
        self.mode = mode
        self.documents = documents
        self.index = None
        self.nodes = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.reranker = None
        self._build_index(progress_callback)
        
    def _build_index(self, progress_callback: Optional[Callable] = None):
        """Build search index"""
        total_steps = 5
        
        if progress_callback:
            progress_callback(1, total_steps, "Configuring chunking parameters...")
        
        chunk_config = self._get_chunk_config()
        
        if progress_callback:
            progress_callback(2, total_steps, "Creating semantic chunks...")
        
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=Settings.embed_model
        )
        
        nodes = splitter.get_nodes_from_documents(self.documents)
        self.nodes = self._smart_truncate_nodes(nodes, chunk_config['max_chars'])
        
        if progress_callback:
            progress_callback(3, total_steps, f"Building vector index ({len(self.nodes)} chunks)...")
        
        self.index = VectorStoreIndex(self.nodes)
        
        if progress_callback:
            progress_callback(4, total_steps, "Initializing hybrid retrievers...")
        
        top_k = chunk_config['top_k']
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
            progress_callback(5, total_steps, "Loading reranking model...")
        
        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2",
            top_n=8 if self.mode == "corporate" else 6
        )
    
    def _get_chunk_config(self) -> Dict:
        """Get chunking configuration"""
        if self.mode == "corporate":
            return {'max_chars': 1200, 'top_k': 15, 'similarity_cutoff': 0.15}
        else:
            return {'max_chars': 900, 'top_k': 8, 'similarity_cutoff': 0.18}
    
    def _smart_truncate_nodes(self, nodes: List, max_chars: int) -> List:
        """Truncate nodes at sentence boundaries"""
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
    
    def generate_summary(
        self, 
        groq_api_key: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> str:
        """Generate document summary"""
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
            
            word_count = len(summary.split())
            if word_count < 30:
                return "Summary unavailable: Insufficient substantive content."
            
            return summary
            
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def _filter_summary_docs(self) -> List[Document]:
        """Filter out non-content sections"""
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
        """Build summary context"""
        context = ""
        for doc in docs[:10]:
            chunk = doc.text[:1000].strip()
            if len(context) + len(chunk) > max_chars:
                break
            context += "\n\n" + chunk
        return context.strip()
    
    def _get_summary_prompt(self, context: str) -> str:
        """Get summary prompt"""
        base = """Create a concise summary (100-150 words).
Focus on main themes. No bullet points. Natural language.
Do NOT start with "This document discusses"."""
        
        specific = "\nBusiness focus." if self.mode == "corporate" else "\nAcademic focus."
        
        return f"{base}{specific}\n\nDOCUMENT:\n{context}\n\nSUMMARY:"
    
    def _get_corporate_sentence_limit(self, question: str) -> int:
        """Determine sentence limit for corporate mode"""
        q = question.lower().strip()
        
        if any(q.startswith(p) for p in ["what is", "define", "who is"]):
            return 3
        
        if any(k in q for k in ["what are", "list", "types", "kinds", "categories", "styles", "phases", "stages", "steps"]):
            return 5
        
        if any(q.startswith(p) for p in ["how does", "how do", "why", "explain", "describe"]):
            return 5
        
        if any(k in q for k in ["compare", "contrast", "difference", "vs", "versus", "similar"]):
            return 6
        
        return 4
    
    def ask_question(self, question: str, groq_api_key: str, return_metadata: bool = False):
        """Answer question with document grounding"""
        if not question or len(question.strip()) < 3:
            error = "Please ask a more specific question."
            if return_metadata:
                return AnswerResult(error, [], "low", 0, 0)
            return error
        
        if not self.index:
            error = "Error: Index not initialized."
            if return_metadata:
                return AnswerResult(error, [], "low", 0, 0)
            return error
        
        try:
            sentence_limit = 0
            if self.mode == "corporate":
                sentence_limit = self._get_corporate_sentence_limit(question)
            
            expanded_queries = self._expand_query(question)
            retrieved_nodes = self._hybrid_retrieve(expanded_queries)
            
            if not retrieved_nodes:
                result = AnswerResult(
                    answer="This information is not covered in the provided documents.",
                    sources=[],
                    confidence="low",
                    retrieved_chunks=0,
                    sentence_limit=sentence_limit
                )
                return result if return_metadata else result.answer
            
            is_list_question = self._is_list_question(question)
            if is_list_question and len(retrieved_nodes) < 5:
                additional_nodes = self._get_additional_context(question, retrieved_nodes)
                if additional_nodes:
                    retrieved_nodes.extend(additional_nodes)
            
            context, sources = self._build_context_with_sources(retrieved_nodes, question)
            
            llm = Groq(
                model="llama-3.1-8b-instant",
                api_key=groq_api_key,
                temperature=0.0,
                max_tokens=300
            )
            
            prompt = self._get_answer_prompt(context, question, sentence_limit)
            response = llm.complete(prompt)
            answer = str(response).strip()
            
            if not answer or len(answer.strip()) < 10:
                error = "Unable to generate answer. Please rephrase."
                if return_metadata:
                    return AnswerResult(error, [], "low", len(retrieved_nodes), sentence_limit)
                return error
            
            answer = self._post_process_answer(answer, question)
            confidence = self._assess_confidence(retrieved_nodes, answer)
            is_negative_response = self._is_negative_response(answer)
            
            if sources and not is_negative_response:
                answer += f"\n\n*📚 Sources: {', '.join(sorted(sources))}*"
            
            result = AnswerResult(
                answer=answer,
                sources=list(sources) if not is_negative_response else [],
                confidence=confidence,
                retrieved_chunks=len(retrieved_nodes),
                sentence_limit=sentence_limit
            )
            
            return result if return_metadata else result.answer
            
        except Exception as e:
            error = "Error processing question. Please try rephrasing."
            if return_metadata:
                return AnswerResult(error, [], "low", 0, 0)
            return error
    
    def _is_list_question(self, question: str) -> bool:
        """Detect list questions"""
        q = question.lower()
        indicators = ["what are", "list", "types of", "kinds of", "styles of", "phases", "stages", "steps"]
        return any(i in q for i in indicators)
    
    def _get_additional_context(self, question: str, existing_nodes: List) -> List:
        """Get additional context for incomplete answers"""
        existing_ids = {node.node_id for node in existing_nodes}
        broader_query = question.replace("what are", "about").replace("list", "information")
        additional = self.vector_retriever.retrieve(broader_query)
        new_nodes = [n for n in additional if n.node_id not in existing_ids][:3]
        return new_nodes
    
    def _is_negative_response(self, answer: str) -> bool:
        """Detect 'not covered' responses"""
        negatives = [
            "not covered in the documents",
            "not addressed in the documents",
            "this information is not available",
            "not found in the documents",
            "doesn't provide information",
            "this topic is not covered"
        ]
        return any(p in answer.lower() for p in negatives)
    
    def _expand_query(self, question: str) -> List[str]:
        """Expand query with variations"""
        queries = [question]
        q = question.lower().strip()
        
        if q.startswith("what is "):
            term = question[8:].strip()
            queries.append(f"define {term}")
        
        if q.startswith("how to "):
            queries.append(question.replace("how to", "steps for"))
        
        return queries[:3]
    
    def _hybrid_retrieve(self, queries: List[str]) -> List:
        """Hybrid retrieval with reranking"""
        all_nodes = {}
        
        for query in queries:
            vector_nodes = self.vector_retriever.retrieve(query)
            bm25_nodes = self.bm25_retriever.retrieve(query)
            
            for node in vector_nodes + bm25_nodes:
                node_id = node.node_id
                if node_id not in all_nodes:
                    all_nodes[node_id] = node
                elif node.score > all_nodes[node_id].score:
                    all_nodes[node_id] = node
        
        if not all_nodes:
            return []
        
        nodes_list = list(all_nodes.values())
        reranked = self.reranker.postprocess_nodes(nodes_list, query_str=queries[0])
        return reranked
    
    def _build_context_with_sources(self, nodes: List, question: str) -> Tuple[str, set]:
        """Build context and extract sources"""
        question_words = len(question.split())
        base_chars = 3000 if question_words < 10 else 5500
        max_context_chars = int(base_chars * 1.5) if self.mode == "academic" else base_chars
        
        context_parts = []
        sources = set()
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
                page = meta.get("page", "")
                
                if filename:
                    source_str = filename
                    if page:
                        source_str += f" (p.{page})"
                    sources.add(source_str)
        
        context = "\n\n---\n\n".join(context_parts)
        return context, sources
    
    def _get_answer_prompt(self, context: str, question: str, sentence_limit: int = 0) -> str:
        """Get answer prompt"""
        if self.mode == "corporate":
            return f"""You are a precise document assistant. Answer ONLY using the context.

RULES:
1. Maximum {sentence_limit} sentences
2. No bullet points - use prose
3. For lists: Cover ALL items briefly
4. If partial: "The document mentions [X] but doesn't provide details on [Y]."
5. If not found: "This information is not covered in the documents."
6. NO external knowledge
7. Be direct - no "According to..."

CONTEXT:
{context}

QUESTION: {question}

ANSWER (max {sentence_limit} sentences):"""
        
        else:
            cap = self._academic_sentence_cap(question)
            return f"""Academic assistant. Detailed answer using ONLY the context.

REQUIREMENTS:
- 2-3 paragraphs for complex topics
- Max {cap} sentences
- If incomplete: "The document covers [X] but not [Y]."
- If not covered: "This topic is not covered."
- NO external knowledge

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    def _academic_sentence_cap(self, question: str) -> int:
        """Sentence limit for academic mode"""
        q = question.lower()
        
        if q.startswith(("what is", "define")):
            return 3
        
        if any(q.startswith(k) for k in ["explain", "discuss", "analyze", "evaluate"]):
            return 10
        
        if any(w in q for w in ["compare", "contrast", "difference"]):
            return 7
        
        return 5
    
    def _post_process_answer(self, answer: str, question: str) -> str:
        """Post-process answer"""
        if self.mode == "corporate":
            sentence_limit = self._get_corporate_sentence_limit(question)
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            
            if len(sentences) > sentence_limit:
                answer = " ".join(sentences[:sentence_limit]).strip()
            
            if not answer.endswith(('.', '!', '?')):
                match = re.search(r'[.!?](?!.*[.!?])', answer, re.S)
                if match:
                    answer = answer[:match.end()].strip()
                else:
                    answer += "."
        
        elif self.mode == "academic":
            cap = self._academic_sentence_cap(question)
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            
            if len(sentences) > cap:
                answer = " ".join(sentences[:cap]).strip()
            
            if not answer.endswith(('.', '!', '?')):
                match = re.search(r'[.!?](?!.*[.!?])', answer, re.S)
                if match:
                    answer = answer[:match.end()].strip()
                else:
                    answer += "."
            
            words = answer.split()
            if len(words) > 100:
                sentences = re.split(r'(?<=[.!?])\s+', answer)
                if len(sentences) > 4:
                    para_count = 3 if len(words) > 180 else 2
                    para_size = len(sentences) // para_count
                    
                    paragraphs = []
                    for i in range(para_count):
                        start_idx = i * para_size
                        end_idx = start_idx + para_size if i < para_count - 1 else len(sentences)
                        para = " ".join(sentences[start_idx:end_idx]).strip()
                        if para:
                            paragraphs.append(para)
                    
                    answer = "\n\n".join(paragraphs)
        
        return answer.strip()
    
    def _assess_confidence(self, nodes: List, answer: str) -> str:
        """Assess confidence"""
        if not nodes:
            return "low"
        
        top_scores = [n.score for n in nodes[:3] if hasattr(n, 'score')]
        if not top_scores:
            return "medium"
        
        avg_score = sum(top_scores) / len(top_scores)
        
        uncertainty = ["not covered", "doesn't provide", "not mentioned", "not addressed"]
        has_uncertainty = any(p in answer.lower() for p in uncertainty)
        
        chunk_count = len(nodes)
        word_count = len(answer.split())
        
        if avg_score > 0.75 and chunk_count >= 3 and not has_uncertainty and word_count > 30:
            return "high"
        elif avg_score > 0.55 and chunk_count >= 2 and word_count > 20:
            return "medium"
        else:
            return "low"


# =============================================================================
# DOCUMENT LOADER
# =============================================================================
def load_documents(
    file_paths: List[str],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> List[Document]:
    """Load documents with metadata and progress tracking"""
    documents = []
    total_files = len(file_paths)
    
    for idx, path in enumerate(file_paths, 1):
        if progress_callback:
            progress_callback(idx, total_files, f"Loading {os.path.basename(path)}...")
        
        filename = os.path.basename(path)
        
        try:
            if path.lower().endswith('.pdf'):
                reader = PyMuPDFReader()
                docs = reader.load(file_path=path)
                
                for i, doc in enumerate(docs):
                    doc.metadata = {
                        "filename": filename,
                        "page": i + 1,
                        "source": path,
                        "file_type": "pdf"
                    }
                    documents.append(doc)
            
            elif path.lower().endswith('.txt'):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                doc = Document(
                    text=text,
                    metadata={
                        "filename": filename,
                        "source": path,
                        "file_type": "txt"
                    }
                )
                documents.append(doc)
            
            elif path.lower().endswith('.docx'):
                with open(path, 'rb') as f:
                    text = f.read().decode('utf-8', errors='ignore')
                
                doc = Document(
                    text=text,
                    metadata={
                        "filename": filename,
                        "source": path,
                        "file_type": "docx"
                    }
                )
                documents.append(doc)
        
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            continue
    
    return documents

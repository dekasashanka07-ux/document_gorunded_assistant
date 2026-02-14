# -*- coding: utf-8 -*-
"""
Enhanced Generic Document Assistant – Production Ready
Multi-chunk, document-grounded QA with STRICT hallucination prevention
Hybrid retrieval + reranking + semantic chunking + source attribution
"""
# =============================================================================
# IMPORTS
# =============================================================================
import os
import re
from typing import List, Optional, Dict, Tuple
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
    confidence: str  # "high", "medium", "low"
    retrieved_chunks: int


# =============================================================================
# DOCUMENT ASSISTANT CLASS
# =============================================================================
class DocumentAssistant:
    """
    Enhanced document-grounded QA assistant with:
    - Hybrid retrieval (Vector + BM25)
    - Cross-encoder reranking
    - Source attribution
    - Intelligent chunking
    - Mode-aware responses
    """
    
    def __init__(self, documents: List[Document], mode: str = "corporate"):
        """
        Initialize assistant with documents
        
        Args:
            documents: List of LlamaIndex Document objects with metadata
            mode: "corporate" (crisp) or "academic" (detailed)
        """
        self.mode = mode
        self.documents = documents
        self.index = None
        self.nodes = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.reranker = None
        self._build_index()
        
    def _build_index(self):
        """Build search index with semantic chunking and hybrid retrievers"""
        # Configure chunking parameters based on mode
        chunk_config = self._get_chunk_config()
        
        # Semantic chunking with sentence boundary respect
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=Settings.embed_model
        )
        
        nodes = splitter.get_nodes_from_documents(self.documents)
        
        # Smart truncation: preserve sentence boundaries
        self.nodes = self._smart_truncate_nodes(nodes, chunk_config['max_chars'])
        
        # Build vector index
        self.index = VectorStoreIndex(self.nodes)
        
        # Initialize retrievers
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
        
        # Initialize reranker for hybrid fusion
        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2",
            top_n=8 if self.mode == "corporate" else 6
        )
    
    def _get_chunk_config(self) -> Dict:
        """Get chunking configuration based on mode"""
        if self.mode == "corporate":
            return {
                'max_chars': 1200,
                'top_k': 15,
                'similarity_cutoff': 0.15
            }
        else:  # academic
            return {
                'max_chars': 900,
                'top_k': 8,
                'similarity_cutoff': 0.18
            }
    
    def _smart_truncate_nodes(self, nodes: List, max_chars: int) -> List:
        """
        Truncate nodes while preserving sentence boundaries
        
        Args:
            nodes: List of parsed nodes
            max_chars: Maximum characters per chunk
            
        Returns:
            List of truncated nodes
        """
        processed_nodes = []
        
        for node in nodes:
            if len(node.text) <= max_chars:
                processed_nodes.append(node)
                continue
            
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', node.text)
            truncated = ""
            
            for sent in sentences:
                if len(truncated) + len(sent) + 1 <= max_chars:
                    truncated += sent + " "
                else:
                    break
            
            # Only add if we have meaningful content
            if len(truncated.strip()) > 50:
                node.text = truncated.strip()
                processed_nodes.append(node)
        
        return processed_nodes
    
    def generate_summary(self, groq_api_key: str) -> str:
        """
        Generate intelligent document summary
        
        Args:
            groq_api_key: Groq API key
            
        Returns:
            Concise summary (100-150 words)
        """
        try:
            llm = Groq(
                model="llama-3.1-8b-instant",
                api_key=groq_api_key,
                temperature=0.2,
                max_tokens=400
            )
            
            # Filter out junk content
            filtered_docs = self._filter_summary_docs()
            
            if not filtered_docs:
                return "Summary unavailable: Document contains primarily structural content (TOC, exercises, etc.)."
            
            # Build context with smart truncation
            context = self._build_summary_context(filtered_docs, max_chars=8000)
            
            # Mode-aware summary prompt
            prompt = self._get_summary_prompt(context)
            
            response = llm.complete(prompt)
            summary = str(response).strip()
            
            # Validate quality
            word_count = len(summary.split())
            if word_count < 30:
                return "Summary unavailable: Insufficient substantive content in document."
            
            return summary
            
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def _filter_summary_docs(self) -> List[Document]:
        """Filter out non-content sections"""
        # Mode-specific skip patterns
        if self.mode == "corporate":
            skip_patterns = [
                "table of contents", "objectives", "front matter",
                "copyright", "disclaimer", "revision history"
            ]
        else:  # academic
            skip_patterns = [
                "learning outcomes", "understand the", "identify the",
                "describe the", "unit objectives", "block introduction",
                "after studying this unit", "check your progress",
                "reflection and action", "terminal questions",
                "further reading", "suggested readings", "key words",
                "glossary", "contents", "course coordinator"
            ]
        
        filtered = []
        for doc in self.documents:
            text_lower = doc.text.lower()
            # Skip if it's mostly junk
            junk_matches = sum(1 for p in skip_patterns if p in text_lower)
            if junk_matches < 2:  # Allow some structural content
                filtered.append(doc)
        
        return filtered
    
    def _build_summary_context(self, docs: List[Document], max_chars: int) -> str:
        """Build summary context with smart sampling"""
        context = ""
        docs_used = 0
        max_docs = 10  # Don't use too many docs
        
        for doc in docs[:max_docs]:
            # Take meaningful chunk from each doc
            chunk = doc.text[:1000].strip()
            
            if len(context) + len(chunk) > max_chars:
                break
            
            context += "\n\n" + chunk
            docs_used += 1
        
        return context.strip()
    
    def _get_summary_prompt(self, context: str) -> str:
        """Get mode-appropriate summary prompt"""
        base_instructions = """Create a concise, informative summary.

REQUIREMENTS:
- 100-150 words
- One cohesive paragraph (no bullet points)
- Focus on main themes and key information
- Ignore: tables of contents, objectives, exercises, references
- Use natural, flowing language
- Do NOT include phrases like "This document discusses" - just state the content directly
"""
        
        if self.mode == "corporate":
            specific = "\n- Business/practical focus\n- Highlight actionable information\n"
        else:
            specific = "\n- Academic/conceptual focus\n- Highlight key theories and frameworks\n"
        
        return f"""{base_instructions}{specific}
DOCUMENT TEXT:
{context}

SUMMARY:"""
    
    def ask_question(
        self, 
        question: str, 
        groq_api_key: str,
        return_metadata: bool = False
    ) -> str:
        """
        Answer question with strict document grounding
        
        Args:
            question: User question
            groq_api_key: Groq API key
            return_metadata: If True, return AnswerResult object
            
        Returns:
            Answer string (or AnswerResult if return_metadata=True)
        """
        # Validate inputs
        if not question or len(question.strip()) < 3:
            return "Please ask a more specific question (at least 3 characters)."
        
        if not self.index:
            return "Error: Index not initialized. Please reinitialize the assistant."
        
        try:
            # Query expansion for better retrieval
            expanded_queries = self._expand_query(question)
            
            # Hybrid retrieval with reranking
            retrieved_nodes = self._hybrid_retrieve(expanded_queries)
            
            if not retrieved_nodes:
                return "This information is not covered in the provided documents."
            
            # Build context and extract sources
            context, sources = self._build_context_with_sources(
                retrieved_nodes,
                question
            )
            
            # Generate answer
            llm = Groq(
                model="llama-3.1-8b-instant",
                api_key=groq_api_key,
                temperature=0.0,
                max_tokens=300
            )
            
            prompt = self._get_answer_prompt(context, question)
            response = llm.complete(prompt)
            answer = str(response).strip()
            
            # Validate answer
            if not answer or len(answer.strip()) < 10:
                return "Unable to generate a proper answer from the documents. Please rephrase your question."
            
            # Post-process answer
            answer = self._post_process_answer(answer, question)
            
            # Add source attribution
            if sources:
                answer += f"\n\n*📚 Sources: {', '.join(sorted(sources))}*"
            
            # Determine confidence
            confidence = self._assess_confidence(retrieved_nodes, answer)
            
            if return_metadata:
                return AnswerResult(
                    answer=answer,
                    sources=list(sources),
                    confidence=confidence,
                    retrieved_chunks=len(retrieved_nodes)
                )
            
            return answer
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)  # Log for debugging
            return "An error occurred while processing your question. Please try rephrasing or contact support."
    
    def _expand_query(self, question: str) -> List[str]:
        """Expand query with variations for better retrieval"""
        queries = [question]
        q_lower = question.lower().strip()
        
        # Add common variations
        if q_lower.startswith("what is "):
            term = question[8:].strip()
            queries.append(f"define {term}")
            queries.append(f"{term} means")
        
        if q_lower.startswith("how to "):
            queries.append(question.replace("how to", "steps for"))
            queries.append(question.replace("how to", "process of"))
        
        if q_lower.startswith("why "):
            queries.append(question.replace("why", "reason for"))
        
        return queries[:3]  # Limit to avoid too many retrievals
    
    def _hybrid_retrieve(self, queries: List[str]) -> List:
        """
        Perform hybrid retrieval with reranking
        
        Args:
            queries: List of query variations
            
        Returns:
            Reranked nodes
        """
        all_nodes = {}
        
        for query in queries:
            # Vector retrieval
            vector_nodes = self.vector_retriever.retrieve(query)
            
            # BM25 retrieval
            bm25_nodes = self.bm25_retriever.retrieve(query)
            
            # Combine and deduplicate
            for node in vector_nodes + bm25_nodes:
                node_id = node.node_id
                if node_id not in all_nodes:
                    all_nodes[node_id] = node
                elif node.score > all_nodes[node_id].score:
                    all_nodes[node_id] = node
        
        if not all_nodes:
            return []
        
        # Rerank with primary query
        nodes_list = list(all_nodes.values())
        reranked = self.reranker.postprocess_nodes(
            nodes_list,
            query_str=queries[0]
        )
        
        return reranked
    
    def _build_context_with_sources(
        self, 
        nodes: List, 
        question: str
    ) -> Tuple[str, set]:
        """
        Build context string and extract source citations
        
        Args:
            nodes: Retrieved nodes
            question: User question
            
        Returns:
            Tuple of (context_string, sources_set)
        """
        # Dynamic context size based on question
        question_words = len(question.split())
        base_chars = 3000 if question_words < 10 else 5500
        max_context_chars = int(base_chars * 1.5) if self.mode == "academic" else base_chars
        
        context_parts = []
        sources = set()
        total_chars = 0
        
        for node in nodes:
            txt = node.node.text.strip()
            
            # Check if adding this would exceed limit
            if total_chars + len(txt) > max_context_chars:
                break
            
            context_parts.append(txt)
            total_chars += len(txt)
            
            # Extract source metadata
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
    
    def _get_answer_prompt(self, context: str, question: str) -> str:
        """Get mode-appropriate answer prompt"""
        if self.mode == "corporate":
            return f"""You are a precise document assistant. Answer ONLY using the provided context.

STRICT RULES:
1. Maximum 3 concise sentences
2. No bullet points or lists - use flowing prose
3. If information is partial, state: "The document mentions [available info] but doesn't provide complete details on [missing aspect]."
4. If not found, respond EXACTLY: "This information is not covered in the documents."
5. NO external knowledge, speculation, or assumptions
6. Be direct - no phrases like "According to the document" or "The text states"
7. Just state the facts naturally

CONTEXT:
{context}

QUESTION: {question}

ANSWER (max 3 sentences):"""
        
        else:  # academic
            return f"""You are an academic document assistant. Provide a detailed, accurate answer using ONLY the context provided.

REQUIREMENTS:
- Natural paragraph structure (2-3 paragraphs for complex topics)
- Clear, explanatory style suitable for learning
- Use proper terminology from the context
- If information is incomplete, explicitly state: "The document covers [X] but does not address [Y]."
- If not covered, respond: "This topic is not covered in the provided documents."
- NO external knowledge or speculation
- Maximum {self._academic_sentence_cap(question)} sentences
- Break longer answers into logical paragraphs

CONTEXT:
{context}

QUESTION: {question}

DETAILED ANSWER:"""
    
    def _academic_sentence_cap(self, question: str) -> int:
        """Determine sentence limit for academic answers"""
        q = question.lower().strip()
        
        # Definition questions: brief
        if q.startswith(("what is", "define", "meaning of")):
            return 3
        
        # Explanation questions: detailed
        if any(q.startswith(kw) for kw in [
            "explain", "discuss", "comment", "examine", "assess",
            "evaluate", "analyze", "analyse", "critically assess",
            "critique", "review", "elaborate", "describe in detail"
        ]):
            return 10  # Allow detailed explanations
        
        # Comparison questions: medium
        if any(word in q for word in ["compare", "contrast", "difference", "similar"]):
            return 7
        
        # Default: moderate
        return 5
    
    def _post_process_answer(self, answer: str, question: str) -> str:
        """Post-process answer for quality and consistency"""
        # Academic mode: enforce sentence limits and paragraph breaks
        if self.mode == "academic":
            cap = self._academic_sentence_cap(question)
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            
            # Trim to sentence cap
            if len(sentences) > cap:
                answer = " ".join(sentences[:cap]).strip()
            
            # Ensure complete sentences
            if not answer.endswith(('.', '!', '?')):
                # Find last complete sentence
                match = re.search(r'[.!?](?!.*[.!?])', answer, re.S)
                if match:
                    answer = answer[:match.end()].strip()
                else:
                    answer += "."
            
            # Add paragraph breaks for longer answers
            words = answer.split()
            if len(words) > 100:
                sentences = re.split(r'(?<=[.!?])\s+', answer)
                if len(sentences) > 4:
                    # Split into 2-3 paragraphs
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
        """Assess answer confidence based on retrieval quality"""
        if not nodes:
            return "low"
        
        # Check average score of top nodes
        top_scores = [n.score for n in nodes[:3] if hasattr(n, 'score')]
        if not top_scores:
            return "medium"
        
        avg_score = sum(top_scores) / len(top_scores)
        
        # Check answer quality indicators
        has_disclaimer = any(phrase in answer.lower() for phrase in [
            "not covered", "doesn't provide", "document mentions but"
        ])
        
        if avg_score > 0.8 and not has_disclaimer:
            return "high"
        elif avg_score > 0.5:
            return "medium"
        else:
            return "low"


# =============================================================================
# DOCUMENT LOADER WITH METADATA
# =============================================================================
def load_documents(file_paths: List[str]) -> List[Document]:
    """
    Load documents with rich metadata
    
    Args:
        file_paths: List of file paths to load
        
    Returns:
        List of Document objects with metadata
    """
    documents = []
    
    for path in file_paths:
        filename = os.path.basename(path)
        
        try:
            if path.lower().endswith('.pdf'):
                # Use PyMuPDF for clean extraction
                reader = PyMuPDFReader()
                docs = reader.load(file_path=path)
                
                # Add metadata to each page
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
                # Basic DOCX support (you might want to add python-docx)
                with open(path, 'rb') as f:
                    # Placeholder - implement proper DOCX parsing if needed
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

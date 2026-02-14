# -*- coding: utf-8 -*-
"""
Enhanced Generic Document Assistant – Production Ready V4
Multi-chunk, document-grounded QA with STRICT hallucination prevention
Hybrid retrieval + reranking + semantic chunking + ACCURATE source attribution
NEW: Answer-aware source filtering - only shows pages that contributed to answer
FIXED: Source over-attribution resolved
"""
# =============================================================================
# IMPORTS
# =============================================================================
import os
import re
from typing import List, Optional, Dict, Tuple, Callable
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
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
    model_name="BAAI/bge-small-en-v1.5",
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
        """Build vector index with improved chunking and retrieval"""
        total_steps = 5
        
        if progress_callback:
            progress_callback(1, total_steps, "Step 1/5: Splitting documents into chunks...")
        
        text_splitter = SentenceSplitter(
            chunk_size=400,
            chunk_overlap=100
        )
        
        nodes = text_splitter.get_nodes_from_documents(self.documents)
        
        if progress_callback:
            progress_callback(2, total_steps, "Step 2/5: Processing chunks...")
        
        chunk_config = self._get_chunk_config()
        self.nodes = self._smart_truncate_nodes(nodes, chunk_config['max_chars'])
        
        if progress_callback:
            progress_callback(3, total_steps, f"Step 3/5: Building vector index ({len(self.nodes)} chunks)...")
        
        self.index = VectorStoreIndex(self.nodes)
        
        if progress_callback:
            progress_callback(4, total_steps, "Step 4/5: Initializing hybrid retrievers...")
        
        top_k = 15
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
            progress_callback(5, total_steps, "Step 5/5: Loading reranking model...")
        
        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2",
            top_n=10 if self.mode == "corporate" else 8
        )
    
    def _get_chunk_config(self) -> Dict:
        """Get chunking configuration"""
        if self.mode == "corporate":
            return {'max_chars': 1200, 'top_k': 15, 'similarity_cutoff': 0.15}
        else:
            return {'max_chars': 900, 'top_k': 15, 'similarity_cutoff': 0.18}
    
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
        base = """Create a single-paragraph summary (100-120 words max).
Focus ONLY on the main topic and key themes.
Use natural, flowing prose - no lists or segmentation.
Do NOT start with "This document discusses" or "The document..."."""
    
        specific = "\nBusiness focus: What's the document about and why it matters." if self.mode == "corporate" else "\nAcademic focus: Core arguments and contributions."
    
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
    
    def _is_simple_question(self, question: str) -> bool:
        """Detect simple factual questions requiring brief answers"""
        question_lower = question.lower().strip()
        simple_triggers = [
            'name', 'list', 'who sang', 'who wrote', 'who is', 'what is',
            'what are', 'which', 'what year', 'when did', 'when was',
            'how many', 'name a', 'name an', 'give me', 'tell me the'
        ]
        return any(trigger in question_lower for trigger in simple_triggers)
    
    def ask_question(self, question: str, groq_api_key: str, return_metadata: bool = False):
        """Answer question with document grounding"""
        if not question or len(question.strip()) < 3:
            error = "Please ask a more specific question."
            if return_metadata:
                return AnswerResult(error, [])
            return error
        
        if not self.index:
            error = "Error: Index not initialized."
            if return_metadata:
                return AnswerResult(error, [])
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
                    sources=[]
                )
                return result if return_metadata else result.answer
            
            is_list_question = self._is_list_question(question)
            if is_list_question and len(retrieved_nodes) < 5:
                additional_nodes = self._get_additional_context(question, retrieved_nodes)
                if additional_nodes:
                    retrieved_nodes.extend(additional_nodes)
            
            context, source_map = self._build_context_with_sources(retrieved_nodes, question)
            
            # Dynamic max_tokens based on question type
            is_simple_question = self._is_simple_question(question)
            max_tokens = 80 if is_simple_question else 300
            
            llm = Groq(
                model="llama-3.1-8b-instant",
                api_key=groq_api_key,
                temperature=0.0,
                max_tokens=max_tokens
            )
            
            prompt = self._get_answer_prompt(context, question, sentence_limit)
            response = llm.complete(prompt)
            answer = str(response).strip()
            
            if not answer or len(answer.strip()) < 10:
                error = "Unable to generate answer. Please rephrase."
                if return_metadata:
                    return AnswerResult(error, [])
                return error
            
            answer = self._post_process_answer(answer, question)
            is_negative_response = self._is_negative_response(answer)
            
            # FIXED: Filter sources based on answer content
            filtered_sources = self._filter_relevant_sources(answer, source_map, question)

            # Add accurate source attribution with multi-doc awareness
            if filtered_sources and not is_negative_response:
                # Count unique documents
                unique_docs = set()
                for source_str in filtered_sources:
                    # Extract document name (before " (p.")
                    doc_name = source_str.split(' (p.')[0] if ' (p.' in source_str else source_str
                    unique_docs.add(doc_name)
    
                # CASE 1: Multiple documents - show document names with section counts
                if len(unique_docs) > 1:
                    doc_citations = []
                    for doc in sorted(unique_docs):
                        # Count sections from this doc
                        doc_sections = [s for s in filtered_sources if s.startswith(doc)]
                        count = len(doc_sections)
                        doc_citations.append(f"{doc} ({count} section{'s' if count != 1 else ''})")
                    answer += f"\n\n📚 Sources: {', '.join(doc_citations)}"
    
                # CASE 2: Single document - show section count only
                else:
                    section_count = len(filtered_sources)
                    answer += f"\n\n✅ Found in {section_count} document section{'s' if section_count != 1 else ''}"

            
            result = AnswerResult(
                answer=answer,
                sources=list(filtered_sources) if not is_negative_response else []
            )
            
            return result if return_metadata else result.answer
            
        except Exception as e:
            error = "Error processing question. Please try rephrasing."
            if return_metadata:
                return AnswerResult(error, [])
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
        """Generic query expansion - works for ANY document"""
        queries = [question]
        q = question.lower().strip()
        
        if q.startswith("what is "):
            term = question[8:].strip()
            queries.append(f"define {term}")
            queries.append(f"explain {term}")
            queries.append(f"{term} meaning")
        
        if q.startswith("what are "):
            term = question[9:].strip()
            queries.append(f"list of {term}")
            queries.append(f"types of {term}")
        
        if q.startswith("how to "):
            queries.append(question.replace("how to", "steps for"))
            queries.append(question.replace("how to", "process of"))
        
        if q.startswith("why "):
            queries.append(question.replace("why", "reasons for"))
        
        filler_words = ["the", "a", "an", "is", "are", "was", "were"]
        words = q.split()
        key_terms = " ".join([w for w in words if w not in filler_words])
        if key_terms != q:
            queries.append(key_terms)
        
        return queries[:4]
    
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
    
    def _build_context_with_sources(self, nodes: List, question: str) -> Tuple[str, Dict[str, List[str]]]:
        """Build context - adaptive based on question length
        
        Returns:
            Tuple of (context_string, source_map)
            source_map: Dict mapping source strings to list of chunk texts
        """
        question_words = len(question.split())
        
        if question_words < 5:
            base_chars = 3000
        elif question_words < 12:
            base_chars = 5000
        else:
            base_chars = 7000
        
        max_context_chars = int(base_chars * 1.5) if self.mode == "academic" else base_chars
        
        context_parts = []
        source_map = {}  # Map source string to list of chunk texts
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
                    if page and str(page).isdigit():
                        source_str += f" (p.{page})"
                    
                    # Store the chunk text with its source
                    if source_str not in source_map:
                        source_map[source_str] = []
                    source_map[source_str].append(txt)
        
        context = "\n\n---\n\n".join(context_parts)
        return context, source_map
    
    def _filter_relevant_sources(self, answer: str, source_map: Dict[str, List[str]], question: str) -> set:
        """
        Filter sources to only include those that contributed to the answer
        
        Args:
            answer: The generated answer text
            source_map: Dict mapping source strings to their chunk texts
            question: The original question
            
        Returns:
            Set of relevant source strings
        """
        # If negative response, return empty set
        if self._is_negative_response(answer):
            return set()
        
        # Extract key terms from answer (proper nouns, numbers, important words)
        answer_lower = answer.lower()
        
        # Extract capitalized words (proper nouns like "Nirvana", "Beatles")
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer)
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\b', answer)
        
        # Extract significant words (4+ characters, not common words)
        common_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'their', 'which', 'these', 'also', 'such', 'when', 'what', 'where', 'about', 'more', 'into', 'through', 'during', 'between'}
        significant_words = [w for w in re.findall(r'\b\w{4,}\b', answer_lower) if w not in common_words]
        
        # Combine all key terms
        answer_keywords = set([w.lower() for w in proper_nouns] + [w.lower() for w in numbers] + significant_words)
        
        # Check if it's a simple question
        is_simple = self._is_simple_question(question)
        
        relevant_sources = set()
        source_scores = {}
        
        for source_str, chunks in source_map.items():
            chunk_text = " ".join(chunks).lower()
            
            # Count keyword matches
            matches = sum(1 for keyword in answer_keywords if keyword in chunk_text)
            
            # Calculate match ratio
            match_ratio = matches / len(answer_keywords) if answer_keywords else 0
            
            source_scores[source_str] = match_ratio
            
            # Threshold based on question type
            if is_simple:
                # For simple questions, require at least 30% keyword match
                if match_ratio >= 0.3:
                    relevant_sources.add(source_str)
            else:
                # For complex questions, require at least 20% match
                if match_ratio >= 0.2:
                    relevant_sources.add(source_str)
        
        # If no sources passed filtering, take top scoring sources
        if not relevant_sources and source_scores:
            # Sort by score and take top N
            sorted_sources = sorted(source_scores.items(), key=lambda x: x[1], reverse=True)
            top_n = 2 if is_simple else 3
            relevant_sources = set([src for src, score in sorted_sources[:top_n] if score > 0])
        
        # If still no sources, fall back to first 1-2 sources
        if not relevant_sources and source_map:
            relevant_sources = set(list(source_map.keys())[:1 if is_simple else 2])
        
        return relevant_sources
    
    def _get_answer_prompt(self, context: str, question: str, sentence_limit: int) -> str:
        """Generate answer prompt with STRICT brevity control for simple questions"""
        
        is_simple = self._is_simple_question(question)
        
        if is_simple:
            instruction = """🚨 CRITICAL: This is a SIMPLE FACTUAL question. You MUST answer in EXACTLY 1-2 sentences MAXIMUM.

MANDATORY RULES:
1. For "name/list" questions: State ONLY the items, NO explanations
2. For "who/what/when" questions: Answer directly in ONE sentence
3. NEVER add background context or extra information
4. STOP after stating the fact

EXAMPLES OF CORRECT BREVITY:
❌ WRONG: "Spotify and Apple Music are two notable platforms. They provide instant access..."
✅ CORRECT: "Spotify and Apple Music."

❌ WRONG: "Queen sang 'Bohemian Rhapsody.' The song was released in 1975 and became..."
✅ CORRECT: "Queen sang 'Bohemian Rhapsody.'"

❌ WRONG: "Nirvana and Pearl Jam are two bands. Nirvana emerged as a defining band..."
✅ CORRECT: "Nirvana and Pearl Jam."

BE EXTREMELY BRIEF. STOP AFTER ANSWERING."""
        else:
            instruction = """This is an EXPLANATORY question. Answer in 2-4 sentences.

Provide specific details from the context to fully answer the question.
Include relevant context and examples where applicable.
Be comprehensive but concise."""
        
        limit_instruction = ""
        if sentence_limit > 0 and not is_simple:
            limit_instruction = f"\nLimit your answer to {sentence_limit} sentences."
        
        prompt = f"""{instruction}{limit_instruction}

IMPORTANT:
- Base your answer ONLY on the document context below
- If the answer isn't in the context, state this clearly in ONE sentence
- Do NOT make up information or add details not present in the context

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        return prompt
    
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
        """Post-process answer with AGGRESSIVE truncation for simple questions"""
        
        # FIRST: Handle simple questions with extreme brevity
        is_simple = self._is_simple_question(question)
        
        if is_simple:
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            
            # For list/name questions, try to keep just the list
            question_lower = question.lower().strip()
            if any(trigger in question_lower for trigger in ['name', 'list']):
                # Keep only first sentence (the actual list)
                answer = sentences[0].strip() if sentences else answer
            else:
                # For other simple questions, max 2 sentences
                answer = " ".join(sentences[:2]).strip() if len(sentences) > 2 else answer
            
            # Ensure proper ending
            if not answer.endswith(('.', '!', '?')):
                answer += "."
            
            return answer.strip()
        
        # EXISTING logic for complex questions
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


# =============================================================================
# DOCUMENT LOADER
# =============================================================================
def load_documents(
    file_paths: List[str],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> List[Document]:
    """
    Load documents with metadata and progress tracking
    
    Args:
        file_paths: List of file paths to load
        progress_callback: Optional callback(current, total, message)
        
    Returns:
        List of Document objects with metadata
    """
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
                    page_num = i + 1
                    
                    doc.metadata = {
                        "filename": filename,
                        "page": page_num,
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

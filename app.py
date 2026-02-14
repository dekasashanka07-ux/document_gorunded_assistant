"""
Enhanced Document Assistant - Streamlit App V2
Production-ready with improved UX, progress tracking, and coverage indicator
"""
import streamlit as st
import os
import tempfile
import uuid
import shutil
from datetime import datetime
from typing import Optional
import document_assistant as da
import fitz

# =============================================================================
# CONFIGURATION
# =============================================================================
MAX_MB = 20
MAX_BYTES = MAX_MB * 1024 * 1024
MAX_PDF_PAGES = 800

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
@st.cache_data(show_spinner=False)
def load_readme() -> str:
    """Load README file for about section"""
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return "README not found."

def validate_pdf_pages(file_bytes: bytes, filename: str) -> Optional[str]:
    """Validate PDF page count"""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = len(doc)
        doc.close()
        
        if page_count > MAX_PDF_PAGES:
            return f"'{filename}' has {page_count} pages (max: {MAX_PDF_PAGES}). Please split the file."
        return None
    except Exception as e:
        return f"Error validating '{filename}': {str(e)}"

def validate_file_size(uploaded_file) -> Optional[str]:
    """Validate file size"""
    if uploaded_file.size > MAX_BYTES:
        size_mb = uploaded_file.size / (1024 * 1024)
        return f"'{uploaded_file.name}' is {size_mb:.1f}MB (max: {MAX_MB}MB)"
    return None

def cleanup_temp_folder(folder_path: str):
    """Safely cleanup temporary folder"""
    if folder_path and os.path.isdir(folder_path):
        try:
            shutil.rmtree(folder_path, ignore_errors=True)
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CHANGED: Removed confidence-related CSS, kept coverage indicator styling
st.markdown("""
<style>
    .main-title {
        white-space: nowrap;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .block-container {
        padding-top: 1.8rem;
    }
    div[data-testid="column"] button {
        margin-top: 18px;
    }
    button[kind="secondary"] {
        padding: 0.35rem 0.75rem !important;
        font-size: 0.9rem !important;
        white-space: nowrap;
        min-width: 120px;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .status-ready {
        background-color: #d4edda;
        color: #155724;
    }
    .status-not-ready {
        background-color: #f8d7da;
        color: #721c24;
    }
    .coverage-info {
        font-size: 0.85rem;
        margin-top: 0.5rem;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 3px solid #007bff;
    }
    .coverage-success {
        background-color: #d4edda;
        color: #155724;
        border-left-color: #28a745;
    }
    .coverage-info-neutral {
        background-color: #d1ecf1;
        color: #0c5460;
        border-left-color: #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "chat" not in st.session_state:
    st.session_state.chat = []
if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = None
if "doc_folder" not in st.session_state:
    st.session_state.doc_folder = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())
if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "current_mode" not in st.session_state:
    st.session_state.current_mode = None

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<h1 class="main-title">📄 Document-Grounded Assistant</h1>', unsafe_allow_html=True)

# Status indicator
status = "Ready ✓" if st.session_state.initialized else "Not Initialized"
status_class = "status-ready" if st.session_state.initialized else "status-not-ready"
st.markdown(
    f'<span class="status-badge {status_class}">{status}</span>',
    unsafe_allow_html=True
)

# Action buttons
spacer, b1, b2 = st.columns([7, 1.5, 1.5])

with b1:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

with b2:
    if st.button("🔄 Reset All", use_container_width=True):
        cleanup_temp_folder(st.session_state.doc_folder)
        
        st.session_state.initialized = False
        st.session_state.chat = []
        st.session_state.doc_summary = None
        st.session_state.assistant = None
        st.session_state.doc_folder = None
        st.session_state.doc_count = 0
        st.session_state.current_mode = None
        st.session_state.uploader_key = str(uuid.uuid4())
        
        st.success("✓ Assistant reset. Upload documents again.")
        st.rerun()

# =============================================================================
# SIDEBAR - DOCUMENT UPLOAD & CONFIGURATION
# =============================================================================
with st.sidebar:
    st.header("📂 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload documents (.pdf, .txt, .docx)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key=st.session_state.uploader_key,
        help=f"Max {MAX_MB}MB per file, {MAX_PDF_PAGES} pages for PDFs"
    )
    
    # File validation
    validation_errors = []
    if uploaded_files:
        for file in uploaded_files:
            size_error = validate_file_size(file)
            if size_error:
                validation_errors.append(size_error)
                continue
            
            if file.name.lower().endswith('.pdf'):
                file_bytes = file.read()
                file.seek(0)
                page_error = validate_pdf_pages(file_bytes, file.name)
                if page_error:
                    validation_errors.append(page_error)
    
    if validation_errors:
        for error in validation_errors:
            st.error(error)
        st.stop()
    
    st.divider()
    
    # Mode selection
    st.header("⚙️ Configuration")
    
    doc_mode_label = st.selectbox(
        "Answer Mode",
        [
            "Corporate (Business/Training - Crisp)",
            "Academic (University/College - Detailed)"
        ],
        index=0,
        help="Corporate: Short, actionable answers. Academic: Detailed explanations."
    )
    doc_mode = "academic" if "Academic" in doc_mode_label else "corporate"
    
    st.divider()
    
    # API Key
    st.header("🔑 API Configuration")
    
    user_api_key = st.text_input(
        "Groq API Key (Optional)",
        type="password",
        value="",
        help="Enter your own key for unlimited usage. Leave empty to use limited fallback."
    )
    
    if user_api_key:
        groq_api_key = user_api_key
        using_fallback = False
        st.success("✓ Using your API key")
    else:
        groq_api_key = os.getenv("GROQ_API_KEY")
        using_fallback = True
        st.info("ℹ️ Using fallback key (10 questions/day)")
    
    # Rate limiting for fallback
    if using_fallback:
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
            st.session_state.last_reset = datetime.now().date()
        
        if datetime.now().date() > st.session_state.last_reset:
            st.session_state.query_count = 0
            st.session_state.last_reset = datetime.now().date()
        
        remaining = 10 - st.session_state.query_count
        st.metric("Questions Remaining Today", remaining)
        
        if st.session_state.query_count >= 10:
            st.error("❌ Daily limit reached. Please add your own API key.")
            st.stop()
    
    st.divider()
    
    # Initialize button
    init_button = st.button(
        "🚀 Initialize Assistant",
        use_container_width=True,
        type="primary"
    )
    
    if init_button:
        if not uploaded_files:
            st.error("❌ Please upload at least one document.")
        elif not groq_api_key:
            st.error("❌ No API key available. Please enter your Groq API key.")
        else:
            try:
                # Step 1: Save files with progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📁 Saving uploaded files...")
                progress_bar.progress(10)
                
                base_tmp = os.path.join(tempfile.gettempdir(), "doc_assistant")
                os.makedirs(base_tmp, exist_ok=True)
                
                folder_id = str(uuid.uuid4())[:8]
                doc_path = os.path.join(base_tmp, folder_id)
                os.makedirs(doc_path, exist_ok=True)
                
                file_paths = []
                for file in uploaded_files:
                    file_bytes = file.read()
                    dest_path = os.path.join(doc_path, file.name)
                    
                    with open(dest_path, "wb") as out:
                        out.write(file_bytes)
                    
                    file_paths.append(dest_path)
                
                st.session_state.doc_folder = doc_path
                st.session_state.doc_count = len(file_paths)
                
                progress_bar.progress(20)
                
                # Step 2: Load documents with progress callback
                def load_progress(current, total, message):
                    pct = 20 + int((current / total) * 20)  # 20-40%
                    progress_bar.progress(pct)
                    status_text.text(f"📄 {message}")
                
                status_text.text("📄 Loading documents...")
                documents = da.load_documents(file_paths, progress_callback=load_progress)
                
                progress_bar.progress(40)
                
                # Step 3: Build index with progress callback
                def index_progress(current, total, message):
                    pct = 40 + int((current / total) * 40)  # 40-80%
                    progress_bar.progress(pct)
                    status_text.text(f"🔧 {message}")
                
                status_text.text(f"🔧 Building index ({doc_mode} mode)...")
                st.session_state.assistant = da.DocumentAssistant(
                    documents,
                    mode=doc_mode,
                    progress_callback=index_progress
                )
                st.session_state.current_mode = doc_mode
                
                progress_bar.progress(80)
                st.session_state.initialized = True
                
                # Step 4: Generate summary with progress callback
                def summary_progress(current, total, message):
                    pct = 80 + int((current / total) * 20)  # 80-100%
                    progress_bar.progress(pct)
                    status_text.text(f"📝 {message}")
                
                status_text.text("📝 Generating summary...")
                st.session_state.doc_summary = st.session_state.assistant.generate_summary(
                    groq_api_key,
                    progress_callback=summary_progress
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Initialization complete!")
                
                st.success(f"✅ Initialized with {len(file_paths)} document(s) in {doc_mode.upper()} mode!")
                st.balloons()
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                cleanup_temp_folder(st.session_state.doc_folder)
                st.session_state.doc_folder = None

# =============================================================================
# ABOUT SECTION
# =============================================================================
with st.expander("ℹ️ About This Assistant", expanded=False):
    readme_content = load_readme()
    st.markdown(readme_content)

# Display current configuration
if st.session_state.initialized:
    with st.expander("📊 Current Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Mode:** {st.session_state.current_mode.capitalize()}")
            st.markdown(f"**Documents:** {st.session_state.doc_count}")
        with col2:
            st.markdown(f"**API:** {'Personal' if not using_fallback else 'Fallback'}")
            if using_fallback:
                st.markdown(f"**Queries Used:** {st.session_state.query_count}/10")

# =============================================================================
# DOCUMENT SUMMARY
# =============================================================================
if st.session_state.doc_summary is not None:
    with st.expander("📄 Document Summary", expanded=False):
        st.markdown(st.session_state.doc_summary)

# =============================================================================
# HELPER FUNCTION FOR COVERAGE INDICATOR
# =============================================================================
def get_coverage_indicator(answer: str, sources: list) -> tuple:
    """
    Generate coverage indicator based on answer content and sources
    Returns: (html_string, css_class)
    """
    # Check for negative response indicators
    negative_phrases = [
        "not covered in the documents",
        "not addressed in the documents",
        "not mentioned in the documents",
        "not found in the documents",
        "doesn't provide information",
        "this topic is not covered",
        "this information is not available"
    ]
    
    has_negative = any(phrase in answer.lower() for phrase in negative_phrases)
    unique_sources = len(sources)
    
    if has_negative or unique_sources == 0:
        return "📋 <strong>Topic not covered in provided documents</strong>", "coverage-info-neutral"
    elif unique_sources == 1:
        return "✅ <strong>Found in 1 document section</strong>", "coverage-success"
    else:
        return f"✅ <strong>Found in {unique_sources} document sections</strong>", "coverage-success"

# =============================================================================
# CHAT INTERFACE
# =============================================================================
if not st.session_state.initialized:
    st.info("👈 **Get Started:** Upload documents in the sidebar and click 'Initialize Assistant'")
    
    st.markdown("### How to Use")
    st.markdown("""
    1. **Upload** your documents (PDF, TXT, or DOCX)
    2. **Select** answer mode (Corporate or Academic)
    3. **Add** your Groq API key (optional, for unlimited use)
    4. **Click** Initialize Assistant
    5. **Ask** questions about your documents
    """)
    
else:
    # CHANGED: Display chat history with coverage indicator
    for item in st.session_state.chat:
        role = item["role"]
        msg = item["message"]
        
        with st.chat_message(role):
            st.markdown(msg)
            
               
    # Chat input
    user_question = st.chat_input("💬 Ask a question about your documents...")
    
    if user_question:
        # Add user message
        st.session_state.chat.append({
            "role": "user",
            "message": user_question
        })
        
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # CHANGED: Generate response with coverage indicator
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get answer WITH metadata
                    result = st.session_state.assistant.ask_question(
                        user_question,
                        groq_api_key,
                        return_metadata=True
                    )
                    
                    answer = result.answer
                    sources = result.sources
                    
                    # Increment query count if using fallback
                    if using_fallback:
                        st.session_state.query_count += 1
                    
                except Exception as e:
                    answer = f"❌ Error: {str(e)}\n\nPlease try rephrasing your question."
                    sources = []
            
            # Display answer
            st.markdown(answer)
            
        
        # CHANGED: Add assistant response to chat with coverage indicator
        st.session_state.chat.append({
            "role": "assistant",
            "message": answer            
        })
        
        st.rerun()

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<hr style="margin-top: 2rem; margin-bottom: 0.5rem; border: 0; border-top: 1px solid #eee;" />
<div style="text-align: center; font-size: 11px; color: #555; padding: 1rem 0;">
    📄 Created &amp; developed by <strong style="color: #9e50ba;">Sashanka Deka</strong>
    <span style="color: #999;"> | </span>
    <a href="https://x.com/sashanka_d" target="_blank" style="color: #1DA1F2; text-decoration: none; margin: 0 4px;">𝕏</a>
    <span style="color: #ccc; margin: 0 2px;">•</span>
    <a href="https://substack.com/@sashankadeka" target="_blank" style="color: #FF6719; text-decoration: none; margin: 0 4px;">Substack</a>
    <span style="color: #ccc; margin: 0 2px;">•</span>
    <a href="https://www.linkedin.com/in/sashanka-deka" target="_blank" style="color: #0077B5; text-decoration: none; margin: 0 4px;">LinkedIn</a>
</div>
""", unsafe_allow_html=True)

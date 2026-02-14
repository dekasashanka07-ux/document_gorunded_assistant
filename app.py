"""
Enhanced Document Assistant - Streamlit App
Production-ready with improved UX and error handling
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
    """
    Validate PDF page count
    
    Args:
        file_bytes: PDF file bytes
        filename: Name of file
        
    Returns:
        Error message if invalid, None if valid
    """
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
    """
    Validate file size
    
    Returns:
        Error message if invalid, None if valid
    """
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

# Custom CSS
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
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
        # Cleanup
        cleanup_temp_folder(st.session_state.doc_folder)
        
        # Reset state
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
    
    # File uploader
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
            # Size check
            size_error = validate_file_size(file)
            if size_error:
                validation_errors.append(size_error)
                continue
            
            # PDF page check
            if file.name.lower().endswith('.pdf'):
                file_bytes = file.read()
                file.seek(0)  # Reset file pointer
                page_error = validate_pdf_pages(file_bytes, file.name)
                if page_error:
                    validation_errors.append(page_error)
    
    # Display validation errors
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
        
        # Reset daily
        if datetime.now().date() > st.session_state.last_reset:
            st.session_state.query_count = 0
            st.session_state.last_reset = datetime.now().date()
        
        # Display usage
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
                with st.spinner("⏳ Processing documents..."):
                    # Create temp directory
                    base_tmp = os.path.join(tempfile.gettempdir(), "doc_assistant")
                    os.makedirs(base_tmp, exist_ok=True)
                    
                    folder_id = str(uuid.uuid4())[:8]
                    doc_path = os.path.join(base_tmp, folder_id)
                    os.makedirs(doc_path, exist_ok=True)
                    
                    # Save files
                    file_paths = []
                    for file in uploaded_files:
                        file_bytes = file.read()
                        dest_path = os.path.join(doc_path, file.name)
                        
                        with open(dest_path, "wb") as out:
                            out.write(file_bytes)
                        
                        file_paths.append(dest_path)
                    
                    st.session_state.doc_folder = doc_path
                    st.session_state.doc_count = len(file_paths)
                
                with st.spinner(f"🔧 Building index ({doc_mode} mode)..."):
                    # Load documents
                    documents = da.load_documents(file_paths)
                    
                    # Initialize assistant
                    st.session_state.assistant = da.DocumentAssistant(
                        documents,
                        mode=doc_mode
                    )
                    st.session_state.current_mode = doc_mode
                
                st.session_state.initialized = True
                
                with st.spinner("📝 Generating summary..."):
                    st.session_state.doc_summary = st.session_state.assistant.generate_summary(
                        groq_api_key
                    )
                
                st.success(f"✅ Initialized with {len(file_paths)} document(s) in {doc_mode.upper()} mode!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                # Cleanup on failure
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
# CHAT INTERFACE
# =============================================================================
if not st.session_state.initialized:
    st.info("👈 **Get Started:** Upload documents in the sidebar and click 'Initialize Assistant'")
    
    # Show instructions
    st.markdown("### How to Use")
    st.markdown("""
    1. **Upload** your documents (PDF, TXT, or DOCX)
    2. **Select** answer mode (Corporate or Academic)
    3. **Add** your Groq API key (optional, for unlimited use)
    4. **Click** Initialize Assistant
    5. **Ask** questions about your documents
    """)
    
else:
    # Display chat history
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)
    
    # Chat input
    user_question = st.chat_input("💬 Ask a question about your documents...")
    
    if user_question:
        # Add user message
        st.session_state.chat.append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    answer = st.session_state.assistant.ask_question(
                        user_question,
                        groq_api_key
                    )
                    
                    # Increment query count if using fallback
                    if using_fallback:
                        st.session_state.query_count += 1
                    
                except Exception as e:
                    answer = f"❌ Error: {str(e)}\n\nPlease try rephrasing your question."
            
            st.markdown(answer)
        
        # Add assistant response to chat
        st.session_state.chat.append(("assistant", answer))
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

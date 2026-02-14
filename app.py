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
            if file.name.lower().endswith('.pdf'):*

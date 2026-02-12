import streamlit as st
import os
import tempfile
import uuid
import shutil
from datetime import datetime
import document_assistant as da
import fitz

@st.cache_data(show_spinner=False)
def load_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return "README not found."

# -------------------------------------------------
# Upload limits
# -------------------------------------------------
MAX_MB = 20
MAX_BYTES = MAX_MB * 1024 * 1024
MAX_PDF_PAGES = 800

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Document Assistant", layout="wide")

st.markdown("""
<style>
.main-title { white-space: nowrap; font-size: 2.2rem; font-weight: 700; margin-bottom: 0.25rem; }
.block-container { padding-top: 1.8rem; }
div[data-testid="column"] button { margin-top: 18px; }
button[kind="secondary"] { padding: 0.35rem 0.75rem !important; font-size: 0.9rem !important; white-space: nowrap; min-width: 120px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Session state init
# -------------------------------------------------
if "initialized" not in st.session_state: st.session_state.initialized = False
if "chat" not in st.session_state: st.session_state.chat = []
if "doc_summary" not in st.session_state: st.session_state.doc_summary = None
if "doc_folder" not in st.session_state: st.session_state.doc_folder = None
if "uploader_key" not in st.session_state: st.session_state.uploader_key = str(uuid.uuid4())
if "assistant" not in st.session_state: st.session_state.assistant = None

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<h1 class="main-title">📄 Document-Grounded Assistant</h1>', unsafe_allow_html=True)
spacer, b1, b2 = st.columns([7,1.5,1.5])

with b1:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat = []

with b2:
    if st.button("Reset Assistant", use_container_width=True):
        st.session_state.initialized = False
        st.session_state.chat = []
        st.session_state.doc_summary = None
        st.session_state.assistant = None
        if st.session_state.doc_folder and os.path.isdir(st.session_state.doc_folder):
            shutil.rmtree(st.session_state.doc_folder, ignore_errors=True)
        st.session_state.doc_folder = None
        st.session_state.uploader_key = str(uuid.uuid4())
        st.success("Assistant reset. Upload documents again.")

# -------------------------------------------------
# Sidebar – Upload + Mode + API Key
# -------------------------------------------------

with st.expander("About this assistant", expanded=False):
    st.markdown(load_readme())

st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload (.pdf, .txt, .docx)",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
    key=st.session_state.uploader_key
)

# Size guard
if uploaded_files:
    for f in uploaded_files:
        if f.size > MAX_BYTES:
            st.sidebar.error(f"'{f.name}' exceeds {MAX_MB}MB limit.")
            st.stop()

# -------------------------------------------------
# MODE SELECTION - NOW WITH COMPLIANCE
# -------------------------------------------------
doc_mode_label = st.sidebar.selectbox(
    "Answer Mode",
    [
        "Corporate (Business/Training/Explanatory – crisp answers)",
        "Academic (University/College Modules – detailed concepts)",
        "Compliance (Policy/Legal/HR/Code of Conduct – exact rules)"
    ],
    index=0
)

# Map UI selection to mode string
if "Corporate" in doc_mode_label:
    doc_mode = "corporate"
elif "Academic" in doc_mode_label:
    doc_mode = "academic"
else:
    doc_mode = "compliance"

# Show mode hint
mode_hints = {
    "corporate": "✅ Concise answers, may combine within same topic",
    "academic": "📚 Structured, explanatory, sentence-capped",
    "compliance": "⚖️ Exact policy language, no cross-section merging"
}
st.sidebar.caption(mode_hints[doc_mode])

# API Key input + fallback
user_api_key = st.sidebar.text_input("Enter your Groq API Key (for unlimited use)", type="password", value="")

if user_api_key:
    groq_api_key = user_api_key
    using_fallback = False
else:
    groq_api_key = os.getenv("GROQ_API_KEY")
    using_fallback = True
    st.sidebar.info("No key entered → using fallback (10 questions/day limit).")

# Limit logic
if using_fallback:
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
        st.session_state.last_reset = datetime.now().date()

    if datetime.now().date() > st.session_state.last_reset:
        st.session_state.query_count = 0
        st.session_state.last_reset = datetime.now().date()

    if st.session_state.query_count >= 10:
        st.error("Daily limit reached (10 questions). Please enter your own Groq API key.")
        st.stop()

# -------------------------------------------------
# Initialize Assistant
# -------------------------------------------------
if st.sidebar.button("Initialize Assistant"):
    if not uploaded_files:
        st.sidebar.error("Upload at least one document.")
    else:
        try:
            base_tmp = os.path.join(tempfile.gettempdir(), "doc_assistant")
            os.makedirs(base_tmp, exist_ok=True)
            folder_id = str(uuid.uuid4())[:8]
            doc_path = os.path.join(base_tmp, folder_id)
            os.makedirs(doc_path, exist_ok=True)

            file_paths = []
            for f in uploaded_files:
                file_bytes = f.read()

                if f.name.lower().endswith(".pdf"):
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    if len(doc) > MAX_PDF_PAGES:
                        st.sidebar.error(f"'{f.name}' has too many pages (> {MAX_PDF_PAGES}). Split the file.")
                        st.stop()

                dest_path = os.path.join(doc_path, f.name)
                with open(dest_path, "wb") as out:
                    out.write(file_bytes)
                file_paths.append(dest_path)

            st.session_state.doc_folder = doc_path

            documents = da.load_documents(file_paths)

            with st.spinner(f"Initializing in {doc_mode.upper()} mode..."):
                st.session_state.assistant = da.DocumentAssistant(documents, mode=doc_mode)
            st.session_state.initialized = True

            with st.spinner("Generating summary..."):
                st.session_state.doc_summary = st.session_state.assistant.generate_summary(groq_api_key)

            st.sidebar.success(f"Ready in {doc_mode.upper()} mode.")
        except Exception as e:
            st.sidebar.error(f"Initialization failed: {str(e)}")

# -------------------------------------------------
# Summary Display
# -------------------------------------------------
if st.session_state.doc_summary is not None:
    with st.expander("Document Summary (~120 words)", expanded=False):
        st.markdown(st.session_state.doc_summary)

# -------------------------------------------------
# Chat Interface
# -------------------------------------------------
if not st.session_state.initialized:
    st.info("👈 Upload documents and click Initialize Assistant to begin.")
else:
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)

    user_question = st.chat_input("Ask something from your document...")
    if user_question:
        st.session_state.chat.append(("user", user_question))
        with st.chat_message("user"): 
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.assistant.ask_question(user_question, groq_api_key)
                if using_fallback: 
                    st.session_state.query_count += 1
            st.markdown(answer)

        st.session_state.chat.append(("assistant", answer))

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("""
<hr style="margin-top: 1rem; margin-bottom: 0.5rem; border: 0; border-top: 1px solid #eee;" />
<div style="text-align: center; font-size: 11px; color: #555;">
📄 Created &amp; developed by <strong style="color: #9e50ba;"> Sashanka Deka</strong>
<span style="color: #999;"> | </span>
<a href="https://x.com/sashanka_d" target="_blank" style="color: #1DA1F2; text-decoration: none; margin: 0 4px;">𝕏</a>
<span style="color: #ccc; margin: 0 2px;">•</span>
<a href="https://substack.com/@sashankadeka" target="_blank" style="color: #FF6719; text-decoration: none; margin: 0 4px;">Substack</a>
<span style="color: #ccc; margin: 0 2px;">•</span>
<a href="https://www.linkedin.com/in/sashanka-deka" target="_blank" style="color: #0077B5; text-decoration: none; margin: 0 4px;">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
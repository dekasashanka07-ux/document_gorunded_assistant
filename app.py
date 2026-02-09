import streamlit as st
import os
import tempfile
import uuid
import shutil
from datetime import datetime # For daily reset
import document_assistant as da # Your updated backend
# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Document Assistant", layout="wide")

st.markdown("""
<style>

/* Keep title in one line */
.main-title {
    white-space: nowrap;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

/* Reduce empty vertical space */
.block-container {
    padding-top: 1.8rem;
}

/* Align buttons cleanly (no stretching) */
div[data-testid="column"] button {
    margin-top: 18px;
}

</style>
""", unsafe_allow_html=True)



# -------------------------------------------------
# Session state init
# -------------------------------------------------
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
    st.session_state.assistant = None # New: Store the DocumentAssistant instance
# -------------------------------------------------
# Header + Controls
# -------------------------------------------------
# -------- Title (full width) --------
st.markdown(
    """
    <h1 class="main-title">📄 Document-Grounded Assistant</h1>
    """,
    unsafe_allow_html=True
)

# -------- Buttons row --------
header_spacer, header_mid, header_right = st.columns([6, 1, 1])

with header_mid:
    if st.button("Clear Chat"):
        st.session_state.chat = []

with header_right:
    if st.button("Reset Assistant"):
        # Reset assistant state
        st.session_state.initialized = False
        st.session_state.chat = []
        st.session_state.doc_summary = None
        st.session_state.assistant = None
        # Delete temp document folder
        if st.session_state.doc_folder and os.path.isdir(st.session_state.doc_folder):
            shutil.rmtree(st.session_state.doc_folder, ignore_errors=True)
        st.session_state.doc_folder = None
        # Reset uploader completely
        st.session_state.uploader_key = str(uuid.uuid4())
        st.success("Assistant reset. Upload documents again.")

# -------------------------------------------------
# Sidebar – Upload + Mode + API Key
# -------------------------------------------------
st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload (.pdf, .txt, .docx)",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
    key=st.session_state.uploader_key
)
doc_mode_label = st.sidebar.selectbox(
    "Answer Mode",
    ["Corporate (Business/Training/Short Legal – crisp answers)",
     "Academic (University/IGNOU Modules – detailed concepts)"],
    index=0
)
doc_mode = "academic" if "Academic" in doc_mode_label else "corporate"

# API Key input + fallback
user_api_key = st.sidebar.text_input(
    "Enter your Groq API Key (for unlimited use)",
    type="password",
    value=""
)

if user_api_key:
    groq_api_key = user_api_key
    using_fallback = False
else:
    groq_api_key = os.getenv("GROQ_API_KEY")
    using_fallback = True
    st.sidebar.info("No key entered → using fallback (10 questions/day limit).")


# Limit logic (only when fallback is used)
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
            # Save uploaded documents
            file_paths = []
            for f in uploaded_files:
                dest_path = os.path.join(doc_path, f.name)
                with open(dest_path, "wb") as out:
                    out.write(f.read())
                file_paths.append(dest_path)
            st.session_state.doc_folder = doc_path
           
            # Load documents using the new helper (better PDF handling)
            documents = da.load_documents(file_paths)
           
            # Initialize the DocumentAssistant class with mode
            with st.spinner(f"Initializing in {doc_mode.upper()} mode..."):
                st.session_state.assistant = da.DocumentAssistant(documents, mode=doc_mode)
            st.session_state.initialized = True
           
            # Generate summary using the new method
            with st.spinner("Generating summary..."):
                st.session_state.doc_summary = st.session_state.assistant.generate_summary(groq_api_key)
                # Increment counter if fallback
                
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
    # Display chat history
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)
    # Input box
    user_question = st.chat_input("Ask something from your document...")
    if user_question:
        # Store user question
        st.session_state.chat.append(("user", user_question))
        # Show latest user question immediately
        with st.chat_message("user"):
            st.markdown(user_question)
        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.assistant.ask_question(user_question, groq_api_key)
                # Increment counter if fallback
                if using_fallback:
                    st.session_state.query_count += 1
            st.markdown(answer)
        # Store assistant answer
        st.session_state.chat.append(("assistant", answer))
# -------------------------------------------------
# Footer – Branding (REQUIRED)
# -------------------------------------------------
st.markdown(
    """
    <hr style="margin-top: 1rem; margin-bottom: 0.5rem; border: 0; border-top: 1px solid #eee;" />
    <div style="text-align: center; font-size: 11px; color: #555;">
        📄 Conceived &amp; created by
        <strong style="color: #9e50ba;"> Sashanka Deka</strong>
        <span style="color: #999;"> | </span>
        <a href="https://x.com/sashanka_d" target="_blank"
           style="color: #1DA1F2; text-decoration: none; margin: 0 4px;">𝕏</a>
        <span style="color: #ccc; margin: 0 2px;">•</span>
        <a href="https://substack.com/@sashankadeka" target="_blank"
           style="color: #FF6719; text-decoration: none; margin: 0 4px;">Substack</a>
        <span style="color: #ccc; margin: 0 2px;">•</span>
        <a href="https://www.linkedin.com/in/sashanka-deka" target="_blank"
           style="color: #0077B5; text-decoration: none; margin: 0 4px;">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)
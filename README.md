# Document-Grounded Assistant

A document-grounded QA assistant that answers **strictly** from uploaded files. Built with Streamlit for easy use—no coding required after setup.

This app is designed for reading and verifying information in reports, policies, study material, and long documents without relying on external knowledge.

## Why Two Answer Modes? (Corporate vs. Academic)

The app offers **two tailored modes** to match different document types and user needs:

- **Corporate Mode** (default for business/policy/training docs)  
  Gives crisp, professional answers using bullets, lists, numbered steps, and short paragraphs. Ideal for quick scans of reports, guidelines, contracts, or training materials where clarity and structure matter most.

- **Academic Mode** (for university notes, textbooks, learning modules, essays)  
  Delivers detailed, natural paragraphs with proper sentence flow, logical breaks, and controlled length (e.g., limited sentences for definitions, more for explanations). Presents document-grounded answers in clear explanatory paragraphs suited for study material, with controlled length for readability.

You select the mode once when initializing—no need to switch mid-chat.

## Key Features

- **PageIndex Retrieval** — Two-level hierarchical retrieval: page-level summary index + chunk-level store. Queries hit page summaries first, then fetch only relevant chunks from matched pages — producing coherent, page-bounded context for the LLM rather than scattered fragments.

- **Accurate Page Citations** — Both Corporate and Academic modes report the exact page numbers the LLM drew from, not all retrieved pages. Academic mode cites pages as `Page 3 • Page 7`. Corporate mode does the same.

- **Strict Grounding** — Answers only use text directly from your documents. If the info isn't there, it clearly says *"This information is not covered in the provided documents."* The assistant does not infer beyond the provided material.

- **Two Answer Tiers per Mode**  
  - *Corporate:* Fact / List / Comparison / Explanation — each with its own format and length rules  
  - *Academic:* Short / Default / Long — scaling from 3-sentence definitions to 2-paragraph scholarly analysis

- **Smart Document Handling** — Supports PDF (native page boundaries via PyMuPDF), TXT, and DOCX (500-word page windows). Structural pages like table of contents and index pages are automatically filtered out.

- **Summary Generation** — Auto-creates a concise overview of the core content on initialization. Ignores tables of contents, objectives, and glossaries.

- **Bring Your Own Groq API Key (BYOGAK)** — Use your free Groq key for unlimited questions. Falls back to a 10 questions/day limit if no key is provided — prevents abuse on free hosting.

- **Deterministic Responses** — Temperature 0.0 and constrained prompt engineering reduce variation and prevent speculative answers.

---

## Known Limitations

- Very visual PDFs (heavy images, complex overlays) may lose minor details during text extraction.
- While grounding minimises errors, responses depend on document quality and extraction accuracy — results may not always be perfectly precise.
- Free Groq tier has rate limits — use your own key for heavy use. `llama-3.1-8b-instant` is used by default (separate 1M token daily limit).
- App is single-session — multi-user needs paid hosting.

---

## Data Handling

- Documents are processed only within the running session.
- They are not stored for training or shared across users.
- The assistant only receives extracted text required to answer a query.
- Closing or resetting the app removes all session data.

---

## Installation & Local Run

1. Clone the repo
```bash
git clone https://github.com/dekasashanka/07-ux-document-grounded-assistant.git
cd document-grounded-assistant

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Run the app:
   ```bash
   streamlit run app.py

Upload documents via the sidebar, choose mode, initialize, and start asking questions.

## How to Get Your Free Groq API Key (Simple Steps for Everyone)

Groq gives fast, free access to powerful AI models like Llama-3.1-8B (the brain behind this app). No credit card needed for the free tier.

1. Go to: https://console.groq.com  
2. Click "Sign up" or "Log in" (use Google, GitHub, or email—easiest with Google/GitHub).  
3. Once logged in, go to: https://console.groq.com/keys  
4. Click **Create API Key** (or similar button).  
5. Give it a name (e.g., "My Document Assistant").  
6. Click **Submit** or **Create**.  
7. Copy the key (it starts with `gsk_...`) — paste it into the sidebar box in the app when asked.  
   - Keep it private—don't share it.  
   - If you lose it, just create a new one.

Free tier has generous daily limits for personal use. For heavy usage, upgrade is optional.

## Deployment (Optional – Make It Public in Minutes)

Want to share the app online for free?

- **Streamlit Community Cloud** (easiest & recommended):  
  1. Push your code to GitHub.  
  2. Go to https://share.streamlit.io  
  3. Click "New app" → connect your repo → select `app.py` as main file.  
  4. Deploy — live in ~2-5 minutes!  
  (Add your fallback Groq key in app settings > Secrets if needed.)

For detailed steps, see: https://docs.streamlit.io/deploy/streamlit-community-cloud

| Component              | Technology                                        |
| ---------------------- | ------------------------------------------------- |
| UI Framework           | Streamlit                                         |
| Retrieval Architecture | PageIndex (two-level: page summary + chunk store) |
| Embedding Model        | BAAI/bge-small-en-v1.5 via HuggingFace            |
| LLM                    | llama-3.1-8b-instant via Groq API                 |
| PDF Extraction         | PyMuPDF (fitz)                                    |
| DOCX Extraction        | python-docx                                       |
| Vector Index           | LlamaIndex VectorStoreIndex                       |

## License

MIT License – feel free to fork, modify, and share.

Developed by Sashanka Deka  
Questions? Open an issue or reach out on [LinkedIn](https://www.linkedin.com/in/sashanka-deka) / [X](https://x.com/sashanka_d).


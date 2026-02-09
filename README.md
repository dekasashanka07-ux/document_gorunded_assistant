# Document-Grounded Assistant

A reliable, hallucination-free AI assistant that answers questions **strictly from your uploaded documents**. Built with Streamlit for easy use—no coding required after setup.

This app is useful for professionals, students, researchers, or anyone who needs accurate insights from reports, policies, study materials, legal docs, or books—without fabricated or external information.

## Why Two Answer Modes? (Corporate vs. Academic)

The app offers **two tailored modes** to match different document types and user needs:

- **Corporate Mode** (default for business/policy/training docs)  
  Gives crisp, professional answers using bullets, lists, numbered steps, and short paragraphs. Ideal for quick scans of reports, guidelines, contracts, or training materials where clarity and structure matter most.

- **Academic Mode** (for university notes, textbooks, IGNOU-style modules, essays)  
  Delivers detailed, natural paragraphs with proper sentence flow, logical breaks, and controlled length (e.g., limited sentences for definitions, more for explanations). Presents document-grounded answers in clear explanatory paragraphs suited for study material, with controlled length for readability.

You select the mode once when initializing—no need to switch mid-chat.

## Key Features

- **Strict Grounding** — Answers only use text directly from your documents. If the info isn't there, it clearly says: "Not covered in the documents." The assistant does not infer beyond the provided material.
- **Hybrid Retrieval** — Combines semantic search + keyword matching for better accuracy on complex or fragmented layouts (slides, tables, dense text).
- **Smart Document Handling** — Supports PDF (clean text extraction), TXT, DOCX. Semantic chunking keeps context meaningful.
- **Summary Generation** — Auto-creates a concise ~120-word overview of the core content (ignores tables of contents, objectives, glossaries).
- **Bring Your Own Groq API Key (BYOGAK)** — Use your free Groq key for unlimited questions; fallback to a simple 20 questions/day limit if no key is provided (prevents abuse on free hosting).
- **No Hallucinations** — Temperature=0.0 + strict prompt engineering ensures factual, grounded responses.

## Known Limitations

- Very visual PDFs (heavy images, complex overlays) may lose minor details during text extraction.
- While grounding minimizes errors, responses depend on document quality and extraction accuracy, so results may not always be perfectly precise.
- Free Groq tier has rate limits—use your own key for heavy use.
- App is single-session; multi-user needs paid hosting.

## Installation & Local Run (For Developers)

1. Clone the repo:  
   ```bash
   git clone https://github.com/dekasashanka07-ux/document_grounded_assistant.git
   cd document_grounded_assistant

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

## License

MIT License – feel free to fork, modify, and share.

Made with ❤️ by Sashanka Deka  
Questions? Open an issue or reach out on [LinkedIn](https://www.linkedin.com/in/sashanka-deka) / [X](https://x.com/sashanka_d).


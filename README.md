# DocuMind

A fintech-ready RAG (Retrieval-Augmented Generation) system for intelligent document querying. Upload any PDF, ask natural language questions, and get accurate, evidence-grounded answers.

## Features

- **RAG Pipeline** — Extract → Chunk → Embed → Index → Retrieve → Generate
- **Hybrid Inference** — Groq API (cloud) or TinyLlama (local)
- **FAISS Vector Search** — Fast semantic similarity matching
- **Privacy-Aware** — Local model option for sensitive financial documents
- **Modern UI** — Streamlit-based chat interface

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS (faiss-cpu) |
| PDF Parsing | pdfplumber |
| LLM (Cloud) | Groq API (openai/gpt-oss-120b) |
| LLM (Local) | TinyLlama 1.1B (Q4 quantized) |
| UI | Streamlit |

## Architecture

```
PDF Upload → Text Extraction → Chunking → Embedding → FAISS Index
                                                          ↓
User Question → Embedding → Vector Search → Top-K Chunks → LLM → Answer
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/DocuMind.git
cd DocuMind

# Install dependencies
pip install -r requirements.txt

# Set up environment
copy .env.example .env
# Add your GROQ_API_KEY to .env (get one free at https://console.groq.com/keys)

# Run the app
streamlit run app.py
```

## How It Works

1. **Upload a PDF** — The app extracts text using pdfplumber
2. **Chunking** — Text is split into overlapping chunks (1000 chars, 200 overlap)
3. **Embedding** — Chunks are embedded locally using sentence-transformers
4. **Indexing** — Embeddings are stored in a FAISS vector index
5. **Query** — Your question is embedded and matched against the index
6. **Generate** — Top chunks are sent to the LLM for a grounded answer

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Your Groq API key | Required for cloud mode |
| `GROQ_MODEL` | Groq model to use | `openai/gpt-oss-120b` |

## Use Cases

- Query invoices, contracts, and financial reports
- Extract key details from compliance documents
- Analyze policy documents with natural language
- Offline document analysis for privacy-sensitive data

## Project Structure

```
DocuMind/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── run.bat             # Windows launcher
├── test_smoke.py       # Smoke tests
└── models/             # Local LLM models (auto-downloaded)
```

## License

MIT

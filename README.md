# 🧠 RAG Scholar

A **Retrieval‑Augmented Generation (RAG)** research assistant that lets you query Wikipedia or arXiv papers through an LLM‑powered QA interface.  The project ships with both a **Streamlit** GUI and a **command‑line tool** and is built entirely on open‑source components.

---

## Key Features

| Capability                         | Details                                                                                                                     |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Multisource ingestion**    | Choose between Wikipedia search or arXiv API.                                                                               |
| **Local vector store**       | FAISS index persisted under `data/` for fast, offline retrieval.                                                          |
| **Pluggable embeddings**     | Default:`sentence‑transformers/all‑MiniLM‑L6‑v2`, but any SBERT model will work.                                      |
| **Model flexibility**        | Supports Hugging Face seq2seq (e.g. Flan‑T5)**or** causal LMs (e.g. Falcon‑7B‑Instruct); optional OpenAI backend. |
| **Custom prompt**            | Few‑shot prompt with context injection + citations.                                                                        |
| **Interactive UI**           | Streamlit front‑end with live source inspectors.                                                                           |
| **Robust CLI**               | Batch or interactive shell for headless environments.                                                                       |
| **Configurable & stateless** | Re‑initialise knowledge base on demand or reuse a saved index.                                                             |

---

## System Architecture

```text
┌────────────┐      ┌─────────────────────┐      ┌──────────────┐
│  User UI   │──┐   │ LangChain RAG Chain │──┐   │ Large Language│
│ (Streamlit │  │   │  RetrievalQA (LLM +│  │   │   Model       │
│   or CLI)  │  │   │   Retriever)       │  │   │ (HF / OpenAI) │
└────────────┘  │   └─────────────────────┘  │   └──────────────┘
                │                            │
                │   ┌──────────────────────┐ │
                └──►│   FAISS VectorStore  │◄┘
                    └──────────────────────┘
                             ▲
                             │
                ┌────────────────────────────┐
                │  Document Loaders (Wiki /   │
                │  arXiv) + Text Splitter     │
                └────────────────────────────┘
```

---

## Quick Start

### 1 · Clone & Install

```bash
git clone https://github.com/your‑handle/rag‑scholar.git
cd rag‑scholar

python -m venv .venv && source .venv/bin/activate  # on Windows: .\.venv\Scripts\activate

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2 · Environment Variables

Create a `.env` file (read via **python‑dotenv**) and set:

```ini
# Optional — only if you tick "Use OpenAI" in the UI
OPENAI_API_KEY="sk‑..."

# For private Hugging Face models
HUGGINGFACEHUB_API_TOKEN="hf_..."
```

### 3 · Launch the Streamlit app

```bash
streamlit run app.py
```

The first launch downloads documents, builds embeddings, and writes a FAISS index to `data/vector_db_<source>_<topic>/`. Subsequent runs reuse the index unless you tick “Initialize/Reload Knowledge Base”.

### 4 · CLI Usage (headless servers)

```bash
python rag_scholar.py \
    --source wikipedia \
    --topic "quantum computing" \
    --question "What is a topological qubit?"
```

Add `--openai` to route generation through the OpenAI API.

---

## Configuration Flags

| Flag           | Type                        | Default                | Description                                |
| -------------- | --------------------------- | ---------------------- | ------------------------------------------ |
| `--source`   | `wikipedia` · `arxiv` | `wikipedia`          | Data corpus to ingest.                     |
| `--topic`    | *str*                     | `"machine learning"` | Search query (Wiki) or arXiv title filter. |
| `--openai`   | toggle                      | *False*              | Switch from Hugging Face to OpenAI LLM.   |
| `--ui`       | toggle                      | *False*              | Run Streamlit interface.                   |
| `--question` | *str*                     | –                     | Ask a single question in CLI mode.         |

---

## RAG Pipeline Internals

1. **Loader** pulls raw documents via *LangChain loaders*.
2. **RecursiveCharacterTextSplitter** chunks text (`chunk_size=1000`, overlap 100).
3. **CustomSentenceTransformerEmbeddings** encodes chunks → 384‑d vectors.
4. **FAISS** stores vectors locally; persisted across sessions.
5. **RetrievalQA** fetches top‑*k* (`k=4`) chunks, injects into a custom prompt, and streams completion from the selected LLM.
6. Sources are returned alongside answers for full transparency.

---

## Repository Layout

```
.
├── app.py              # Streamlit front‑end
├── requirements.txt    # Python deps (≈ python 3.10+)
├── data/               # Persisted FAISS indices
└── README.md           # (you are here)
```

---

## Logging

`logging` is configured at **INFO** level by default.

```
2025‑06‑14 12:34:56,789 ‑ RAGScholar ‑ INFO ‑ Loaded 256 documents
```

Set the environment variable `LOGLEVEL=DEBUG` for verbose output.

---

## Development Notes

- **GPU**:  If you have CUDA installed, change `model.to("cuda")` in `RAGScholar.__init__()` to unlock GPU inference.
- **Vector DB schema changes**:  A mismatching index triggers a rebuild; see `load_vectorstore()`.
- **Adding new data sources**:  Implement another `Loader` and extend `source_type` logic.

---

Made with LangChain v0.2.0

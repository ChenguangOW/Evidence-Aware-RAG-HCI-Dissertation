# Evidence-Aware RAG Academic Assistant
## MSc HCI Dissertation Source Code

This is the actual software prototype for the dissertation:

**Designing and Evaluating an Evidence-Aware Interface for RAG-Based Academic Question Answering**

It implements:

- PDF/TXT academic document upload
- document text extraction
- overlapping chunking
- **Sentence-Transformers + FAISS** semantic retrieval
- TF-IDF fallback retrieval
- OpenAI grounded answer generation
- local extractive fallback when no API key is available
- three HCI interface conditions:
  - A — answer only
  - B — answer + citations
  - C — claim-by-claim evidence + support cue
- optional user-evaluation logging to CSV

---

## Python version

Recommended: **Python 3.11 or 3.12**

---

## Quick run (portable version)

This version is easiest to install and will run using TF-IDF if the semantic backend is unavailable.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Full dissertation architecture

To use the intended **Sentence-Transformers + FAISS** retrieval backend:

```bash
pip install -r requirements-full.txt
streamlit run app.py
```

The first time Sentence-Transformers runs, it may download:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Choose **Sentence-Transformers + FAISS** in the sidebar.

---

## OpenAI API

The application can run without an API key.

Without a key, it uses a local extractive fallback.

For LLM generation:

### macOS / Linux

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
streamlit run app.py
```

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
streamlit run app.py
```

The default model is:

```text
gpt-5.6-luna
```

You can change it from the Streamlit sidebar.

---

## Test before the viva

Run:

```bash
python demo_cli.py
```

Then run automated tests:

```bash
pytest -q
```

---

## Project structure

```text
Evidence_Aware_RAG_Dissertation_Code/
├── app.py
├── demo_cli.py
├── requirements.txt
├── requirements-full.txt
├── README.md
├── data/
├── src/
│   └── evidence_rag/
│       ├── __init__.py
│       ├── document.py
│       ├── retrieval.py
│       ├── generation.py
│       ├── evidence.py
│       └── logging_utils.py
└── tests/
    └── test_core.py
```

---

## Dissertation architecture

```text
Academic PDF
     ↓
PDF text extraction
     ↓
Overlapping document chunks
     ↓
Sentence-BERT embeddings
     ↓
FAISS vector index
     ↓
Top-k retrieved evidence
     ↓
LLM grounded generation
     ↓
┌───────────────────────────────┐
│ A: Basic answer               │
│ B: Answer + citations         │
│ C: Claim + evidence alignment │
└───────────────────────────────┘
     ↓
User evaluation
```

---

## Important methodological note

The **Strongly Supported / Partially Supported / Unsupported** labels are currently heuristic display cues based on claim-to-evidence textual similarity.

They should be described in the dissertation as an **interface prototype mechanism**, not as calibrated factual confidence scores.

A stronger future implementation could replace the heuristic with a trained natural-language-inference (NLI) model and validate thresholds on an annotated entailment dataset.

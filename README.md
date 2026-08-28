# Evidence-Aware RAG Academic Assistant

[![Tests](https://github.com/ChenguangOW/Evidence-Aware-RAG-HCI-Dissertation/actions/workflows/tests.yml/badge.svg)](https://github.com/ChenguangOW/Evidence-Aware-RAG-HCI-Dissertation/actions/workflows/tests.yml)

![Evidence-Aware interface](docs/evidence-aware-interface.jpg)

MSc HCI dissertation prototype for:

**Designing and Evaluating an Evidence-Aware Interface for RAG-Based Academic Question Answering**

This project investigates how interface design can help users understand, verify, and critically evaluate answers produced by retrieval-augmented generation (RAG) systems.

## Research focus

The main contribution is not a new RAG algorithm. The project compares three interface conditions for academic question answering:

- **Version A — Basic Answer:** answer only
- **Version B — Citation-Based:** answer plus source citations
- **Version C — Evidence-Aware:** answer split into claims, with claim-level source evidence and heuristic support cues

The design goal is **calibrated trust**: helping users judge when an answer is well supported and when it should be checked more carefully.

## Features

- PDF/TXT academic document upload
- page-aware text extraction
- overlapping document chunking
- **Sentence-Transformers + FAISS** semantic retrieval
- TF-IDF retrieval fallback
- grounded answer generation using the OpenAI Responses API
- local extractive fallback when no API key is available
- three HCI interface conditions
- claim-to-evidence alignment for Version C
- heuristic support labels:
  - Strongly Supported
  - Partially Supported
  - Unsupported / Check Source
- optional evaluation logging to CSV
- automated unit tests and GitHub Actions CI

## System architecture

```text
Academic PDF / TXT
        ↓
Text extraction
        ↓
Overlapping page-aware chunks
        ↓
Sentence-BERT embeddings
        ↓
FAISS vector index
        ↓
Top-k retrieved passages
        ↓
Grounded answer generation
        ↓
┌───────────────────────────────┐
│ A: Answer only                │
│ B: Answer + citations         │
│ C: Claim + evidence alignment │
└───────────────────────────────┘
        ↓
User evaluation
```

## Project structure

```text
Evidence-Aware-RAG-HCI-Dissertation/
├── .github/
│   └── workflows/
│       └── tests.yml
├── app.py
├── demo_cli.py
├── requirements.txt
├── requirements-full.txt
├── README.md
├── run_mac.sh
├── run_windows.bat
├── sample_notes.txt
├── data/
├── docs/
│   └── evidence-aware-interface.jpg
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

## Python version

Recommended: **Python 3.11 or 3.12**.

## Quick start

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

Streamlit normally opens at:

```text
http://localhost:8501
```

## Full semantic retrieval setup

The portable requirements install the core system and TF-IDF fallback. To use the dissertation's preferred semantic retrieval architecture:

```bash
pip install -r requirements-full.txt
streamlit run app.py
```

Then select:

```text
Sentence-Transformers + FAISS
```

in the sidebar.

The first run may download:

```text
sentence-transformers/all-MiniLM-L6-v2
```

## OpenAI generation

The application does **not** require an API key to run. Without a key, it uses a local extractive fallback so that the prototype remains demonstrable offline.

For generative RAG, set an OpenAI API key.

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

The model name can also be changed directly in the Streamlit sidebar.

## How to run the three experimental conditions

1. Upload the same academic document.
2. Click **Index document**.
3. Enter an academic question.
4. Select one of the three interface conditions:
   - A — Basic Answer
   - B — Citation-Based
   - C — Evidence-Aware
5. Ask the question and record the participant response if conducting the study.

For controlled comparison, use the same underlying document and question set across conditions.

## Evaluation data

The optional participant form in the app stores responses in:

```text
data/evaluation_log.csv
```

This file is ignored by Git so participant data are not accidentally committed to the repository.

Recommended quantitative measures include:

- trust
- clarity
- verification ease
- confidence
- unsupported-claim detection accuracy
- task completion time

Any real participant study should follow the relevant university ethics and consent requirements.

## Testing

Run the command-line smoke test:

```bash
python demo_cli.py
```

Run unit tests:

```bash
pytest -q
```

The current core test suite checks:

- page-aware chunking
- retrieval relevance
- claim splitting
- support-label thresholds
- local no-API generation
- claim-to-evidence alignment

## Continuous integration

GitHub Actions is configured in:

```text
.github/workflows/tests.yml
```

On pushes and pull requests to `main`, CI runs on Python 3.11 and 3.12 and performs:

1. dependency installation
2. Python source compilation
3. unit tests
4. command-line smoke test

## Important methodological limitation

The **Strongly Supported / Partially Supported / Unsupported** labels are currently **heuristic interface cues based on textual similarity**.

They are **not** calibrated probabilities, entailment guarantees, or definitive factual verification.

In the dissertation, they should be described as a prototype mechanism for investigating evidence-aware interaction design. A stronger future implementation could replace the similarity heuristic with a trained natural-language-inference model and calibrate the thresholds on an annotated dataset.

## Reproducibility note

Two retrieval modes are intentionally provided:

- **Sentence-Transformers + FAISS** — preferred dissertation architecture
- **TF-IDF + cosine similarity** — portable fallback for environments where semantic-model dependencies are unavailable

This separation makes the HCI interface prototype easier to reproduce while keeping the intended RAG architecture available for full demonstrations.

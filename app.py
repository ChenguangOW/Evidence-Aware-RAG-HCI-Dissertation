from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidence_rag.document import load_uploaded_file, chunk_pages
from evidence_rag.retrieval import make_retriever
from evidence_rag.generation import generate_answer
from evidence_rag.evidence import analyse_answer
from evidence_rag.logging_utils import append_evaluation

st.set_page_config(page_title="Evidence-Aware RAG Academic Assistant", page_icon="📚", layout="wide")

st.title("Evidence-Aware RAG Academic Assistant")
st.caption("MSc HCI dissertation prototype — comparing answer-only, citation-based, and evidence-aware interfaces.")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = None

with st.sidebar:
    st.header("System setup")
    uploaded = st.file_uploader("Upload an academic PDF or TXT", type=["pdf", "txt"])
    backend_label = st.selectbox(
        "Retrieval backend",
        ["Sentence-Transformers + FAISS", "TF-IDF + cosine similarity"],
        index=0,
        help="Sentence-Transformers + FAISS matches the dissertation architecture. TF-IDF is a lightweight fallback.",
    )
    top_k = st.slider("Retrieved passages (top-k)", 2, 8, 4)
    api_key = st.text_input("OpenAI API key (optional)", type="password", value="", help="Leave blank to run the local extractive fallback.")
    model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))

    if st.button("Index document", use_container_width=True, disabled=uploaded is None):
        try:
            with st.spinner("Extracting and indexing document..."):
                pages = load_uploaded_file(uploaded)
                chunks = chunk_pages(pages)
                backend = "sbert" if backend_label.startswith("Sentence") else "tfidf"
                retriever = make_retriever(backend)
                try:
                    retriever.build(chunks)
                except RuntimeError:
                    if backend == "sbert":
                        st.warning("Sentence-Transformers/FAISS is unavailable. Falling back to TF-IDF so the prototype can still run.")
                        retriever = make_retriever("tfidf")
                        retriever.build(chunks)
                    else:
                        raise
                st.session_state.retriever = retriever
                st.session_state.document_name = uploaded.name
                st.session_state.last_answer = None
                st.session_state.last_retrieved = None
            st.success(f"Indexed {len(chunks)} chunks using {st.session_state.retriever.name}.")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

if st.session_state.retriever is None:
    st.info("Upload a document and click **Index document** to begin.")
    st.stop()

st.success(f"Document ready: **{st.session_state.document_name}**  \nRetriever: **{st.session_state.retriever.name}**")

condition = st.radio("Interface condition", ["A — Basic Answer", "B — Citation-Based", "C — Evidence-Aware"], horizontal=True)
question = st.text_input("Academic question", placeholder="What are the main principles of user-centred design?")

if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Retrieving source evidence..."):
            retrieved = st.session_state.retriever.retrieve(question, top_k=top_k)
        with st.spinner("Generating grounded answer..."):
            answer, generation_mode = generate_answer(
                question=question,
                retrieved=retrieved,
                api_key=api_key.strip() or os.getenv("OPENAI_API_KEY"),
                model=model.strip() or "gpt-5.6-luna",
            )
        st.session_state.last_answer = answer
        st.session_state.last_retrieved = retrieved
        st.session_state.last_question = question
        st.session_state.last_generation_mode = generation_mode
    except Exception as exc:
        st.error(f"Question answering failed: {exc}")

answer = st.session_state.last_answer
retrieved = st.session_state.last_retrieved

if answer and retrieved:
    st.divider()
    st.caption(f"Generation mode: {st.session_state.last_generation_mode}")

    if condition.startswith("A"):
        st.subheader("AI Answer")
        with st.container(border=True):
            st.write(answer)
        st.caption("No source evidence is shown in Version A.")

    elif condition.startswith("B"):
        st.subheader("AI Answer")
        with st.container(border=True):
            st.write(answer)
        st.subheader("Sources")
        for i, item in enumerate(retrieved, start=1):
            with st.expander(f"[{i}] Page {item.page} — retrieval score {item.score:.3f}"):
                st.write(item.text)

    else:
        st.subheader("AI Answer")
        with st.container(border=True):
            st.write(answer)
        st.subheader("Claim-by-Claim Evidence")
        analysis = analyse_answer(answer, retrieved)
        if not analysis:
            st.warning("No sentence-like claims were detected.")
        else:
            for i, item in enumerate(analysis, start=1):
                with st.container(border=True):
                    st.markdown(f"### Claim {i}")
                    st.write(item.claim)
                    if item.label == "Strongly Supported":
                        st.success(f"{item.label} — similarity {item.similarity:.2f}")
                    elif item.label == "Partially Supported":
                        st.warning(f"{item.label} — similarity {item.similarity:.2f}")
                    else:
                        st.error(f"{item.label} — similarity {item.similarity:.2f}")
                    st.markdown(f"**Best source evidence — Page {item.page}**")
                    st.info(item.evidence)
        st.caption("The support label is a heuristic interface cue based on textual similarity. It is not a calibrated probability or definitive factual proof.")

    st.divider()
    st.subheader("Participant Evaluation (optional)")
    participant_id = st.text_input("Participant ID", placeholder="P01")
    c1, c2, c3 = st.columns(3)
    with c1:
        trust = st.slider("Trust", 1, 5, 3)
    with c2:
        clarity = st.slider("Clarity", 1, 5, 3)
    with c3:
        verification = st.slider("Verification ease", 1, 5, 3)
    comment = st.text_area("Comment")

    if st.button("Save evaluation response"):
        if not participant_id.strip():
            st.error("Enter an anonymous participant ID first.")
        else:
            append_evaluation(
                csv_path=ROOT / "data" / "evaluation_log.csv",
                participant_id=participant_id,
                condition=condition,
                question=st.session_state.last_question,
                trust=trust,
                clarity=clarity,
                verification_ease=verification,
                comment=comment,
            )
            st.success("Saved to data/evaluation_log.csv")

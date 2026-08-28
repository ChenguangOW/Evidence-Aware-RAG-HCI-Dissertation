from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.document import Page, chunk_pages
from evidence_rag.retrieval import TfidfRetriever
from evidence_rag.generation import generate_answer
from evidence_rag.evidence import analyse_answer

text = """
User-centred design is an iterative design approach that focuses on users and their needs.
Designers should understand users, tasks, and environments before making design decisions.
Evaluation with users can reveal usability problems and improve a system.
"""

chunks = chunk_pages([Page(1, text)])
retriever = TfidfRetriever()
retriever.build(chunks)

question = "What is user-centred design?"
retrieved = retriever.retrieve(question, top_k=3)
answer, mode = generate_answer(question, retrieved, api_key=None)

print("QUESTION:", question)
print("ANSWER:", answer)
print("MODE:", mode)
print("\nCLAIM EVIDENCE:")
for item in analyse_answer(answer, retrieved):
    print(
        f"- {item.label} | page={item.page} | similarity={item.similarity:.3f}\n"
        f"  claim: {item.claim}\n"
        f"  evidence: {item.evidence[:150]}..."
    )

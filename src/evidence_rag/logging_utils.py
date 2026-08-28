from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


FIELDNAMES = [
    "timestamp_utc",
    "participant_id",
    "condition",
    "question",
    "trust",
    "clarity",
    "verification_ease",
    "comment",
]


def append_evaluation(
    csv_path: str | Path,
    participant_id: str,
    condition: str,
    question: str,
    trust: int,
    clarity: int,
    verification_ease: int,
    comment: str,
) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "participant_id": participant_id.strip(),
                "condition": condition,
                "question": question.strip(),
                "trust": int(trust),
                "clarity": int(clarity),
                "verification_ease": int(verification_ease),
                "comment": comment.strip(),
            }
        )

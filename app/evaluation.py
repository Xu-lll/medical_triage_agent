from __future__ import annotations

import json
from pathlib import Path

from .agent import MedicalTriageAgent
from .database import DEFAULT_DB_PATH, ROOT, init_db
from .vector_store import DEFAULT_VECTOR_DIR, build_vector_store


DEFAULT_EVAL_PATH = ROOT / "data" / "eval_cases.json"


def evaluate(
    eval_path: Path | str = DEFAULT_EVAL_PATH,
    use_vector: bool = True,
    rebuild: bool = False,
) -> dict:
    if rebuild:
        init_db(DEFAULT_DB_PATH)
        build_vector_store(reset=True)

    cases = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    agent = MedicalTriageAgent(
        DEFAULT_DB_PATH,
        use_llm=False,
        use_vector=use_vector,
        vector_dir=DEFAULT_VECTOR_DIR,
    )

    rows = []
    top1_hits = 0
    top3_hits = 0
    for case in cases:
        docs = agent.retriever.search(case["query"], k=5)
        ranked = [dept for dept, _ in agent._rank_departments(docs)[:3]]
        expected = case["department"]
        top1 = ranked[0] if ranked else ""
        top1_ok = top1 == expected
        top3_ok = expected in ranked
        top1_hits += int(top1_ok)
        top3_hits += int(top3_ok)
        rows.append(
            {
                "query": case["query"],
                "expected": expected,
                "top1": top1,
                "top3": ranked,
                "top1_ok": top1_ok,
                "top3_ok": top3_ok,
                "evidence": [doc.title for doc in docs[:3]],
            }
        )

    total = len(cases)
    return {
        "total": total,
        "top1_accuracy": round(top1_hits / total, 4) if total else 0,
        "top3_recall": round(top3_hits / total, 4) if total else 0,
        "vector_db": str(DEFAULT_VECTOR_DIR),
        "rows": rows,
    }

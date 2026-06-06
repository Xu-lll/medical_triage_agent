from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .database import DEFAULT_DB_PATH, load_records
from .vector_store import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_VECTOR_DIR,
    build_vector_store,
    load_vector_store,
)


@dataclass(frozen=True)
class RetrievedDoc:
    id: str
    title: str
    category: str
    department: str
    severity: str
    content: str
    source: str
    score: float
    keyword_hits: list[str]


class HybridMedicalRetriever:
    """Hybrid retriever: Chroma vector recall + keyword overlap."""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        vector_dir: Path | str = DEFAULT_VECTOR_DIR,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        use_vector: bool = True,
    ):
        self.records = load_records(db_path)
        if not self.records:
            raise RuntimeError("知识库为空，请先运行 scripts/init_db.py")
        self.records_by_id = {record["id"]: record for record in self.records}
        self.use_vector = use_vector
        self.store = None
        if use_vector:
            self.store = load_vector_store(vector_dir, model=model, base_url=base_url)
            if self.store._collection.count() == 0:
                self.store = build_vector_store(
                    db_path=db_path,
                    persist_directory=vector_dir,
                    model=model,
                    base_url=base_url,
                )

    def search(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        vector_scores = self._vector_scores(query, k=max(k, len(self.records)))
        docs: list[RetrievedDoc] = []
        for record in self.records:
            hits = [kw for kw in record["keywords"] if kw and kw in query]
            keyword_score = min(1.0, len(hits) / max(2, len(record["keywords"]) * 0.45))
            vector_score = vector_scores.get(record["id"], 0.0)
            score = 0.72 * vector_score + 0.28 * keyword_score
            if hits:
                score += 0.08 * math.log1p(len(hits))
            docs.append(
                RetrievedDoc(
                    id=record["id"],
                    title=record["title"],
                    category=record["category"],
                    department=record["department"],
                    severity=record["severity"],
                    content=record["content"],
                    source=record["source"],
                    score=round(score, 4),
                    keyword_hits=hits,
                )
            )
        return sorted(docs, key=lambda item: item.score, reverse=True)[:k]

    def _vector_scores(self, query: str, k: int) -> dict[str, float]:
        if not self.use_vector or self.store is None:
            return {record["id"]: 0.0 for record in self.records}
        scored = self.store.similarity_search_with_score(query, k=k)
        scores: dict[str, float] = {}
        for doc, distance in scored:
            record_id = doc.metadata.get("id")
            if not record_id:
                continue
            scores[str(record_id)] = 1.0 / (1.0 + max(float(distance), 0.0))
        return scores


def format_docs(docs: Iterable[RetrievedDoc]) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        hits = "、".join(doc.keyword_hits) if doc.keyword_hits else "无直接关键词命中"
        lines.append(
            f"[{i}] {doc.title} | {doc.department} | risk={doc.severity} | "
            f"score={doc.score} | hits={hits}\n{doc.content}\n来源：{doc.source}"
        )
    return "\n\n".join(lines)

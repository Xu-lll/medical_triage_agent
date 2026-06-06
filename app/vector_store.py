from __future__ import annotations

import warnings
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from .database import DEFAULT_DB_PATH, ROOT, load_records


DEFAULT_VECTOR_DIR = ROOT / "data" / "chroma_medical"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "deepseek-r1:7b"
COLLECTION_NAME = "medical_knowledge"


def build_embeddings(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> OllamaEmbeddings:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OllamaEmbeddings(model=model, base_url=base_url)


def build_vector_store(
    db_path: Path | str = DEFAULT_DB_PATH,
    persist_directory: Path | str = DEFAULT_VECTOR_DIR,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    reset: bool = False,
) -> Chroma:
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    records = load_records(db_path)
    docs = [
        Document(
            page_content=_record_text(record),
            metadata={
                "id": record["id"],
                "title": record["title"],
                "category": record["category"],
                "department": record["department"],
                "severity": record["severity"],
                "keywords": "、".join(record["keywords"]),
                "content": record["content"],
                "source": record["source"],
            },
        )
        for record in records
    ]
    ids = [record["id"] for record in records]

    embeddings = build_embeddings(model=model, base_url=base_url)
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    if reset:
        try:
            store.delete_collection()
        except Exception:
            pass
        store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )
    if store._collection.count() == 0:
        store.add_documents(docs, ids=ids)
    return store


def load_vector_store(
    persist_directory: Path | str = DEFAULT_VECTOR_DIR,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=build_embeddings(model=model, base_url=base_url),
        persist_directory=str(persist_directory),
    )


def _record_text(record: dict) -> str:
    return "\n".join(
        [
            f"标题：{record['title']}",
            f"类别：{record['category']}",
            f"推荐科室：{record['department']}",
            f"风险等级：{record['severity']}",
            f"关键词：{'、'.join(record['keywords'])}",
            f"医学知识：{record['content']}",
            f"来源：{record['source']}",
        ]
    )

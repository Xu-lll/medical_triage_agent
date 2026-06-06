from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "medical_knowledge.db"
DEFAULT_SEED_PATH = ROOT / "data" / "medical_seed.json"


SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    department TEXT NOT NULL,
    severity TEXT NOT NULL,
    keywords TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL
);
"""


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(
    db_path: Path | str = DEFAULT_DB_PATH,
    seed_path: Path | str = DEFAULT_SEED_PATH,
) -> None:
    db_path = Path(db_path)
    seed_path = Path(seed_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(seed_path.read_text(encoding="utf-8"))

    with connect(db_path) as conn:
        conn.execute(SCHEMA)
        conn.execute("DELETE FROM knowledge")
        conn.executemany(
            """
            INSERT INTO knowledge
            (id, category, title, department, severity, keywords, content, source)
            VALUES (:id, :category, :title, :department, :severity, :keywords_json, :content, :source)
            """,
            (_prepare_record(item) for item in records),
        )
        conn.commit()


def _prepare_record(item: dict) -> dict:
    prepared = dict(item)
    prepared["keywords_json"] = json.dumps(item.get("keywords", []), ensure_ascii=False)
    return prepared


def load_records(db_path: Path | str = DEFAULT_DB_PATH) -> list[dict]:
    with connect(db_path) as conn:
        rows: Iterable[sqlite3.Row] = conn.execute(
            "SELECT id, category, title, department, severity, keywords, content, source FROM knowledge"
        )
        records = [dict(row) for row in rows]
    for record in records:
        record["keywords"] = json.loads(record["keywords"])
    return records

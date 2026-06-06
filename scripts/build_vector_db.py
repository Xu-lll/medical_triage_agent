from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import DEFAULT_DB_PATH, init_db  # noqa: E402
from app.vector_store import DEFAULT_MODEL, DEFAULT_VECTOR_DIR, build_vector_store  # noqa: E402


if __name__ == "__main__":
    init_db(DEFAULT_DB_PATH)
    store = build_vector_store(reset=True)
    print(f"向量数据库已构建：{DEFAULT_VECTOR_DIR}")
    print(f"Embedding 模型：{DEFAULT_MODEL}")
    print(f"文档数量：{store._collection.count()}")

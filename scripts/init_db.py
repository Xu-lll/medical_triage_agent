from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import DEFAULT_DB_PATH, init_db  # noqa: E402


if __name__ == "__main__":
    init_db(DEFAULT_DB_PATH)
    print(f"数据库已初始化：{DEFAULT_DB_PATH}")

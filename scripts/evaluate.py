from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation import evaluate  # noqa: E402


if __name__ == "__main__":
    metrics = evaluate(rebuild=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

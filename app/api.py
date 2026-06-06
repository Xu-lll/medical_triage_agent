from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .agent import MedicalTriageAgent
from .database import DEFAULT_DB_PATH, init_db
from .langchain_agent import LangChainMedicalAgent
from .retriever import format_docs
from .vector_store import DEFAULT_MODEL, DEFAULT_VECTOR_DIR, build_vector_store


APP_DIR = Path(__file__).resolve().parent
RESPONSE_LOG_PATH = APP_DIR.parent / "data" / "response_times.jsonl"
MAX_CHAT_SESSIONS = 64
_response_log_lock = threading.Lock()
_session_lock = threading.Lock()
_session_agents: dict[str, MedicalTriageAgent] = {}
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app = FastAPI(title="智能医疗分诊与建议智能体", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


@app.on_event("startup")
def ensure_database_ready():
    if not Path(DEFAULT_DB_PATH).exists():
        init_db(DEFAULT_DB_PATH)


@app.middleware("http")
async def record_response_time(request: Request, call_next):
    started_at = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
    _append_response_time(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
        }
    )
    return response


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    session_id: str | None = Field(default=None, max_length=80)
    knowledge_mode: str = Field(default="rag", pattern="^(rag|llm)$")
    use_llm: bool = True
    use_agent_executor: bool = False
    show_trace: bool = True


@lru_cache(maxsize=1)
def get_core_agent() -> MedicalTriageAgent:
    return MedicalTriageAgent(DEFAULT_DB_PATH, model=DEFAULT_MODEL, use_llm=True)


@lru_cache(maxsize=1)
def get_react_agent() -> LangChainMedicalAgent:
    return LangChainMedicalAgent(DEFAULT_DB_PATH, model=DEFAULT_MODEL)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok", "model": DEFAULT_MODEL, "chat_sessions": len(_session_agents)}


@app.post("/api/chat")
def chat(payload: ChatRequest):
    started_at = time.perf_counter()
    session_id = _normalize_session_id(payload.session_id)
    if payload.use_agent_executor:
        result = get_react_agent().invoke(payload.query)
        docs = result["docs"]
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return {
            "session_id": session_id,
            "answer": result["answer"],
            "trace": result["steps"] if payload.show_trace else [],
            "docs": [_doc_to_dict(doc) for doc in _visible_docs(docs)],
            "context": format_docs(docs),
            "response_time_ms": elapsed_ms,
        }

    agent = get_session_agent(session_id)
    original_llm = agent.llm
    if not payload.use_llm:
        agent.llm = None
    result = agent.ask(
        payload.query,
        show_trace=payload.show_trace,
        knowledge_mode=payload.knowledge_mode,
    )
    agent.llm = original_llm
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    return {
        "session_id": session_id,
        "answer": result.answer,
        "trace": result.trace,
        "docs": [_doc_to_dict(doc) for doc in _visible_docs(result.docs)],
        "context": format_docs(result.docs),
        "response_time_ms": elapsed_ms,
        "agent_response_time_ms": result.response_time_ms,
    }


@app.post("/api/rebuild")
def rebuild():
    get_core_agent.cache_clear()
    get_react_agent.cache_clear()
    with _session_lock:
        _session_agents.clear()
    init_db(DEFAULT_DB_PATH)
    store = build_vector_store(reset=True)
    return {"status": "rebuilt", "vector_dir": str(DEFAULT_VECTOR_DIR), "documents": store._collection.count()}


@app.get("/api/metrics")
def metrics():
    from .evaluation import evaluate

    return evaluate(use_vector=True, rebuild=False)


@app.post("/api/session/{session_id}/reset")
def reset_session(session_id: str):
    session_id = _normalize_session_id(session_id)
    with _session_lock:
        _session_agents.pop(session_id, None)
    return {"status": "reset", "session_id": session_id}


def _doc_to_dict(doc):
    return {
        "id": doc.id,
        "title": doc.title,
        "department": doc.department,
        "severity": doc.severity,
        "score": doc.score,
        "keyword_hits": doc.keyword_hits,
        "source": doc.source,
        "content": doc.content,
    }


def _visible_docs(docs):
    visible = [doc for doc in docs if doc.keyword_hits or doc.score > 0]
    return visible or docs[:1]


def _append_response_time(record: dict) -> None:
    RESPONSE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _response_log_lock:
        with RESPONSE_LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(line + "\n")


def _normalize_session_id(session_id: str | None) -> str:
    value = (session_id or "default").strip()
    value = "".join(char for char in value if char.isalnum() or char in {"-", "_"})
    return value[:80] or "default"


def get_session_agent(session_id: str) -> MedicalTriageAgent:
    with _session_lock:
        agent = _session_agents.get(session_id)
        if agent is None:
            if len(_session_agents) >= MAX_CHAT_SESSIONS:
                oldest_session = next(iter(_session_agents))
                _session_agents.pop(oldest_session, None)
            agent = MedicalTriageAgent(DEFAULT_DB_PATH, model=DEFAULT_MODEL, use_llm=True)
            _session_agents[session_id] = agent
        return agent


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)

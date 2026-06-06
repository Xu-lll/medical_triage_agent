from __future__ import annotations

import argparse

from .agent import MedicalTriageAgent
from .database import DEFAULT_DB_PATH, init_db


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="智能医疗分诊与建议智能体")
    parser.add_argument("--init-db", action="store_true", help="初始化 SQLite 医学知识库")
    parser.add_argument("--build-vector-db", action="store_true", help="构建 Chroma 向量数据库")
    parser.add_argument("--query", type=str, help="单轮问诊输入")
    parser.add_argument("--trace", action="store_true", help="显示 ReAct 风格推理/工具调用轨迹")
    parser.add_argument("--no-llm", action="store_true", help="不调用 Ollama LLM，仅使用模板兜底回答")
    parser.add_argument("--agent-executor", action="store_true", help="使用 LangChain Tool + AgentExecutor ReAct 智能体")
    parser.add_argument("--model", default="deepseek-r1:7b", help="Ollama 模型名")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.init_db:
        init_db(DEFAULT_DB_PATH)
        print(f"数据库已初始化：{DEFAULT_DB_PATH}")

    if args.build_vector_db:
        from .vector_store import DEFAULT_VECTOR_DIR, build_vector_store

        build_vector_store(DEFAULT_DB_PATH, model=args.model, reset=True)
        print(f"向量数据库已构建：{DEFAULT_VECTOR_DIR}")

    if args.agent_executor:
        from .langchain_agent import LangChainMedicalAgent

        react_agent = LangChainMedicalAgent(DEFAULT_DB_PATH, model=args.model)
        if args.query:
            result = react_agent.invoke(args.query)
            if args.trace:
                print("\n===== AgentExecutor Steps =====")
                for step in result["steps"]:
                    print(f"Action: {step['tool']}\nInput: {step['tool_input']}\nObservation: {step['observation']}\n")
                print("===== Answer =====")
            print(result["answer"])
            return

    agent = MedicalTriageAgent(DEFAULT_DB_PATH, model=args.model, use_llm=not args.no_llm)
    if args.query:
        result = agent.ask(args.query, show_trace=args.trace)
        if result.trace:
            print("\n===== ReAct Trace =====")
            print("\n".join(result.trace))
            print("\n===== Answer =====")
        print(result.answer)
        return

    print("智能医疗分诊 Demo。输入症状描述，输入 exit 退出。")
    while True:
        query = input("\n患者描述> ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue
        print(agent.ask(query).answer)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import warnings
from pathlib import Path

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.llms import Ollama

from .agent import MedicalTriageAgent, screen_emergency
from .database import DEFAULT_DB_PATH
from .retriever import RetrievedDoc, format_docs
from .vector_store import DEFAULT_MODEL, DEFAULT_OLLAMA_URL, DEFAULT_VECTOR_DIR


REACT_PROMPT = PromptTemplate.from_template(
    """你是一个医疗分诊 ReAct Agent。你只能调用工具获得依据，不能直接编造医学事实。

可用工具：
{tools}

工具名称：{tool_names}

必须遵循格式：
Question: 用户问题
Thought: 你下一步要做什么
Action: 工具名称
Action Input: 工具输入
Observation: 工具返回
... 可以重复 Thought/Action/Action Input/Observation
Thought: 我已经获得足够依据
Final Answer: 最终回答

要求：
1. 先调用 emergency_screen，再调用 medical_search，再调用 department_rank。
2. 如果信息不足，可以调用 followup_questions。
3. 最终回答必须使用 compose_answer 工具生成，不要自己写医疗建议。
4. 中文输出。

Question: {input}
{agent_scratchpad}"""
)


class LangChainMedicalAgent:
    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        vector_dir: Path | str = DEFAULT_VECTOR_DIR,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        use_vector: bool = True,
    ):
        self.core = MedicalTriageAgent(
            db_path=db_path,
            vector_dir=vector_dir,
            model=model,
            base_url=base_url,
            use_llm=False,
            use_vector=use_vector,
        )
        self.last_docs: list[RetrievedDoc] = []
        self.last_query = ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.llm = Ollama(
                model=model,
                base_url=base_url,
                temperature=0.0,
                num_ctx=4096,
            )
        self.core.llm = self.llm
        self.tools = self._build_tools()
        runnable = create_react_agent(self.llm, self.tools, REACT_PROMPT)
        self.executor = AgentExecutor(
            agent=runnable,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=8,
            return_intermediate_steps=True,
        )

    def invoke(self, query: str) -> dict:
        self.last_query = query
        self.last_docs = []
        try:
            result = self.executor.invoke({"input": query})
            answer = str(result.get("output", "")).strip()
            steps = self._serialize_steps(result.get("intermediate_steps", []))
            tool_names = {step["tool"] for step in steps}
            if (
                not answer
                or "Agent stopped" in answer
                or "medical_search" not in tool_names
                or "compose_answer" not in tool_names
            ):
                guarded = self._invoke_guarded_tools(query)
                return {
                    "answer": guarded["answer"],
                    "steps": steps + guarded["steps"],
                    "docs": self.last_docs,
                }
            return {"answer": answer, "steps": steps, "docs": self.last_docs}
        except Exception as exc:
            guarded = self._invoke_guarded_tools(query)
            guarded["steps"].insert(0, {"tool": "agent_executor_error", "observation": str(exc)})
            return {
                "answer": guarded["answer"],
                "steps": guarded["steps"],
                "docs": self.last_docs,
            }

    def _invoke_guarded_tools(self, query: str) -> dict:
        steps = []
        for tool_name, tool_input in [
            ("emergency_screen", query),
            ("medical_search", query),
            ("department_rank", query),
            ("followup_questions", query),
            ("compose_answer", query),
        ]:
            tool = next(item for item in self.tools if item.name == tool_name)
            observation = tool.run(tool_input)
            steps.append(
                {
                    "tool": tool_name,
                    "tool_input": tool_input,
                    "observation": str(observation)[:1200],
                }
            )
        return {"answer": steps[-1]["observation"], "steps": steps}

    def _build_tools(self) -> list[Tool]:
        return [
            Tool.from_function(
                name="emergency_screen",
                description="识别患者描述中的急症红旗词，输入为患者自然语言描述。",
                func=self._tool_emergency_screen,
            ),
            Tool.from_function(
                name="medical_search",
                description="调用 Chroma 向量库和关键词混合检索医学知识，输入为患者自然语言描述。",
                func=self._tool_medical_search,
            ),
            Tool.from_function(
                name="department_rank",
                description="根据最近一次检索结果输出 Top-3 推荐科室，输入任意文本。",
                func=self._tool_department_rank,
            ),
            Tool.from_function(
                name="followup_questions",
                description="当问诊信息不足时生成追问问题，输入为患者自然语言描述。",
                func=self._tool_followup_questions,
            ),
            Tool.from_function(
                name="compose_answer",
                description="基于最近一次检索结果生成最终分诊回答，输入为患者自然语言描述。",
                func=self._tool_compose_answer,
            ),
        ]

    def _tool_emergency_screen(self, query: str) -> str:
        hits = screen_emergency(query)
        return json.dumps(
            {
                "emergency_hits": hits,
                "has_emergency_risk": bool(hits),
                "next_action": "medical_search",
            },
            ensure_ascii=False,
        )

    def _tool_medical_search(self, query: str) -> str:
        self.last_query = query
        self.last_docs = self.core.retriever.search(query, k=5)
        return format_docs(self.last_docs)

    def _tool_department_rank(self, _: str = "") -> str:
        docs = self.last_docs or self.core.retriever.search(self.last_query, k=5)
        ranked = self.core._rank_departments(docs)
        return json.dumps(
            [{"department": dept, "score": round(score, 4)} for dept, score in ranked[:3]],
            ensure_ascii=False,
        )

    def _tool_followup_questions(self, query: str) -> str:
        docs = self.last_docs or self.core.retriever.search(query, k=3)
        top_departments = [dept for dept, _ in self.core._rank_departments(docs)[:2]]
        questions = ["症状从什么时候开始，是否持续加重？", "有没有发热、胸痛、呼吸困难、意识改变等危险信号？"]
        if "消化内科" in top_departments or "普外科" in top_departments:
            questions.append("腹痛具体位置在哪里，是否伴呕吐、黑便或血便？")
        elif "呼吸内科" in top_departments:
            questions.append("是否有咳痰、气促、血氧下降或基础肺病？")
        else:
            questions.append("是否有既往病史、过敏史或正在使用的药物？")
        return "\n".join(f"{idx}. {item}" for idx, item in enumerate(questions, 1))

    def _tool_compose_answer(self, query: str) -> str:
        docs = self.last_docs or self.core.retriever.search(query, k=5)
        fallback = self.core._compose_dialog_answer(query, query, docs)
        return self.core._llm_dialog_answer(query, query, docs, fallback) if self.core.llm else fallback

    @staticmethod
    def _serialize_steps(steps: list) -> list[dict]:
        serialized = []
        for action, observation in steps:
            serialized.append(
                {
                    "tool": getattr(action, "tool", ""),
                    "tool_input": getattr(action, "tool_input", ""),
                    "observation": str(observation)[:1200],
                }
            )
        return serialized

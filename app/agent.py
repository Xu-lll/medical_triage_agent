from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

try:
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_community.llms import Ollama
except Exception:  # pragma: no cover
    ConversationBufferWindowMemory = None
    Ollama = None

from .database import DEFAULT_DB_PATH
from .retriever import HybridMedicalRetriever, RetrievedDoc, format_docs
from .vector_store import DEFAULT_MODEL, DEFAULT_OLLAMA_URL, DEFAULT_VECTOR_DIR


EMERGENCY_WORDS = [
    "晕厥",
    "昏厥",
    "意识障碍",
    "意识改变",
    "意识不清",
    "说话不清",
    "一侧无力",
    "口角歪斜",
    "偏瘫",
    "抽搐",
    "大出血",
    "出血不止",
    "血氧低",
    "血氧下降",
    "剧烈腹痛",
    "突发最严重头痛",
    "口唇发紫",
    "呼吸明显费力",
    "轻生",
    "自伤",
]

EMERGENCY_COMBINATIONS = [
    ("胸痛", ["大汗", "出汗", "冷汗", "左肩", "左臂", "肩背", "下颌", "压榨", "呼吸困难", "气短", "晕厥"]),
    ("呼吸困难", ["胸痛", "血氧", "口唇发紫", "说话困难", "喘憋", "明显费力"]),
    ("腹痛", ["剧烈", "右下腹", "反跳痛", "黑便", "血便", "呕血", "持续加重"]),
    ("头痛", ["突发", "最严重", "一侧无力", "说话不清", "意识", "视物异常"]),
]

SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "emergency": 4}


@dataclass
class AgentAnswer:
    answer: str
    trace: list[str] = field(default_factory=list)
    docs: list[RetrievedDoc] = field(default_factory=list)
    response_time_ms: float = 0.0


class MedicalTriageAgent:
    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        memory_k: int = 4,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        vector_dir: Path | str = DEFAULT_VECTOR_DIR,
        use_llm: bool = True,
        use_vector: bool = True,
    ):
        self.retriever = HybridMedicalRetriever(
            db_path=db_path,
            vector_dir=vector_dir,
            model=model,
            base_url=base_url,
            use_vector=use_vector,
        )
        self.long_summary = ""
        self.llm = None
        if use_llm and Ollama:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.llm = Ollama(
                    model=model,
                    base_url=base_url,
                    temperature=0.2,
                    num_ctx=4096,
                )
        self.memory = None
        if ConversationBufferWindowMemory:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.memory = ConversationBufferWindowMemory(
                    k=memory_k, return_messages=False
                )
        self.turns: list[tuple[str, str]] = []

    def ask(self, user_input: str, show_trace: bool = False) -> AgentAnswer:
        started_at = time.perf_counter()
        trace = [
            "Thought: 先判断是否有急症红旗，再调用医学知识库做混合检索。",
            "Action: hybrid_medical_search(query)",
        ]
        docs = self.retriever.search(user_input, k=5)
        trace.append("Observation:\n" + format_docs(docs))
        trace.append("Thought: 根据检索结果、风险等级和历史对话生成分诊建议。")

        base_answer = self._compose_answer(user_input, docs)
        answer = self._llm_answer(user_input, docs, base_answer) if self.llm else ""
        if not answer:
            answer = base_answer
        self._save_context(user_input, answer)
        if not show_trace:
            trace = []
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        return AgentAnswer(answer=answer, trace=trace, docs=docs, response_time_ms=round(elapsed_ms, 2))

    def _llm_answer(self, query: str, docs: list[RetrievedDoc], base_answer: str) -> str:
        prompt = self._build_prompt(query, docs, base_answer)
        try:
            raw = self.llm.invoke(prompt)
        except Exception:
            return ""
        supplement = self._guard_answer(
            self._strip_deepseek_think(str(raw)).strip(), require_disclaimer=False
        )
        if not supplement:
            return base_answer
        return f"{base_answer}\n\n【LLM 补充说明】\n{supplement}"

    def _build_prompt(
        self, query: str, docs: list[RetrievedDoc], deterministic_draft: str
    ) -> str:
        history = "\n".join(
            f"患者：{user}\n助手：{assistant}" for user, assistant in self.turns[-3:]
        )
        emergency_hit = screen_emergency(query)
        context = format_docs(docs)
        return f"""你是一个用于线上问诊分诊的医疗智能体。请严格基于【检索知识】和【系统计算的分诊草稿】生成补充说明，不要编造诊断。

要求：
1. 不要重复完整报告，只输出“补充说明”。
2. 必须沿用【系统计算的分诊草稿】里的优先级、推荐科室和 Top-3，不要自行改高或改低风险。
3. 如果出现急症红旗词，必须建议急诊或拨打急救电话。
4. 不要建议“开具”任何药物，不要给出处方剂量；用药只能提示遵医嘱、咨询医生/药师或阅读说明书。
5. 不得新增【检索知识】和【系统计算的分诊草稿】之外的症状、疾病或危险信号。
6. 不要出现英文夹杂。
7. 用中文回答，控制在 180 字以内。

【急症红旗命中】
{", ".join(emergency_hit) if emergency_hit else "无"}

【长期摘要】
{self.long_summary or "无"}

【近期对话】
{history or "无"}

【患者本轮描述】
{query}

【检索知识】
{context}

【系统计算的分诊草稿】
{deterministic_draft}

请给出补充说明："""

    @staticmethod
    def _strip_deepseek_think(text: str) -> str:
        if "</think>" in text:
            return text.split("</think>", 1)[1]
        return text

    @staticmethod
    def _guard_answer(text: str, require_disclaimer: bool = True) -> str:
        if require_disclaimer and "仅供分诊参考" not in text and "不能替代医生" not in text:
            text = "以下内容仅供分诊参考，不能替代医生面诊或急救判断。\n\n" + text
        unsafe_phrases = {
            "建议医生开具对乙酰氨基酚": "如需使用退热止痛药，请咨询医生或药师，并阅读药品说明书",
            "建议开具对乙酰氨基酚": "如需使用退热止痛药，请咨询医生或药师，并阅读药品说明书",
            "建议使用对乙酰氨基酚": "如需使用退热止痛药，请咨询医生或药师，并阅读药品说明书",
            "或体重下降": "",
            "、体重下降": "",
            "体重下降等": "",
        }
        for unsafe, safe in unsafe_phrases.items():
            text = text.replace(unsafe, safe)
        return text

    def _compose_answer(self, query: str, docs: list[RetrievedDoc]) -> str:
        emergency_hit = screen_emergency(query)
        top = docs[0]
        top_departments = self._rank_departments(docs)
        has_evidence = any(doc.keyword_hits or doc.score > 0 for doc in docs)
        if not has_evidence and not emergency_hit:
            return "\n".join(
                [
                    "以下内容仅用于线上分诊和健康信息参考，不能替代医生面诊或急救判断。",
                    "",
                    "【优先级】信息不足，建议补充描述后再评估",
                    "目前没有从知识库中匹配到足够可靠的症状线索，因此不直接推荐急诊科。",
                    "可补充症状开始时间、部位、严重程度、体温、诱因、既往病史和正在使用的药物；如不确定可先选择全科/普通内科或线上问诊初筛。",
                    "",
                    "【Top-3 分诊推荐】",
                    "1. 全科/普通内科（综合匹配度 0.00）",
                    "",
                    "【初步建议】",
                    "- 若出现胸痛伴大汗或放射痛、呼吸明显费力、意识改变、一侧无力、说话不清、大出血、血氧下降、持续高热或症状快速加重，请及时急诊。",
                    "- 用药请遵医嘱或药品说明书；儿童、孕妇、老人和慢病患者建议更谨慎。",
                ]
            )
        significant_docs = [
            doc for doc in docs if doc.score >= max(0.22, top.score * 0.45)
        ] or [top]
        risk = max(significant_docs, key=lambda doc: SEVERITY_RANK.get(doc.severity, 0))

        lines = [
            "以下内容仅用于线上分诊和健康信息参考，不能替代医生面诊或急救判断。",
            "",
        ]
        if emergency_hit:
            lines.extend(
                [
                    "【优先级】急症优先",
                    f"你的描述中出现了可能需要立即处理的信号：{', '.join(emergency_hit) or risk.title}。",
                    "建议立即前往急诊科，症状明显或持续加重时请拨打当地急救电话。",
                    "",
                ]
            )
        elif risk.severity in {"high", "emergency"}:
            lines.extend(
                [
                    "【优先级】建议尽快就医",
                    f"知识库提示需要重点排查：{risk.title}。",
                    f"推荐科室：{risk.department}；如疼痛、发热或呼吸困难加重，请改走急诊。",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "【优先级】可先门诊/线上问诊评估",
                    f"最匹配的方向是：{top.title}。",
                    f"推荐科室：{top.department}。",
                    "",
                ]
            )

        lines.extend(
            [
                "【Top-3 分诊推荐】",
                *[
                    f"{idx}. {dept}（综合匹配度 {score:.2f}）"
                    for idx, (dept, score) in enumerate(top_departments[:3], start=1)
                ],
                "",
                "【依据】",
            ]
        )
        for doc in docs[:3]:
            hits = "、".join(doc.keyword_hits) if doc.keyword_hits else "语义相似"
            lines.append(f"- {doc.title}：{doc.content}（命中：{hits}；来源：{doc.source}）")

        lines.extend(
            [
                "",
                "【初步建议】",
                "- 记录体温、症状开始时间、诱因、既往病史和正在使用的药物。",
                "- 若出现胸痛、呼吸困难、意识改变、持续高热、严重脱水、血便或症状快速加重，请及时急诊。",
                "- 用药请遵医嘱或药品说明书；儿童、孕妇、老人和慢病患者建议更谨慎。",
            ]
        )

        if self.long_summary:
            lines.extend(["", f"【长期摘要】{self.long_summary}"])

        return "\n".join(lines)

    @staticmethod
    def _rank_departments(docs: list[RetrievedDoc]) -> list[tuple[str, float]]:
        if not docs:
            return []
        top_score = docs[0].score
        significant_docs = [
            doc for doc in docs if doc.score >= max(0.08, top_score * 0.55)
        ] or [docs[0]]
        significant_docs = [
            doc for doc in significant_docs if doc.keyword_hits or doc.score > 0
        ]
        if not significant_docs:
            return [("全科/普通内科", 0.0)]
        scores: dict[str, float] = {}
        for doc in significant_docs:
            severity_weight = 1.0 + 0.15 * SEVERITY_RANK.get(doc.severity, 0)
            keyword_bonus = 1.0 + 0.08 * len(doc.keyword_hits)
            scores[doc.department] = scores.get(doc.department, 0.0) + doc.score * severity_weight * keyword_bonus
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def _save_context(self, query: str, answer: str) -> None:
        self.turns.append((query, answer[:180]))
        if len(self.turns) > 8:
            older = self.turns[:-4]
            symptoms = []
            for user_text, _ in older[-4:]:
                symptoms.append(user_text[:40])
            self.long_summary = "；".join(symptoms)
            self.turns = self.turns[-4:]
        if self.memory:
            self.memory.save_context({"input": query}, {"output": answer})


def screen_emergency(query: str) -> list[str]:
    normalized = re.sub(r"\s+", "", query)
    hits = [word for word in EMERGENCY_WORDS if word in normalized]
    for primary, companions in EMERGENCY_COMBINATIONS:
        if primary in normalized:
            matched = [word for word in companions if word in normalized]
            if matched:
                hits.append(f"{primary}+{matched[0]}")
    return list(dict.fromkeys(hits))

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

GENERAL_MEDICAL_WORDS = [
    "症状",
    "病",
    "疼",
    "痛",
    "痒",
    "发热",
    "发烧",
    "咳",
    "吐",
    "泻",
    "晕",
    "药",
    "医院",
    "医生",
    "科",
    "检查",
    "治疗",
    "怎么办",
    "严重",
    "几天",
    "昨天",
    "今天",
    "前天",
    "便血",
    "血便",
    "大便带血",
    "口腔",
    "嘴",
    "嘴唇",
    "牙龈",
    "水泡",
    "疱疹",
    "溃疡",
]

MEDICATION_WORDS = ["吃什么药", "用什么药", "该吃药", "吃药", "用药", "药", "布洛芬", "对乙酰氨基酚"]
DEPARTMENT_WORDS = ["哪个科", "什么科", "哪个诊室", "什么诊室", "挂什么号", "挂哪个号", "去哪个科", "看什么科", "该去什么诊室"]


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

    def ask(
        self,
        user_input: str,
        show_trace: bool = False,
        knowledge_mode: str = "rag",
    ) -> AgentAnswer:
        started_at = time.perf_counter()
        context_query = self._contextual_query(user_input)
        if knowledge_mode == "llm":
            trace = [
                "Thought: 使用纯 LLM 对话模式，不调用本地医学知识库。",
                f"Context: {context_query}",
            ]
            answer = self._llm_free_dialog_answer(user_input, context_query) if self.llm else ""
            if not answer:
                answer = "纯 LLM 模式需要启用并连接本地 Ollama 模型。请勾选“使用 LLM”，或切换到 RAG 知识库增强模式。"
            self._save_context(user_input, answer)
            if not show_trace:
                trace = []
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            return AgentAnswer(answer=answer, trace=trace, docs=[], response_time_ms=round(elapsed_ms, 2))

        search_query = self._expand_query_terms(context_query)
        trace = [
            "Thought: 先识别用户意图，并结合近期对话补全本轮检索输入。",
            f"Action: hybrid_medical_search({search_query})",
        ]
        docs = self.retriever.search(search_query, k=5)
        docs = self._filter_context_docs(search_query, docs)
        trace.append("Observation:\n" + format_docs(docs))
        trace.append("Thought: 根据意图、检索证据、风险等级和历史对话生成对话式回复。")

        fallback_answer = self._compose_dialog_answer(user_input, search_query, docs)
        answer = self._llm_dialog_answer(user_input, search_query, docs, fallback_answer) if self.llm else ""
        if not answer:
            answer = fallback_answer
        self._save_context(user_input, answer)
        if not show_trace:
            trace = []
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        return AgentAnswer(answer=answer, trace=trace, docs=docs, response_time_ms=round(elapsed_ms, 2))

    def _compose_dialog_answer(
        self, query: str, context_query: str, docs: list[RetrievedDoc]
    ) -> str:
        emergency_hit = screen_emergency(context_query)
        has_evidence = any(doc.keyword_hits or doc.score > 0 for doc in docs)
        has_context = bool(self.turns)
        has_medical_signal = self._has_medical_signal(query) or has_evidence

        if not has_medical_signal and not has_context:
            return (
                "你好，我可以像问诊助手一样和你多轮沟通，帮你梳理症状、判断大致就诊方向，"
                "也可以回答用药注意事项和就医准备。\n\n"
                "你可以先告诉我：哪里不舒服、从什么时候开始、有没有发热/疼痛/呼吸困难等情况。"
            )

        if emergency_hit:
            return self._compose_answer(context_query, docs)

        if self._is_department_question(query):
            return self._compose_department_answer(context_query, docs)

        if self._is_medication_question(query):
            return self._compose_medication_answer(query, context_query, docs)

        if not has_evidence:
            return self._compose_followup_only_answer(has_context)

        risk = self._select_risk(docs)
        top = docs[0]
        questions = self._filter_followup_questions(context_query, self._followup_questions(docs))
        if risk.severity in {"high", "emergency"}:
            opening = "收到，这条信息我会作为补充。" if has_context and not self._has_specific_symptom(query) else "根据你目前描述的信息，我先不直接下诊断，但需要更谨慎一些。"
            return "\n".join(
                [
                    opening,
                    f"目前更需要排查：{risk.title}；建议优先考虑 {risk.department}，症状明显或加重时及时急诊。",
                    "",
                    "为了进一步判断，请继续补充：",
                    *[f"{idx}. {item}" for idx, item in enumerate(questions, start=1)],
                    "",
                    "如果出现胸痛伴大汗或放射痛、呼吸明显费力、意识改变、一侧无力、说话不清、血氧下降等情况，请立即急诊。",
                ]
            )

        opening = "收到，这条信息我会作为补充；你继续补充后我可以继续判断。" if has_context and not self._has_specific_symptom(query) else "我先根据现有信息做一个阶段性判断，后面你补充细节后我可以继续调整建议。"
        return "\n".join(
            [
                opening,
                f"目前更像是：{top.title} 相关问题；可先考虑 {top.department}。",
                "【优先级】可先门诊/线上问诊评估。",
                "如果发热持续升高、精神状态变差、剧烈头痛、反复呕吐、皮疹、颈部僵硬或症状继续加重，建议尽快线下就医。",
                "",
                "接下来只需要再确认：",
                *[f"{idx}. {item}" for idx, item in enumerate(questions[:2], start=1)],
                "",
                "用药不要自行叠加多种药物；如果你告诉我年龄、体温、既往病史和正在用的药，我可以继续帮你整理更安全的就医/用药咨询要点。",
            ]
        )

    def _compose_medication_answer(
        self, query: str, context_query: str, docs: list[RetrievedDoc]
    ) -> str:
        emergency_hit = screen_emergency(context_query)
        if emergency_hit:
            return self._compose_answer(context_query, docs)

        has_evidence = any(doc.keyword_hits or doc.score > 0 for doc in docs)
        if not has_evidence:
            return (
                "可以聊用药，但我需要先知道具体症状，不能只根据“吃什么药”直接推荐药物。\n\n"
                "请补充：主要哪里不舒服、持续多久、体温多少、年龄、是否怀孕/慢病/药物过敏、现在已经吃过什么药。"
            )

        top = docs[0]
        questions = self._followup_questions(docs)
        return "\n".join(
            [
                f"你问到用药，我先按安全原则说：目前信息更接近 {top.title}，但不能在这里给出处方或固定剂量。",
                "如果是轻症，可以先以休息、补液、观察体温和症状变化为主；退热止痛、止咳、抗过敏等药物需要结合年龄、基础病、过敏史和说明书/医生药师建议选择。",
                "",
                "继续判断前，请你补充：",
                *[f"{idx}. {item}" for idx, item in enumerate(questions, start=1)],
                "4. 年龄、是否孕期/哺乳期、肝肾疾病、胃溃疡或药物过敏史？",
                "",
                "如果出现呼吸困难、意识改变、持续高热不退、明显脱水、胸痛伴大汗或症状快速加重，请及时就医或急诊。",
            ]
        )

    def _compose_department_answer(self, context_query: str, docs: list[RetrievedDoc]) -> str:
        departments = self._rank_departments(docs)
        primary = departments[0][0] if departments else "全科/普通内科"
        oral_query = any(word in context_query for word in ["口腔", "嘴", "嘴唇", "牙龈", "口疮", "水泡", "疱疹", "溃疡"])
        if "发热" in context_query or "发烧" in context_query:
            primary = "全科/发热门诊/呼吸内科" if primary in {"全科/普通内科", "全科/呼吸内科"} else primary
        if oral_query and primary in {"全科/普通内科", "皮肤科/口腔科"}:
            primary = "口腔科"
        emergency_hit = screen_emergency(context_query)
        if emergency_hit:
            return self._compose_answer(context_query, docs)
        if oral_query:
            return "\n".join(
                [
                    f"按你目前描述，可以优先挂：{primary}。",
                    "如果医院分得更细，可选择口腔黏膜科；如果水泡在嘴唇或口周成簇，也可考虑皮肤科。",
                    "若伴发热、吞咽困难、喝水困难、眼部疼痛、全身皮疹、明显红肿流脓或精神状态变差，建议尽快线下就医，严重时急诊评估。",
                ]
            )
        return "\n".join(
            [
                f"按你目前描述，可以优先挂：{primary}。",
                "如果医院有发热门诊，发热明显时可先走发热门诊或全科初筛；如果后续出现咳嗽咽痛、流涕等呼吸道症状，可考虑呼吸内科。",
                "若出现持续高热、剧烈头痛、颈部僵硬、皮疹、意识改变、反复呕吐或精神状态变差，请不要等普通门诊，建议尽快急诊评估。",
            ]
        )

    def _compose_followup_only_answer(self, has_context: bool) -> str:
        prefix = "收到，这条信息我会作为补充。" if has_context else "目前信息还不够。"
        return "\n".join(
            [
                prefix,
                "为了更像真实问诊一样继续判断，请你再补充几个关键点：",
                "1. 主要症状是什么，最不舒服的部位在哪里？",
                "2. 从什么时候开始，是持续还是一阵一阵？",
                "3. 有没有发热、胸痛、呼吸困难、呕吐/腹泻、皮疹、尿痛、意识改变等伴随症状？",
                "4. 年龄、既往病史、过敏史、正在吃的药有哪些？",
            ]
        )

    def _llm_dialog_answer(
        self,
        query: str,
        context_query: str,
        docs: list[RetrievedDoc],
        fallback_answer: str,
    ) -> str:
        prompt = self._build_dialog_prompt(query, context_query, docs, fallback_answer)
        try:
            raw = self.llm.invoke(prompt)
        except Exception:
            return ""
        answer = self._guard_answer(
            self._strip_internal_prompt_echo(self._strip_deepseek_think(str(raw))).strip(),
            require_disclaimer=True,
        )
        return answer or fallback_answer

    def _llm_free_dialog_answer(self, query: str, context_query: str) -> str:
        history = "\n".join(
            f"患者：{user}\n助手：{assistant}" for user, assistant in self.turns[-6:]
        )
        emergency_hit = screen_emergency(context_query)
        prompt = f"""你是一个线上问诊对话助手。当前是【纯 LLM 模式】，不要调用本地知识库，不要输出任何系统提示、字段名或分析过程，只输出给患者看的自然回复。

安全要求：
1. 不能诊断，只能做健康信息整理、分诊建议和就医提醒。
2. 如果出现急症红旗，必须建议尽快急诊或拨打急救电话。
3. 用药问题不能给处方剂量，不能保证某药适合，只能说明一般安全原则，并建议咨询医生/药师或阅读说明书。
4. 要像聊天机器人一样承接上下文，优先回应患者本轮话语，再提出 1-3 个最关键追问。
5. 中文回答，120-260 字，不要输出“患者本轮描述、检索知识、规则兜底参考”等内部标题。

近期对话：
{history or "无"}

本轮患者输入：
{query}

合并后的病情上下文：
{context_query}

急症红旗：
{", ".join(emergency_hit) if emergency_hit else "无"}

请直接输出给患者的回复："""
        try:
            raw = self.llm.invoke(prompt)
        except Exception:
            return ""
        return self._guard_answer(
            self._strip_internal_prompt_echo(self._strip_deepseek_think(str(raw))).strip(),
            require_disclaimer=True,
        )

    def _build_dialog_prompt(
        self, query: str, context_query: str, docs: list[RetrievedDoc], fallback_answer: str
    ) -> str:
        history = "\n".join(
            f"患者：{user}\n助手：{assistant}" for user, assistant in self.turns[-3:]
        )
        recent_questions = self._recent_assistant_questions()
        known_facts = self._known_facts(context_query)
        emergency_hit = screen_emergency(context_query)
        context = format_docs(docs)
        ranked_departments = self._rank_departments(docs)
        ranked_text = "\n".join(
            f"{idx}. {dept}（匹配度 {score:.2f}）"
            for idx, (dept, score) in enumerate(ranked_departments[:3], start=1)
        ) or "暂无可靠科室排序"
        has_evidence = any(doc.keyword_hits or doc.score > 0 for doc in docs)
        emergency_text = ", ".join(emergency_hit) if emergency_hit else "无"
        intent = self._dialog_intent(query, has_evidence)
        return f"""你是一个 LLM 主导的线上问诊智能体，不是固定模板报告生成器。你需要像真实问诊助手一样，根据多轮上下文和检索知识，与患者自然对话。

下面 <internal_context> 中的内容是系统内部参考，绝对不要原样输出，不要输出其中的标题、字段名或“规则兜底参考”。你最终只能输出 <final_answer> 中面向患者的一段自然回复。

要求：
1. 由你直接生成完整回复，包括理解患者输入、阶段性判断、下一步追问和安全建议；不要照抄固定模板。
2. 回复要自然、简短、有对话感，不要每轮都输出同样结构的大报告。
3. 必须优先回应患者本轮输入，承认他已经补充的信息，并基于新信息更新判断；不要机械重复上一轮回答。
4. 每次都先给一句阶段性建议或处理方向，再追问。不能只提问不建议。
5. 不要重复“近期助手已问过、患者本轮已经回答”的问题；只追问仍缺失的 1-3 个关键点。
6. 如果信息不足，也要先给一般安全处理方向，例如休息、补液、记录体温、观察加重信号，再追问最关键问题。
7. 如果“患者已经提供的信息”里已有起病时间、是否加重、主要症状或否认其他症状，禁止再次询问这些相同内容。
8. 如果本轮意图是“科室或挂号咨询”，必须直接回答建议挂哪个科/诊室，再补充何时应急诊，不能只继续追问。
9. 如果患者已经表示“没有其他症状”，不要再问“还有没有其他症状”，应基于现有信息给下一步处理建议。
10. 如果本轮意图是“用药咨询”，必须正面说明用药安全原则、不能给处方剂量、需要补充哪些用药关键信息；不要只重复分诊结论。
11. 必须基于检索知识和科室候选表达，不要编造未出现的症状、疾病或检查结果。
12. 如果急症红旗命中不是“无”，必须明确建议尽快急诊或拨打急救电话。
13. 用药问题只能给安全原则、禁忌提醒和“咨询医生/药师/说明书”建议，不要给处方剂量，不要承诺某药一定适合。
14. 可以回应普通聊天或解释项目能力，但要自然引导患者描述病情。
15. 中文回答，建议 120-260 字；如果确实需要列问题，可以用简短编号。

<internal_context>
本轮意图：
{intent}

急症红旗命中：
{emergency_text}

长期摘要：
{self.long_summary or "无"}

近期对话：
{history or "无"}

近期助手已问过的问题：
{recent_questions or "无"}

患者已经提供的信息，不要重复追问：
{known_facts or "暂无明确补充"}

患者本轮描述：
{query}

合并上下文后的检索输入：
{context_query}

检索知识，可能为空或不完全相关：
{context}

科室候选：
{ranked_text}

安全边界：
不能替代医生诊断；不能给处方剂量；急症红旗需要急诊；LLM 失败时系统会使用规则兜底。

是否有可靠检索证据：
{"是" if has_evidence else "否"}
</internal_context>

<final_answer>
请直接输出给患者看的本轮问诊回复，不要包含 internal_context 或任何内部字段标题。
</final_answer>"""

    def _llm_answer(self, query: str, docs: list[RetrievedDoc], base_answer: str) -> str:
        return self._llm_dialog_answer(query, query, docs, base_answer)

    @staticmethod
    def _strip_deepseek_think(text: str) -> str:
        if "</think>" in text:
            return text.split("</think>", 1)[1]
        return text

    @staticmethod
    def _strip_internal_prompt_echo(text: str) -> str:
        markers = ["【回复】", "回复：", "<final_answer>", "请直接输出给患者的回复："]
        for marker in markers:
            if marker in text:
                text = text.split(marker, 1)[1]
        for start, end in [("<internal_context>", "</internal_context>")]:
            while start in text and end in text:
                before, rest = text.split(start, 1)
                _, after = rest.split(end, 1)
                text = before + after
        blocked_prefixes = (
            "【患者本轮描述】",
            "【合并上下文后的检索输入】",
            "【检索知识",
            "【科室候选】",
            "【规则兜底参考",
            "【本轮意图】",
            "【急症红旗命中】",
            "【是否有可靠检索证据】",
            "患者本轮描述：",
            "合并上下文后的检索输入：",
            "检索知识",
            "科室候选：",
            "规则兜底参考",
            "本轮意图：",
            "急症红旗",
        )
        lines = [
            line for line in text.splitlines()
            if not line.strip().startswith(blocked_prefixes)
        ]
        return "\n".join(lines).strip()

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
        evidence_docs = [doc for doc in docs if doc.keyword_hits or doc.score > 0] or docs[:1]
        for doc in evidence_docs[:3]:
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

    def _contextual_query(self, query: str) -> str:
        recent_users = [user for user, _ in self.turns[-3:]]
        if not recent_users:
            return query
        if self._looks_like_new_topic(query):
            return query
        return "；".join([*recent_users, query])

    @staticmethod
    def _expand_query_terms(query: str) -> str:
        expansions = []
        if "便血" in query or "大便带血" in query:
            expansions.extend(["血便", "黑便", "消化道出血"])
        if "拉肚子" in query:
            expansions.append("腹泻")
        if "肚子疼" in query:
            expansions.append("腹痛")
        if "头疼" in query:
            expansions.append("头痛")
        if "发烧" in query:
            expansions.append("发热")
        if "口腔" in query and "水泡" in query:
            expansions.extend(["口腔水泡", "口腔起泡", "口腔溃疡"])
        if "嘴里" in query and ("水泡" in query or "起泡" in query):
            expansions.extend(["口腔水泡", "嘴里起泡", "口腔黏膜"])
        if not expansions:
            return query
        return "；".join([query, *expansions])

    def _looks_like_new_topic(self, query: str) -> bool:
        if not self.turns:
            return True
        return any(phrase in query for phrase in ["换个问题", "另一个问题", "重新开始", "不是这个", "新的症状"])

    def _filter_context_docs(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        filtered: list[RetrievedDoc] = []
        medication_query = self._is_medication_question(query)
        digestive_query = any(word in query for word in ["腹", "肚", "泻", "便", "呕", "恶心", "胃", "黑便", "血便"])
        respiratory_query = any(word in query for word in ["咳", "鼻", "咽", "痰", "喘", "呼吸", "胸"])
        neuro_query = any(word in query for word in ["头疼", "头痛", "偏头痛", "畏光", "头晕", "意识"])
        urinary_query = any(word in query for word in ["尿", "腰痛", "血尿"])
        skin_query = any(word in query for word in ["皮", "疹", "痒", "风团", "红斑"])
        child_query = any(word in query for word in ["儿童", "宝宝", "孩子", "小孩", "婴儿"])
        nasal_query = any(word in query for word in ["鼻", "鼻塞", "流涕", "黄鼻涕", "面部", "嗅觉"])
        generic_hits = {"发热", "发烧", "疼", "痛", "头晕"}
        digestive_depts = {"消化内科", "普外科", "肝胆外科"}
        respiratory_depts = {"呼吸内科", "耳鼻喉科"}

        for doc in docs:
            if doc.id == "system_fever_headache" and any(item.id == "triage_fever_headache" for item in docs):
                continue
            if doc.category == "drug" and not medication_query:
                continue
            if doc.id == "disease_sinusitis" and not nasal_query:
                continue
            hits = set(doc.keyword_hits)
            if hits and hits <= generic_hits:
                if doc.department in digestive_depts and not digestive_query:
                    continue
                if doc.department in respiratory_depts and not (respiratory_query or neuro_query):
                    continue
                if doc.department == "泌尿外科" and not urinary_query:
                    continue
                if doc.department == "皮肤科" and not skin_query:
                    continue
                if doc.department == "儿科" and not child_query:
                    continue
            if not hits and doc.score <= 0:
                continue
            filtered.append(doc)

        if filtered:
            if neuro_query and any(word in query for word in ["发热", "发烧"]):
                fever_headache = [doc for doc in filtered if doc.id == "triage_fever_headache"]
                others = [doc for doc in filtered if doc.id != "triage_fever_headache"]
                return [*fever_headache, *others] if fever_headache else filtered
            return filtered
        if neuro_query and any(word in query for word in ["发热", "发烧"]):
            return [
                RetrievedDoc(
                    id="system_fever_headache",
                    title="发热伴头痛评估",
                    category="system_note",
                    department="全科/呼吸内科",
                    severity="medium",
                    content="发热伴头痛可见于呼吸道感染、流感样疾病等，也需要警惕持续高热、剧烈头痛、颈部僵硬、皮疹、意识改变或反复呕吐等危险信号。",
                    source="system",
                    score=0.2,
                    keyword_hits=["发热", "头痛"],
                )
            ]
        return docs[:1]

    def _recent_assistant_questions(self) -> str:
        questions: list[str] = []
        for _, assistant in self.turns[-3:]:
            for part in re.split(r"[？?]\s*", assistant):
                text = part.strip()
                if not text:
                    continue
                if len(text) > 80:
                    text = text[-80:]
                if any(word in text for word in ["是否", "有没有", "从什么时候", "多久", "严重程度", "哪里", "多少"]):
                    questions.append(text + "？")
        return "\n".join(dict.fromkeys(questions[-5:]))

    @staticmethod
    def _known_facts(context_query: str) -> str:
        facts: list[str] = []
        if any(word in context_query for word in ["头疼", "头痛"]):
            facts.append("已有症状：头疼/头痛")
        if any(word in context_query for word in ["发热", "发烧"]):
            facts.append("已有症状：发热")
        if "昨天" in context_query:
            facts.append("起病时间：昨天开始")
        if "3天" in context_query or "三天" in context_query:
            facts.append("起病时间：约 3 天")
        if "加重" in context_query or "没有好转" in context_query:
            facts.append("变化趋势：加重或没有好转")
        if "好转" in context_query and "没有好转" not in context_query:
            facts.append("变化趋势：有好转")
        if "发热最影响" in context_query:
            facts.append("最影响患者的是：发热")
        if "没有别的症状" in context_query or "无其他症状" in context_query:
            facts.append("患者表示暂时没有其他伴随症状")
        if "便血" in context_query or "血便" in context_query:
            facts.append("伴随症状：便血/血便")
        return "\n".join(dict.fromkeys(facts))

    def _has_medical_signal(self, query: str) -> bool:
        return self._has_specific_symptom(query) or any(word in query for word in GENERAL_MEDICAL_WORDS)

    def _has_specific_symptom(self, query: str) -> bool:
        if screen_emergency(query):
            return True
        for record in self.retriever.records:
            if any(keyword and keyword in query for keyword in record["keywords"]):
                return True
        return False

    def _dialog_intent(self, query: str, has_evidence: bool) -> str:
        if self._is_department_question(query):
            return "科室或挂号咨询"
        if self._is_medication_question(query):
            return "用药咨询"
        if self.turns and not self._has_specific_symptom(query) and self._has_medical_signal(query):
            return "病情补充或追问回答"
        if self._has_specific_symptom(query) or has_evidence:
            return "症状描述与分诊咨询"
        if self._has_medical_signal(query):
            return "健康问题咨询但信息不足"
        return "普通聊天或能力咨询"

    @staticmethod
    def _is_medication_question(query: str) -> bool:
        return any(word in query for word in MEDICATION_WORDS)

    @staticmethod
    def _is_department_question(query: str) -> bool:
        return any(word in query for word in DEPARTMENT_WORDS)

    @staticmethod
    def _select_risk(docs: list[RetrievedDoc]) -> RetrievedDoc:
        top = docs[0]
        significant_docs = [
            doc for doc in docs if (doc.keyword_hits or doc.score > 0) and doc.score >= max(0.22, top.score * 0.45)
        ] or [top]
        return max(significant_docs, key=lambda doc: SEVERITY_RANK.get(doc.severity, 0))

    def _followup_questions(self, docs: list[RetrievedDoc]) -> list[str]:
        departments = [dept for dept, _ in self._rank_departments(docs)[:2]]
        questions = [
            "这些症状从什么时候开始，最近是在好转还是加重？",
            "目前最影响你的症状是哪一个，严重程度大概 0-10 分是多少？",
        ]
        if "呼吸内科" in departments or "耳鼻喉科" in departments:
            questions.append("有没有体温、咳痰颜色、气促、胸痛、血氧下降或接触流感/新冠患者？")
        elif "消化内科" in departments or "普外科" in departments or "肝胆外科" in departments:
            questions.append("腹痛具体位置在哪里，是否伴呕吐、腹泻、黑便、血便或进食后加重？")
        elif "泌尿外科" in departments:
            questions.append("有没有发热、腰痛、血尿、尿量减少，或者近期是否饮水少？")
        elif "皮肤科" in departments:
            questions.append("皮疹范围多大，是否瘙痒、起风团、水疱，是否接触过新食物或药物？")
        elif "精神心理科" in departments:
            questions.append("这种状态持续多久了，是否影响睡眠/工作，是否出现自伤或轻生想法？")
        else:
            questions.append("有没有既往病史、药物过敏、怀孕/哺乳、儿童或老年等特殊情况？")
        return questions

    @staticmethod
    def _filter_followup_questions(context_query: str, questions: list[str]) -> list[str]:
        filtered = []
        for question in questions:
            if any(word in context_query for word in ["昨天", "前天", "3天", "三天"]) and "什么时候开始" in question:
                continue
            if any(word in context_query for word in ["加重", "没有好转", "好转"]) and "好转还是加重" in question:
                continue
            if "最影响" in context_query and "最影响" in question:
                continue
            if any(word in context_query for word in ["没有别的症状", "没有其他症状", "无其他症状"]) and ("有没有" in question or "是否" in question):
                continue
            filtered.append(question)
        return filtered or ["年龄、体温最高多少、是否有慢性病或正在使用的药物？"]

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

# 智能医疗分诊与建议智能体

这是一个和简历项目描述对应的小型可运行 Demo：基于 SQLite 医学知识库、Chroma 向量数据库、Ollama 本地 DeepSeek R1、关键词 + 向量混合检索、LangChain Tool / AgentExecutor，以及 FastAPI Web 前端，实现线上问诊场景下的分诊推荐、医学问答和初步建议。

> 免责声明：本项目仅用于课程、简历和工程演示，不提供诊断结论，不能替代医生面诊或急救判断。

## 功能

- 医学知识库：40 条疾病、症状、药品、分诊规则，覆盖常见线上问诊科室。
- 向量数据库：使用 `langchain_chroma.Chroma` 持久化本地医学知识向量。
- 本地 LLM：使用 Ollama 中的 `deepseek-r1:7b` 进行 RAG 答案生成。
- 混合检索：关键词命中 + Chroma 向量召回，相比纯关键词更能处理自然语言描述。
- ReAct Agent：提供 `emergency_screen`、`medical_search`、`department_rank`、`followup_questions`、`compose_answer` 等 LangChain Tools，并接入 `AgentExecutor`。
- 分诊建议：输出急诊/门诊优先级、Top-3 科室推荐、依据与来源。
- 急症筛查：对胸痛、呼吸困难、腹痛等症状使用“明确红旗词 + 症状组合”判断，减少普通症状被误判为急诊。
- 响应耗时记录：每个 HTTP 请求都会写入 `data/response_times.jsonl`，聊天接口同时返回 `response_time_ms`，前端会直接显示本次响应时间。
- 多轮对话：Web 端支持连续追问、清空会话；后端按 `session_id` 隔离不同浏览器的上下文。
- 对话记忆：每个会话使用 `ConversationBufferWindowMemory` 保存近期问诊上下文，并保留简易长期摘要。
- ReAct 演示：`--trace` 可展示工具调用和观察结果，方便答辩展示 Agent 流程。
- Web 前端：FastAPI + 原生 HTML/CSS/JS，展示答案、检索证据、Agent 轨迹和评测指标。

## 快速运行

在你的 `langchain` 环境中运行：

```powershell
conda activate langchain
cd E:\毕业设计\medical_triage_agent
python scripts\build_vector_db.py
python -m app.cli --query "发热咳嗽两天，肌肉酸痛，很乏力" --trace
```

使用真正的 LangChain Tool / AgentExecutor：

```powershell
python -m app.cli --agent-executor --query "胸痛半小时，出汗，还有点呼吸困难" --trace
```

也可以进入交互模式：

```powershell
python -m app.cli
```

启动 Web 服务：

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

然后打开：

```text
http://127.0.0.1:8000
```

如果希望同一局域网或公网用户访问，请查看 [DEPLOYMENT.md](DEPLOYMENT.md)。

如果 Ollama 没启动，先打开 Ollama 服务；确认模型名：

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

如果只是想演示检索和规则兜底，不调用大模型：

```powershell
python -m app.cli --query "最近尿频尿急尿痛，有一点血尿" --no-llm
```

## 测试

```powershell
cd E:\毕业设计\medical_triage_agent
D:\Anaconda3\envs\langchain\python.exe -m unittest discover -s tests -v
```

当前回归测试覆盖：
- 胸痛伴出汗、左肩痛、呼吸困难时仍推荐急诊。
- 普通感冒样描述不会被强制推荐急诊。
- 饭后胸口灼热、反酸、嗳气更倾向消化内科，不会因“胸口不适”自动急诊。
- Agent 返回对象会记录 `response_time_ms`。

## 响应时间日志

启动 Web 服务后，每一次 HTTP 响应都会追加一行 JSON 到：

```text
data/response_times.jsonl
```

示例字段：

```json
{"timestamp":"2026-06-06T08:00:00.000000+00:00","method":"POST","path":"/api/chat","status_code":200,"elapsed_ms":123.45}
```

`/api/chat` 响应体还会包含：
- `response_time_ms`：本次聊天接口总耗时，单位毫秒。
- `agent_response_time_ms`：核心分诊 Agent 内部耗时，单位毫秒。使用 `AgentExecutor` 时仅返回接口总耗时。

浏览器页面的“分诊建议”区域会显示本次响应时间；也可以通过响应头 `X-Response-Time-Ms` 查看任意接口耗时。

## 急症筛查策略

系统不再把所有单个泛化词都直接升为急诊。例如“胸痛”只有在伴随大汗、冷汗、左肩/左臂/肩背/下颌放射痛、压榨感、呼吸困难、气短或晕厥等组合信号时，才进入“急症优先”。“呼吸困难”会结合血氧下降、口唇发紫、说话困难、喘憋、明显费力等信号判断。检索到急诊规则但用户描述缺少这些红旗信号时，会降为“建议尽快就医”，避免“不管输入什么都建议去急诊”。

## 评测

项目内置 20 条标注问诊集：

```powershell
python scripts\evaluate.py
```

当前本地评测结果：

- Top-1 分诊准确率：1.00
- Top-3 分诊召回率：1.00
- 知识库文档数：40

## 数据来源

知识库为人工整理的小规模结构化样例，参考了以下公开医学健康页面，没有直接抓取网页正文：

- CDC Flu Signs and Symptoms: https://www.cdc.gov/flu/signs-symptoms/index.html
- CDC Stroke Signs and Symptoms: https://www.cdc.gov/stroke/signs-symptoms/index.html
- CDC Asthma Signs and Symptoms: https://www.cdc.gov/asthma/signs-symptoms/index.html
- CDC Diabetes Signs and Symptoms: https://www.cdc.gov/diabetes/signs-symptoms/index.html
- CDC Hand, Foot, and Mouth Disease: https://www.cdc.gov/hand-foot-mouth/signs-symptoms/index.html
- MedlinePlus Chest Pain: https://medlineplus.gov/chestpain.html
- MedlinePlus Common Cold: https://medlineplus.gov/commoncold.html
- MedlinePlus Migraine: https://medlineplus.gov/migraine.html
- MedlinePlus GERD: https://medlineplus.gov/gerd.html
- MedlinePlus Back Pain: https://medlineplus.gov/backpain.html
- MedlinePlus Depression: https://medlineplus.gov/depression.html
- MedlinePlus Drug Information: https://medlineplus.gov/druginformation.html

## 后续可扩展

- 将前端改成 Vue/React，并增加多轮会话列表。
- 加入更大规模医学问诊数据集和人工复核标签。
- 接入专门的中文医学 embedding，替换当前 DeepSeek R1 embedding 方案。

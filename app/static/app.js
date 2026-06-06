const form = document.getElementById("chatForm");
const chatLogEl = document.getElementById("chatLog");
const docsEl = document.getElementById("docs");
const traceEl = document.getElementById("trace");
const metricsEl = document.getElementById("metrics");
const responseTimeEl = document.getElementById("responseTime");
const clearSessionBtn = document.getElementById("clearSession");
const sessionId = getOrCreateSessionId();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const queryEl = document.getElementById("query");
  const query = queryEl.value.trim();
  if (!query) return;

  appendMessage("user", query);
  const pendingMessage = appendMessage("assistant", "正在理解你的描述，并结合上下文整理回复...");
  responseTimeEl.textContent = "响应时间：-";
  docsEl.innerHTML = "";
  traceEl.innerHTML = "";
  queryEl.value = "";

  const payload = {
    query,
    session_id: sessionId,
    knowledge_mode: document.getElementById("knowledgeMode").value,
    use_llm: document.getElementById("useLlm").checked,
    use_agent_executor: document.getElementById("useAgent").checked,
    show_trace: document.getElementById("showTrace").checked
  };

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  pendingMessage.querySelector("pre").textContent = data.answer || "未生成回答";
  const responseTime = data.response_time_ms || res.headers.get("X-Response-Time-Ms");
  responseTimeEl.textContent = responseTime ? `响应时间：${responseTime} ms` : "响应时间：-";
  renderDocs(data.docs || []);
  renderTrace(data.trace || []);
  chatLogEl.scrollTop = chatLogEl.scrollHeight;
});

clearSessionBtn.addEventListener("click", async () => {
  await fetch(`/api/session/${encodeURIComponent(sessionId)}/reset`, {method: "POST"});
  chatLogEl.innerHTML = "";
  appendMessage("assistant", "会话已清空，可以开始新的问诊。");
  docsEl.innerHTML = "";
  traceEl.innerHTML = "";
  responseTimeEl.textContent = "响应时间：-";
});

document.getElementById("metricsBtn").addEventListener("click", async () => {
  metricsEl.textContent = "正在评测...";
  const res = await fetch("/api/metrics");
  const data = await res.json();
  metricsEl.textContent = JSON.stringify(data, null, 2);
});

function renderDocs(docs) {
  docsEl.innerHTML = docs.map((doc) => `
    <article class="item">
      <strong>${escapeHtml(doc.title)} · ${escapeHtml(doc.department)}</strong>
      <div class="meta">risk=${escapeHtml(doc.severity)} score=${doc.score} hits=${escapeHtml((doc.keyword_hits || []).join("、") || "语义相似")}</div>
      <div>${escapeHtml(doc.content)}</div>
      <div class="meta">${escapeHtml(doc.source)}</div>
    </article>
  `).join("");
}

function renderTrace(trace) {
  traceEl.innerHTML = trace.map((step) => {
    if (typeof step === "string") {
      return `<article class="item"><pre>${escapeHtml(step)}</pre></article>`;
    }
    return `<article class="item">
      <strong>${escapeHtml(step.tool || "trace")}</strong>
      <div class="meta">${escapeHtml(String(step.tool_input || ""))}</div>
      <pre>${escapeHtml(String(step.observation || ""))}</pre>
    </article>`;
  }).join("");
}

function appendMessage(role, text) {
  const emptyMessage = chatLogEl.querySelector(".message.assistant pre");
  if (emptyMessage && emptyMessage.textContent === "等待输入...") {
    chatLogEl.innerHTML = "";
  }
  const item = document.createElement("article");
  item.className = `message ${role}`;
  item.innerHTML = `<pre>${escapeHtml(text)}</pre>`;
  chatLogEl.appendChild(item);
  chatLogEl.scrollTop = chatLogEl.scrollHeight;
  return item;
}

function getOrCreateSessionId() {
  const key = "medical-triage-session-id";
  let value = localStorage.getItem(key);
  if (!value) {
    value = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    localStorage.setItem(key, value);
  }
  return value;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

# 部署与 GitHub 同步指南

## 1. 让同一局域网的人访问

如果只是让同一 Wi-Fi 或校园网内的同学访问，把启动命令中的 `127.0.0.1` 改成 `0.0.0.0`：

```powershell
D:\Anaconda3\envs\langchain\python.exe -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

然后查询本机局域网 IP：

```powershell
ipconfig
```

假设 IPv4 地址是 `192.168.1.23`，别人打开：

```text
http://192.168.1.23:8000
```

如果打不开，通常是 Windows 防火墙没有放行 8000 端口，或者对方不在同一个网络。

## 2. 临时公网访问

适合答辩、演示或短时间给别人看：

- Cloudflare Tunnel
- ngrok
- frp

这类工具会把本地 `http://127.0.0.1:8000` 映射成一个公网 HTTPS 地址。优点是快，缺点是本机必须一直开着，链接也可能变化。

## 3. 正式部署到云服务器

如果希望别人长期访问，建议部署到云服务器、Render、Railway、Fly.io 等平台。

基础启动命令：

```bash
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

云服务器注意点：

- 本项目默认使用 Ollama 的 `deepseek-r1:7b`，云服务器上也要安装 Ollama 并拉取同名模型。
- 如果服务器资源不够，可以在网页上取消“使用 LLM”，系统仍可用规则和知识库检索生成分诊建议。
- 首次启动时 API 会自动初始化 SQLite 知识库；向量库为空时会自动构建，但第一次请求可能较慢。
- 医疗项目不要直接收集真实个人敏感信息；公网演示时建议明确用途是课程/简历 Demo。

## 4. 同步到 GitHub

当前机器如果没有安装 Git，可以先安装 Git for Windows：

```text
https://git-scm.com/download/win
```

安装后在项目目录运行：

```powershell
cd E:\毕业设计\medical_triage_agent
git init
git add .
git commit -m "Improve triage chat demo"
```

在 GitHub 网页端新建一个空仓库，例如：

```text
medical_triage_agent
```

然后绑定远程仓库并推送：

```powershell
git branch -M main
git remote add origin https://github.com/你的用户名/medical_triage_agent.git
git push -u origin main
```

如果你更习惯图形界面，也可以用 GitHub Desktop：

1. File -> Add local repository。
2. 选择 `E:\毕业设计\medical_triage_agent`。
3. 填写提交说明并 Commit。
4. 点击 Publish repository。

## 5. 推荐不要上传的文件

项目已经添加 `.gitignore`，默认排除：

- Python 缓存：`__pycache__/`
- 本地日志：`*.log`
- 响应耗时日志：`data/response_times.jsonl`
- 本地生成的 SQLite / Chroma 向量库文件

保留 `data/medical_seed.json` 即可重建知识库。

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from src.main.python.eval.qa_evaluation import (
    build_csv_export,
    build_markdown_report,
    create_evaluation_job,
    get_items,
    get_job,
    list_jobs,
    minimax_config,
    pause_evaluation_job,
    run_evaluation_job,
)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("", response_class=HTMLResponse)
def evaluation_page() -> str:
    return _EVALUATION_HTML


@router.post("/jobs")
async def create_job_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: str = Form(default=""),
    allow_local_fallback: bool = Form(default=True),
) -> dict[str, object]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    try:
        job_id = create_evaluation_job(
            file.filename or "evaluation.csv",
            content,
            {"name": name, "allow_local_fallback": allow_local_fallback},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(run_evaluation_job, job_id)
    return {"job_id": job_id, "status": "pending"}


@router.get("/jobs")
def list_jobs_endpoint() -> dict[str, object]:
    return {"jobs": list_jobs()}


@router.get("/config")
def config_endpoint() -> dict[str, object]:
    config = minimax_config()
    return {
        "judge": "dashscope-minimax",
        "minimax_configured": bool(config.api_key),
        "minimax_model": config.model,
        "minimax_base_url": config.base_url,
        "dashscope_workspace_configured": bool(config.workspace),
        "retry_count": config.retry_count,
        "timeout": config.timeout,
    }


@router.get("/jobs/{job_id}")
def get_job_endpoint(job_id: str) -> dict[str, object]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="evaluation job not found")
    return job


@router.get("/jobs/{job_id}/items")
def get_items_endpoint(job_id: str, limit: int = 500, offset: int = 0) -> dict[str, object]:
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="evaluation job not found")
    return {"items": get_items(job_id, limit=limit, offset=offset)}


@router.post("/jobs/{job_id}/resume")
def resume_job_endpoint(job_id: str, background_tasks: BackgroundTasks) -> dict[str, object]:
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="evaluation job not found")
    background_tasks.add_task(run_evaluation_job, job_id)
    return {"job_id": job_id, "status": "evaluating"}


@router.post("/jobs/{job_id}/pause")
def pause_job_endpoint(job_id: str) -> dict[str, object]:
    if not pause_evaluation_job(job_id):
        raise HTTPException(status_code=404, detail="evaluation job not found")
    return {"job_id": job_id, "status": "paused"}


@router.get("/jobs/{job_id}/report.md")
def download_markdown_report(job_id: str) -> PlainTextResponse:
    try:
        report = build_markdown_report(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    headers = {"Content-Disposition": f'attachment; filename="evaluation-{job_id}.md"'}
    return PlainTextResponse(report, media_type="text/markdown; charset=utf-8", headers=headers)


@router.get("/jobs/{job_id}/results.csv")
def download_csv_results(job_id: str) -> Response:
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="evaluation job not found")
    csv_text = build_csv_export(job_id)
    headers = {"Content-Disposition": f'attachment; filename="evaluation-{job_id}.csv"'}
    return Response(csv_text, media_type="text/csv; charset=utf-8", headers=headers)


_EVALUATION_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>QA 效果评估系统</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #667085;
      --border: #d8dee9;
      --brand: #2563eb;
      --brand-dark: #1d4ed8;
      --danger: #b42318;
      --ok: #067647;
      --warn: #b54708;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      padding: 20px 28px;
    }
    h1 { margin: 0 0 6px; font-size: 24px; }
    p { margin: 0; color: var(--muted); }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      gap: 18px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
    }
    h2 {
      font-size: 18px;
      margin: 0 0 14px;
    }
    .upload {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto;
      gap: 12px;
      align-items: end;
    }
    label { display: grid; gap: 8px; color: var(--muted); font-size: 13px; }
    input[type="file"], input[type="text"] {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 12px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    .checkbox {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text);
      font-size: 14px;
      align-self: center;
    }
    button, .button {
      border: 0;
      border-radius: 6px;
      background: var(--brand);
      color: white;
      padding: 11px 14px;
      font: inherit;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
    }
    button:hover, .button:hover { background: var(--brand-dark); }
    button:disabled { background: #98a2b3; cursor: wait; }
    .grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
    }
    .metric {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      min-height: 86px;
    }
    .metric b { display: block; font-size: 24px; margin-bottom: 8px; }
    .metric span { color: var(--muted); font-size: 13px; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      border-bottom: 1px solid var(--border);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
      font-size: 13px;
    }
    th { color: var(--muted); font-weight: 600; }
    .status {
      display: inline-flex;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      background: #eef2ff;
      color: #3538cd;
    }
    .bad { color: var(--danger); }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
    .muted { color: var(--muted); }
    .notice {
      margin-top: 12px;
      color: var(--warn);
      font-size: 13px;
    }
    .judge {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
      margin-bottom: 14px;
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 860px) {
      main { padding: 14px; }
      .upload { grid-template-columns: 1fr; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      table { display: block; overflow-x: auto; white-space: nowrap; }
    }
  </style>
</head>
<body>
  <header>
    <h1>QA 效果评估系统</h1>
    <p>上传 CSV、XLSX、JSON 或 JSONL，系统会按 Gold QA、RAG 忠实度、弱监督质量自动分模式评估。</p>
  </header>
  <main>
    <section>
      <h2>上传评估文件</h2>
      <div class="judge" id="judgeStatus">正在读取 MiniMax 裁判配置...</div>
      <form id="uploadForm" class="upload">
        <label>评估文件
          <input id="fileInput" type="file" accept=".csv,.xlsx,.json,.jsonl" required />
        </label>
        <label class="checkbox">
          <input id="fallbackInput" type="checkbox" checked />
          MiniMax 失败时使用本地降级评估
        </label>
        <button id="uploadButton" type="submit">开始评估</button>
      </form>
      <p class="notice" id="message"></p>
    </section>

    <section id="overview" hidden>
      <h2>报告总览</h2>
      <div class="grid" id="metrics"></div>
      <div class="actions" id="jobActions"></div>
      <div class="actions" id="downloads"></div>
      <p class="notice" id="warnings"></p>
    </section>

    <section>
      <h2>最近任务</h2>
      <div id="jobs"></div>
    </section>

    <section id="details" hidden>
      <h2>单题明细</h2>
      <div id="items"></div>
    </section>
  </main>

  <script>
    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");
    const uploadButton = document.getElementById("uploadButton");
    const message = document.getElementById("message");
    let currentJobId = null;
    let timer = null;

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!fileInput.files.length) return;
      uploadButton.disabled = true;
      message.textContent = "正在上传并创建评估任务...";
      const data = new FormData();
      data.append("file", fileInput.files[0]);
      data.append("allow_local_fallback", document.getElementById("fallbackInput").checked ? "true" : "false");
      try {
        const response = await fetch("/evaluation/jobs", { method: "POST", body: data });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "上传失败");
        currentJobId = payload.job_id;
        message.textContent = "任务已创建，正在评估...";
        await refreshAll();
        startPolling(currentJobId);
      } catch (error) {
        message.textContent = error.message;
      } finally {
        uploadButton.disabled = false;
      }
    });

    function startPolling(jobId) {
      clearInterval(timer);
      timer = setInterval(async () => {
        const job = await loadJob(jobId);
        if (job && ["completed", "failed", "cancelled"].includes(job.status)) {
          clearInterval(timer);
        }
      }, 1500);
    }

    async function refreshAll() {
      await refreshConfig();
      const response = await fetch("/evaluation/jobs");
      const payload = await response.json();
      renderJobs(payload.jobs || []);
      if (currentJobId) await loadJob(currentJobId);
    }

    async function refreshConfig() {
      const target = document.getElementById("judgeStatus");
      try {
        const response = await fetch("/evaluation/config");
        const config = await response.json();
        target.innerHTML = `
          <span>Judge：阿里百炼 MiniMax 单题裁判，模型 ${escapeHtml(config.minimax_model || "-")}</span>
          <span class="${config.minimax_configured ? "ok" : "bad"}">${config.minimax_configured ? "已配置 DASHSCOPE_API_KEY" : "未配置 DASHSCOPE_API_KEY"}</span>
        `;
      } catch (error) {
        target.textContent = "无法读取 MiniMax 裁判配置";
      }
    }

    async function loadJob(jobId) {
      currentJobId = jobId;
      const response = await fetch(`/evaluation/jobs/${jobId}`);
      if (!response.ok) return null;
      const job = await response.json();
      renderOverview(job);
      await loadItems(jobId);
      await refreshJobsOnly();
      return job;
    }

    async function refreshJobsOnly() {
      const response = await fetch("/evaluation/jobs");
      const payload = await response.json();
      renderJobs(payload.jobs || []);
    }

    async function loadItems(jobId) {
      const response = await fetch(`/evaluation/jobs/${jobId}/items?limit=200`);
      const payload = await response.json();
      renderItems(payload.items || []);
    }

    function renderJobs(jobs) {
      const target = document.getElementById("jobs");
      if (!jobs.length) {
        target.innerHTML = '<p class="muted">还没有评估任务。</p>';
        return;
      }
      target.innerHTML = `
        <table>
          <thead><tr><th>任务</th><th>创建时间</th><th>状态</th><th>样本</th><th>失败</th><th>综合分</th><th>复核率</th><th>操作</th></tr></thead>
          <tbody>
            ${jobs.map(job => `
              <tr>
                <td>${escapeHtml(job.name || job.file_name)}</td>
                <td>${formatTime(job.created_at)}</td>
                <td><span class="status">${job.status}</span></td>
                <td>${job.total_count || 0}</td>
                <td class="${job.failed_count ? "bad" : ""}">${job.failed_count || 0}</td>
                <td>${percent(job.summary && job.summary.composite_quality_estimate)}</td>
                <td>${percent(job.summary && job.summary.needs_review_rate)}</td>
                <td>
                  <button type="button" onclick="loadJob('${job.id}')">查看</button>
                  ${jobActionButton(job)}
                </td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    }

    function renderOverview(job) {
      const summary = job.summary || {};
      document.getElementById("overview").hidden = false;
      document.getElementById("metrics").innerHTML = [
        metric("已完成 / 失败", `${summary.completed_count || 0} / ${summary.failed_count || 0}`),
        metric("综合质量估计分", percent(summary.composite_quality_estimate)),
        metric("Gold 准确率", nullablePercent(summary.gold_accuracy)),
        metric("RAG 忠实率", nullablePercent(summary.rag_faithful_rate)),
        metric("Weak 可用率", nullablePercent(summary.weak_usable_rate)),
        metric("人工复核率", percent(summary.needs_review_rate)),
      ].join("");
      document.getElementById("downloads").innerHTML = `
        <a class="button" href="/evaluation/jobs/${job.id}/report.md">下载 Markdown 报告</a>
        <a class="button" href="/evaluation/jobs/${job.id}/results.csv">下载 CSV 明细</a>
      `;
      document.getElementById("jobActions").innerHTML = jobActionButton(job);
      const modes = summary.mode_distribution || {};
      const warnings = summary.warnings || [];
      document.getElementById("warnings").textContent =
        `样本构成：Gold ${modes.gold_qa || 0}，RAG ${modes.rag_faithfulness || 0}，Weak ${modes.weak_qa_quality || 0}。` +
        (warnings.length ? " " + warnings.join(" ") : "");
    }

    async function pauseJob(jobId) {
      await fetch(`/evaluation/jobs/${jobId}/pause`, { method: "POST" });
      if (currentJobId === jobId) await loadJob(jobId);
      await refreshJobsOnly();
    }

    async function resumeJob(jobId) {
      await fetch(`/evaluation/jobs/${jobId}/resume`, { method: "POST" });
      currentJobId = jobId;
      await loadJob(jobId);
      startPolling(jobId);
    }

    function jobActionButton(job) {
      if (!job || ["completed", "failed", "cancelled"].includes(job.status)) return "";
      if (job.status === "paused") {
        return `<button type="button" onclick="resumeJob('${job.id}')">继续</button>`;
      }
      return `<button type="button" onclick="pauseJob('${job.id}')">暂停</button>`;
    }

    function renderItems(items) {
      document.getElementById("details").hidden = false;
      const target = document.getElementById("items");
      if (!items.length) {
        target.innerHTML = '<p class="muted">暂无明细。</p>';
        return;
      }
      target.innerHTML = `
        <table>
          <thead><tr><th>行</th><th>模式</th><th>裁判</th><th>分数</th><th>主错误</th><th>复核</th><th>问题</th><th>原因</th></tr></thead>
          <tbody>
            ${items.map(item => `
              <tr>
                <td>${item.row_index}</td>
                <td>${item.evaluation_mode}</td>
                <td>${escapeHtml((item.result && item.result.judge_provider) || "minimax")}</td>
                <td class="${item.is_pass ? "ok" : "bad"}">${score(item.score)}</td>
                <td>${escapeHtml(item.primary_error_type || "")}</td>
                <td>${item.needs_human_review ? '<span class="warn">是</span>' : "否"}</td>
                <td>${escapeHtml(item.question || "")}</td>
                <td>${escapeHtml((item.result && item.result.reason) || item.failure_reason || "")}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    }

    function metric(label, value) {
      return `<div class="metric"><b>${value}</b><span>${label}</span></div>`;
    }
    function percent(value) {
      return value === undefined || value === null ? "-" : `${(Number(value) * 100).toFixed(1)}%`;
    }
    function nullablePercent(value) {
      return value === undefined || value === null ? "无样本" : percent(value);
    }
    function score(value) {
      return value === undefined || value === null ? "-" : Number(value).toFixed(2);
    }
    function formatTime(value) {
      if (!value) return "-";
      return new Date(Number(value) * 1000).toLocaleString("zh-CN", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      });
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[ch]);
    }

    refreshAll();
  </script>
</body>
</html>
"""

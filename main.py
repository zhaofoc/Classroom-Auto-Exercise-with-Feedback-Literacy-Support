"""
Quiz Pipeline — FastAPI Backend
================================
Stage 1 → Stage 2 → Stage 4(round1) → 学生作答 → Stage 5 → Stage 4(round2)

教师页面：/           上传文件、运行 Pipeline、查看题目
学生页面：/student    等待题目就绪、作答、查看结果

SSE 频道：
  GET /stream/teacher   → 教师进度事件
  GET /stream/student   → 学生状态事件（题目就绪 / 第二轮就绪）

REST：
  POST /upload/sampdoc
  POST /upload/syl
  POST /run
  GET  /questions           → 获取当前轮次题目
  POST /submit              → 学生提交答案
  GET  /skillmap            → 获取能力图谱（Stage 5 结果）
  GET  /questions/round2    → 获取第二轮题目
  GET  /status              → 当前 pipeline 状态
"""

import asyncio
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from docx import Document as DocxDocument
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

import os

MODEL       = "gpt-4o"
BASE_DIR    = Path("./pipeline_workspace")
SAMPDOC_DIR = BASE_DIR / "sampdoc"
SYL_DIR     = BASE_DIR / "syl"
PROS_DIR    = BASE_DIR / "pros"
EVAL_DIR    = BASE_DIR / "eval"
TUTOR_DIR   = BASE_DIR / "tutor"   # 教师风格画像 / 反馈质量提示，跨轮次/跨运行持久保存，不随 STATE.reset() 清空
SUPPORTED   = {".txt", ".json", ".docx", ".doc"}
MAX_RETRIES = 3

for d in [SAMPDOC_DIR, SYL_DIR, PROS_DIR, EVAL_DIR, TUTOR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TUTOR_STYLE_PATH = TUTOR_DIR / "tutor_style.json"        # 持续演进的教师风格画像
TUTOR_TIPS_PATH  = TUTOR_DIR / "tutor_tips_latest.json"  # 最近一次的反馈质量提示（仅展示给教师）


def _load_tutor_style() -> dict:
    """读取当前教师风格画像；不存在则返回空字典。"""
    if TUTOR_STYLE_PATH.exists():
        try:
            return json.loads(TUTOR_STYLE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

# ── Global pipeline state ──────────────────────────────────────────────────────
class PipelineState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.status          = "idle"       # idle | running | waiting_answers | analyzing | deciding | done | error
        self.stage           = 0            # 0-5
        self.stage_msg       = ""
        self.error_msg       = ""
        self.questions_r1    = []           # list of course question sets (round 1, kept for API compat)
        self.questions_r2    = []           # round 2 questions (kept for API compat)
        self.questions_current = []         # questions for the currently active round
        self.current_round   = 0           # which round we are on (1-indexed)
        self.user_answers    = None         # answers for the current round (reset each round)
        self.all_answers_history: list[dict] = []  # all rounds' answers, in order
        self.skillmap        = None
        self.continuation_decisions: list[dict] = []  # per-round agent decisions
        self.stop_reason     = ""           # why we stopped (if early)
        # -- Teacher-in-the-loop feedback review
        self.feedback_status = "none"       # none | pending | approved
        self.feedback_draft  = None         # AI-generated draft, awaiting teacher review: {feedback, highlight, round, stopped_early}
        self.feedback_final  = None         # teacher-approved final version (same shape), this is what students see
        self.tutor_tips      = None         # latest TutorAdviseAgent output (teacher-facing only)
        self.tutor_style     = None         # in-memory cache of TutorStyle.json (persisted on disk, survives reset via _load_tutor_style)
        # SSE subscriber queues
        self.teacher_queues: list[asyncio.Queue] = []
        self.student_queues: list[asyncio.Queue] = []

    # -- broadcast helpers
    async def broadcast(self, queues: list, event: str, data: dict):
        payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        for q in list(queues):
            await q.put(payload)

    async def teacher(self, event: str, data: dict):
        await self.broadcast(self.teacher_queues, event, data)

    async def student(self, event: str, data: dict):
        await self.broadcast(self.student_queues, event, data)

STATE = PipelineState()

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Quiz Pipeline API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── File utilities ─────────────────────────────────────────────────────────────

def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        for enc in ["utf-8", "gbk", "utf-16", "latin-1"]:
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return f"[Failed to decode: {path.name}]"
    elif ext == ".json":
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return f"[Empty file: {path.name}]"
        try:
            return _json_to_text(json.loads(raw))
        except json.JSONDecodeError as e:
            return f"[JSON parse failed: {path.name} — {e}]"
    elif ext == ".docx":
        doc = DocxDocument(str(path))
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts)
    elif ext == ".doc":
        try:
            with tempfile.TemporaryDirectory() as tmp:
                r = subprocess.run(
                    ["libreoffice", "--headless", "--convert-to", "docx",
                     "--outdir", tmp, str(path)],
                    capture_output=True, timeout=60
                )
                converted = Path(tmp) / (path.stem + ".docx")
                if r.returncode == 0 and converted.exists():
                    return read_file(converted)
        except Exception:
            pass
        r = subprocess.run(["antiword", str(path)], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else f"[Failed to read: {path.name}]"
    return f"[Unsupported format: {path.name}]"


def _json_to_text(data, indent=0) -> str:
    prefix = "  " * indent
    lines = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{k}:")
                lines.append(_json_to_text(v, indent + 1))
            else:
                lines.append(f"{prefix}{k}: {v}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(_json_to_text(item, indent))
            else:
                lines.append(f"{prefix}- {item}")
    else:
        lines.append(f"{prefix}{data}")
    return "\n".join(lines)


def read_directory(directory: Path) -> dict:
    results = {}
    for fpath in sorted(directory.iterdir()):
        if fpath.suffix.lower() in SUPPORTED:
            try:
                results[fpath.name] = read_file(fpath)
            except Exception as e:
                results[fpath.name] = f"[Read error: {e}]"
    return results


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        end = -1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end])
    return json.loads(text)


# ── Stage 1 ────────────────────────────────────────────────────────────────────

def _run_stage1(api_key: str) -> str:
    docs = read_directory(SAMPDOC_DIR)
    if not docs:
        raise ValueError("No readable files found in sampdoc/")

    saved = {"value": ""}

    @tool
    def save_style_criteria(criteria_json: str) -> str:
        """将风格标准保存为 JSON 文件。
        Args:
            criteria_json: 符合 JSON 格式的风格标准字符串
        """
        PROS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROS_DIR / "document_style_criteria.json"
        try:
            parsed = json.loads(criteria_json)
            text = json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            text = criteria_json
        out_path.write_text(text, encoding="utf-8")
        saved["value"] = text
        return f"Saved to {out_path}"

    llm = ChatOpenAI(model=MODEL, temperature=0, api_key=api_key)
    system_prompt = (
        "你是 DocumentStyleAnalyzerAgent，专业的文档风格分析师。\n\n"
        "用户会把若干文档发给你。你需要：\n"
        "1. 阅读所有文档，提取格式模板和写作风格。\n"
        "2. 整理成一个 JSON 对象。\n"
        "3. 调用 save_style_criteria 工具保存。\n\n"
        "JSON 必须包含：\n"
        "【language_style】tone / sentence_length / vocabulary / paragraph_structure / common_patterns\n"
        "【document_template】document_sections / heading_convention / list_and_table_usage / "
        "recurring_elements / meta_info / layout_notes\n"
        "【document_semantics】who_author / who_reader / what / when / where / why / how_reader_uses\n\n"
        "直接输出 JSON，不需要额外解释。"
        "JSON 中所有文本性描述内容（value）必须使用英文（English）书写，字段名（key）保持不变。"
    )
    agent = create_react_agent(model=llm, tools=[save_style_criteria], prompt=system_prompt)
    content = "\n\n".join(f"=== {name} ===\n{text}" for name, text in docs.items())
    agent.invoke({"messages": [{"role": "user", "content": f"以下是需要分析的文档内容：\n\n{content}"}]})

    if not saved["value"]:
        p = PROS_DIR / "document_style_criteria.json"
        if p.exists():
            saved["value"] = p.read_text(encoding="utf-8")

    return saved["value"]


# ── Stage 2 ────────────────────────────────────────────────────────────────────

def _extract_learning_objectives(llm, text: str) -> dict:
    system = (
        "你是课程分析专家。从课程大纲/标准文档中提取学习目标，按布鲁姆分类法分类。\n"
        "仅输出 JSON，不要有任何其他文字，JSON 中所有文本性内容须使用英文（English）书写：\n"
        '{"course_name":"","grade_or_level":"","subject_domains":[],'
        '"objectives":[{"id":"LO-001","domain":"","objective":"","bloom_level":"","keywords":[]}],'
        '"summary":""}'
    )
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=f"请分析：\n\n{text}")])
    return _parse_json_response(resp.content)


def _extract_task_design(llm, objectives: dict, original_text: str) -> dict:
    system = (
        "你是教育设计专家。根据学习目标和课程标准，生成任务风格标准与结构化任务树。\n"
        "仅输出 JSON，不要有任何其他文字，JSON 中所有文本性内容须使用英文（English）书写：\n"
        '{"task_style_criteria":{"cognitive_level":"","task_types":[],"interaction_style":"",'
        '"language_complexity":"","feedback_style":"","rationale":""},'
        '"structured_task_tree":{"root_title":"","nodes":[],"learning_path_description":""}}'
    )
    user = (
        f"学习目标：\n{json.dumps(objectives, ensure_ascii=False, indent=2)}\n\n"
        f"原始课程文档：\n{original_text}"
    )
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return _parse_json_response(resp.content)


def _run_stage2(api_key: str) -> list:
    files = [f for f in sorted(SYL_DIR.iterdir()) if f.suffix.lower() in SUPPORTED]
    if not files:
        raise ValueError("No readable files found in syl/")

    PROS_DIR.mkdir(exist_ok=True)
    analyses = []
    for file in files:
        text = read_file(file)
        objectives = _extract_learning_objectives(ChatOpenAI(model=MODEL, temperature=0, api_key=api_key), text)
        task_design = _extract_task_design(ChatOpenAI(model=MODEL, temperature=0, api_key=api_key), objectives, text)
        result = {"source_file": file.name, "learning_objectives": objectives, "student_analysis": task_design}
        out_path = PROS_DIR / (file.stem + "_analysis.json")
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        analyses.append(result)
    return analyses


# ── Stage 4 ────────────────────────────────────────────────────────────────────

def _build_mcq_system(has_skillmap: bool) -> str:
    skillmap_note = (
        "\n你还会收到一份【学生能力图谱（skillmap）】，记录了该学生的强项、弱项和推荐确认技能。"
        "出题时须优先覆盖学生的弱项和推荐确认技能对应的学习目标，帮助学生针对性巩固薄弱环节。\n"
        if has_skillmap else ""
    )
    return (
        "你是 EvaluationQuestionGeneratorAgent，专业的课堂评价题目设计专家。\n\n"
        "你会收到：\n"
        "1. 课程学习目标与任务设计（JSON）\n"
        "2. 文本语言风格标准（JSON）\n"
        f"{skillmap_note}"
        "\n重要提示：这些题目是给学生作答的，必须以学生为中心设计。\n"
        "出题前须先读懂 document_semantics：\n"
        "- who_reader 决定题目的语气和表达方式\n"
        "- what / why 决定题目考察的侧重点与深度\n\n"
        "你的任务：生成恰好 5 道单选题，供学生在课堂作答。\n\n"
        "出题要求（严格执行）：\n"
        "- 题目数量：恰好 5 道，不多不少\n"
        "- 题目类型：全部为单选题，每题提供 A、B、C、D 四个选项，只有一个正确答案\n"
        "- 难度覆盖低、中、高三个层次（布鲁姆分类法），分布约为 2:2:1\n"
        "- 每道题须标注对应的学习目标ID\n"
        "- 题目表述的语气须与 who_reader 身份高度一致，亲切自然\n\n"
        "题目、选项、解析等所有学生可见文本必须使用英文（English）书写。\n"
        "仅输出以下 JSON 格式，不要有任何其他文字：\n"
        '{"course_name":"","source_file":"","round":1,'
        '"generated_questions":['
        '{"id":"Q001","bloom_level":"","difficulty":"","objective_ids":["LO-001"],'
        '"question":"","options":["A. ...","B. ...","C. ...","D. ..."],'
        '"answer":"A","explanation":""}],'
        '"question_count":5}'
    )


MCQ_VALIDATOR_SYSTEM = """你是 EvaluationQuestionValidatorAgent，负责对学生单选题做最小范围的硬规则校验。

只检查以下三条硬规则，其他任何问题一律 PASS：
规则1【目标相关性】：5道题作为整体，不得与课程学习目标完全无关。
规则2【内容安全】：题目和选项中不得含有歧视性词汇或不良价值观引导。
规则3【受众正确】：题目必须是面向学生提问的，不得写成面向教师或家长的表述。

仅当某条规则被明确、显著违反时才返回 REJECT。
必须严格按以下 JSON 格式输出（reason 与 suggestions 中的文本须使用英文 English 书写）：{"result": "PASS或REJECT", "reason": "", "suggestions": []}"""


def _validate_mcq_format(questions: dict) -> tuple:
    qs = questions.get("generated_questions", [])
    if len(qs) != 5:
        return False, f"Incorrect number of questions: expected 5, got {len(qs)}"
    for q in qs:
        if len(q.get("options", [])) != 4:
            return False, f"Question {q.get('id')} does not have exactly 4 options"
        if q.get("answer", "").upper() not in ("A", "B", "C", "D"):
            return False, f"Question {q.get('id')} has an invalid answer"
    return True, ""


def _run_stage4(api_key: str, analyses: list, style_text: str, skillmap=None, round_num=1) -> list:
    all_results = []

    for analysis in analyses:
        source = analysis.get("source_file", "unknown")
        questions = None
        content_feedback = []

        for attempt in range(1, MAX_RETRIES + 1):
            gen_llm = ChatOpenAI(model=MODEL, temperature=0.5, api_key=api_key)
            user_parts = [
                "【课程学习目标与任务设计】\n" + json.dumps(analysis, ensure_ascii=False, indent=2),
                "【语言风格标准】\n" + style_text,
            ]
            if skillmap:
                user_parts.append("【学生能力图谱（skillmap）】\n" + json.dumps(skillmap, ensure_ascii=False, indent=2))
            if content_feedback:
                user_parts.append("【上一版本审核未通过，请根据以下意见修改题目】\n" + "\n".join(f"- {s}" for s in content_feedback))

            try:
                resp = gen_llm.invoke([
                    SystemMessage(content=_build_mcq_system(skillmap is not None)),
                    HumanMessage(content="\n\n".join(user_parts)),
                ])
                questions = _parse_json_response(resp.content)
                questions["round"] = round_num
            except Exception:
                questions = None
                continue

            is_valid, error_msg = _validate_mcq_format(questions)
            if not is_valid:
                questions = None
                continue

            val_llm = ChatOpenAI(model=MODEL, temperature=0, api_key=api_key)
            objectives = analysis.get("learning_objectives", {}).get("objectives", [])
            user_content = (
                "【待审核题目】\n" + json.dumps(questions, ensure_ascii=False, indent=2)
                + "\n\n【课程学习目标】\n" + json.dumps(objectives, ensure_ascii=False, indent=2)
            )
            val_resp = val_llm.invoke([SystemMessage(content=MCQ_VALIDATOR_SYSTEM), HumanMessage(content=user_content)])
            try:
                verdict = _parse_json_response(val_resp.content)
            except Exception:
                verdict = {"result": "PASS"}

            if verdict.get("result", "PASS").upper() == "PASS":
                break
            else:
                content_feedback = verdict.get("suggestions", [])
                questions = None

        if questions is None and analyses:
            # fallback: one more attempt without validator
            try:
                gen_llm = ChatOpenAI(model=MODEL, temperature=0.5, api_key=api_key)
                resp = gen_llm.invoke([
                    SystemMessage(content=_build_mcq_system(skillmap is not None)),
                    HumanMessage(content="\n\n".join([
                        "【课程学习目标与任务设计】\n" + json.dumps(analysis, ensure_ascii=False, indent=2),
                        "【语言风格标准】\n" + style_text,
                    ])),
                ])
                questions = _parse_json_response(resp.content)
                questions["round"] = round_num
            except Exception:
                continue

        if questions:
            all_results.append(questions)

    filename = "evaquest_round2.json" if round_num == 2 else "evaquest.json"
    (EVAL_DIR / filename).write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_results


# ── Stage 5 ────────────────────────────────────────────────────────────────────

CAPABILITY_ANALYZER_SYSTEM = """你是 StudentCapabilityAnalyzerAgent，专业的学生能力诊断专家。

你会收到：
1. 一套课堂测试题目（JSON）
2. 学生的答题记录（JSON）
3. 学习目标说明（JSON）

分析维度：
- 强项（strengths）：答对率 ≥ 80% 的学习目标
- 弱项（weaknesses）：答对率 ≤ 40% 或全错的学习目标
- 推荐确认技能（recommended_to_verify）：答对率在 41%～79% 之间

仅输出以下 JSON 格式，不要有任何其他文字：
{
  "student_performance_summary": {"total_questions": 5, "correct_count": 0, "accuracy_rate": "0%"},
  "strengths": [{"objective_id": "LO-001", "objective_description": "", "evidence": ""}],
  "weaknesses": [{"objective_id": "LO-002", "objective_description": "", "evidence": ""}],
  "recommended_to_verify": [{"objective_id": "LO-003", "objective_description": "", "evidence": ""}],
  "recommendations": ["建议1"]
}
所有文本性字段（objective_description / evidence / recommendations 等）必须使用英文（English）书写。"""


def _run_stage5(api_key: str, user_answers: dict) -> dict:
    evaquest_path = EVAL_DIR / "evaquest.json"
    if not evaquest_path.exists():
        raise ValueError("evaquest.json not found")

    questions_all = json.loads(evaquest_path.read_text(encoding="utf-8"))

    _, analyses = _load_pros_files()
    objective_map = {}
    for analysis in analyses:
        for obj in analysis.get("learning_objectives", {}).get("objectives", []):
            objective_map[obj["id"]] = obj.get("objective", "")

    all_skillmaps = []
    for course_qs, course_ans in zip(questions_all, user_answers.get("per_course", [])):
        llm = ChatOpenAI(model=MODEL, temperature=0, api_key=api_key)
        user_content = (
            "【课堂测试题目】\n" + json.dumps(course_qs, ensure_ascii=False, indent=2)
            + "\n\n【学生答题记录】\n" + json.dumps(course_ans, ensure_ascii=False, indent=2)
            + "\n\n【学习目标参考说明】\n" + json.dumps(objective_map, ensure_ascii=False, indent=2)
        )
        resp = llm.invoke([SystemMessage(content=CAPABILITY_ANALYZER_SYSTEM), HumanMessage(content=user_content)])
        try:
            skillmap = _parse_json_response(resp.content)
            skillmap["source_file"] = course_qs.get("source_file", "unknown")
            all_skillmaps.append(skillmap)
        except Exception:
            pass

    output = all_skillmaps[0] if len(all_skillmaps) == 1 else all_skillmaps

    # Save per-round file: skillmap_r01.json, skillmap_r02.json ...
    existing = sorted(PROS_DIR.glob("skillmap_r*.json"))
    next_round = len(existing) + 1
    round_path = PROS_DIR / f"skillmap_r{next_round:02d}.json"
    round_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    # Also overwrite skillmap.json as the latest (backward compat)
    (PROS_DIR / "skillmap.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _load_pros_files():
    style_path = PROS_DIR / "document_style_criteria.json"
    style_text = style_path.read_text(encoding="utf-8") if style_path.exists() else ""
    analyses = []
    for p in sorted(PROS_DIR.glob("*_analysis.json")):
        try:
            analyses.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return style_text, analyses


# ── Stage 6a: Progress Benefit Agent ──────────────────────────────────────────

PROGRESS_BENEFIT_SYSTEM = """你是 ProgressBenefitAgent，专门评估学生继续练习的潜在收益。

你会收到：
1. 学生所有历史答题记录（按轮次排列）
2. 当前能力图谱（strengths / weaknesses / recommended_to_verify）
3. 学习目标列表（含布鲁姆分类层次）
4. 当前已完成轮次数

你的任务：
- 分析当前弱项和待确认技能中，还有哪些布鲁姆高层次目标（分析/综合/评价，即 bloom_level 为 analyze/evaluate/create）尚未掌握
- 评估再做一轮能带来的学习提升空间
- 如果弱项多且涉及高层次认知目标，收益分应高；如果弱项少或已接近掌握，收益分应低

评分标准（0-100）：
- 90-100：大量高层次弱项未覆盖，极高提升空间
- 60-89：有若干弱项，继续练习有明显价值
- 30-59：弱项较少，收益有限
- 0-29：几乎无弱项，或已多轮无实质进步

仅输出以下 JSON，不要有任何其他文字：
{"benefit_score": 75, "reasoning": "简短说明（English，1-2句）", "key_gaps": ["LO-002", "LO-005"]}"""


def _run_progress_benefit_agent(api_key: str, all_history: list, skillmap: dict,
                                 analyses: list, current_round: int) -> dict:
    """评估再做一轮的潜在收益（0-100）。"""
    objective_list = []
    for analysis in analyses:
        for obj in analysis.get("learning_objectives", {}).get("objectives", []):
            objective_list.append(obj)

    user_content = (
        f"【已完成轮次数】{current_round}\n\n"
        "【历史答题记录（按轮次）】\n"
        + json.dumps(all_history, ensure_ascii=False, indent=2)
        + "\n\n【当前能力图谱】\n"
        + json.dumps(skillmap, ensure_ascii=False, indent=2)
        + "\n\n【学习目标（含布鲁姆层次）】\n"
        + json.dumps(objective_list, ensure_ascii=False, indent=2)
    )

    llm = ChatOpenAI(model=MODEL, temperature=0, api_key=api_key)
    resp = llm.invoke([SystemMessage(content=PROGRESS_BENEFIT_SYSTEM), HumanMessage(content=user_content)])
    try:
        result = _parse_json_response(resp.content)
    except Exception:
        result = {"benefit_score": 50, "reasoning": "Score parsing failed, using default value", "key_gaps": []}

    # Save to file
    out_path = EVAL_DIR / f"benefit_r{current_round:02d}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# ── Stage 6b: Burnout Risk Agent ───────────────────────────────────────────────

BURNOUT_RISK_SYSTEM = """你是 BurnoutRiskAgent，专门评估学生继续练习的倦怠风险。

你会收到：
1. 学生所有历史答题记录（按轮次排列）
2. 当前已完成轮次数

你的任务，综合评估以下三个维度：
- 重复错误：同一知识点（objective_id）在多轮中反复答错，说明陷入困境
- 连续错误率：最近一轮中连续答错的题目比例（连续3题及以上答错视为高风险信号）
- 轮次惩罚：已完成轮次越多，边际倦怠风险越高（第1轮基础风险低，第8轮以上风险显著升高）

评分标准（0-100，越高越危险）：
- 0-20：低风险，状态良好
- 21-50：中等风险，可继续但需关注
- 51-75：较高风险，建议谨慎
- 76-100：高风险，强烈建议停止

仅输出以下 JSON，不要有任何其他文字：
{"burnout_score": 40, "reasoning": "简短说明（English，1-2句）", "risk_flags": ["repeated_error:LO-003", "high_consecutive_errors"]}"""


def _run_burnout_risk_agent(api_key: str, all_history: list, current_round: int) -> dict:
    """评估继续练习的倦怠风险（0-100）。"""
    user_content = (
        f"【已完成轮次数】{current_round}\n\n"
        "【历史答题记录（按轮次）】\n"
        + json.dumps(all_history, ensure_ascii=False, indent=2)
    )

    llm = ChatOpenAI(model=MODEL, temperature=0, api_key=api_key)
    resp = llm.invoke([SystemMessage(content=BURNOUT_RISK_SYSTEM), HumanMessage(content=user_content)])
    try:
        result = _parse_json_response(resp.content)
    except Exception:
        result = {"burnout_score": 50, "reasoning": "Score parsing failed, using default value", "risk_flags": []}

    # Save to file
    out_path = EVAL_DIR / f"burnout_r{current_round:02d}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# ── Stage 6c: Feedback Text Generator Agent ───────────────────────────────────

FEEDBACK_GENERATOR_SYSTEM = """你是 FeedbackTextGeneratorAgent，负责在学生完成所有练习轮次后，生成一段温暖、真诚、有建设性的总结反馈。

你会收到：
1. 所有轮次的答题记录
2. 所有轮次的能力图谱（skillmap）历史
3. 所有轮次的收益/风险评分记录
4. 最终停止原因（系统内部判断）
5. 总完成轮次数

你的任务：
- 用第二人称（"你"）直接和学生说话
- 语气温暖鼓励，像一位关心学生的老师，不要像系统报告
- 不要提及"收益分"、"风险分"、"Agent"等技术词汇
- 结构建议：① 开头肯定努力 ② 指出进步或亮点 ③ 点出需要继续加强的方向 ④ 鼓励收尾
- 长度控制在 100-180 字之间，自然分段
- 如果提供了【教师个人反馈风格画像】，请在语气、句式、结构习惯上尽量贴近该教师的偏好，
  但仍必须遵守以上关于结构与字数的硬性要求

反馈正文（feedback）与亮点总结（highlight）必须使用英文（English）书写，即使以上指示使用中文描述。
仅输出以下 JSON，不要有任何其他文字：
{"feedback": "反馈正文（英文 English，纯文本，可用\\n分段）", "highlight": "一句话亮点总结（英文 English，简短）"}"""


def _run_feedback_generator(api_key: str, all_history: list, all_skillmaps: list,
                             all_decisions: list, total_rounds: int, stop_reason: str,
                             tutor_style: dict | None = None) -> dict:
    """生成面向学生的自然语言反馈草稿（AI 初稿，需教师审核后才会展示给学生）。"""
    tutor_style_block = ""
    if tutor_style:
        tutor_style_block = (
            "\n\n【教师个人反馈风格画像（请尽量贴近这种风格）】\n"
            + json.dumps(tutor_style, ensure_ascii=False, indent=2)
        )

    user_content = (
        f"【总完成轮次数】{total_rounds}\n\n"
        f"【停止原因（内部参考，不要直接引用）】{stop_reason}\n\n"
        "【各轮答题记录】\n"
        + json.dumps(all_history, ensure_ascii=False, indent=2)
        + "\n\n【各轮能力图谱历史】\n"
        + json.dumps(all_skillmaps, ensure_ascii=False, indent=2)
        + "\n\n【各轮收益/风险评分记录】\n"
        + json.dumps(all_decisions, ensure_ascii=False, indent=2)
        + tutor_style_block
    )

    llm = ChatOpenAI(model=MODEL, temperature=0.4, api_key=api_key)
    resp = llm.invoke([SystemMessage(content=FEEDBACK_GENERATOR_SYSTEM), HumanMessage(content=user_content)])
    try:
        result = _parse_json_response(resp.content)
    except Exception:
        result = {
            "feedback": "You've completed this round of practice, and your effort deserves recognition! Keep up your curiosity and drive to learn — you're on the right track.",
            "highlight": "Completed all practice rounds"
        }

    # Save draft to file (final version is written separately, only after teacher approval)
    (EVAL_DIR / "feedback_draft.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


# ── Stage 6d: Tutor Style Analyzer Agent (独立、只做风格提炼，不评判对错) ───────────
# 无论教师是否修改了 AI 原稿，都会被调用：完全批准也是一种表态（说明这个版本符合/不违背
# 教师的风格取向），同样是可以用来强化或校准风格画像的证据，只是证据强度和"主动改写"不同。

STYLE_ANALYZER_SYSTEM = """你是 TutorStyleAnalyzerAgent，负责通过对比"AI生成的反馈初稿"与"教师最终确认的版本"，
提炼并维护该教师的个人反馈风格画像，供后续 AI 生成反馈时参考模仿。

你会收到：
1. 【教师历史风格画像】——之前累积总结的 JSON，可能为空（{}）
2. 【是否有实质修改】——true 表示教师主动改写过内容，false 表示教师完全原样批准了 AI 草稿
3. 【AI 原始草稿】
4. 【教师最终确认的版本】

你的任务：
- 若【是否有实质修改】为 true：对比两版本之间的具体改动（增删改的词句、语气变化、结构调整、
  举例方式、称呼习惯等），提炼**稳定、可复用的风格取向**（"教师喜欢怎么表达"）。不要把偶然的
  一次性修改当成风格特征。
- 若【是否有实质修改】为 false（教师完全原样批准）：这本身也是一次有效的风格信号——说明 AI
  这次呈现的语气/句式/结构是符合该教师认可范围的。请据此**强化**历史画像中与本次文本相符的
  特征的置信度（在 confidence_notes 中体现），但不要凭空编造新的风格特征——没有改动就没有
  "新证据"，只有"对已有判断的确认"。
- 将本次发现（或确认）与历史画像**融合更新**：保留仍然成立的旧特征、强化被反复印证的特征、
  补充新发现的特征；若新证据与旧画像冲突，可调整表述或在 confidence_notes 中注明
- 严禁对改动（或未改动）的好坏做任何评判（不要输出"这样改是否合适""是否应该保留"之类的评价，
  那不是你的职责，只需要客观描述教师的表达偏好）

仅输出以下 JSON 格式的、融合更新后的完整画像，不要有任何其他文字：
{
  "tone_preference": "",
  "sentence_style": "",
  "structure_preference": "",
  "vocabulary_notes": "",
  "common_edits": ["教师常做的修改类型，例如：偏好补充具体的知识点名称"],
  "example_phrases": ["教师偏好使用的表达片段（若有明显重复出现的可提炼，没有可留空数组）"],
  "confidence_notes": "简短说明（English）哪些特征已被多次印证、哪些还只是初步观察",
  "last_updated_round": 0
}
所有文本性字段（tone_preference / sentence_style / structure_preference / vocabulary_notes / common_edits / example_phrases / confidence_notes 等）必须使用英文（English）书写。"""


def _run_style_analyzer(api_key: str, old_style: dict, ai_text: str, teacher_text: str,
                         edited: bool, round_num: int = 0) -> dict:
    """独立 Agent：只提炼教师的风格取向，更新并持久化 TutorStyle.json。无论教师是否修改都会调用。"""
    llm = ChatOpenAI(model=MODEL, temperature=0.2, api_key=api_key)
    user_content = (
        "【教师历史风格画像】\n" + json.dumps(old_style or {}, ensure_ascii=False, indent=2)
        + f"\n\n【是否有实质修改】{'true' if edited else 'false'}"
        + "\n\n【AI 原始草稿】\n" + ai_text
        + "\n\n【教师最终确认的版本】\n" + teacher_text
    )
    resp = llm.invoke([SystemMessage(content=STYLE_ANALYZER_SYSTEM), HumanMessage(content=user_content)])
    try:
        result = _parse_json_response(resp.content)
    except Exception:
        result = old_style or {}
    result["last_updated_round"] = round_num
    TUTOR_STYLE_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# ── Stage 6e: Tutor Advise Agent (独立、中立评审，三视角结构化输出) ────────────────
# 同样无论教师是否修改都会调用：即便教师完全批准，也可以从三个视角评价"批准"这个行为本身
# （例如 AI 原文其实有可改进之处但教师未察觉/选择不改，这本身也是值得记录的反馈素养信号）。

TUTOR_ADVISE_SYSTEM = """你是 TutorAdviseAgent，一名中立、专业的反馈质量评审员与教练。
你的评审对象是"AI生成的反馈草稿"与"教师最终确认版本"之间的关系——教师可能做了大量修改，
也可能一字未改直接批准；无论哪种情况，这都是一次可以被评价的教师行为，你都需要给出评价。

一份好的学生反馈应当满足以下教育学原则（评价【教育学反馈策略】视角时以此为依据，
每条前面的【短标签】是给你自己检索用的，**输出时必须写出原则的具体内容，禁止只写编号**）：
1. 【语气与人称】用第二人称（"你"）直接和学生说话，语气温暖鼓励，像关心学生的老师，而不是系统报告
2. 【避免技术术语】不包含"收益分"、"风险分"、"Agent"等技术词汇，不暴露系统内部机制
3. 【结构完整】① 开头肯定努力 ② 指出进步或亮点 ③ 点出需要继续加强的方向 ④ 鼓励收尾
4. 【长度适中】约100-180字，不过长导致学生失去耐心，也不过短导致内容空洞
5. 【内容具体有依据】不是空洞套话，知识点归因准确，不能凭空指责或表扬学生
6. 【保护自信心】不打击学生自信心，不使用说教、指责、贴负面标签的语言
7. 【保护隐私与感受】保护学生的心理感受和隐私，不做不当比较（如与其他学生比较）

请从以下三个独立视角分别输出评价条目：

【视角一：behavioral（教师行为描绘）】
客观、中立地描述教师这次实际做了什么操作——增加了什么内容、删除了什么、调整了语气/结构，
或者完全保留了 AI 版本、未做任何改动。这个视角本身应尽量保持中立描述（stance 用 "neutral"），
但如果某个具体行为在客观上明显造成了信息缺失、引入了错误内容，可标记为 "concern"；
如果某行为体现出良好、一贯的操作习惯，也可以标记为 "good"。

【视角二：pedagogical（教育学反馈策略）】
依据上面 7 条教育学原则，评价教师的改动或"未改动的选择"是否符合好反馈的标准。
"good" = 符合/强化了某条原则；"concern" = 违反了某条原则、降低了反馈质量。若某处改动同时
涉及多条原则，请拆成多个独立条目分别说明，不要把多条原则合并写进一个 principle 字段。
如果教师完全批准，但 AI 原文本身在某条原则上有欠缺，也应该在这里指出——这提示教师审核时
可以更关注这一类问题。

**重要格式要求**：面向教师的所有文字（observation / principle / impact / note / suggestion /
overall_comment）都必须是可以脱离本提示词、独立读懂的自然语言。绝对不允许出现"违反了原则1"
"符合原则3和6"这类只写编号、不写内容的表述——教师看不到这份提示词里的编号列表，只写编号
等于没说清楚。正确示例："『保护自信心』：这句话使用了'你怎么又错了'这类说教式表达，容易让
学生感到被指责"；错误示例："违反了原则6"。

【视角三：growth（教师职业发展 · 反馈素养）】
这个视角不是让你随意发挥"教练式建议"，而是要依据以下两套教育学界公认的理论框架来分析
（分析时参照这些理论的内核，但输出给教师的文字必须转化成通俗、口语化的英文（English），不要堆砌
人名、年份或生硬的学术术语）：

(a) Hattie & Timperley 反馈层级理论：有效的反馈应该回答三个问题——目标是什么（这次练习
    在检验什么能力）、目前进展如何（学生实际表现出的水平）、下一步怎么做（具体可执行的
    改进方向）。同时，反馈内容可以作用在四个层级：① 任务层级（只说对错本身）② 过程层级
    （解题思路、方法、知识点运用）③ 自我调节层级（学习策略、复习方法、元认知）④ 自我层级
    （空泛的表扬或打气，如"你真棒""继续加油"）。研究发现，作用在"过程"和"自我调节"层级
    的反馈对学习效果提升最大；只停留在"自我层级"的空泛表扬效果最弱，容易让学生只关注
    "被夸奖"本身而不是真正该学的东西。→ 评价时可以看：教师的编辑是把反馈往"过程/自我调节"
    层级拉，还是往"自我层级"的空泛表扬退？是否清楚交代了目标、进展、下一步这三块？

(b) 教师反馈素养的三个维度 + 一个持续改进的习惯：① 设计维度——反馈是否是学生能真正"拿去用"
    的（比如给出的方向具体到学生下次能照做，而不是抽象笼统）② 关系维度——反馈过程中是否
    体现了信任、敏感和真诚，让学生愿意相信这份反馈是为他好，而不是被审判③ 务实维度——
    有限的篇幅和精力有没有花在真正影响学习效果的地方，而不是面面俱到却都很浅④ 持续反思
    的习惯——教师是否愿意批判性地审视和调整自己一贯的反馈方式（这正是本系统持续积累
    TutorStyle 教师风格画像、逐轮优化生成的意义所在）。→ 评价时可以看：这次的编辑/批准，
    体现出教师在哪个维度比较成熟，哪个维度还可以打磨。

结合以上两套框架，评价这次处理反映出该教师在"反馈素养"上哪些方面已经比较成熟（"good"），
哪些方面还有提升空间（"concern"），并给出具体、可操作、鼓励性的下一步建议——建议要具体到
"下次可以怎么写"，不能是"要更用心"这种空话。这个视角关注的是教师本人的成长轨迹，不是本次
反馈文本对错，语气应像一位善意的同行导师，而不是考官。

保持中立：不因为"这是教师改的"就默认合理，也不因为"AI是系统生成的"就默认更权威。
教师和 AI 都可能出错，都要如实评价。若教师完全批准且各方面都无问题，三个视角也可以只输出
"good"条目，不必强行制造"concern"。

仅输出以下 JSON 格式，不要有任何其他文字：
{
  "behavioral": [
    {"stance": "good|concern|neutral", "observation": "客观描述教师这次做了什么", "note": "补充说明（可选，无则留空字符串）"}
  ],
  "pedagogical": [
    {"stance": "good|concern", "observation": "具体的改动或保留内容", "principle": "『原则名称』：原则的具体内容是什么（完整自然语言，不能只写编号）", "impact": "对学生的可能影响"}
  ],
  "growth": [
    {"stance": "good|concern", "observation": "该项反映出的反馈素养现状", "lens": "对应上面理论中的哪个层级/维度（用通俗英文 English 说明，例如'process-level feedback'或'relational dimension'，不要写生硬的人名缩写）", "suggestion": "给教师的下一步提升建议（教练式语气，具体可操作，不能是空话）"}
  ],
  "overall_comment": "一句话总评（English，综合三个视角，可以是肯定，也可以是委婉但明确的提醒）"
}
面向教师的所有文字字段（observation / principle / impact / note / suggestion / lens / overall_comment 等）必须使用英文（English）书写，即使以上说明使用中文撰写。"""


def _run_tutor_advise(api_key: str, ai_text: str, teacher_text: str,
                       edited: bool, round_num: int = 0) -> dict:
    """独立 Agent：三视角评审（不影响 TutorStyle），结果仅展示给教师。无论教师是否修改都会调用。"""
    llm = ChatOpenAI(model=MODEL, temperature=0.2, api_key=api_key)
    user_content = (
        f"【是否有实质修改】{'true' if edited else 'false'}"
        + "\n\n【AI 原始草稿】\n" + ai_text
        + "\n\n【教师最终确认的版本】\n" + teacher_text
    )
    resp = llm.invoke([SystemMessage(content=TUTOR_ADVISE_SYSTEM), HumanMessage(content=user_content)])
    try:
        result = _parse_json_response(resp.content)
    except Exception:
        result = {"behavioral": [], "pedagogical": [], "growth": [], "overall_comment": "Analysis failed, please try again later."}
    result["round"] = round_num
    result["edited"] = edited
    TUTOR_TIPS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# ── Continuation decision ──────────────────────────────────────────────────────

MAX_ROUNDS = 10


def _should_continue(benefit_result: dict, burnout_result: dict, current_round: int) -> tuple[bool, str]:
    """
    决定是否继续下一轮。
    规则：benefit_score < burnout_score * 0.8 → 停止
    返回 (should_continue: bool, reason: str)
    """
    benefit = benefit_result.get("benefit_score", 50)
    burnout = burnout_result.get("burnout_score", 50)
    threshold = burnout * 0.8

    if current_round >= MAX_ROUNDS:
        return False, f"Reached the maximum number of rounds ({MAX_ROUNDS})"
    if benefit < threshold:
        return False, (
            f"Benefit score ({benefit}) < risk score x0.8 ({threshold:.1f}), "
            f"so continuing is not worthwhile. {burnout_result.get('reasoning', '')}"
        )
    return True, (
        f"Benefit score ({benefit}) >= risk score x0.8 ({threshold:.1f}), so continuing is worthwhile. "
        f"{benefit_result.get('reasoning', '')}"
    )


# ── Pipeline runner (async) ────────────────────────────────────────────────────

async def _await_teacher_feedback_approval(
    loop, api_key: str, all_history: list, all_skillmap_history: list,
    decisions: list, total_rounds: int, stop_reason: str, stopped_early: bool,
):
    """
    生成 AI 反馈初稿 → 通知教师审核 → 阻塞等待教师提交最终版本。
    教师提交（POST /feedback/approve）会：
      1. 把 STATE.feedback_status 设为 "approved"
      2. 把最终版本推送给学生（学生只会看到教师确认后的版本）
      3. 若教师做了实质性修改，触发 Style Analyzer / Tutor Advise 两个独立 Agent
    本函数只负责生成初稿 + 挂起等待，不负责后续推送，实际推送在 /feedback/approve 里完成。
    """
    tutor_style = _load_tutor_style()
    feedback_result = await loop.run_in_executor(
        None, lambda: _run_feedback_generator(
            api_key, all_history, all_skillmap_history, decisions,
            total_rounds, stop_reason, tutor_style,
        )
    )
    STATE.feedback_draft = {
        "feedback":      feedback_result.get("feedback", ""),
        "highlight":     feedback_result.get("highlight", ""),
        "round":         total_rounds,
        "stopped_early": stopped_early,
    }
    STATE.feedback_status = "pending"
    STATE.status = "awaiting_feedback_approval"

    await STATE.teacher("feedback_pending", {
        "message": "The AI has generated a feedback draft. It will be sent to the student once you review and approve it.",
        "draft": STATE.feedback_draft,
    })
    await STATE.student("waiting_feedback", {
        "message": "You have finished this round of practice. Your teacher is finalizing your feedback, please wait...",
    })

    # Block until the teacher approves via POST /feedback/approve
    while STATE.feedback_status != "approved":
        await asyncio.sleep(1)

    STATE.status = "done"


async def _run_pipeline(api_key: str):
    loop = asyncio.get_event_loop()
    STATE.status = "running"

    async def stage_msg(stage: int, msg: str, status: str = "progress"):
        STATE.stage = stage
        STATE.stage_msg = msg
        await STATE.teacher("stage", {"stage": stage, "status": status, "message": msg})

    try:
        # ── Stage 1 ──────────────────────────────────────────────
        await stage_msg(1, "Stage 1 — Analyzing reference document style...")
        style_text = await loop.run_in_executor(None, lambda: _run_stage1(api_key))
        await stage_msg(1, "Stage 1 complete ✓ — Style criteria extracted", "done")

        # ── Stage 2 ──────────────────────────────────────────────
        await stage_msg(2, "Stage 2 — Analyzing syllabus and extracting learning objectives...")
        analyses = await loop.run_in_executor(None, lambda: _run_stage2(api_key))
        await stage_msg(2, f"Stage 2 complete ✓ — Processed {len(analyses)} syllabus file(s)", "done")

        await STATE.student("waiting", {"message": "Syllabus analysis complete. Generating questions, please wait..."})

        style_text_loaded, analyses_loaded = await loop.run_in_executor(None, _load_pros_files)
        skillmap = None

        # ── Multi-round loop (up to MAX_ROUNDS) ──────────────────
        for round_num in range(1, MAX_ROUNDS + 1):
            STATE.current_round = round_num

            # Generate questions for this round
            await stage_msg(4, f"Stage 4 — Generating assessment questions (round {round_num})...")
            questions = await loop.run_in_executor(
                None, lambda rn=round_num, sm=skillmap: _run_stage4(
                    api_key, analyses_loaded, style_text_loaded, sm, rn
                )
            )

            # Keep round-1 / round-2 refs for backward-compat API endpoints
            if round_num == 1:
                STATE.questions_r1 = questions
            elif round_num == 2:
                STATE.questions_r2 = questions
            STATE.questions_current = questions

            await stage_msg(4, f"Stage 4 complete ✓ — Round {round_num} questions generated, waiting for student answers", "done")

            # Notify student
            event = "questions_ready" if round_num == 1 else "round_ready"
            await STATE.student(event, {
                "message": f"Round {round_num} questions are ready, please start answering!",
                "questions": questions,
                "round": round_num,
            })

            STATE.status = "waiting_answers"
            STATE.user_answers = None  # reset for this round
            await STATE.teacher("waiting_answers", {
                "message": f"Round {round_num} questions have been sent, waiting for the student to submit answers...",
                "round": round_num,
            })

            # Wait for student answers
            while STATE.user_answers is None:
                await asyncio.sleep(1)

            # Append this round's answers to history
            round_record = {"round": round_num, "answers": STATE.user_answers}
            STATE.all_answers_history.append(round_record)
            (EVAL_DIR / f"user_answers_r{round_num}.json").write_text(
                json.dumps(round_record, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            STATE.status = "analyzing"
            await STATE.teacher("analyzing", {
                "message": f"Received round {round_num} answers, analyzing skill map...",
                "round": round_num,
            })

            # ── Stage 5: update skillmap ──────────────────────────
            await stage_msg(5, f"Stage 5 — Analyzing round {round_num} skill map...")
            skillmap = await loop.run_in_executor(
                None, lambda: _run_stage5(api_key, STATE.user_answers)
            )
            STATE.skillmap = skillmap
            await stage_msg(5, f"Stage 5 complete ✓ — Round {round_num} skill map generated", "done")
            await STATE.teacher("skillmap_ready", {"skillmap": skillmap, "round": round_num})

            # ── Decide whether to continue ────────────────────────
            # (skip decision after final allowed round)
            if round_num >= MAX_ROUNDS:
                STATE.stop_reason = f"Reached the maximum number of rounds ({MAX_ROUNDS})"
                await STATE.teacher("decision", {
                    "round": round_num,
                    "continue": False,
                    "reason": STATE.stop_reason,
                    "benefit_score": None,
                    "burnout_score": None,
                })
                break

            STATE.status = "deciding"
            await STATE.teacher("deciding", {
                "message": f"Evaluating whether to continue to round {round_num + 1}...",
                "round": round_num,
            })

            benefit_result = await loop.run_in_executor(
                None, lambda: _run_progress_benefit_agent(
                    api_key, STATE.all_answers_history, skillmap, analyses_loaded, round_num
                )
            )
            burnout_result = await loop.run_in_executor(
                None, lambda: _run_burnout_risk_agent(
                    api_key, STATE.all_answers_history, round_num
                )
            )

            go_on, reason = _should_continue(benefit_result, burnout_result, round_num)

            decision = {
                "round": round_num,
                "continue": go_on,
                "reason": reason,
                "benefit_score": benefit_result.get("benefit_score"),
                "benefit_reasoning": benefit_result.get("reasoning"),
                "burnout_score": burnout_result.get("burnout_score"),
                "burnout_reasoning": burnout_result.get("reasoning"),
                "key_gaps": benefit_result.get("key_gaps", []),
                "risk_flags": burnout_result.get("risk_flags", []),
            }
            STATE.continuation_decisions.append(decision)
            await STATE.teacher("decision", decision)

            if not go_on:
                STATE.stop_reason = reason
                # Generate feedback draft and wait for teacher review/approval before it reaches the student
                all_skillmap_history = [
                    json.loads(p.read_text(encoding="utf-8"))
                    for p in sorted(PROS_DIR.glob("skillmap_r*.json"))
                ]
                await _await_teacher_feedback_approval(
                    loop, api_key, STATE.all_answers_history,
                    all_skillmap_history, STATE.continuation_decisions,
                    round_num, reason, True,
                )
                break

            # Notify student a new round is coming
            await STATE.student("next_round_coming", {
                "message": f"Round {round_num} analysis complete. Generating round {round_num + 1} questions for you...",
                "round": round_num + 1,
                "benefit_score": benefit_result.get("benefit_score"),
                "burnout_score": burnout_result.get("burnout_score"),
            })

        else:
            # Loop completed naturally (hit MAX_ROUNDS without break)
            STATE.stop_reason = f"已达最大轮次上限（{MAX_ROUNDS} 轮）"

        total_rounds = len(STATE.all_answers_history)

        # Generate feedback only for max-rounds end (early-stop already handled inside loop)
        if STATE.stop_reason.startswith("Reached"):
            all_skillmap_history = [
                json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(PROS_DIR.glob("skillmap_r*.json"))
            ]
            await _await_teacher_feedback_approval(
                loop, api_key, STATE.all_answers_history,
                all_skillmap_history, STATE.continuation_decisions,
                total_rounds, STATE.stop_reason, False,
            )

        STATE.status = "done"

        await STATE.teacher("done", {
            "message": f"Pipeline complete 🎉 ({total_rounds} round(s) finished)",
            "total_rounds": total_rounds,
            "stop_reason": STATE.stop_reason,
        })

    except Exception as e:
        STATE.status = "error"
        STATE.error_msg = str(e)
        await STATE.teacher("error", {"message": f"Pipeline failed: {e}"})
        await STATE.student("error", {"message": "An error occurred. Please contact your teacher."})


# ── HTTP endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("teacher.html")


@app.get("/student")
async def student_page():
    return FileResponse("student.html")


@app.get("/status")
async def get_status():
    return {
        "status":            STATE.status,
        "stage":             STATE.stage,
        "stage_msg":         STATE.stage_msg,
        "current_round":     STATE.current_round,
        "total_rounds_done": len(STATE.all_answers_history),
        "max_rounds":        MAX_ROUNDS,
        "stop_reason":       STATE.stop_reason,
        "has_questions_r1":  len(STATE.questions_r1) > 0,
        "has_questions_r2":  len(STATE.questions_r2) > 0,
        "has_skillmap":      STATE.skillmap is not None,
        "pipeline_available": PIPELINE_AVAILABLE,
        "feedback_status":   STATE.feedback_status,   # none | pending | approved
        "has_feedback_draft": STATE.feedback_draft is not None,
    }


@app.post("/upload/sampdoc")
async def upload_sampdoc(files: list[UploadFile] = File(...)):
    shutil.rmtree(SAMPDOC_DIR, ignore_errors=True)
    SAMPDOC_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        dest = SAMPDOC_DIR / f.filename
        dest.write_bytes(await f.read())
        saved.append(f.filename)
    return {"saved": saved, "count": len(saved)}


@app.post("/upload/syl")
async def upload_syl(files: list[UploadFile] = File(...)):
    shutil.rmtree(SYL_DIR, ignore_errors=True)
    SYL_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        dest = SYL_DIR / f.filename
        dest.write_bytes(await f.read())
        saved.append(f.filename)
    return {"saved": saved, "count": len(saved)}


@app.get("/files")
async def list_files():
    sampdoc = [f.name for f in SAMPDOC_DIR.iterdir() if f.suffix.lower() in SUPPORTED] if SAMPDOC_DIR.exists() else []
    syl     = [f.name for f in SYL_DIR.iterdir() if f.suffix.lower() in SUPPORTED] if SYL_DIR.exists() else []
    return {"sampdoc": sampdoc, "syl": syl}


@app.post("/run")
async def run_pipeline(payload: dict = Body(...)):
    # Any state other than idle/done/error means a pipeline is mid-flight (including
    # waiting_answers / analyzing / deciding / awaiting_feedback_approval) — resetting
    # STATE underneath a running task would corrupt it (e.g. an in-flight
    # _await_teacher_feedback_approval loop would never see "approved" again).
    if STATE.status not in ("idle", "done", "error"):
        return {"error": f"A pipeline is already in progress (current status: {STATE.status}). Please wait for it to finish first."}
    api_key = (payload.get("api_key") or "").strip()
    if not api_key:
        return {"error": "Please provide an OpenAI API Key"}
    if not PIPELINE_AVAILABLE:
        return {"error": "Missing backend dependencies: please install langchain-openai langgraph python-docx"}

    STATE.reset()
    # Clear per-run artifacts left over from a previous pipeline run — these are never
    # cleaned up elsewhere (only sampdoc/syl get rmtree'd on upload), so stale files like
    # feedback_final.json / skillmap_r*.json would otherwise leak into the new run and make
    # e.g. GET /feedback report "ready" with last round's content before this run produces
    # its own. TUTOR_DIR (TutorStyle.json / tutor_tips) is intentionally NOT cleared — that's
    # meant to persist and accumulate across runs.
    for d in (PROS_DIR, EVAL_DIR):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    asyncio.create_task(_run_pipeline(api_key))
    return {"ok": True, "message": "Pipeline started"}


@app.get("/questions")
async def get_questions():
    if not STATE.questions_r1:
        return {"ready": False, "questions": []}
    return {"ready": True, "questions": STATE.questions_r1}


@app.get("/questions/round2")
async def get_questions_r2():
    if not STATE.questions_r2:
        return {"ready": False, "questions": []}
    return {"ready": True, "questions": STATE.questions_r2}


@app.get("/questions/current")
async def get_questions_current():
    """返回当前轮次的题目（适用于多轮场景）。"""
    if not STATE.questions_current:
        return {"ready": False, "questions": [], "round": STATE.current_round}
    return {
        "ready": True,
        "questions": STATE.questions_current,
        "round": STATE.current_round,
    }


@app.get("/feedback")
async def get_feedback():
    """返回教师确认后的最终反馈文本（学生看到的版本）。审核通过前该文件不存在。"""
    p = EVAL_DIR / "feedback_final.json"
    if not p.exists():
        return {"ready": False, "feedback": None}
    try:
        return {"ready": True, **json.loads(p.read_text(encoding="utf-8"))}
    except Exception:
        return {"ready": False, "feedback": None}


@app.get("/feedback/draft")
async def get_feedback_draft():
    """返回当前待教师审核的 AI 反馈初稿（教师页面用于渲染审核界面）。"""
    if STATE.feedback_status != "pending" or not STATE.feedback_draft:
        return {"ready": False, "draft": None}
    return {"ready": True, "draft": STATE.feedback_draft}


@app.post("/feedback/approve")
async def approve_feedback(payload: dict = Body(...)):
    """
    教师审核提交（无论是否修改，都走这个接口）：
    payload: {
      "api_key": "...",           ← 触发风格/建议分析 Agent 需要
      "feedback": "教师最终确认的正文",
      "highlight": "教师最终确认的一句话亮点"
    }
    - 教师提交的版本即为最终版本，直接推送给学生和教师
    - 若教师对 AI 原稿做了实质性修改（diff 非空），才会触发 Style Analyzer 与 Tutor Advise
      两个相互独立的 Agent；若教师原样批准，则跳过（无 diff 可分析）
    """
    if STATE.feedback_status != "pending" or not STATE.feedback_draft:
        return {"error": "There is no feedback currently pending review"}

    api_key = (payload.get("api_key") or "").strip()
    teacher_feedback = (payload.get("feedback") or "").strip()
    teacher_highlight = (payload.get("highlight") or "").strip()
    if not teacher_feedback:
        return {"error": "Feedback text cannot be empty"}

    ai_draft = STATE.feedback_draft
    round_num = ai_draft.get("round", STATE.current_round)
    ai_text = ai_draft.get("feedback", "").strip()
    edited = teacher_feedback != ai_text

    final = {
        "feedback":      teacher_feedback,
        "highlight":     teacher_highlight or ai_draft.get("highlight", ""),
        "round":         round_num,
        "stopped_early": ai_draft.get("stopped_early", False),
        "edited":        edited,
    }
    STATE.feedback_final  = final
    STATE.feedback_status = "approved"
    (EVAL_DIR / "feedback_final.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 推送最终版本：学生现在才第一次看到反馈（先推送，不必等两个分析 Agent 跑完）
    await STATE.student("done", {
        "message":       final["feedback"],
        "highlight":     final["highlight"],
        "round":         final["round"],
        "stopped_early": final["stopped_early"],
    })
    await STATE.teacher("feedback_approved", {
        "message": "Feedback approved and sent to the student.",
        "feedback": final,
        "edited": edited,
    })

    # 无论教师是否修改了 AI 原稿，都要调用这两个独立 Agent——"完全批准"本身也是一种表态，
    # 同样是可供 Style Analyzer 提炼/强化风格画像、供 Tutor Advise 三视角评审的有效信号。
    tips = None
    if api_key and PIPELINE_AVAILABLE:
        loop = asyncio.get_event_loop()
        old_style = _load_tutor_style()
        # 两个 Agent 相互独立并发运行：互不读取对方的输出，互不影响
        style_task = loop.run_in_executor(
            None, lambda: _run_style_analyzer(api_key, old_style, ai_text, teacher_feedback, edited, round_num)
        )
        advise_task = loop.run_in_executor(
            None, lambda: _run_tutor_advise(api_key, ai_text, teacher_feedback, edited, round_num)
        )
        new_style, tips = await asyncio.gather(style_task, advise_task)
        STATE.tutor_style = new_style
        STATE.tutor_tips = tips
        await STATE.teacher("tutor_tips", {"tips": tips, "round": round_num})

    return {"ok": True, "edited": edited, "tips": tips}


@app.get("/tutor-tips")
async def get_tutor_tips():
    """返回最近一次 Tutor Advise Agent 的评审结果（仅供教师查看）。"""
    if not STATE.tutor_tips:
        return {"ready": False, "tips": None}
    return {"ready": True, "tips": STATE.tutor_tips}


@app.get("/tutor-style")
async def get_tutor_style():
    """返回当前累积的教师风格画像（TutorStyle.json）。"""
    style = STATE.tutor_style or _load_tutor_style()
    return {"ready": bool(style), "style": style}


@app.get("/decisions")
async def get_decisions():
    """返回所有轮次的 Agent 决策记录。"""
    return {
        "decisions": STATE.continuation_decisions,
        "current_round": STATE.current_round,
        "stop_reason": STATE.stop_reason,
    }


@app.post("/submit")
async def submit_answers(payload: dict = Body(...)):
    """
    payload: {
      "round": 1,                      ← 新增，标识提交的是哪一轮
      "per_course": [
        {
          "source_file": "...",
          "course_name": "...",
          "answers": [
            {"id": "Q001", "user_answer": "A", "correct_answer": "B",
             "is_correct": false, "objective_ids": [...], "bloom_level": "", "difficulty": ""}
          ],
          "score": 3,
          "total": 5
        }
      ]
    }
    """
    STATE.user_answers = payload
    round_num = payload.get("round", STATE.current_round)
    # Save to disk (keyed by round)
    (EVAL_DIR / f"user_answers_r{round_num}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Also overwrite the legacy path for Stage 5 compatibility
    (EVAL_DIR / "user_answers.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    await STATE.teacher("answers_received", {
        "message": f"The student has submitted round {round_num} answers!",
        "round": round_num,
    })
    return {"ok": True}


@app.get("/skillmap")
async def get_skillmap():
    if not STATE.skillmap:
        return {"ready": False, "skillmap": None}
    return {"ready": True, "skillmap": STATE.skillmap}


# ── SSE streams ────────────────────────────────────────────────────────────────

async def _sse_generator(queues_list: list) -> AsyncGenerator:
    q: asyncio.Queue = asyncio.Queue()
    queues_list.append(q)
    try:
        # Send current state immediately on connect
        yield f"event: connected\ndata: {json.dumps({'status': STATE.status})}\n\n"
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=30)
                yield msg
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
    finally:
        if q in queues_list:
            queues_list.remove(q)


@app.get("/stream/teacher")
async def stream_teacher():
    return StreamingResponse(
        _sse_generator(STATE.teacher_queues),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/stream/student")
async def stream_student():
    return StreamingResponse(
        _sse_generator(STATE.student_queues),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

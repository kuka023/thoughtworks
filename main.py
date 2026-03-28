from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import shutil
import base64
from datetime import datetime
from rag import rag

load_dotenv()

UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("./data/policies", exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    rag.load()
    yield


app = FastAPI(lifespan=lifespan)

PRODUCT_TITLE = "AI 报销审核助手"
PRODUCT_SUBTITLE = "员工提交 · 财务审核 · 管理洞察 一体化智能平台"

client = OpenAI(
    api_key=os.getenv("ARK_API_KEY"),
    base_url="https://ark.volces.com/api/v3",
)
MODEL = os.getenv("ARK_MODEL", "doubao-pro-32k")
VISION_MODEL = os.getenv("ARK_VISION_MODEL", "doubao-vision-pro-32k")

# ── 内存数据库 ──
SUBMISSIONS = [
    {
        "id": "EXP-0315-001",
        "employee": "张伟", "department": "销售部", "city": "北京",
        "submit_time": "2024-03-15 09:23",
        "trip_start": "2024-03-10", "trip_end": "2024-03-14",
        "total_amount": 3280.0, "status": "待审核", "risk": "high",
        "images": [],
        "items": [
            {"type": "交通费", "invoice_type": "增值税普通发票", "amount": 1200.0,
             "date": "2024-03-10", "vendor": "中国南方航空", "invoice_id": "044001800012",
             "company_header": "某科技有限公司"},
            {"type": "住宿费", "invoice_type": "增值税普通发票", "amount": 1680.0,
             "date": "2024-03-13", "vendor": "北京喜来登酒店", "invoice_id": "012003400089",
             "company_header": "张伟"},
            {"type": "餐饮费", "invoice_type": "增值税专用发票", "amount": 400.0,
             "date": "2024-03-15", "vendor": "全聚德餐厅", "invoice_id": "033002100045",
             "company_header": "某科技有限公司"},
        ],
        "issues": ["住宿费发票抬头为个人姓名「张伟」，非公司名称", "餐饮费票据日期(3月15日)超出出差范围(3月10-14日)"],
    },
    {
        "id": "EXP-0315-002",
        "employee": "李娜", "department": "产品部", "city": "深圳",
        "submit_time": "2024-03-15 10:45",
        "trip_start": "2024-03-12", "trip_end": "2024-03-15",
        "total_amount": 2150.0, "status": "待审核", "risk": "low",
        "images": [],
        "items": [
            {"type": "交通费", "invoice_type": "增值税普通发票", "amount": 850.0,
             "date": "2024-03-12", "vendor": "深圳北站", "invoice_id": "011002300078",
             "company_header": "某科技有限公司"},
            {"type": "住宿费", "invoice_type": "增值税普通发票", "amount": 980.0,
             "date": "2024-03-13", "vendor": "深圳万豪酒店", "invoice_id": "021004500023",
             "company_header": "某科技有限公司"},
            {"type": "餐饮费", "invoice_type": "增值税普通发票", "amount": 320.0,
             "date": "2024-03-14", "vendor": "海底捞餐厅", "invoice_id": "031001200067",
             "company_header": "某科技有限公司"},
        ],
        "issues": [],
    },
    {
        "id": "EXP-0315-003",
        "employee": "王强", "department": "技术部", "city": "成都",
        "submit_time": "2024-03-15 14:22",
        "trip_start": "2024-03-08", "trip_end": "2024-03-12",
        "total_amount": 4890.0, "status": "待审核", "risk": "medium",
        "images": [],
        "items": [
            {"type": "交通费", "invoice_type": "增值税普通发票", "amount": 1600.0,
             "date": "2024-03-08", "vendor": "中国国际航空", "invoice_id": "044001800099",
             "company_header": "某科技有限公司"},
            {"type": "住宿费", "invoice_type": "增值税普通发票", "amount": 2400.0,
             "date": "2024-03-09", "vendor": "成都锦江宾馆", "invoice_id": "021004500089",
             "company_header": "某科技有限公司"},
            {"type": "办公用品", "invoice_type": "增值税专用发票", "amount": 890.0,
             "date": "2024-03-11", "vendor": "成都办公超市", "invoice_id": "031001200099",
             "company_header": "某科技有限公司"},
        ],
        "issues": ["办公用品单笔金额(¥890)超过政策上限(¥800)"],
    },
    {
        "id": "EXP-0314-004",
        "employee": "赵敏", "department": "交付部", "city": "上海",
        "submit_time": "2024-03-14 16:08",
        "trip_start": "2024-03-11", "trip_end": "2024-03-14",
        "total_amount": 1890.0, "status": "已通过", "risk": "low",
        "images": [],
        "items": [
            {"type": "交通费", "invoice_type": "增值税普通发票", "amount": 890.0,
             "date": "2024-03-11", "vendor": "上海虹桥站", "invoice_id": "011002300099",
             "company_header": "某科技有限公司"},
            {"type": "住宿费", "invoice_type": "增值税普通发票", "amount": 780.0,
             "date": "2024-03-12", "vendor": "上海如家酒店", "invoice_id": "021004500099",
             "company_header": "某科技有限公司"},
            {"type": "餐饮费", "invoice_type": "增值税普通发票", "amount": 220.0,
             "date": "2024-03-13", "vendor": "南京路餐厅", "invoice_id": "031001200111",
             "company_header": "某科技有限公司"},
        ],
        "issues": [],
    },
    {
        "id": "EXP-0314-005",
        "employee": "陈磊", "department": "销售部", "city": "北京",
        "submit_time": "2024-03-14 11:30",
        "trip_start": "2024-03-05", "trip_end": "2024-03-08",
        "total_amount": 5600.0, "status": "已退回", "risk": "high",
        "images": [],
        "items": [
            {"type": "交通费", "invoice_type": "增值税普通发票", "amount": 2200.0,
             "date": "2024-03-05", "vendor": "中国东方航空", "invoice_id": "044001800111",
             "company_header": "某科技有限公司"},
            {"type": "住宿费", "invoice_type": "增值税普通发票", "amount": 2800.0,
             "date": "2024-03-06", "vendor": "北京华尔道夫酒店", "invoice_id": "012003400111",
             "company_header": "某科技有限公司"},
            {"type": "餐饮费", "invoice_type": "增值税普通发票", "amount": 600.0,
             "date": "2024-03-07", "vendor": "俏江南餐厅", "invoice_id": "031001200222",
             "company_header": "某科技有限公司"},
        ],
        "issues": ["住宿费单晚¥2800超出差标准上限(¥800/晚)", "华尔道夫五星级酒店超差标，需总监审批"],
    },
]

DASHBOARD = {
    "total_submissions": 487,
    "total_amount": 1285400,
    "avg_audit_minutes": 4.2,
    "anomaly_rate": 8.3,
    "target_anomaly_rate": 10.0,
    "passed": 401,
    "rejected": 46,
    "pending": 40,
    "cities": [
        {"name": "北京", "count": 156, "amount": 412000, "anomaly_rate": 9.2},
        {"name": "深圳", "count": 143, "amount": 389000, "anomaly_rate": 7.1},
        {"name": "成都", "count": 98, "amount": 267000, "anomaly_rate": 8.8},
        {"name": "上海", "count": 90, "amount": 217400, "anomaly_rate": 7.9},
    ],
    "top_issues": [
        {"type": "发票抬头错误", "count": 89, "pct": 18.3},
        {"type": "日期范围超出", "count": 67, "pct": 13.8},
        {"type": "金额超限", "count": 52, "pct": 10.7},
    ],
    "trend": {"last_month_count": 423, "this_month_count": 487, "growth_pct": 15.1},
}


# ── Pydantic 模型 ──
class CheckRequest(BaseModel):
    employee: str
    trip_start: str
    trip_end: str
    items: list
    total_amount: float


class SubmitRequest(BaseModel):
    employee: str
    department: str
    city: str
    trip_start: str
    trip_end: str
    total_amount: float
    items: list
    image_ids: list = []


# ── 问题分类工具 ──
_TITLE_MAP = [
    (["个人姓名", "抬头"], "发票抬头错误"),
    (["日期", "超出", "范围"], "票据日期超出出差范围"),
    (["重复", "发票"], "发票重复提交"),
    (["五星", "华尔道夫", "奢华"], "住宿超出标准"),
    (["超标", "超出差", "超限", "超过政策"], "金额超标预警"),
    (["类型", "不匹配", "专票", "餐饮"], "发票类型不符"),
]

def classify_issue(text: str) -> tuple[str, str]:
    """返回 (标题, 风险等级)"""
    high_kw = ["个人姓名", "重复", "五星", "华尔道夫", "超标", "超限", "不符"]
    risk = "high" if any(k in text for k in high_kw) else "medium"
    for kws, title in _TITLE_MAP:
        if all(k in text for k in kws) or any(k in text for k in kws[:1]):
            return title, risk
    return "审核异常", risk


# ── 工具函数 ──
def stream_llm(system: str, user: str, prefix_events: list = None):
    """SSE 流：先发 prefix_events（如 citations），再流式输出 LLM"""
    def generate():
        try:
            if prefix_events:
                for ev in prefix_events:
                    yield f"data: {json.dumps(ev)}\n\n"
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {json.dumps({'content': delta.content})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


# ── 路由 ──
@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/config")
def get_config():
    return {"product_title": PRODUCT_TITLE, "product_subtitle": PRODUCT_SUBTITLE}


# ── 图片上传 & OCR ──
@app.post("/upload/invoice")
async def upload_invoice(file: UploadFile = File(...)):
    """接收发票图片，调用视觉模型 OCR 识别，返回结构化字段"""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".pdf", ".webp"}:
        raise HTTPException(status_code=400, detail="仅支持 JPG/PNG/PDF/WEBP")

    # 保存文件
    file_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, file_id)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 读取图片并 base64 编码
    with open(save_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    },
                    {
                        "type": "text",
                        "text": """请识别这张发票图片，提取以下字段，以JSON格式返回（无法识别的字段留空字符串）：
{
  "invoice_type": "发票类型（增值税普通发票/增值税专用发票/定额发票/其他）",
  "company_header": "发票抬头（购买方名称）",
  "amount": 金额数字（仅数字，不含符号）,
  "date": "开票日期（格式：YYYY-MM-DD）",
  "vendor": "销售方名称",
  "invoice_id": "发票号码",
  "items_desc": "货物或服务描述（用于判断费用类型）"
}
只返回JSON，不要其他说明。"""
                    }
                ]
            }],
            max_tokens=500,
        )
        raw = resp.choices[0].message.content.strip()
        # 提取 JSON
        start = raw.find("{")
        end = raw.rfind("}") + 1
        ocr_result = json.loads(raw[start:end]) if start >= 0 else {}
    except Exception as e:
        # 视觉模型不可用时返回 mock 数据（演示用）
        ocr_result = {
            "invoice_type": "增值税普通发票",
            "company_header": "（OCR演示）某科技有限公司",
            "amount": 680.0,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "vendor": "（OCR演示）某酒店",
            "invoice_id": f"DEMO{datetime.now().strftime('%H%M%S')}",
            "items_desc": "住宿服务",
            "_note": f"视觉模型调用失败({e})，已返回演示数据"
        }

    return {"file_id": file_id, "filename": file.filename, "ocr": ocr_result}


# ── 报销单 ──
@app.get("/submissions")
def get_submissions():
    return SUBMISSIONS


@app.post("/submissions/submit")
def submit_expense(req: SubmitRequest):
    new_id = f"EXP-NEW-{len(SUBMISSIONS)+1:03d}"
    SUBMISSIONS.insert(0, {
        "id": new_id,
        "employee": req.employee, "department": req.department, "city": req.city,
        "submit_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "trip_start": req.trip_start, "trip_end": req.trip_end,
        "total_amount": req.total_amount, "status": "待审核", "risk": "low",
        "images": req.image_ids,
        "items": req.items, "issues": [],
    })
    return {"success": True, "id": new_id}


@app.post("/submissions/check")
def check_expense(req: CheckRequest):
    system = """你是企业报销合规预审AI，帮助员工在提交前发现问题。

输出格式（Markdown）：

## 预检结果

**风险等级**：🟢 低风险 / 🟡 中风险 / 🔴 高风险（三选一）

**问题清单**

（每个问题一行：❌ 问题描述。无问题则：✅ 未发现合规问题）

**修改建议**

（每个问题的具体操作方法）

---
*以上为AI预检建议，不影响正式提交。*

核查维度：
1. 发票抬头必须是公司名称，个人姓名=高风险
2. 票据日期须在出差日期范围内，超出=中风险
3. 费用类型与发票类型须匹配
4. 住宿超¥800/晚、餐饮超¥200/人次需提醒
5. 是否有重复发票号"""

    user = f"""员工：{req.employee}
出差日期：{req.trip_start} 至 {req.trip_end}
报销总额：¥{req.total_amount}

票据明细：
{json.dumps(req.items, ensure_ascii=False, indent=2)}"""

    return stream_llm(system, user)


@app.post("/submissions/{submission_id}/audit")
def audit_submission(submission_id: str):
    s = next((x for x in SUBMISSIONS if x["id"] == submission_id), None)
    if not s:
        raise HTTPException(status_code=404, detail="报销单不存在")

    # ── 生成结构化问题卡片（不调用LLM，快速返回）──
    issue_cards = []
    for issue_text in (s.get("issues") or []):
        title, risk = classify_issue(issue_text)
        related = rag.retrieve(issue_text, k=1)
        issue_cards.append({
            "title": title,
            "risk": risk,
            "description": issue_text,
            "rag_source": related[0]["source"] if related else None,
            "rag_rule": related[0]["rule"] if related else None,
        })

    # 无问题时加一条"合规"卡
    if not issue_cards:
        issue_cards.append({
            "title": "合规性检查通过",
            "risk": "low",
            "description": "系统未发现明显合规问题，建议直接通过。",
            "rag_source": None, "rag_rule": None,
        })

    # ── LLM 流式生成退回话术 ──
    system = """你是财务专员助手，起草一段给员工的退回通知（如无问题则写通过通知）。
要求：语气友善专业，承认员工辛苦，指出具体问题并说明需要修正的内容，100-150字，用引号包裹，不要其他说明。"""

    issues_desc = "\n".join(f"- {i['description']}" for i in issue_cards if i["risk"] != "low")
    user = f"""员工：{s['employee']}（{s['department']}）
报销单：{s['id']}，金额：¥{s['total_amount']}
问题列表：
{issues_desc if issues_desc else '无问题，合规通过'}"""

    prefix = [{"structured_issues": issue_cards}]
    return stream_llm(system, user, prefix_events=prefix)


# ── 政策管理 ──
@app.get("/policies/status")
def policy_status():
    return rag.status()


@app.post("/policies/reload")
def reload_policies():
    count = rag.load()
    return {"success": True, "total_chunks": count, "files": rag.loaded_files}


@app.post("/policies/upload")
async def upload_policy(file: UploadFile = File(...)):
    """上传新的政策 Excel 文件到策略库"""
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="仅支持 Excel 文件")
    save_path = os.path.join("./data/policies", file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    count = rag.load()
    return {"success": True, "filename": file.filename, "total_chunks": count}


# ── 仪表盘 ──
@app.get("/dashboard")
def get_dashboard():
    return DASHBOARD


@app.post("/dashboard/analyze")
def analyze_dashboard():
    d = DASHBOARD
    system = """你是企业财务分析师，生成月度报销情况管理层摘要（Markdown，300字以内）。

输出格式：

## 本月报销情况摘要

**整体表现**（1-2句，评价趋势和KPI）

**风险预警**
- 重点关注项1
- 重点关注项2

**下月预测与建议**（预测+1-2条可操作建议）

数字须与提供数据严格一致。"""

    user = f"""本月数据：
- 总报销单数：{d['total_submissions']}份（上月{d['trend']['last_month_count']}份，增长{d['trend']['growth_pct']}%）
- 总报销金额：¥{d['total_amount']:,}
- 平均审核时长：{d['avg_audit_minutes']}分钟/单（AI介入前约12分钟）
- 异常率：{d['anomaly_rate']}%（目标<{d['target_anomaly_rate']}%）
- 审核结果：通过{d['passed']}单，退回{d['rejected']}单，待审{d['pending']}单
- 分公司异常率：北京{d['cities'][0]['anomaly_rate']}%，深圳{d['cities'][1]['anomaly_rate']}%，成都{d['cities'][2]['anomaly_rate']}%，上海{d['cities'][3]['anomaly_rate']}%
- Top异常：{d['top_issues'][0]['type']}({d['top_issues'][0]['count']}例)、{d['top_issues'][1]['type']}({d['top_issues'][1]['count']}例)"""

    return stream_llm(system, user)


app.mount("/static", StaticFiles(directory="static"), name="static")

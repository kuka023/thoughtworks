from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = FastAPI()

# ─────────────────────────────────────────────
# ✏️  面试时只改这里
SYSTEM_PROMPT = """你是一个专业的AI助手。
请根据用户输入，提供结构化的分析和建议。
输出格式：
1. 核心问题
2. 解决方案
3. 执行步骤
"""

INPUT_PLACEHOLDER = "请输入内容..."
OUTPUT_TITLE = "AI 分析结果"
PRODUCT_TITLE = "AI 智能助手"
PRODUCT_SUBTITLE = "由豆包大模型驱动"
# ─────────────────────────────────────────────

client = OpenAI(
    api_key=os.getenv("ARK_API_KEY"),
    base_url="https://ark.volces.com/api/v3",
)
MODEL = os.getenv("ARK_MODEL", "doubao-pro-32k")


class ChatRequest(BaseModel):
    user_input: str


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/config")
def get_config():
    return {
        "product_title": PRODUCT_TITLE,
        "product_subtitle": PRODUCT_SUBTITLE,
        "input_placeholder": INPUT_PLACEHOLDER,
        "output_title": OUTPUT_TITLE,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.user_input.strip():
        raise HTTPException(status_code=400, detail="输入不能为空")

    def generate():
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": req.user_input},
                ],
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


app.mount("/static", StaticFiles(directory="static"), name="static")

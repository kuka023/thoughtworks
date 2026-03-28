@echo off
cd /d %~dp0
echo 正在启动 AI 产品原型...
uvicorn main:app --reload --port 8000

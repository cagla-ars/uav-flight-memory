@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -e .

set PYTHONPATH=src
start "" http://127.0.0.1:8001
uvicorn uav_flight_memory.web.app:app --host 127.0.0.1 --port 8001

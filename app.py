"""
app.py
Root entry point for Heart-Brain Axis Wearable AI & Rapid Triage Server.
Run directly with:
    python app.py
or:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from src.server.app import app
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting Heart-Brain Axis Triage Server on http://0.0.0.0:{port} ...")
    uvicorn.run("src.server.app:app", host="0.0.0.0", port=port, reload=False)

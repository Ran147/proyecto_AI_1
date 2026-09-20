from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

from .models import get_registry  # noqa: E402
from .routers import command, emotion, models, transcribe  # noqa: E402

app = FastAPI(title="JarvisTEC API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

for r in (models.router, command.router, emotion.router, transcribe.router):
    app.include_router(r)


@app.get("/health", tags=["sistema"])
def health():
    return {"status": "ok", "models": list(get_registry().keys())}


FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
if (FRONTEND / "index.html").exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")

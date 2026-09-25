from pathlib import Path
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from .core.errors import ModelError  # noqa: E402
from .core.orchestrator import ModelOrchestrator, get_orchestrator  # noqa: E402
from .routers import command, emotion, models, transcribe  # noqa: E402
from .services import emotion_service, speech_service  # noqa: E402

# Se construye al importar: nombres duplicados o inválidos impiden iniciar la API.
get_orchestrator()

app = FastAPI(title="JarvisTEC API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

for r in (models.router, command.router, emotion.router, transcribe.router):
    app.include_router(r)


@app.exception_handler(ModelError)
def model_error(_: Request, exc: ModelError):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/health", tags=["sistema"])
def health(orch: ModelOrchestrator = Depends(get_orchestrator)):
    return {
        "status": "ok",
        "models": orch.names(),
        "emotion_provider": emotion_service.provider_name(),
        "speech_provider": speech_service.provider_name(),
    }


FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
if (FRONTEND / "index.html").exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")

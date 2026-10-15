"""FastAPI application: API routes plus the static frontend."""
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .core.errors import ModelError
from .core.orchestrator import ModelOrchestrator, get_orchestrator
from .routers import command, emotion, models, transcribe
from .services import emotion_service, speech_service

# Environment variables are read at call time, never at import time.
load_dotenv()

# Built on import: duplicated or invalid model names stop the API from
# starting.
get_orchestrator()

app = FastAPI(title="JarvisTEC API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (
    models.router,
    command.router,
    emotion.router,
    transcribe.router,
):
    app.include_router(router)


@app.exception_handler(ModelError)
def handle_model_error(_: Request, exc: ModelError) -> JSONResponse:
    """Translate orchestrator errors into their HTTP status codes."""
    return JSONResponse(
        status_code=exc.status_code, content={"detail": str(exc)}
    )


@app.get("/health", tags=["system"])
def health(orchestrator: ModelOrchestrator = Depends(get_orchestrator)):
    """Report API status, registered models and active providers."""
    return {
        "status": "ok",
        "models": orchestrator.names(),
        "emotion_provider": emotion_service.provider_name(),
        "speech_provider": speech_service.provider_name(),
        # Engines the interface can pick, and the number of each model for
        # our speech engine ("scratch").
        "speech_engines": speech_service.available_engines(),
        "speech_numbers": speech_service.command_numbers(),
        "emotion_models": emotion_service.available_classifiers(),
    }


class FrontendFiles(StaticFiles):
    """Static frontend that browsers revalidate on every load.

    Without this header a browser may keep running an old app.js after the
    code changes.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
if (FRONTEND_DIR / "index.html").exists():
    app.mount(
        "/",
        FrontendFiles(directory=FRONTEND_DIR, html=True),
        name="frontend",
    )

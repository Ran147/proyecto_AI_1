# JarvisTEC

Voice and vision personal assistant that recognizes the user's emotion and
runs machine learning models from spoken commands. Group project for the
Artificial Intelligence course, Instituto Tecnológico de Costa Rica.

## Quick start

Run every command from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # optional: GOOGLE_API_KEY, see docs/GOOGLE_SETUP.md
uvicorn backend.main:app --reload
```

- Web interface: http://localhost:8000/
- Interactive API documentation (Swagger): http://localhost:8000/docs
- Status and registered models: http://localhost:8000/health

Without `GOOGLE_API_KEY`, `/emotion` and `/transcribe` run in mock mode.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | API status, registered models and active providers |
| GET | `/models` | Metadata of every model |
| GET | `/models/{name}` | Metadata of one model |
| POST | `/models/{name}/predict` | Run a model. Body: `{"inputs": {...}}` |
| POST | `/command` | Transcribed text to the right model. Body: `{"text": "JarvisTEC calidad del vino"}` |
| POST | `/emotion` | Image (field `image`) to dominant emotion and scores |
| POST | `/transcribe` | Audio (field `audio`) to transcribed text |

## Models

| Name | Task | Status |
|---|---|---|
| `wine` | Wine quality classification (low / medium / high) | Trained; see [training/wine/README.md](training/wine/README.md) |
| `bitcoin` | Bitcoin price | Demo, not trained (`"demo": true` in `GET /models`) |

The full roadmap is in [NEXT_STEPS.md](NEXT_STEPS.md).

## Project structure

```
backend/
  main.py               FastAPI application; also serves the frontend
  core/orchestrator.py  single entry point to the models: catalog, prediction, commands
  core/registry.py      model registry; rejects duplicated names at startup
  core/errors.py        model errors mapped to HTTP status codes
  models/__init__.py    MODEL_CLASSES: explicit list of active models
  models/base.py        BaseModel contract implemented by every adapter
  models/<name>/        one package per model: adapter.py + schema.py (inference only)
  routers/              HTTP endpoints; they delegate to the orchestrator
  services/             command routing, emotion detection, speech-to-text
frontend/               HTML, CSS and JavaScript interface
training/               offline training; never imported by the backend
  _template/            notebook with the 6 rubric sections + model card template
  <name>/               one folder per trained model
tests/                  pytest suite; tests/models/ has one file per trained model
artifacts/              trained models (.joblib + .meta.json); git-ignored
data/                   datasets; git-ignored
docs/                   project documentation
```

## Language policy

Code, identifiers, API names, JSON keys, comments and documentation are in
English. Text shown to or spoken to the end user stays in Spanish: interface
labels, voice command phrases, model display titles, the `message` field and
the error `detail` messages, because the frontend shows them verbatim.

## Development

```bash
python -m pytest                     # tests
pip install ruff && ruff check .     # PEP 8 lint (79-character lines)
```

Tests that need a trained artifact are skipped when it is missing; generate it
by running the model's notebook in `training/<name>/`.

## Documentation

- [docs/PROJECT.md](docs/PROJECT.md): course requirements, grading and project status
- [NEXT_STEPS.md](NEXT_STEPS.md): model-by-model work plan and definition of done
- [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md): how to train and integrate a model
- [docs/GOOGLE_SETUP.md](docs/GOOGLE_SETUP.md): temporary Google Cloud setup for vision and voice
- [docs/course_project_specification.pdf](docs/course_project_specification.pdf): original assignment (Spanish)

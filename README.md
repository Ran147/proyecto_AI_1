# JarvisTEC

Voice and vision personal assistant that recognizes the user's emotion and
runs machine learning models from spoken commands. Group project for the
Artificial Intelligence course (IC-6200), Instituto Tecnológico de Costa Rica.

Team: Anthony Jafeth Arias Robleto, Randy Espinoza Downing, Kenneth Jhoan
Araya Arguedas.

## Quick start

Run every command from the repository root. Tested with Python 3.14+.

### Windows (Git Bash)
```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m uvicorn backend.main:app --reload
```

- Web interface: http://localhost:8000/ (use Chrome or Edge for voice)
- Interactive API documentation (Swagger): http://localhost:8000/docs
- Status and registered models: http://localhost:8000/health

That is all: the ten trained models, the emotion networks and the speech
models are committed in `artifacts/`, and no API key is needed.

- **Voice** has two engines, switched with the **Speech** button in the
  header. The default is the local model on the server (Vosk, offline, any
  browser, English and Spanish models): the page records WAV audio and
  `POST /transcribe` returns the text. The other is the browser's own speech
  recognition (Web Speech API, Chrome or Edge, needs internet).
- **Emotion** runs on a local model in the backend: the YuNet face detector,
  then the FER+ and HSEmotion expression classifiers averaged, through OpenCV.
- **Our own networks, trained from scratch** (optional): an emotion
  classifier trained on FER-2013 and a spoken-number recognizer trained on
  Speech Commands. The pretrained models above stay the default; to use ours,
  add to `.env`:

  ```bash
  EMOTION_MODEL=scratch     # our emotion network instead of FER+ and HSEmotion
  SPEECH_PROVIDER=scratch   # our speech network: say "JarvisTEC, three" (the number of a model)
  ```

  and restart the server. See [training/emotion_cnn/](training/emotion_cnn/)
  and [training/speech_commands/](training/speech_commands/) for how they
  were trained and evaluated.
- With `GOOGLE_API_KEY` in `.env` (copy `.env.example`), emotion and
  `POST /transcribe` switch to the Google Cloud APIs; see
  [docs/GOOGLE_SETUP.md](docs/GOOGLE_SETUP.md).

## Using it

Say or type a command that starts with "JarvisTEC" (or "Jarvis"), or click one
in the list. Commands are understood in English and in Spanish; the
**Commands** button in the header tells the speech recognition which one to
listen for. The interface and the answers are in English.

Spanish examples: "JarvisTEC, calidad del vino", "precio del carro",
"abandono de clientes", "riesgo de derrame cerebral", "precio del aguacate",
"precio del bitcoin", "grasa corporal", "recomiéndame una película", "tipo
de hepatitis", "tipo de cirrosis".

| Model | Main command | Also understands |
|---|---|---|
| `wine` | JarvisTEC, wine quality | how good is this wine; classify wine |
| `car_price` | JarvisTEC, car price | how much is my car worth; used car price |
| `churn` | JarvisTEC, customer churn | will the customer leave |
| `stroke` | JarvisTEC, stroke risk | risk of a stroke |
| `avocado` | JarvisTEC, avocado price | avocado forecast; price of avocados |
| `bitcoin` | JarvisTEC, bitcoin price | bitcoin price tomorrow |
| `body_fat` | JarvisTEC, body fat | how much body fat do I have |
| `movies` | JarvisTEC, recommend a movie | what movie should I watch |
| `hepatitis` | JarvisTEC, hepatitis type | classify hepatitis |
| `cirrhosis` | JarvisTEC, cirrhosis stage | type of cirrhosis |

- A command without data answers for an example case. The **Data** button
  next to each command opens a form with the model's inputs; edited values
  also apply to later spoken or typed commands.
- The percentage in an answer is the model's estimate for that case, not the
  model's accuracy. The accuracy of each model is in its notebook and model
  card.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | API status, registered models and active providers |
| GET | `/models` | Metadata of every model: inputs, allowed values, defaults, commands |
| GET | `/models/{name}` | Metadata of one model |
| POST | `/models/{name}/predict` | Run a model. Body: `{"inputs": {...}}`; missing inputs take defaults |
| POST | `/command` | Text to the right model. Body: `{"text": "JarvisTEC wine quality"}` |
| POST | `/emotion` | Image (field `image`; optional `model`: `ensemble`, `ferplus`, `hsemotion` or `scratch`) to dominant emotion and scores |
| POST | `/transcribe` | Audio (field `audio`, WAV; field `language`: `en` or `es`; optional `engine`) to text, with the local model, Google or, when configured, ours |

`POST /transcribe` never returns an invented text: if the server has neither
the local model nor a Google key, it answers 503 and says so.

## Models

| Name | Task | Model | Test result (baseline) |
|---|---|---|---|
| `wine` | Wine quality: low / medium / high | Random forest | Macro F1 0.59 (0.20) |
| `car_price` | Used car resale price | Gradient boosting on the log price | R2 0.88; RMSE 1.79 lakh (5.08) |
| `churn` | Telecom customer churn | Gradient boosting, tuned threshold | Churn F1 0.63, ROC-AUC 0.85 (F1 0) |
| `stroke` | Stroke risk | Random forest, tuned threshold | ROC-AUC 0.83, recall 0.70 (0.50) |
| `avocado` | Avocado price next week | Ridge on lagged prices | RMSE 0.079 USD national series (naive 0.076) |
| `bitcoin` | Bitcoin close next day | Random forest on log returns | RMSE 69.98 USD (naive 69.73) |
| `body_fat` | Body fat percentage | Ridge, two measurements | R2 0.67; RMSE 4.27 points (7.45) |
| `movies` | Movie recommendation | Item-based collaborative filtering | NDCG@10 0.140 with one seed (popularity 0.165) |
| `hepatitis` | Hepatitis C category | Random forest, balanced | Macro F1 0.59 (0.23) |
| `cirrhosis` | Cirrhosis stage 1-4 | Random forest, balanced | Accuracy 0.51 (0.37) |

Each model has a notebook with the six sections of the grading rubric
(`training/<name>/notebook.ipynb`) and a model card
(`training/<name>/README.md`). The results are reported next to their
baseline, including the cases where the model does not beat it (`bitcoin`,
`avocado` on the national series, `movies` with a single seed movie).

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
  <name>/               notebook and model card of each model
artifacts/              trained models (.joblib + .meta.json), emotion networks (.onnx), speech models,
                        and our two networks trained from scratch (emotion_cnn, speech_commands)
scripts/                dataset and model downloads, emotion evaluation, timing
tests/                  pytest suite; tests/models/ has one file per model
data/                   datasets; git-ignored (scripts/download_datasets.py)
docs/                   project documentation and the written report (docs/report/)
```

## Development

```bash
python -m pytest                     # 354 tests
pip install ruff && ruff check backend tests scripts   # PEP 8 lint (79 columns)
```

To retrain a model, download the data and run its notebook; it rewrites the
files in `artifacts/`. Run one notebook at a time: several at once can
exhaust the memory.

```bash
python scripts/download_datasets.py
jupyter nbconvert --to notebook --execute --inplace training/wine/notebook.ipynb
```

Other scripts: `scripts/download_emotion_models.py` and
`scripts/download_speech_models.py` (re-fetch the emotion networks and the
speech models, with hash check), `scripts/evaluate_emotion_model.py` (emotion
classifiers on FER-2013), `scripts/evaluate_emotion_pipeline.py` (the
whole emotion pipeline, face detector included),
`scripts/measure_response_times.py`. Training the two networks from scratch
needs PyTorch: `requirements-train.txt`.

## Documentation

- [docs/CHANGES.md](docs/CHANGES.md): what changed in the last big update, for the team
- [docs/PROJECT.md](docs/PROJECT.md): course requirements, grading and project status
- [docs/report/main.tex](docs/report/main.tex): the written report (Overleaf source)
- [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md): how to train and integrate a model
- [docs/GOOGLE_SETUP.md](docs/GOOGLE_SETUP.md): optional Google Cloud setup for vision and voice
- [docs/course_project_specification.pdf](docs/course_project_specification.pdf): original assignment (Spanish)

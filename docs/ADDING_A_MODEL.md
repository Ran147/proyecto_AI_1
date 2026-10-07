# Adding a model to JarvisTEC

Every model has two separate halves:

| Half | Location | Contents |
|---|---|---|
| Training (offline) | `training/<name>/` | notebook and model card (dataset, configuration, evaluation) |
| Inference (backend) | `backend/models/<name>/` | adapter (`adapter.py`) and input schema (`schema.py`) |

The bridge between them is the artifact `artifacts/<name>.joblib`. The backend
never imports anything from `training/`.

Routers and the command router are **never** edited: the orchestrator
(`backend/core/orchestrator.py`) is the only component that registers and runs
models. Adding the class to `MODEL_CLASSES` is enough.

```
HTTP request / voice command
        |
routers/models.py, routers/command.py  -- HTTP only
        |
core/orchestrator.py  -- finds the model, validates the inputs with its
        |                schema, loads it (once), runs predict() and builds
        |                the response
core/registry.py      -- one instance per `name`; rejects duplicates at startup
        |
models/<name>/adapter.py  -- loads artifacts/<name>.joblib and predicts
```

The reference implementation is the `wine` model: `training/wine/`,
`backend/models/wine/` and `tests/models/test_wine.py`.

## 1. Training (`training/<name>/`)

Copy `training/_template/` to `training/<name>/`. The notebook already has the
6 rubric sections in order:

1. Problem analysis (0.5 pt)
2. Data understanding (0.5 pt)
3. Data exploration (0.5 pt)
4. Model (2 pts)
5. Evaluation (1 pt)
6. Conclusion (0.5 pt)

Steps:

1. Put the CSV in `data/<dataset>/` (git-ignored) and record its source in
   `training/<name>/README.md`.
2. Edit the **Configuration** cell (model name, `../../data/...` path, target
   column, task type, time series or not).
3. Run the notebook top to bottom and write the text marked **TODO**. The
   rubric grades the analysis and the conclusions, not only the code.
4. The export cell writes `artifacts/<name>.joblib` (a full `Pipeline`,
   preprocessing included) and `artifacts/<name>.meta.json` (test metrics,
   hyperparameters, features, a `default_input` and the scikit-learn
   version).
5. Copy the real metrics from `meta.json` into the model card. Never write
   numbers that did not come from a run.

Notes:

- Use snake_case, lowercase column names (e.g. rename `pH` to `ph`): they
  become the API input fields.
- Time series (bitcoin, avocado...): set `IS_TIME_SERIES = True` and
  `DATE_COLUMN` for a chronological split. Predicting "tomorrow" usually needs
  lag features (previous days' values); add that cell in section 4.
- The template is a starting point: add models, tuning or plots when the
  problem needs them.
- The movie recommender does not fit the template (it is not classic
  supervised learning); sections 4 and 5 must be adapted.

## 2. Inference (`backend/models/<name>/`)

Create a package with three files.

`schema.py` lists the inputs the model accepts. It is validated **before** the
model is loaded (invalid input returns 422), and the `input_fields` shown by
`GET /models` are generated from it.

```python
"""Input schema of the wine quality model."""
from typing import Literal

from pydantic import BaseModel as Schema
from pydantic import Field


class WineInputs(Schema):
    """Same snake_case columns the pipeline was trained on."""

    type: Literal["red", "white"] = Field(description="wine type")
    alcohol: float = Field(ge=0, le=100, description="alcohol, % vol.")
```

`adapter.py` is the class the orchestrator runs.

```python
"""Adapter of the wine quality model."""
from typing import Any

import joblib
import pandas as pd

from ..base import BaseModel
from .schema import WineInputs


class Wine(BaseModel):
    """Classifies wine quality as low, medium or high."""

    name = "wine"  # unique, lowercase, no spaces: /models/{name}
    title = "Wine quality"  # display name (UI)
    description = "Classifies wine quality from its chemistry."
    task_type = "classification"
    keywords = ["wine quality", "classify wine"]  # voice phrases, lowercase
    input_schema = WineInputs
    default_input = {"type": "white", "alcohol": 10.4}  # from meta.json

    def load(self) -> None:
        self.pipeline = joblib.load(self.artifact_path())
        self.pipeline.set_params(model__n_jobs=1)

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        label = self.pipeline.predict(pd.DataFrame([inputs]))[0]
        return {"quality": str(label)}

    def speak(self, result: dict[str, Any]) -> str:
        return f"I estimate this wine is of {result['quality']} quality."
```

`__init__.py` exports the class.

```python
"""Wine quality: Random Forest trained in training/wine/."""
from .adapter import Wine

__all__ = ["Wine"]
```

## 3. Register

Add the class to `MODEL_CLASSES` in `backend/models/__init__.py`:

```python
from .wine import Wine

MODEL_CLASSES = [..., Wine]
```

When a model replaces another one with the same `name`, remove the old class
from the list and delete its package; the API does not start with two models
sharing a name.

Rules checked by the registry at startup:

- `name` is unique, lowercase and has no spaces.
- A `keywords` phrase cannot belong to two models (the command would be
  ambiguous).

Other rules:

- `predict` receives validated inputs and returns a JSON-serializable dict
  (convert numpy types with `float()`, `int()` or `str()`).
- `load` runs once, on the first prediction. If it fails (e.g. the `.joblib`
  is missing), the API answers 503 for that model and the others keep
  working.
- Set `n_jobs=1` on the loaded estimator: every request predicts a single
  row, so a thread pool only adds latency.
- `default_input` lets a voice command work without data (e.g. "JarvisTEC,
  wine quality").
- `keywords`: 3-5 natural English phrases a user would say, lowercase, each
  containing the topic word. `keywords_es`: the same in Spanish (accents are
  ignored when matching). The router matches whole words and forgives
  small typos in long words; add the model to the tables in
  `tests/test_routing.py`.
- Datasets are not committed (`data/` is git-ignored;
  `scripts/download_datasets.py` fetches them). The trained artifacts in
  `artifacts/` are committed: keep them small (compress large forests) and
  commit the `.joblib` together with its `.meta.json`.

## 4. Test

```bash
python -m pytest
uvicorn backend.main:app --reload
```

In http://localhost:8000/docs try `POST /models/wine/predict` and
`POST /command` with `{"text": "JarvisTEC wine quality"}`.

Every trained model has its tests in `tests/models/test_<name>.py` (pattern:
`tests/models/test_wine.py`). Tests that run the model are marked `skipif`
when the artifact is missing; the
registration, validation (422) and missing-artifact (503) tests always run.

To test an adapter without an artifact, `tests/test_orchestrator.py` shows how
to build a `ModelOrchestrator` from custom classes and inject it into the API
with `app.dependency_overrides`.

## 5. Deliverables per model

- [ ] `training/<name>/notebook.ipynb` with the 6 sections
- [ ] `training/<name>/README.md` with dataset, reproduction steps and real metrics
- [ ] Trained artifact in `artifacts/` (or instructions to regenerate it)
- [ ] `backend/models/<name>/` package registered in `MODEL_CLASSES`
- [ ] 2-3 voice command phrases
- [ ] 1-2 scientific papers for the bibliography
- [ ] Overleaf section

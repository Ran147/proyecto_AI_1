# JarvisTEC - Status and remaining steps

The build plan this file used to hold is finished: the ten models are
trained, integrated and tested. What changed and why is in
[docs/CHANGES.md](docs/CHANGES.md); the requirements and their status are in
[docs/PROJECT.md](docs/PROJECT.md).

## Models

| # | Model | Backend `name` | Task | Status |
|---|---|---|---|---|
| 1 | Wine quality | `wine` | Classification | Done |
| 2 | Car price | `car_price` | Regression | Done |
| 3 | Telco customer churn | `churn` | Classification | Done |
| 4 | Stroke risk | `stroke` | Classification | Done |
| 5 | Avocado price | `avocado` | Time series | Done |
| 6 | Bitcoin price | `bitcoin` | Time series | Done |
| 7 | Body fat | `body_fat` | Regression | Done |
| 8 | Movie recommendation | `movies` | Recommendation | Done |
| 9 | Hepatitis C | `hepatitis` | Classification | Done |
| 10 | Cirrhosis | `cirrhosis` | Classification | Done |

"Done" means, for every model:

- `training/<name>/notebook.ipynb` executed, with the six rubric sections
  and the analysis written from its own outputs;
- `artifacts/<name>.joblib` and `.meta.json` committed, and reproduced by
  re-running the notebook (fixed seed);
- `training/<name>/README.md` model card with the real test metrics;
- `backend/models/<name>/` adapter and input schema, registered in
  `MODEL_CLASSES`;
- `tests/models/test_<name>.py` passing: default prediction, invalid input
  (422), missing artifact (503), every voice command;
- checked against a running server: `POST /models/<name>/predict`,
  `POST /command` and the command button with its data form.

## Before the delivery

- [ ] Upload `docs/report/` to Overleaf and add the screenshots (the only
      items still marked `[PENDING]` in the report).
- [ ] Each team member reads the ten notebooks; anyone may be asked to
      defend any decision.
- [ ] Open every DOI in the report's bibliography once.
- [ ] Confirm the delivery format with the professor.

## Rules worth keeping

- Never write a metric that did not come from a run; the model card copies
  `meta.json`.
- All preprocessing that can live inside the scikit-learn `Pipeline` does, so
  the backend receives raw values. The time-series models build their lag
  features in one function that is identical in the notebook and the
  adapter.
- A model is compared with a baseline on the same test set, and a result
  that does not beat the baseline is reported as such.
- Model `name`s are part of the API routes: do not rename them.
- Voice phrases must be unique across models and contain the topic word;
  `tests/test_routing.py` checks the whole set, including phrases that must
  match nothing.

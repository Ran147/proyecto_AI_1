# Offline training

Everything that produces a model lives here and is **never imported by the
backend**. The backend only consumes the result: the artifact in
`artifacts/`.

```
training/
  _template/            copy to training/<name>/
    notebook.ipynb      the 6 rubric sections (configuration in the first code cell)
    README.md           model card: dataset, reproduction, evaluation results
  <name>/               one folder per trained model
```

`<name>` must match the `name` of the adapter in `backend/models/<name>/`.

## Workflow

1. Download the dataset into `data/<dataset>/` (git-ignored) and record its
   source in `training/<name>/README.md`.
2. Edit the notebook's **Configuration** cell (name, path, target column, task
   type). Paths are relative to `training/<name>/`: `../../data/` and
   `../../artifacts/`.
3. Run the notebook top to bottom. Section 5 (Evaluation) computes the metrics
   on the test set.
4. The export cell saves:
   - `artifacts/<name>.joblib`: the full `Pipeline` (preprocessing + model).
   - `artifacts/<name>.meta.json`: test metrics, hyperparameters, features, a
     `default_input` and the scikit-learn version.
5. Copy the real metrics from `meta.json` into the model card. Never write
   metrics that did not come from a run.

Then create the inference adapter: see
[docs/ADDING_A_MODEL.md](../docs/ADDING_A_MODEL.md).

## Status

- `wine/`: trained (Random Forest, test macro F1 0.59). It is the reference
  example for the other models.
- `bitcoin` in the API is still an **untrained demo**
  (`backend/models/demo_bitcoin/`) and has no folder here.

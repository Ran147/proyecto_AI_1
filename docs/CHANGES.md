# What changed (October 2026 update)

For Anthony and Randy. This is a summary of everything that went into `main`
after the commit "translation and wine model", so nobody has to reconstruct
it from the diff. If you only read one section, read "Try it in two minutes".

## Try it in two minutes

```bash
git pull
python -m venv .venv && .venv\Scripts\activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open http://localhost:8000 in Chrome or Edge. Nothing else is needed: the
trained models are in the repository and no API key is required.

- Click a command, or press **Speak** and say "JarvisTEC, wine quality".
- Open **Data** next to a command to run the model with your own values.
- Turn the camera on and press **Analyze emotion**.
- `python -m pytest` should report 308 passed.

If the page looks old (Spanish labels), reload with Ctrl+F5.

## The big decisions

| Decision | Before | Now | Why |
|---|---|---|---|
| Language | Code in English, interface and commands in Spanish | Everything in English, including voice commands and spoken answers | Requirement for the delivery |
| Speech-to-text | Google Cloud (planned: local model) | The browser's speech recognition (Web Speech API) | Google Cloud needs a billing account; the browser needs no key |
| Emotion | Google Vision (planned: local model) | Local pretrained networks in the backend (YuNet + FER+ + HSEmotion) | Same reason; it also keeps `POST /emotion` a real service |
| Trained models | Git-ignored, regenerated from notebooks | Committed in `artifacts/` | Whoever clones can run the application immediately |
| Datasets | Git-ignored | Still git-ignored; `scripts/download_datasets.py` fetches all ten | Not ours to redistribute |
| Google providers | Default | Optional: set `GOOGLE_API_KEY` in `.env` and both switch to Google | Kept in case a key is available for the delivery |

## What was added

### Ten trained models (there was one, plus a fake bitcoin demo)

For each of `wine`, `car_price`, `churn`, `stroke`, `avocado`, `bitcoin`,
`body_fat`, `movies`, `hepatitis`, `cirrhosis`:

- `training/<name>/notebook.ipynb`: executed, six rubric sections, analysis
  written from the notebook's own outputs, our three names in the first cell.
- `training/<name>/README.md`: model card (dataset, configuration, test
  metrics, how to reproduce).
- `artifacts/<name>.joblib` and `.meta.json`: the trained model and its
  metrics.
- `backend/models/<name>/`: `schema.py` (inputs and their limits) and
  `adapter.py` (loads the model, predicts, builds the spoken sentence).
- `tests/models/test_<name>.py`.

The `demo_bitcoin` package was deleted; `bitcoin` is now a trained model.

Results, with the baseline in parentheses:

| Model | Result on the test set |
|---|---|
| wine | Macro F1 0.59 (0.20) |
| car_price | R2 0.88, RMSE 1.79 lakh (5.08) |
| churn | Churn F1 0.63, ROC-AUC 0.85 |
| stroke | ROC-AUC 0.83, recall 0.70, precision 0.18 |
| avocado | RMSE 0.153 USD over all series (naive 0.164); 0.079 on the national series (naive 0.076) |
| bitcoin | RMSE 69.98 USD (naive 69.73): does not beat the naive forecast |
| body_fat | R2 0.67, RMSE 4.27 points (7.45) |
| movies | NDCG@10 0.269 with the full history (popularity 0.165); 0.140 with one seed movie (popularity 0.165) |
| hepatitis | Macro F1 0.59 (0.23) |
| cirrhosis | Accuracy 0.51 (0.37), macro F1 0.44 (0.14) |

Three models do not beat their reference in the scenario the assistant
serves (bitcoin, avocado on the national series, movies with one seed). The
notebooks and the report say so plainly; that is deliberate. Be ready to
explain it in the defense: the naive forecast is the standard hard baseline
for prices, and a recommender that ignores the title is not a recommender.

### Application

- **Data forms.** Each command has a **Data** button that opens a form built
  from the model's input schema. Edited values are also used by later spoken
  or typed commands (`POST /command` takes `inputs_by_model`).
- **Data inside the sentence.** "red wine quality", "organic avocado price",
  "recommend me a movie like Jurassic Park" (`inputs_from_command` in the
  adapters).
- **Example cases.** Without data, `wine`, `churn` and `cirrhosis` answer for
  a real, clear example taken from their held-out test set (so the demo shows
  a confident answer); the other models answer for a typical case.
- **Browser voice** and **English speech synthesis**.
- **Emotion from three frames**, averaged; the emotion is cleared when the
  camera is turned off.

### Command router (`backend/services/intent_router.py`)

Rewritten. A phrase matches when its words appear as whole words, in order;
small typos are forgiven only in long words and only if every word matches.
"JarvisTEC, gold price" now matches nothing instead of the car model.
`tests/test_routing.py` checks natural phrasings, recognizer slips and
phrases that must not match.

### Emotion service (`backend/services/emotion_service.py`)

Local provider: YuNet finds the face, FER+ and HSEmotion classify it and
their probabilities are averaged. We measured it
(`scripts/evaluate_emotion_model.py`, 7178 FER-2013 test faces): accuracy
0.565 for FER+ alone, 0.601 for the average, and sadness recall from 0.24 to
0.32. Sadness and fear remain hard; say so if asked.

### Speech service (`backend/services/speech_service.py`)

Without a Google key the server has no speech engine: `POST /transcribe`
answers 503 with an explanation instead of returning a fake text. The mock
provider only runs when `SPEECH_PROVIDER=mock` is set explicitly.

### Report (`docs/report/`)

`main.tex` (English, the seven parts the assignment asks for), `figures/`
(one evaluation figure per model) and a compiled preview PDF. It must still
go to Overleaf and get the screenshots.

### Scripts (`scripts/`)

- `download_datasets.py`: the ten datasets, at the paths the notebooks expect.
- `download_emotion_models.py`: the three ONNX files, with hash check.
- `evaluate_emotion_model.py`: the emotion table of the report.
- `measure_response_times.py`: the response-time table of the report.

### Robustness

- Inputs that used to crash with a 500 (infinite bitcoin price, impossible
  date, a 2-pixel-high image) now answer 422.
- `requirements.txt` pins the libraries that read the trained models.
- The frontend is served with `Cache-Control: no-cache`.

## Where to look for what

| I want to... | Go to |
|---|---|
| Understand one model end to end | `training/<name>/notebook.ipynb`, then `backend/models/<name>/adapter.py` |
| See every result in one place | `README.md`, section Models; `docs/report/main.tex`, section 5 |
| Change or add a voice phrase | `keywords` in the adapter, then `tests/test_routing.py` |
| Add a model | `docs/ADDING_A_MODEL.md` |
| Retrain a model | `python scripts/download_datasets.py`, run the notebook, commit the new artifact |
| Use Google instead | `docs/GOOGLE_SETUP.md` |

## What each of us still has to do

- Read the ten notebooks. Any of us can be asked about any model.
- Read `docs/report/main.tex` once.
- Run the application with your own camera and microphone.

## Known limits (do not hide them)

- Voice needs Chrome or Edge and an internet connection.
- Movie titles and commands are English only.
- Car prices are in Indian rupees; the body fat model was trained on men
  only; avocado data ends in March 2018 and bitcoin data in July 2017.
- The medical models are estimates, not diagnoses; hepatitis and cirrhosis
  have very few test patients per class.

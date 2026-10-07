# JarvisTEC - Project Requirements and Status

Single reference for what the course requires and where the project stands.
The original assignment is [course_project_specification.pdf](course_project_specification.pdf);
this document is an English summary of it. The model-by-model work plan is in
[NEXT_STEPS.md](../NEXT_STEPS.md).

- **Course:** Artificial Intelligence, I Semester 2026, Computer Engineering,
  Instituto Tecnológico de Costa Rica, San Carlos Local Technological Campus
- **Professor:** Ing. Efrén Jiménez Delgado

## 1. Project summary

JarvisTEC is a personal assistant driven by voice and image. It reads the
user's face and voice, interprets natural-language machine learning commands
(the assignment's example is "JarvisTEC, tipo de cambio para mañana"; this
project's commands are in English, e.g. "JarvisTEC, bitcoin price") and answers by running the
corresponding model. The assistant needs no real face: an animation, image or
audio effect may represent it.

## 2. Administrative rules

- Groups of at most 3 members.
- Final delivery: week 11 (it may be finished and presented earlier).
- Documentation must be written in Overleaf, without exception.
- The whole project must live in a Bitbucket, GitLab or GitHub repository.

## 3. Requirements

| # | Requirement | Details |
|---|---|---|
| R1 | User interface | A platform for user-machine communication (free design; the reference is a J.A.R.V.I.S.-style interface). |
| R2 | Emotion recognition agent | Recognize the person's emotion from a live camera photo or a video frame. |
| R3 | Speech-to-text agent | Convert audio to text and execute the instruction it contains. |
| R4 | 10 machine learning models | Chosen from the course list (section 7), or from an external dataset approved by the professor. |
| R5 | REST API | Endpoints for every feature: one per model, plus vision and voice. |
| R6 | Voice commands | A set of commands per model, e.g. `JarvisTEC "tipo de cambio para mañana"`. |
| R7 | Scientific document | Overleaf document with the structure in section 5. |

**Alternative datasets (R4).** Instead of the course list, the group may use a
dataset tied to a real need (company, institution, community or
organization):

- The problem, the dataset origin and nature (public, institutional or own)
  and the proposed model must be presented to the professor **before**
  development starts.
- The dataset must allow the same analysis as the other models.
- One external dataset may cover several models if each one solves a
  different task (e.g. classification and regression) and the professor
  approves it.
- The professor has the final word on the dataset and the approach.

**External APIs (R2, R3).** The course allows Google or Azure APIs for
emotions and speech-to-text in the week-10 delivery only; the assignment notes
mention Azure Face and Speech-to-Text. **Project decision:** speech-to-text
uses the speech recognition built into Chrome and Edge (Web Speech API, served
by Google's speech service), which needs no API key or billing account. The
Google Cloud Speech-to-Text and Vision providers remain in the backend and
take over when `GOOGLE_API_KEY` is set ([GOOGLE_SETUP.md](GOOGLE_SETUP.md)).
Emotion recognition runs on a local model in the backend (YuNet face
detector, then the FER+ and HSEmotion expression classifiers averaged; all
pretrained and run with OpenCV), which also needs no key.

## 4. Grading

| Item | Weight | Criteria |
|---|---|---|
| Model creation | 30 % | 5 points per model: problem analysis 0.5, data understanding 0.5, data exploration 0.5, model 2, evaluation 1, conclusion 0.5 |
| Application | 10 % | 1 point per model completed and integrated into the application |
| REST API | 10 % | 1 point per model completed and exposed through the API |
| Document | 10 % | Complete 10 points; incomplete 3 points; not submitted 0 |

Delivery: requirements 1-4 of the scope in **week 10**.

## 5. Scientific document (Overleaf)

Written from an applied scientific point of view, with every model justified
by scientific papers. Required sections:

1. Cover page
2. Proposed solution
3. ML architectures
4. Justification and explanation of the developed model
5. Analysis of the results
6. Bibliography
7. Appendices

## 6. Status

Everything below is implemented, tested and committed, including the trained
models. The history of the last big update is in [CHANGES.md](CHANGES.md).

### R1. User interface - done

- Arc-reactor style HUD (`frontend/index.html`, `frontend/style.css`),
  in English.
- Reactor animation driven by the live microphone level (Web Audio API).
- Camera on/off and frame capture (WebRTC).
- Voice input through the browser's speech recognition, and typed input.
- A data form per model (button **Data**), generated from `GET /models`, to
  run a model with custom values.
- Text response, command list, conversation log and raw API response viewer.
- The answer is read aloud with the browser's speech synthesis.

### R5. REST API - done

- FastAPI server with CORS; serves the frontend (`backend/main.py`).
- `GET /health`, `GET /models`, `GET /models/{name}`,
  `POST /models/{name}/predict`, `POST /command`, `POST /emotion`,
  `POST /transcribe`.
- Single model orchestrator (`backend/core/orchestrator.py`), explicit model
  registry with duplicate detection, one package per model with an input
  schema. See [ADDING_A_MODEL.md](ADDING_A_MODEL.md).
- The 10 trained models are exposed (`GET /models`).
- `POST /command` accepts `inputs_by_model`, so the interface can send the
  values edited in each model's data form together with a spoken command.
- Invalid inputs answer 422 with the reason, a missing model file 503.

### R2. Emotion recognition - done (local model)

- Camera frame capture, `POST /emotion`, probability bars and empathetic
  reply prefixes in the UI.
- Local provider (`backend/services/emotion_service.py`): YuNet finds the
  largest face in the frame and two pretrained networks classify its
  expression into 8 emotions, FER+ and HSEmotion; their probabilities are
  averaged. It is the default without a Google key. `EMOTION_MODEL` in `.env`
  selects `ensemble` (default), `ferplus` or `hsemotion`.
- Evaluation on the 7178 FER-2013 test faces
  (`scripts/evaluate_emotion_model.py`, original labels, classifiers only):

  | Model | Accuracy | Macro recall | Sadness | Anger | Fear | Neutral |
  |---|---|---|---|---|---|---|
  | FER+ | 0.565 | 0.485 | 0.241 | 0.381 | 0.062 | 0.912 |
  | HSEmotion | 0.560 | 0.523 | 0.407 | 0.511 | 0.384 | 0.564 |
  | Both averaged (served) | 0.601 | 0.551 | 0.316 | 0.443 | 0.194 | 0.857 |

  FER+ alone calls most sad and angry faces neutral; the average recovers
  part of them without losing the neutral and happy ones. The images are
  48x48 grayscale, outside HSEmotion's training domain, so its numbers here
  are a lower bound for webcam frames. The combination was chosen on this
  same test set, so the table describes it rather than validating it.
- The interface analyzes 3 frames and averages their scores, so a blink or a
  half-formed expression weighs less.
- Tested by the team with a real camera in Chrome.
- Known limit: subtle or posed expressions remain hard; sadness and fear are
  recognized in fewer than a third of the test faces.
- Optional Google Vision provider, covered by tests with simulated
  responses.

### R3. Speech-to-text and commands - done (browser recognition)

- Speech recognition in the browser (`frontend/app.js`, Web Speech API,
  language `en-US`). Only the transcribed text is sent to `POST /command`.
  It works in Chrome and Edge and needs an internet connection. Tested by the
  team with a real microphone in Chrome.
- Command router (`backend/services/intent_router.py`): text normalization,
  tolerant wake word ("JarvisTEC", "Jarvis tech", "yarvis tek"), whole-word
  phrase matching and per-word typo tolerance. Phrases about other topics
  ("gold price") match no model; `tests/test_routing.py` covers both sides.
- Some commands carry data in the sentence ("organic avocado price",
  "recommend me a movie like Jurassic Park").
- `POST /transcribe` runs Google Speech-to-Text when `GOOGLE_API_KEY` is
  set. Without a key the server has no speech engine and answers 503 with an
  explanation; it never returns an invented text.

### R4, R6. Machine learning models and commands - 10 of 10 done

| # | Model | Dataset | Main command |
|---|---|---|---|
| 1 | Wine quality | Kaggle `rajyellow46/wine-quality` | JarvisTEC, wine quality |
| 2 | Car price | GitHub `amankharwal/Website-data` (`car data.csv`) | JarvisTEC, car price |
| 3 | Telco customer churn | GitHub `IBM/telco-customer-churn-on-icp4d` | JarvisTEC, customer churn |
| 4 | Stroke risk | Kaggle `fedesoriano/stroke-prediction-dataset` | JarvisTEC, stroke risk |
| 5 | Avocado price | Kaggle `neuromusic/avocado-prices` | JarvisTEC, avocado price |
| 6 | Bitcoin price | Kaggle `team-ai/bitcoin-price-prediction` | JarvisTEC, bitcoin price |
| 7 | Body fat | Kaggle `fedesoriano/body-fat-prediction-dataset` | JarvisTEC, body fat |
| 8 | Movie recommendation | Google Drive dataset from the assignment | JarvisTEC, recommend a movie |
| 9 | Hepatitis C | Kaggle `fedesoriano/hepatitis-c-dataset` | JarvisTEC, hepatitis type |
| 10 | Cirrhosis | Kaggle `fedesoriano/cirrhosis-prediction-dataset` | JarvisTEC, cirrhosis stage |

Each model has a notebook with the 6 rubric sections, a model card, a trained
artifact, a backend package, tests and voice commands. Results are in the
[README](../README.md#models) and in each model card.

### R7. Scientific document - written, Overleaf upload pending

- [x] Cover page
- [x] Proposed solution
- [x] ML architectures (vision, voice and the 10 predictive models)
- [x] Justification of the models
- [x] Analysis of the results (metrics, baselines, response times, figures)
- [x] Bibliography
- [ ] Appendices: screenshots still to be added
- [ ] Upload `docs/report/main.tex` and `docs/report/figures/` to Overleaf

## 7. Course dataset list

| Task | Dataset |
|---|---|
| Predict the Bitcoin price | Kaggle `team-ai/bitcoin-price-prediction` |
| Predict a car price | GitHub `amankharwal/Website-data` (`car data.csv`) |
| Recommend a movie | Google Drive (movie dataset) |
| Predict S&P 500 stock prices | Kaggle `camnugent/sandp500` |
| Predict a house price | Kaggle `zillow-prize-1` |
| Classify wine quality | Kaggle `rajyellow46/wine-quality` |
| Predict a company's inventory | Kaggle `demand-forecasting-kernels-only` |
| Predict bike trip fares | Kaggle `chh-ola` |
| Classify mobile customer churn | GitHub `IBM/telco-customer-churn-on-icp4d` |
| Predict flight delays | Kaggle `giovamata/airlinedelaycauses` |
| Classify stroke risk | Kaggle `fedesoriano/stroke-prediction-dataset` |
| Classify the hepatitis type | Kaggle `fedesoriano/hepatitis-c-dataset` |
| Predict a patient's body mass | Kaggle `fedesoriano/body-fat-prediction-dataset` |
| Classify the cirrhosis type | Kaggle `fedesoriano/cirrhosis-prediction-dataset` |
| Predict recovered COVID-19 patients | Kaggle `sudalairajkumar/novel-corona-virus-2019-dataset` |
| Predict the Walmart price index | Kaggle `walmart-recruiting-store-sales-forecasting` |
| Predict the avocado price | Kaggle `neuromusic/avocado-prices` |
| Predict daily crimes in London | Kaggle `jboysen/london-crime` |
| Predict daily crimes in Chicago | Kaggle `chicago/chicago-crime` |
| Predict Rossmann sales | Kaggle `rossmann-store-sales` |

The assignment PDF swaps the links of the hepatitis and body mass tasks; the
table above pairs each task with its correct dataset.

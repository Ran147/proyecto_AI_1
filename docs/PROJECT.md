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
(e.g. "JarvisTEC, tipo de cambio para mañana") and answers by running the
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
mention Azure Face and Speech-to-Text. **Project decision:** emotion
recognition and speech-to-text will run on local models. The current Google
integration is temporary and will be removed once the local models work.

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

### R1. User interface - done

- Arc-reactor style HUD (`frontend/index.html`, `frontend/style.css`).
- Reactor animation driven by the live microphone level (Web Audio API).
- Camera on/off and frame capture (WebRTC).
- Voice input (`MediaRecorder`) and typed input.
- Text response, command list, conversation log and raw API response viewer.
- Browser speech synthesis as an optional extra.

### R5. REST API - architecture done, models in progress

- FastAPI server with CORS; serves the frontend (`backend/main.py`).
- `GET /health`, `GET /models`, `GET /models/{name}`,
  `POST /models/{name}/predict`, `POST /command`, `POST /emotion`,
  `POST /transcribe`.
- Single model orchestrator (`backend/core/orchestrator.py`), explicit model
  registry with duplicate detection, one package per model with an input
  schema. See [ADDING_A_MODEL.md](ADDING_A_MODEL.md).
- Pending: expose the 10 trained models (1 of 10 done).

### R2. Emotion recognition - flow done, local model pending

- Done: camera frame capture, `POST /emotion`, normalized output
  (`happiness`, `sadness`, `anger`, `surprise`, `neutral`), probability bars
  and empathetic reply prefixes in the UI, and a mock mode for development.
- Pending:
  - Remove the Google Vision code (`_call_vision`, `parse_vision_response`,
    `GOOGLE_API_KEY`) from `backend/services/emotion_service.py`.
  - Choose a local architecture (e.g. a CNN trained on FER-2013, or a local
    OpenCV / PyTorch / ONNX model).
  - Add local face detection (e.g. Haar cascade or MediaPipe) to crop the
    face before classification.
  - Run inference locally in the backend.

### R3. Speech-to-text and commands - flow done, local model pending

- Done: browser recording (`audio/webm` or `audio/wav`), `POST /transcribe`,
  command router with text normalization, tolerant wake word ("JarvisTEC",
  "Jarvis", "yarvis tek"), fuzzy keyword matching and model execution, and a
  mock mode for development.
- Pending:
  - Remove the Google Speech-to-Text code (`_call_speech`,
    `parse_speech_response`) from `backend/services/speech_service.py`.
  - Integrate a local offline engine (e.g. Whisper / faster-whisper, Vosk in
    Spanish, or Wav2Vec2).
  - Update `requirements.txt` accordingly.

### R4, R6. Machine learning models - 1 of 10

| # | Model | Dataset | Status |
|---|---|---|---|
| 1 | Wine quality | Kaggle `rajyellow46/wine-quality` | Trained and integrated; team review pending |
| 2 | Car price | GitHub `amankharwal/Website-data` (`car data.csv`) | Not started |
| 3 | Telco customer churn | GitHub `IBM/telco-customer-churn-on-icp4d` | Not started |
| 4 | Stroke risk | Kaggle `fedesoriano/stroke-prediction-dataset` | Not started |
| 5 | Avocado price | Kaggle `neuromusic/avocado-prices` | Not started |
| 6 | Bitcoin price | Kaggle `team-ai/bitcoin-price-prediction` | Demo only (not trained) |
| 7 | Body fat | Kaggle `fedesoriano/body-fat-prediction-dataset` | Not started |
| 8 | Movie recommendation | Google Drive dataset from the assignment | Not started |
| 9 | Hepatitis C | Kaggle `fedesoriano/hepatitis-c-dataset` | Not started |
| 10 | Cirrhosis | Kaggle `fedesoriano/cirrhosis-prediction-dataset` | Not started |

Each model needs a notebook with the 6 rubric sections, a backend package, a
trained artifact and voice commands. Details and the definition of done are
in [NEXT_STEPS.md](../NEXT_STEPS.md).

### R7. Scientific document - not started

- [ ] Cover page
- [ ] Proposed solution
- [ ] ML architectures (vision, voice and the 10 predictive models)
- [ ] Justification of the models
- [ ] Analysis of the results (metrics, confusion matrices, ROC / R2 / RMSE,
      response times)
- [ ] Bibliography (IEEE, ACM, Springer or Scopus papers)
- [ ] Appendices (screenshots, class diagrams, API endpoints)

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

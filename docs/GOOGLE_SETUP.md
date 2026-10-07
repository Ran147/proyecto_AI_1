# Google Cloud setup (face and voice)

> Optional. The project runs without any key: voice is transcribed by the
> browser and emotion by a local model (see [PROJECT.md](PROJECT.md),
> sections R2 and R3). This guide switches both to the Google Cloud APIs,
> which the assignment allows.

JarvisTEC can use two Google APIs:

- **Cloud Vision** (face detection) to read the emotion in a camera frame.
- **Cloud Speech-to-Text** to turn microphone audio into text.

Both are called over REST with an **API key**; no SDK is installed.

## Steps

1. Go to https://console.cloud.google.com and sign in.
2. Create a new project (e.g. `jarvistec`).
3. Enable billing for the project. Google requires it even within the free
   quota; check the current prices and free quotas of Vision and
   Speech-to-Text first, and set a **budget alert** (Billing > Budgets and
   alerts) to avoid surprises.
4. In **APIs and services > Library**, enable:
   - `Cloud Vision API`
   - `Cloud Speech-to-Text API`
5. In **APIs and services > Credentials > Create credentials > API key**,
   create a key.
6. Edit the key and, under **API restrictions**, limit it to those two APIs,
   so a leaked key does less damage.
7. Copy `.env.example` to `.env` and paste the key:
   ```
   GOOGLE_API_KEY=AIza...
   ```
8. Restart `uvicorn`. The indicator in the top-right corner of the interface
   should no longer say "voz/rostro simulados".

**Never** commit `.env` or the key (`.env` is in `.gitignore`). If the key
leaks, delete it in the console and create a new one.

## Sharing the key with the team

Every member can use the same key (share it through a private channel, never
the repository) or create their own key in the same project.

## Without a key

Nothing has to be configured:

- Voice: the interface uses the speech recognition built into Chrome and
  Edge and never calls `/transcribe`. `POST /transcribe` answers 503 and says
  that no speech provider is configured.
- Emotion: `/emotion` runs the local model (YuNet + FER+ + HSEmotion); its
  files are in `artifacts/`.

Simulated providers exist for development and tests only
(`EMOTION_PROVIDER=mock`, `SPEECH_PROVIDER=mock` in `.env`); their responses
include `"provider": "mock"` and the interface shows it. **Do not use them in
the demo or the evaluation.**

## How emotions are interpreted

Google Vision does not return percentages but likelihood levels
(`VERY_UNLIKELY` ... `VERY_LIKELY`) for four emotions: joy, sorrow, anger and
surprise. The backend maps them to 0-1 scores and picks the dominant emotion;
when none reaches `POSSIBLE`, the face is considered **neutral**. This
limitation (4 emotions, no fear or disgust) must be stated in the document.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Google Vision: API key not valid` | Key copied incorrectly, or the API is not enabled / not in the key's restrictions |
| `Google Vision: ... billing` | Billing is not enabled for the project |
| `No se detectó ningún rostro` | Poor lighting, face turned sideways or too far away |
| `No se entendió el audio` | Microphone too far, noise, or audio longer than about 1 minute |
| Empty transcription in Safari | Safari records MP4, which Google does not accept: use Chrome, Edge or Firefox |
| "JarvisTEC" transcribed oddly | Expected: the router tolerates variants ("yarvis tec"), and hint phrases are already sent to Google |

## Switching providers

`backend/services/emotion_service.py` and `speech_service.py` each expose a
single public function (`detect_emotion`, `transcribe`) and a `PROVIDERS`
dictionary. To add a provider (e.g. Azure), write a new function and
register it there; the rest of the system does not change.

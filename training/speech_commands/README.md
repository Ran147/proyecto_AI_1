# Spoken command recognition from scratch (`speech_commands`)

Our own speech engine: a convolutional network trained from random
initialization, with no pretrained weights. It is **optional**: the
pretrained engines (Vosk on the server and the browser's recognition) stay
the default, and the backend uses this network when `.env` has
`SPEECH_PROVIDER=scratch` (`backend/services/speech_service.py`).

## Problem

- Task: audio classification (keyword spotting), 12 classes: the digits
  *zero* to *nine*, `unknown` (any other word) and `silence`.
- What it answers: which model the user asks for, by its number.
  "JarvisTEC, three" runs the third model of the list; "zero" is the tenth.
  When this engine is configured, the interface shows the numbers in the
  command list.
- It is not a general speech-to-text system: it does not understand free
  sentences, Spanish, or data inside the command ("red wine").

## Dataset

- Source: [Speech Commands v0.02](https://www.tensorflow.org/datasets/catalog/speech_commands)
  (Warden, 2018), Google. Download:
  `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz` (2.3 GB),
  unpacked into `data/speech_commands/`.
- License: CC BY 4.0.
- 105 829 one-second WAV clips (16 kHz) of 35 English words, 2 618
  speakers. The official `validation_list.txt` and `testing_list.txt` keep
  every speaker in a single split.

## Configuration

See the **Configuration** cell of the notebook: log-mel features (25 ms
windows every 10 ms, 40 bands), unknown words subsampled to twice an average
digit class and redrawn every epoch, silence clips cut from the background
noise, `SEED = 42`.

## Reproduce

Training needs PyTorch and a GPU is recommended. From the repository root:

```bash
python -m venv .venv-train
.venv-train\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv-train\Scripts\python -m pip install -r requirements-train.txt
```

(Without an NVIDIA GPU, install `torch` from the default index; training
then takes much longer.) `.venv-train` is git-ignored.

Then run `training/speech_commands/notebook.ipynb` with that environment. It
writes `artifacts/speech_commands.onnx` and `artifacts/speech_commands.meta.json`.
The backend runs the ONNX file with OpenCV, so the application itself does
not need PyTorch.

## Evaluation

Test set: the official test list (250 speakers never heard in training)
with the unknown words subsampled to twice an average digit class, plus
silence clips; 5 338 clips, used once (copied from
`artifacts/speech_commands.meta.json`). Best epoch 37 of 40 (validation
accuracy 0.978). 963 532 parameters, all trained from random
initialization; the ONNX file is 3.9 MB.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Most frequent class | 0.1538 | 0.0222 |
| Average spectrum + logistic regression | 0.6000 | 0.6181 |
| **CNN from scratch (ours)** | **0.9760** | **0.9789** |

- Every digit: F1 between 0.963 (four) and 0.990 (six); unknown 0.940;
  silence 0.995.
- Precision of the detected digits: 0.981. Unknown words or silence taken
  for a digit: 4.1 %.

**As the application uses it** (notebook section 5): 600 recordings of
about 2.5 s built from test speakers (a leading word, pauses, a digit and
background noise), with the sliding window and the 0.7 threshold:

| Recordings with a digit (400) | Share |
|---|---|
| Right digit | 0.890 |
| Missed (the user is asked to repeat) | 0.102 |
| Wrong digit | 0.008 |

Recordings without a digit (200) that triggered a command: 0.065.

An extra check with voices of another kind (the Windows synthetic voices
"Microsoft David" and "Zira" saying "Jarvis Tec, *number*", 22 recordings):
all 20 numbers recognized and both sentences without a number ignored. One
of them is `tests/data/jarvis_three.wav`, used by the tests.

## References

- Warden, P. (2018). Speech Commands: A dataset for limited-vocabulary speech
  recognition. arXiv:1804.03209.
- Sainath, T. N., & Parada, C. (2015). Convolutional neural networks for
  small-footprint keyword spotting. Interspeech 2015.
- Park, D. S., et al. (2019). SpecAugment: A simple data augmentation method
  for automatic speech recognition. Interspeech 2019.

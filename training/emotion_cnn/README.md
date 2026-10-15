# Emotion recognition from scratch (`emotion_cnn`)

Our own emotion classifier: a convolutional network trained from random
initialization, with no pretrained weights. It is **optional**: the backend
keeps the pretrained classifiers (FER+ and HSEmotion averaged) as the
default, and uses this network when `.env` has `EMOTION_MODEL=scratch`
(`backend/services/emotion_service.py`). The face itself is found by the
pretrained YuNet detector in both cases.

## Problem

- Task: image classification, 7 classes: anger, disgust, fear, happiness,
  neutral, sadness, surprise.
- What it answers: the dominant expression of the face in front of the
  camera, which JarvisTEC uses to adapt the tone of its answers.

## Dataset

- Source: Kaggle [msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013),
  the FER-2013 dataset (Goodfellow et al., 2013), unpacked into
  `data/fer2013/` (`train/<emotion>/*.jpg`, `test/<emotion>/*.jpg`).
- License: released for the ICML 2013 challenge for research use; check the
  Kaggle page.
- 35 887 grayscale faces of 48x48 pixels: 28 709 train, 7 178 test.

## Configuration

See the **Configuration** cell of the notebook: 90/10 stratified split of
the training images for validation, inverse-frequency class weights, label
smoothing 0.1, AdamW with a one-cycle schedule, GPU data augmentation,
`SEED = 42`.

## Reproduce

Same training environment as `training/speech_commands/README.md`
(`.venv-train` with PyTorch). Run `training/emotion_cnn/notebook.ipynb`; it
writes `artifacts/emotion_cnn.onnx` and `artifacts/emotion_cnn.meta.json`.
The backend runs the ONNX file with OpenCV.

## Evaluation

Test set: the 7 178 FER-2013 test images, used once (copied from
`artifacts/emotion_cnn.meta.json`). Best epoch 56 of 60 (validation macro F1
0.6658). 4.8 million parameters, all trained from random initialization.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Most frequent class | 0.2471 | 0.0566 |
| PCA + logistic regression | 0.3348 | 0.2970 |
| **CNN from scratch (ours)** | **0.6861** | **0.6598** |
| FER+ (pretrained) | 0.5646 | 0.4667 |
| HSEmotion (pretrained on AffectNet) | 0.5599 | 0.5066 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| anger | 0.609 | 0.641 | 0.625 | 958 |
| disgust | 0.497 | 0.721 | 0.588 | 111 |
| fear | 0.578 | 0.497 | 0.535 | 1024 |
| happiness | 0.905 | 0.851 | 0.877 | 1774 |
| neutral | 0.584 | 0.728 | 0.648 | 1233 |
| sadness | 0.615 | 0.496 | 0.549 | 1247 |
| surprise | 0.759 | 0.838 | 0.796 | 831 |

**Inside the application** (`python scripts/evaluate_emotion_pipeline.py`):
420 test faces pasted into a 640x480 frame, found and cropped by YuNet as
with the camera. Our network reads the central 85 % of the detector's box
(chosen on validation faces):

| Classifier | Accuracy (417 detected faces) |
|---|---|
| **scratch (ours)** | **0.643** |
| ensemble (FER+ and HSEmotion) | 0.561 |
| hsemotion | 0.523 |
| ferplus | 0.518 |

Limitations: noisy labels (one annotator per image); 568 test images are
copies of training images (a known issue of FER-2013); grayscale, mostly
posed web photos, so real webcam use will be less accurate; an expression
is not a feeling.

## References

- Goodfellow, I. J., et al. (2013). Challenges in representation learning: A
  report on three machine learning contests. ICONIP 2013.
- Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for
  large-scale image recognition. ICLR 2015.
- Khaireddin, Y., & Chen, Z. (2021). Facial emotion recognition: State of the
  art performance on FER2013. arXiv:2105.03588.
- Barsoum, E., Zhang, C., Ferrer, C. C., & Zhang, Z. (2016). Training deep
  networks for facial expression recognition with crowd-sourced label
  distribution (FER+). ICMI 2016.

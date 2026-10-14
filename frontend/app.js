/* JarvisTEC frontend.
 *
 * API contract:
 *   GET  /models      -> [{ name, title, keywords, default_input, input_fields, input_options }]
 *   POST /command     JSON { text, inputs_by_model } -> { matched, message, model, result, ... }
 *   POST /emotion     multipart field "image" -> { emotion, scores, provider }
 *   POST /transcribe  multipart field "audio" (WAV) -> { text, confidence, provider }
 *                     (the server's local model or Google; the browser's own
 *                     speech recognition is the other engine, see "Microphone")
 */

// Optional API base URL, e.g. ?api=http://localhost:8000
const API = new URLSearchParams(location.search).get("api") || "";
// Language of the answers, written and spoken.
const LANGUAGE = "en-US";
// Commands are understood in English and Spanish; this is what the speech
// recognition listens for.
const COMMAND_LANGUAGES = {
  en: { label: "English", recognition: "en-US" },
  es: { label: "Spanish", recognition: "es-CR" },
};

const $ = (id) => document.getElementById(id);
const el = {
  apiStatus: $("apiStatus"), btnMute: $("btnMute"), btnEngine: $("btnEngine"), btnLang: $("btnLang"),
  video: $("video"), videoWrap: document.querySelector(".video-wrap"),
  btnCam: $("btnCam"), btnEmotion: $("btnEmotion"),
  emotionLabel: $("emotionLabel"), emotionNote: $("emotionNote"), emotionBars: $("emotionBars"),
  reactor: $("reactor"), status: $("status"), response: $("response"),
  raw: $("raw"), rawJson: $("rawJson"),
  btnMic: $("btnMic"), micText: $("micText"),
  form: $("typedForm"), typed: $("typed"),
  commands: $("commands"), log: $("log"),
};

const STATUS = { idle: "Standing by", listening: "Listening", thinking: "Processing", speaking: "Answering" };
const EMOTION_LABELS = {
  happiness: "Happiness", sadness: "Sadness", anger: "Anger", surprise: "Surprise",
  fear: "Fear", disgust: "Disgust", contempt: "Contempt", neutral: "Neutral",
};
// Aliases for providers that use other emotion names (Google Vision).
const EMOTION_ALIAS = { happy: "happiness", joy: "happiness", sad: "sadness", sorrow: "sadness", angry: "anger" };
const EMOTION_PREFIX = {
  sadness: "You seem a little down. ", anger: "I understand you are upset. Let's take it calmly. ",
  happiness: "Glad to see you in a good mood! ", surprise: "", fear: "It's all right, I'm here. ",
};
const NO_EMOTION_NOTE = "Turn the camera on so JarvisTEC can read your expression.";

// speech: "server" records WAV audio and sends it to POST /transcribe (the
// local model or Google); "browser" uses the Web Speech API of Chrome and
// Edge; "none" when neither is available. serverSpeech is the provider the
// server reports, or null when it has no engine.
let state = { camOn: false, stream: null, recording: false, voice: true, emotion: "neutral", busy: false, speech: "none", serverSpeech: null, lang: "en", chips: [] };
const ENGINE_LABELS = { local: "local model", google: "Google", mock: "simulated" };
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let currentUtterance = null;
let rec = { chunks: [], stream: null, ctx: null, raf: 0, node: null, rate: 0 };
// Values the user edited in each model's data form: { model: { field: value } }.
const customInputs = {};

/* ---------- Utilities ---------- */
function setState(s) {
  document.body.dataset.state = s;
  el.status.textContent = STATUS[s] || s;
}
function say(text, isError = false) {
  el.response.textContent = text;
  el.response.classList.toggle("error", isError);
}
function logLine(kind, text) {
  if (el.log.querySelector(".muted")) el.log.innerHTML = "";
  const li = document.createElement("li");
  li.className = kind; li.textContent = text;
  el.log.prepend(li);
}
async function api(path, opts) {
  const r = await fetch(API + path, opts);
  let body = null;
  try { body = await r.json(); } catch { /* response without a JSON body */ }
  if (!r.ok) throw new Error(body?.detail || `Error ${r.status}`);
  return body;
}
function stopSpeaking() {
  // A cancelled utterance still fires "end" later; it must not touch the state.
  currentUtterance = null;
  window.speechSynthesis?.cancel();
}
function speak(text) {
  if (!state.voice || !("speechSynthesis" in window)) return;
  stopSpeaking();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = LANGUAGE;
  const voices = speechSynthesis.getVoices();
  u.voice = voices.find(v => v.lang === LANGUAGE) || voices.find(v => v.lang.startsWith("en")) || null;
  u.onstart = () => { if (currentUtterance === u) setState("speaking"); };
  u.onend = u.onerror = () => {
    if (currentUtterance !== u) return;
    currentUtterance = null;
    setState("idle");
  };
  currentUtterance = u;
  speechSynthesis.speak(u);
}

/* ---------- Connection and commands ---------- */
async function init() {
  try {
    const h = await api("/health");
    el.apiStatus.className = "api ok";
    // Two speech engines: the server's engine (local Vosk model or Google)
    // and the browser's recognition. The local model on the server is preferred
    // as the first option whenever available; the button switches between them.
    if (h.speech_provider in ENGINE_LABELS) state.serverSpeech = h.speech_provider;
    if (state.serverSpeech) state.speech = "server";
    else if (SpeechRecognition) state.speech = "browser";
    state.baseStatus = `API connected · ${h.models.length} models`;
    state.simulatedFace = h.emotion_provider === "mock";
    showSpeechEngine();
    const models = await api("/models");
    renderCommands(models);
  } catch (e) {
    el.apiStatus.className = "api down";
    el.apiStatus.lastElementChild.textContent = "API unavailable";
    el.commands.innerHTML = '<p class="muted">Could not load the commands. Is uvicorn running?</p>';
  }
}
function showSpeechEngine() {
  const engine = state.speech === "browser" ? "browser"
    : state.speech === "server" ? ENGINE_LABELS[state.serverSpeech] : null;
  const notes = [engine ? `speech: ${engine}` : "speech unavailable"];
  if (state.simulatedFace) notes.push("simulated face");
  el.apiStatus.lastElementChild.textContent = [state.baseStatus, ...notes].join(" · ");
  el.apiStatus.title = engine ? "" : "No speech recognition here: use Chrome or Edge, or type the command";
  // The switch only makes sense when both engines exist.
  el.btnEngine.hidden = !(SpeechRecognition && state.serverSpeech);
  el.btnEngine.textContent = `Speech: ${engine}`;
}
function switchSpeechEngine() {
  if (state.recording) return;
  state.speech = state.speech === "browser" ? "server" : "browser";
  showSpeechEngine();
}
// Main phrase of a model in the language of the commands.
function phraseOf(m) {
  return (state.lang === "es" && (m.keywords_es || [])[0]) || m.keywords[0] || m.name;
}
function switchCommandLanguage() {
  if (state.recording) return;
  state.lang = state.lang === "en" ? "es" : "en";
  el.btnLang.textContent = `Commands: ${COMMAND_LANGUAGES[state.lang].label}`;
  state.chips.forEach(({ label, model }) => { label.textContent = `JarvisTEC, ${phraseOf(model)}`; });
  el.typed.placeholder = `Or type a command: JarvisTEC, ${state.lang === "es" ? "calidad del vino" : "wine quality"}`;
}
function renderCommands(models) {
  el.commands.innerHTML = "";
  state.chips = [];
  models.forEach(m => {
    const item = document.createElement("div");
    item.className = "chip-item";
    const b = document.createElement("button");
    b.className = "chip";
    b.innerHTML = "<b></b><small></small>";
    b.querySelector("b").textContent = `JarvisTEC, ${phraseOf(m)}`;
    b.querySelector("small").textContent = m.title;
    b.onclick = () => sendCommand(`JarvisTEC ${phraseOf(m)}`);
    state.chips.push({ label: b.querySelector("b"), model: m });
    item.appendChild(b);
    const fields = Object.keys(m.default_input || {});
    if (fields.length) {
      const toggle = document.createElement("button");
      toggle.className = "chip-data"; toggle.type = "button";
      toggle.textContent = "Data";
      toggle.setAttribute("aria-expanded", "false");
      toggle.title = "Edit the data this model uses";
      const form = buildInputForm(m);
      form.hidden = true;
      toggle.onclick = () => {
        form.hidden = !form.hidden;
        toggle.setAttribute("aria-expanded", String(!form.hidden));
      };
      item.append(toggle, form);
    }
    el.commands.appendChild(item);
  });
}

/* ---------- Data form of each model ---------- */
// The form is generated from GET /models: one control per default_input
// field. Only the values the user changed are sent; the rest keep the
// model's defaults.
function buildInputForm(m) {
  const form = document.createElement("form");
  form.className = "inputs"; form.autocomplete = "off"; form.noValidate = true;
  Object.entries(m.default_input).forEach(([field, value]) => {
    const label = document.createElement("label");
    const name = document.createElement("span");
    name.textContent = field;
    const description = (m.input_fields || {})[field] || "";
    const options = (m.input_options || {})[field];
    let control;
    if (options) {
      control = document.createElement("select");
      options.forEach(o => {
        const opt = document.createElement("option");
        opt.value = String(o); opt.textContent = String(o);
        control.appendChild(opt);
      });
    } else {
      control = document.createElement("input");
      control.type = typeof value === "number" ? "number" : "text";
      // "int, ..." fields take whole numbers.
      if (control.type === "number") control.step = description.startsWith("int") ? "1" : "any";
    }
    control.name = field;
    // Optional lists arrive as null; their description says what they hold.
    const isList = Array.isArray(value) || (value === null && /\bprices\b|\blist\b/i.test(description));
    control.dataset.kind = isList ? "list" : value === null ? "optional" : typeof value;
    control.value = formatValue(value);
    control.dataset.initial = control.value;
    label.title = description || field;
    const hint = document.createElement("small");
    hint.textContent = describeField(description);
    label.append(name, control, hint);
    form.appendChild(label);
  });
  const actions = document.createElement("div");
  actions.className = "inputs-actions";
  const run = document.createElement("button");
  run.type = "submit"; run.textContent = "Run with this data";
  const reset = document.createElement("button");
  reset.type = "button"; reset.className = "ghost"; reset.textContent = "Reset";
  reset.onclick = () => {
    form.querySelectorAll("[name]").forEach(c => { c.value = c.dataset.initial; });
    delete customInputs[m.name];
  };
  actions.append(run, reset);
  form.appendChild(actions);
  // Edited values also apply to spoken and typed commands for this model.
  form.oninput = () => {
    try { customInputs[m.name] = readForm(form); } catch { /* reported on submit */ }
  };
  form.onsubmit = (e) => {
    e.preventDefault();
    try {
      customInputs[m.name] = readForm(form);
    } catch (error) {
      say(error.message, true); logLine("err", error.message);
      el.response.scrollIntoView({ block: "center", behavior: "smooth" });
      return;
    }
    sendCommand(`JarvisTEC ${phraseOf(m)}`);
  };
  return form;
}
function formatValue(value) {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}
// "float, alcohol, % vol." -> "alcohol, % vol."
function describeField(text) {
  if (!text) return "";
  const cut = text.indexOf(", ");
  return cut === -1 ? "" : text.slice(cut + 2);
}
// "1.10, 1.12 1.15" -> [1.10, 1.12, 1.15]; anything that is not a number is
// an error instead of being dropped silently.
function parseNumberList(field, raw) {
  const parts = raw.split(/[\s,;]+/).filter(Boolean);
  const numbers = parts.map(Number);
  if (numbers.some(Number.isNaN)) {
    throw new Error(`${field}: write numbers separated by commas, with a decimal point (for example 1.10, 1.25).`);
  }
  return numbers;
}
function readForm(form) {
  const values = {};
  form.querySelectorAll("[name]").forEach(c => {
    if (c.value === c.dataset.initial) return;  // unchanged: keep the default
    const raw = c.value.trim();
    const kind = c.dataset.kind;
    if (kind === "number") {
      if (raw === "") return;  // emptied: keep the default
      if (Number.isNaN(Number(raw))) throw new Error(`${c.name}: write a number.`);
      values[c.name] = Number(raw);
    } else if (kind === "list") {
      if (raw === "" && c.dataset.initial === "") return;  // optional list left empty
      values[c.name] = parseNumberList(c.name, raw);
    } else if (kind === "optional") {
      if (raw !== "") values[c.name] = raw;  // e.g. a date
    } else if (kind === "boolean") {
      values[c.name] = raw === "true";
    } else {
      values[c.name] = raw;
    }
  });
  return values;
}

async function sendCommand(text) {
  text = text.trim();
  if (!text) return false;
  if (state.busy) {
    logLine("err", "Still working on the previous command; try again in a moment.");
    return false;
  }
  state.busy = true;
  stopSpeaking();
  setState("thinking");
  logLine("you", text);
  say("…");
  el.response.scrollIntoView({ block: "center", behavior: "smooth" });
  try {
    // With the camera on, the emotion is read in parallel with the command.
    const emoP = state.camOn ? analyzeEmotion({ silent: true }) : Promise.resolve(null);
    const data = await api("/command", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, inputs_by_model: customInputs }),
    });
    await emoP;
    showRaw(data);
    if (!data.matched) {
      say(data.message, true); logLine("err", data.message); setState("idle");
    } else {
      const msg = (EMOTION_PREFIX[state.emotion] || "") + data.message;
      say(msg); logLine("bot", msg); speak(msg);
      if (!currentUtterance) setState("idle");
    }
  } catch (e) {
    say(e.message, true); logLine("err", e.message); setState("idle");
  } finally {
    state.busy = false;
  }
  return true;
}
function showRaw(data) {
  el.raw.hidden = false;
  el.rawJson.textContent = JSON.stringify(data, null, 2);
}

/* ---------- Camera and emotion ---------- */
// Without a current reading the assistant must not keep reacting to an old one.
function clearEmotion(note) {
  state.emotion = "neutral";
  document.body.dataset.emotion = "neutral";
  el.emotionLabel.textContent = "Not analyzed";
  el.emotionBars.innerHTML = "";
  if (note) el.emotionNote.textContent = note;
}
async function toggleCamera() {
  if (state.camOn) {
    state.stream.getTracks().forEach(t => t.stop());
    state.camOn = false; el.video.srcObject = null;
    el.videoWrap.classList.remove("on");
    el.btnCam.textContent = "Turn camera on"; el.btnEmotion.disabled = true;
    clearEmotion(NO_EMOTION_NOTE);
    el.emotionNote.classList.remove("warn");
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    el.video.srcObject = state.stream;
    state.camOn = true; el.videoWrap.classList.add("on");
    el.btnCam.textContent = "Turn camera off"; el.btnEmotion.disabled = false;
    el.emotionNote.textContent = 'Press "Analyze emotion" or send a command: it is analyzed automatically.';
    el.emotionNote.classList.remove("warn");
  } catch (e) {
    el.emotionNote.textContent = "Could not access the camera. Check the browser permissions.";
    el.emotionNote.classList.add("warn");
  }
}
function captureFrame() {
  const c = document.createElement("canvas");
  c.width = el.video.videoWidth || 640; c.height = el.video.videoHeight || 480;
  c.getContext("2d").drawImage(el.video, 0, 0, c.width, c.height);
  return new Promise(res => c.toBlob(res, "image/jpeg", 0.85));
}
// One frame can catch a blink or a half-formed expression, so a few frames
// are analyzed and their scores averaged.
const EMOTION_FRAMES = 3;
const EMOTION_FRAME_GAP_MS = 150;
async function readEmotion() {
  const readings = [];
  let lastError = null;
  for (let i = 0; i < EMOTION_FRAMES; i++) {
    if (i) await new Promise(r => setTimeout(r, EMOTION_FRAME_GAP_MS));
    try {
      const fd = new FormData(); fd.append("image", await captureFrame(), "frame.jpg");
      readings.push(await api("/emotion", { method: "POST", body: fd }));
    } catch (e) {
      lastError = e;
    }
  }
  if (!readings.length) throw lastError;
  const scores = {};
  readings.forEach(r => Object.entries(r.scores || {}).forEach(([k, v]) => {
    scores[k] = (scores[k] || 0) + v / readings.length;
  }));
  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  // Providers without scores for every emotion (Google) keep their own verdict.
  const emotion = readings[0].provider === "local" && ranked.length ? ranked[0][0] : readings[0].emotion;
  return { ...readings[0], emotion, scores, frames: readings.length };
}
async function analyzeEmotion({ silent = false } = {}) {
  if (!state.camOn) return null;
  try {
    const data = await readEmotion();
    applyEmotion(data);
    return data;
  } catch (e) {
    clearEmotion(e.message);
    el.emotionNote.classList.add("warn");
    if (!silent) logLine("err", `Emotion: ${e.message}`);
    return null;
  }
}
function applyEmotion(data) {
  const key = EMOTION_ALIAS[data.emotion] || data.emotion || "neutral";
  state.emotion = key;
  document.body.dataset.emotion = key;
  el.emotionLabel.textContent = EMOTION_LABELS[key] || key;
  el.emotionNote.textContent = data.provider === "mock"
    ? "Simulated emotion: the local emotion models are missing."
    : "Dominant emotion detected.";
  el.emotionNote.classList.toggle("warn", data.provider === "mock");
  el.emotionBars.innerHTML = "";
  Object.entries(data.scores || {})
    .sort((a, b) => b[1] - a[1]).slice(0, 5)
    .forEach(([k, v]) => {
      const name = EMOTION_LABELS[EMOTION_ALIAS[k] || k] || k;
      const pct = Math.round(v * 100);
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = '<span></span><div class="track"><div class="fill"></div></div><span class="pct"></span>';
      row.children[0].textContent = name;
      row.querySelector(".fill").style.width = pct + "%";
      row.querySelector(".pct").textContent = pct + "%";
      el.emotionBars.appendChild(row);
    });
}

/* ---------- Microphone ---------- */
const MIC_DENIED = "Could not access the microphone. Check the browser permissions.";
function showListening() {
  state.recording = true;
  el.btnMic.setAttribute("aria-pressed", "true"); el.micText.textContent = "Stop";
  setState("listening");
  say("I'm listening…");
}
async function toggleMic() {
  if (state.recording) return stopRecording();
  if (state.speech === "browser") return startBrowserSpeech();
  if (state.speech === "none") {
    say("This browser has no speech recognition. Use Chrome or Edge, or type the command.", true);
    return;
  }
  try {
    rec.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    say(MIC_DENIED, true);
    return;
  }
  stopSpeaking();
  rec.chunks = [];
  watchLevel(rec.stream);
  captureSamples(rec.stream);
  showListening();
}
function stopRecording() {
  // The browser recognizer finishes in its own "end" event.
  if (recognition) return recognition.stop();
  const wav = encodeWav(rec.chunks, rec.rate);
  releaseMic();
  onRecordingDone(wav);
}

/* Server engine: the raw microphone samples are kept and sent as a WAV file
 * (16-bit PCM, mono, 16 kHz), the format the local model reads. */
const WAV_RATE = 16000;
function captureSamples(stream) {
  rec.rate = rec.ctx.sampleRate;
  const source = rec.ctx.createMediaStreamSource(stream);
  rec.node = rec.ctx.createScriptProcessor(4096, 1, 1);
  rec.node.onaudioprocess = (e) => {
    if (state.recording) rec.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  source.connect(rec.node);
  rec.node.connect(rec.ctx.destination);  // required for the node to run
}
function encodeWav(chunks, rate) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const input = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) { input.set(c, offset); offset += c.length; }
  // Downsample to 16 kHz by averaging the samples that fall in each step.
  const step = rate / WAV_RATE;
  const length = Math.floor(total / step);
  const pcm = new Int16Array(length);
  for (let i = 0; i < length; i++) {
    const from = Math.floor(i * step), to = Math.min(total, Math.floor((i + 1) * step));
    let sum = 0;
    for (let j = from; j < to; j++) sum += input[j];
    const value = Math.max(-1, Math.min(1, sum / Math.max(1, to - from)));
    pcm[i] = value < 0 ? value * 0x8000 : value * 0x7fff;
  }
  const view = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const text = (at, s) => { for (let i = 0; i < s.length; i++) view.setUint8(at + i, s.charCodeAt(i)); };
  text(0, "RIFF"); view.setUint32(4, 36 + pcm.length * 2, true); text(8, "WAVE");
  text(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, WAV_RATE, true); view.setUint32(28, WAV_RATE * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, "data"); view.setUint32(40, pcm.length * 2, true);
  pcm.forEach((v, i) => view.setInt16(44 + i * 2, v, true));
  return new Blob([view], { type: "audio/wav" });
}
function releaseMic() {
  state.recording = false;
  rec.stream?.getTracks().forEach(t => t.stop());
  rec.stream = null;
  rec.node?.disconnect(); rec.node = null;
  cancelAnimationFrame(rec.raf); rec.ctx?.close(); rec.ctx = null;
  el.reactor.style.setProperty("--level", 0);
  el.btnMic.setAttribute("aria-pressed", "false"); el.micText.textContent = "Speak";
}

/* Speech recognition in the browser (Web Speech API): no audio is sent to
 * the backend, only the transcribed text goes to POST /command. It stops by
 * itself after a pause, or when the button is pressed again. */
const SPEECH_ERRORS = {
  "not-allowed": MIC_DENIED,
  "service-not-allowed": MIC_DENIED,
  "audio-capture": "No microphone was found.",
  "no-speech": "I did not hear anything. Try again.",
  "network": "The browser's speech recognition needs an internet connection.",
};
const NOT_UNDERSTOOD = "I could not understand the audio. Try again.";
function startBrowserSpeech() {
  stopSpeaking();
  let finalText = "", error = null;
  recognition = new SpeechRecognition();
  recognition.lang = COMMAND_LANGUAGES[state.lang].recognition;
  recognition.interimResults = true;  // show the words while they are said
  recognition.maxAlternatives = 1;
  recognition.onresult = (e) => {
    let heard = "";
    finalText = "";
    for (const result of e.results) {
      if (result.isFinal) finalText += result[0].transcript;
      heard += result[0].transcript;
    }
    say(heard || "I'm listening…");
  };
  recognition.onerror = (e) => { error = e.error; };
  recognition.onend = async () => {
    recognition = null;
    releaseMic();
    const text = finalText.trim();
    if (text) {
      if (!(await sendCommand(text))) setState("idle");
      return;
    }
    // The browser's recognition needs its vendor's online service; when
    // that fails and the server has an engine, switch to it.
    if (["network", "service-not-allowed"].includes(error) && state.serverSpeech) {
      state.speech = "server";
      showSpeechEngine();
      const notice = `The browser's speech recognition is not available; switched to ${ENGINE_LABELS[state.serverSpeech]}. Press Speak again.`;
      say(notice, true); logLine("err", notice); setState("idle");
      return;
    }
    // "aborted" only happens when the user stops before saying anything.
    const message = SPEECH_ERRORS[error] || NOT_UNDERSTOOD;
    say(message, true); logLine("err", message); setState("idle");
  };
  try {
    recognition.start();
  } catch {
    recognition = null;
    say("Could not start the speech recognition.", true);
    return;
  }
  showListening();
  // Only for the reactor animation; recognition does not depend on it.
  navigator.mediaDevices?.getUserMedia({ audio: true }).then((stream) => {
    if (!state.recording) return stream.getTracks().forEach(t => t.stop());
    rec.stream = stream;
    watchLevel(stream);
  }).catch(() => {});
}
function watchLevel(stream) {
  // The reactor pulses with the real microphone volume.
  rec.ctx = new AudioContext();
  const an = rec.ctx.createAnalyser(); an.fftSize = 256;
  rec.ctx.createMediaStreamSource(stream).connect(an);
  const buf = new Uint8Array(an.frequencyBinCount);
  const tick = () => {
    an.getByteTimeDomainData(buf);
    let peak = 0; for (const v of buf) peak = Math.max(peak, Math.abs(v - 128));
    el.reactor.style.setProperty("--level", Math.min(1, peak / 60).toFixed(2));
    rec.raf = requestAnimationFrame(tick);
  };
  tick();
}
async function onRecordingDone(blob) {
  setState("thinking"); say("Transcribing…");
  try {
    const fd = new FormData(); fd.append("audio", blob, "audio.wav");
    fd.append("language", state.lang);
    const { text, provider } = await api("/transcribe", { method: "POST", body: fd });
    if (!text) throw new Error(NOT_UNDERSTOOD);
    if (provider === "mock") logLine("err", "Simulated transcription (SPEECH_PROVIDER=mock).");
    if (!(await sendCommand(text))) setState("idle");
  } catch (e) {
    say(e.message, true); logLine("err", e.message); setState("idle");
  }
}

/* ---------- Events ---------- */
el.btnCam.onclick = toggleCamera;
el.btnEmotion.onclick = () => analyzeEmotion();
el.btnMic.onclick = toggleMic;
el.btnEngine.onclick = switchSpeechEngine;
el.btnLang.onclick = switchCommandLanguage;
el.form.onsubmit = async (e) => {
  e.preventDefault();
  // The text is kept when the command could not be sent yet.
  if (await sendCommand(el.typed.value)) el.typed.value = "";
};
el.btnMute.onclick = () => {
  state.voice = !state.voice;
  el.btnMute.setAttribute("aria-pressed", String(!state.voice));
  el.btnMute.textContent = state.voice ? "Voice on" : "Voice muted";
  if (!state.voice) { stopSpeaking(); if (document.body.dataset.state === "speaking") setState("idle"); }
};
init();

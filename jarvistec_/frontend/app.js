/* JarvisTEC – frontend.
 * Contrato con la API (para el paso 5):
 *   POST /emotion     (multipart, campo "image")  -> { "emotion": "happiness", "scores": { "happiness": 0.9, ... } }
 *   POST /transcribe  (multipart, campo "audio")  -> { "text": "jarvistec precio del bitcoin" }
 *   POST /command     (json {text})               -> { matched, message, model, result, ... }
 */
const API = new URLSearchParams(location.search).get("api") || "";  // ej: ?api=http://localhost:8000

const $ = (id) => document.getElementById(id);
const el = {
  apiStatus: $("apiStatus"), btnMute: $("btnMute"),
  video: $("video"), videoWrap: document.querySelector(".video-wrap"),
  btnCam: $("btnCam"), btnEmotion: $("btnEmotion"),
  emotionLabel: $("emotionLabel"), emotionNote: $("emotionNote"), emotionBars: $("emotionBars"),
  reactor: $("reactor"), status: $("status"), response: $("response"),
  raw: $("raw"), rawJson: $("rawJson"),
  btnMic: $("btnMic"), micText: $("micText"),
  form: $("typedForm"), typed: $("typed"),
  commands: $("commands"), log: $("log"),
};

const STATUS = { idle: "En espera", listening: "Escuchando", thinking: "Procesando", speaking: "Respondiendo" };
const EMOTION_ES = {
  happiness: "Alegría", sadness: "Tristeza", anger: "Enojo", surprise: "Sorpresa",
  fear: "Miedo", disgust: "Asco", contempt: "Desprecio", neutral: "Neutral",
};
// Alias por si el proveedor (Azure, Google) usa otros nombres
const EMOTION_ALIAS = { happy: "happiness", joy: "happiness", sad: "sadness", sorrow: "sadness", angry: "anger" };
const EMOTION_PREFIX = {
  sadness: "Te noto un poco decaído. ", anger: "Entiendo que estés molesto. Vamos con calma. ",
  happiness: "¡Qué buen ánimo! ", surprise: "", fear: "Tranquilo, aquí estoy. ",
};

let state = { camOn: false, stream: null, recording: false, voice: true, emotion: "neutral", busy: false };
let rec = { media: null, chunks: [], stream: null, ctx: null, raf: 0 };

/* ---------- utilidades ---------- */
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
  try { body = await r.json(); } catch { /* respuesta sin JSON */ }
  if (!r.ok) throw new Error(body?.detail || `Error ${r.status}`);
  return body;
}
function speak(text) {
  if (!state.voice || !("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "es-CR";
  const voices = speechSynthesis.getVoices();
  u.voice = voices.find(v => v.lang === "es-CR") || voices.find(v => v.lang.startsWith("es")) || null;
  u.onstart = () => setState("speaking");
  u.onend = u.onerror = () => setState("idle");
  speechSynthesis.speak(u);
}

/* ---------- conexión y comandos ---------- */
async function init() {
  try {
    const h = await api("/health");
    el.apiStatus.className = "api ok";
    el.apiStatus.lastElementChild.textContent = `API conectada · ${h.models.length} modelos`;
    const models = await api("/models");
    renderCommands(models);
  } catch (e) {
    el.apiStatus.className = "api down";
    el.apiStatus.lastElementChild.textContent = "API no disponible";
    el.commands.innerHTML = '<p class="muted">No se pudo cargar. ¿Está corriendo uvicorn?</p>';
  }
}
function renderCommands(models) {
  el.commands.innerHTML = "";
  models.forEach(m => {
    const b = document.createElement("button");
    b.className = "chip";
    const phrase = m.keywords[0] || m.name;
    b.innerHTML = "<b></b><small></small>";
    b.querySelector("b").textContent = `JarvisTEC, ${phrase}`;
    b.querySelector("small").textContent = m.title;
    b.onclick = () => sendCommand(`JarvisTEC ${phrase}`);
    el.commands.appendChild(b);
  });
}

async function sendCommand(text) {
  text = text.trim();
  if (!text || state.busy) return;
  state.busy = true;
  setState("thinking");
  logLine("you", text);
  say("…");
  try {
    // Si la cámara está encendida, se lee la emoción en paralelo al comando
    const emoP = state.camOn ? analyzeEmotion({ silent: true }) : Promise.resolve(null);
    const data = await api("/command", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    await emoP;
    showRaw(data);
    if (!data.matched) {
      say(data.message, true); logLine("err", data.message); setState("idle");
    } else {
      const msg = (EMOTION_PREFIX[state.emotion] || "") + data.message;
      say(msg); logLine("bot", msg); speak(msg);
      if (!state.voice) setState("idle");
    }
  } catch (e) {
    say(e.message, true); logLine("err", e.message); setState("idle");
  } finally {
    state.busy = false;
  }
}
function showRaw(data) {
  el.raw.hidden = false;
  el.rawJson.textContent = JSON.stringify(data, null, 2);
}

/* ---------- cámara y emoción ---------- */
async function toggleCamera() {
  if (state.camOn) {
    state.stream.getTracks().forEach(t => t.stop());
    state.camOn = false; el.video.srcObject = null;
    el.videoWrap.classList.remove("on");
    el.btnCam.textContent = "Encender cámara"; el.btnEmotion.disabled = true;
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    el.video.srcObject = state.stream;
    state.camOn = true; el.videoWrap.classList.add("on");
    el.btnCam.textContent = "Apagar cámara"; el.btnEmotion.disabled = false;
    el.emotionNote.textContent = "Pulsa «Analizar emoción» o envía un comando: se analiza solo.";
    el.emotionNote.classList.remove("warn");
  } catch (e) {
    el.emotionNote.textContent = "No se pudo acceder a la cámara. Revisa los permisos del navegador.";
    el.emotionNote.classList.add("warn");
  }
}
function captureFrame() {
  const c = document.createElement("canvas");
  c.width = el.video.videoWidth || 640; c.height = el.video.videoHeight || 480;
  c.getContext("2d").drawImage(el.video, 0, 0, c.width, c.height);
  return new Promise(res => c.toBlob(res, "image/jpeg", 0.85));
}
async function analyzeEmotion({ silent = false } = {}) {
  if (!state.camOn) return null;
  try {
    const blob = await captureFrame();
    const fd = new FormData(); fd.append("image", blob, "frame.jpg");
    const data = await api("/emotion", { method: "POST", body: fd });
    applyEmotion(data);
    return data;
  } catch (e) {
    el.emotionNote.textContent = e.message;
    el.emotionNote.classList.add("warn");
    if (!silent) logLine("err", `Emoción: ${e.message}`);
    return null;
  }
}
function applyEmotion(data) {
  const key = EMOTION_ALIAS[data.emotion] || data.emotion || "neutral";
  state.emotion = key;
  document.body.dataset.emotion = key;
  el.emotionLabel.textContent = EMOTION_ES[key] || key;
  el.emotionNote.textContent = "Emoción dominante detectada.";
  el.emotionNote.classList.remove("warn");
  el.emotionBars.innerHTML = "";
  Object.entries(data.scores || {})
    .sort((a, b) => b[1] - a[1]).slice(0, 5)
    .forEach(([k, v]) => {
      const name = EMOTION_ES[EMOTION_ALIAS[k] || k] || k;
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

/* ---------- micrófono ---------- */
async function toggleMic() {
  if (state.recording) return stopRecording();
  try {
    rec.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    say("No se pudo acceder al micrófono. Revisa los permisos del navegador.", true);
    return;
  }
  speechSynthesis?.cancel();
  rec.chunks = [];
  rec.media = new MediaRecorder(rec.stream);          // normalmente webm/opus
  rec.media.ondataavailable = e => e.data.size && rec.chunks.push(e.data);
  rec.media.onstop = onRecordingDone;
  rec.media.start();
  watchLevel(rec.stream);
  state.recording = true;
  el.btnMic.setAttribute("aria-pressed", "true"); el.micText.textContent = "Detener";
  setState("listening");
  say("Te escucho…");
}
function stopRecording() {
  state.recording = false;
  rec.media.stop();
  rec.stream.getTracks().forEach(t => t.stop());
  cancelAnimationFrame(rec.raf); rec.ctx?.close();
  el.reactor.style.setProperty("--level", 0);
  el.btnMic.setAttribute("aria-pressed", "false"); el.micText.textContent = "Hablar";
}
function watchLevel(stream) {
  // El reactor late con el volumen real del micrófono
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
async function onRecordingDone() {
  setState("thinking"); say("Transcribiendo…");
  const blob = new Blob(rec.chunks, { type: rec.media.mimeType || "audio/webm" });
  try {
    const fd = new FormData(); fd.append("audio", blob, "audio.webm");
    const { text } = await api("/transcribe", { method: "POST", body: fd });
    if (!text) throw new Error("No se entendió el audio. Intenta de nuevo.");
    await sendCommand(text);
  } catch (e) {
    say(e.message, true); logLine("err", e.message); setState("idle");
  }
}

/* ---------- eventos ---------- */
el.btnCam.onclick = toggleCamera;
el.btnEmotion.onclick = () => analyzeEmotion();
el.btnMic.onclick = toggleMic;
el.form.onsubmit = (e) => { e.preventDefault(); sendCommand(el.typed.value); el.typed.value = ""; };
el.btnMute.onclick = () => {
  state.voice = !state.voice;
  el.btnMute.setAttribute("aria-pressed", String(!state.voice));
  el.btnMute.textContent = state.voice ? "Voz activada" : "Voz silenciada";
  if (!state.voice) speechSynthesis?.cancel();
};
speechSynthesis?.addEventListener?.("voiceschanged", () => {});
init();

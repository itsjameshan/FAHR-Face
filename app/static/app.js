/* FAHR-Face Live — frontend.
 * Vanilla ES2020, no build step, no external requests. Talks only to the same-origin
 * API described in docs/superpowers/specs/2026-10-26-camera-web-demo-design.md §6.
 */
'use strict';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CAPTURE_MAX_WIDTH = 640;
const CAPTURE_JPEG_QUALITY = 0.85;
const MIN_SPACING_MS = 100;
const BACKOFF_STEPS_MS = [500, 1000, 2000, 5000];
const IOU_THRESHOLD = 0.3;
const EMA_ALPHA = 0.3;
const TRACK_TTL_MS = 1000;
const UPLOAD_REENCODE_MAX_PX = 1600;
const DEFAULT_MAX_UPLOAD_BYTES = 8 * 1024 * 1024;
const TRACK_COLORS = ['#2f80ed', '#27ae60', '#e67e22', '#9b51e0', '#eb5757', '#00a3a3', '#c2a200'];
const DOWNLOAD_COMMAND = 'bash scripts/download_checkpoints.sh';
const HUGGING_FACE_URL = 'https://huggingface.co/fridolin22/FAHR-Face';
const RISK_EXPLANATION =
  'Relative mortality-risk score from FAHR-FaceSurvival (uncalibrated; higher = higher estimated risk; ' +
  'only comparable between faces scored by the same model).';
const SURVIVAL_MISSING_NOTE = 'FAHR-FaceSurvival weights are not publicly released';

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

const byId = (id) => document.getElementById(id);
const byTestId = (id) => document.querySelector(`[data-testid="${id}"]`);

const el = {
  body: document.body,
  chips: {
    detector: byTestId('status-detector'),
    device: byTestId('status-device'),
    age: byTestId('status-age'),
    survival: byTestId('status-survival'),
  },
  banners: {
    insecure: byTestId('banner-insecure-context'),
    detector: byTestId('banner-detector-error'),
    random: byTestId('banner-random-weights'),
    ageMissing: byTestId('banner-age-missing'),
    survival: byTestId('banner-survival'),
    camera: byTestId('banner-camera-error'),
    server: byTestId('banner-server-error'),
  },
  stage: byId('stage'),
  video: byId('video'),
  overlay: byId('overlay'),
  startBtn: byId('start-camera'),
  stopBtn: byId('stop-camera'),
  cameraSelect: byId('camera-select'),
  modeToggle: byId('mode-toggle'),
  snapshotBtn: byId('analyze-snapshot'),
  backToLiveBtn: byId('back-to-live'),
  uploadInput: byId('upload-input'),
  mirrorToggle: byId('mirror-toggle'),
  landmarksToggle: byId('landmarks-toggle'),
  status: byId('status'),
  warnings: byId('warnings'),
  latency: byId('latency'),
  faceList: byId('face-list'),
  panelEmpty: byId('panel-empty'),
  consentDialog: byId('consent-dialog'),
  consentAccept: byId('consent-accept'),
  consentCancel: byId('consent-cancel'),
  cardTemplate: byId('face-card-template'),
};

function setBanner(banner, visible, detail) {
  if (!banner) return;
  banner.hidden = !visible;
  const slot = banner.querySelector('.banner-detail');
  if (slot && detail !== undefined) slot.textContent = detail || '';
}

function setChip(chip, value, stateName) {
  chip.querySelector('.chip-value').textContent = value;
  chip.dataset.state = stateName;
}

let lastStatusText = '';
function setStatus(text, kind = 'info') {
  if (text === lastStatusText && el.status.dataset.kind === kind) return;
  lastStatusText = text;
  el.status.textContent = text;
  el.status.dataset.kind = kind;
}

function setWarnings(list) {
  const text = Array.isArray(list) && list.length ? list.join(' · ') : '';
  if (el.warnings.textContent !== text) el.warnings.textContent = text;
  el.warnings.hidden = !text;
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const formatAge = (v) => (isNum(v) ? v.toFixed(1) : 'n/a');
const formatRisk = (v) => (isNum(v) ? `${v >= 0 ? '+' : '-'}${Math.abs(v).toFixed(2)}` : 'n/a');
const formatPercent = (v) => (isNum(v) ? `${Math.round(v * 100)}%` : 'n/a');
const trackColor = (id) => TRACK_COLORS[id % TRACK_COLORS.length];

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

const state = {
  health: null,
  ageAvailable: null, // null = unknown yet
  survivalAvailable: null,
  survivalReason: '',
  maxUploadBytes: DEFAULT_MAX_UPLOAD_BYTES,
  stream: null,
  consented: false,
  mode: 'continuous', // 'continuous' | 'snapshot'
  frozen: null, // { image, width, height, mirror, source } when a still frame is on the stage
  lastResult: null, // { faces: tracked faces, imageSize }
  secure: window.isSecureContext,
};

// ---------------------------------------------------------------------------
// Tracking + smoothing
// ---------------------------------------------------------------------------

function iou(a, b) {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

function ema(previous, value, alpha) {
  if (!isNum(value)) return previous; // skip nulls: keep the previous smoothed value
  if (!isNum(previous)) return value;
  return alpha * value + (1 - alpha) * previous;
}

class Tracker {
  constructor({ iouThreshold = IOU_THRESHOLD, alpha = EMA_ALPHA, ttlMs = TRACK_TTL_MS } = {}) {
    this.iouThreshold = iouThreshold;
    this.alpha = alpha;
    this.ttlMs = ttlMs;
    this.tracks = [];
    this.nextId = 0;
  }

  reset() {
    this.tracks = [];
    this.nextId = 0;
  }

  /** Match detections to tracks greedily by IoU, update EMAs, expire stale tracks.
   *  Returns the detections of this frame enriched with track id and smoothed values. */
  update(faces, now = performance.now()) {
    const pairs = [];
    this.tracks.forEach((track, ti) => {
      faces.forEach((face, fi) => {
        const score = iou(track.box, face.box);
        if (score >= this.iouThreshold) pairs.push({ ti, fi, score });
      });
    });
    pairs.sort((p, q) => q.score - p.score);

    const usedTracks = new Set();
    const usedFaces = new Set();
    const assignment = new Map(); // face index -> track
    for (const { ti, fi } of pairs) {
      if (usedTracks.has(ti) || usedFaces.has(fi)) continue;
      usedTracks.add(ti);
      usedFaces.add(fi);
      assignment.set(fi, this.tracks[ti]);
    }

    const current = faces.map((face, fi) => {
      let track = assignment.get(fi);
      if (!track) {
        track = { id: this.nextId++, smoothedAge: null, smoothedRisk: null };
        this.tracks.push(track);
      }
      track.box = face.box;
      track.landmarks = face.landmarks || null;
      track.score = face.score;
      track.index = face.index;
      track.rawAge = isNum(face.age) ? face.age : null;
      track.rawRisk = isNum(face.risk_score) ? face.risk_score : null;
      track.smoothedAge = ema(track.smoothedAge, track.rawAge, this.alpha);
      track.smoothedRisk = ema(track.smoothedRisk, track.rawRisk, this.alpha);
      if (face.crop) track.crop = face.crop;
      track.lastSeen = now;
      return track;
    });

    this.tracks = this.tracks.filter((t) => now - t.lastSeen <= this.ttlMs);
    return current;
  }

  /** Tracks still alive (seen within the TTL), most recently seen first. */
  alive(now = performance.now()) {
    return this.tracks.filter((t) => now - t.lastSeen <= this.ttlMs);
  }
}

const tracker = new Tracker();

// ---------------------------------------------------------------------------
// Request gate: never two /api/analyze requests at the same time
// ---------------------------------------------------------------------------

let requestChain = Promise.resolve();
function enqueueRequest(fn) {
  const run = requestChain.then(fn, fn);
  requestChain = run.catch(() => {});
  return run;
}

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function postAnalyze(blob, { includeCrops = true } = {}) {
  const form = new FormData();
  form.append('image', blob, 'frame.jpg');
  const url = `/api/analyze?include_crops=${includeCrops ? 'true' : 'false'}`;
  let response;
  try {
    response = await fetch(url, { method: 'POST', body: form, cache: 'no-store' });
  } catch (err) {
    throw new ApiError(`Cannot reach the server (${err.message})`, 0);
  }
  if (!response.ok) {
    let detail = response.statusText || 'request failed';
    try {
      const data = await response.json();
      if (data && data.detail) detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    } catch (_) {
      /* non-JSON error body */
    }
    throw new ApiError(`HTTP ${response.status}: ${detail}`, response.status);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// Continuous analysis loop with back-off
// ---------------------------------------------------------------------------

class AnalysisLoop {
  constructor({ capture, send, onResult, onError, minSpacingMs = MIN_SPACING_MS, backoffMs = BACKOFF_STEPS_MS }) {
    this.capture = capture;
    this.send = send;
    this.onResult = onResult;
    this.onError = onError;
    this.minSpacingMs = minSpacingMs;
    this.backoffMs = backoffMs;
    this.running = false;
    this.timer = null;
    this.failures = 0;
    this.busy = false;
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.failures = 0;
    this.schedule(0);
  }

  stop() {
    this.running = false;
    clearTimeout(this.timer);
    this.timer = null;
  }

  schedule(delayMs) {
    clearTimeout(this.timer);
    if (!this.running) return;
    this.timer = setTimeout(() => this.tick(), delayMs);
  }

  async tick() {
    if (!this.running || this.busy) return;
    this.busy = true;
    const started = performance.now();
    try {
      const blob = await this.capture();
      if (!this.running) return;
      if (!blob) {
        this.schedule(this.minSpacingMs);
        return;
      }
      const t0 = performance.now();
      const result = await enqueueRequest(() => this.send(blob));
      const latency = performance.now() - t0;
      this.failures = 0;
      if (this.running) this.onResult(result, latency);
      this.schedule(Math.max(0, this.minSpacingMs - (performance.now() - started)));
    } catch (err) {
      const delay = this.backoffMs[Math.min(this.failures, this.backoffMs.length - 1)];
      this.failures += 1;
      if (this.running) this.onError(err, delay);
      this.schedule(delay);
    } finally {
      this.busy = false;
    }
  }
}

// ---------------------------------------------------------------------------
// Health + banners
// ---------------------------------------------------------------------------

async function loadHealth() {
  try {
    const response = await fetch('/api/health', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const health = await response.json();
    applyHealth(health);
    setBanner(el.banners.server, false);
    el.body.dataset.health = 'ok';
    setStatus(state.secure ? 'Ready. Click “Start camera” or upload a photo.' : 'Ready. Upload a photo to analyse it.');
  } catch (err) {
    el.body.dataset.health = 'error';
    for (const chip of Object.values(el.chips)) setChip(chip, 'unreachable', 'error');
    setBanner(el.banners.server, true, `Could not load /api/health (${err.message}). Is the server running?`);
    setStatus('Server status unavailable.', 'error');
  }
}

function applyHealth(health) {
  state.health = health;
  const detector = health.detector || {};
  const models = health.models || {};
  const age = models.age || {};
  const survival = models.survival || {};

  setChip(el.chips.detector, detector.available ? detector.name || 'unknown' : 'unavailable', detector.available ? 'ok' : 'error');
  setChip(el.chips.device, health.device || 'unknown', 'ok');

  if (age.weights === 'checkpoint' && age.available) setChip(el.chips.age, 'ready', 'ok');
  else if (age.weights === 'random') setChip(el.chips.age, 'random weights', 'warn');
  else setChip(el.chips.age, 'missing', 'error');

  if (survival.available) setChip(el.chips.survival, 'ready', 'ok');
  else setChip(el.chips.survival, 'not available', 'off');

  state.ageAvailable = Boolean(age.available);
  state.survivalAvailable = Boolean(survival.available);
  state.survivalReason = survival.error || SURVIVAL_MISSING_NOTE;
  state.maxUploadBytes = (health.limits && health.limits.max_upload_bytes) || DEFAULT_MAX_UPLOAD_BYTES;

  setBanner(el.banners.detector, !detector.available, detector.error || 'The server could not load the face detector.');
  setBanner(el.banners.random, age.weights === 'random' || survival.weights === 'random');
  setBanner(el.banners.ageMissing, age.weights === 'none' || !age.available);
  setBanner(el.banners.survival, !survival.available, state.survivalReason);
}

function applyResponseModels(models) {
  if (!models) return;
  if (models.age && typeof models.age.available === 'boolean') state.ageAvailable = models.age.available;
  if (models.survival && typeof models.survival.available === 'boolean') {
    state.survivalAvailable = models.survival.available;
    if (models.survival.reason) state.survivalReason = models.survival.reason;
  }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

function cameraErrorMessage(err) {
  const name = err && err.name;
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return 'Camera permission was denied. Allow camera access for this site in the browser’s address-bar permissions and try again.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No camera was found. Connect a webcam (or pick another camera) and try again.';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'The camera could not be read. It may be in use by another application — close it and try again.';
    case 'OverconstrainedError':
    case 'ConstraintNotSatisfiedError':
      return 'The selected camera does not support the requested resolution or is no longer available. Pick another camera.';
    case 'SecurityError':
      return 'Camera access is blocked in this context. Open the page via http://localhost or HTTPS.';
    case 'AbortError':
      return 'Camera start was aborted by the browser. Try again.';
    default:
      return `Could not start the camera (${(err && err.message) || name || 'unknown error'}).`;
  }
}

function videoConstraints(deviceId) {
  const video = { width: { ideal: 1280 }, height: { ideal: 720 } };
  if (deviceId) video.deviceId = { exact: deviceId };
  else video.facingMode = 'user';
  return { video, audio: false };
}

async function startCamera(deviceId) {
  if (!state.secure || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setBanner(el.banners.insecure, true);
    return;
  }
  setBanner(el.banners.camera, false);
  setStatus('Starting camera…');
  el.startBtn.disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia(videoConstraints(deviceId));
    releaseStream();
    state.stream = stream;
    el.video.srcObject = stream;
    await el.video.play().catch(() => {}); // autoplay is allowed for muted video
    el.stage.classList.add('live');
    el.stopBtn.disabled = false;
    el.snapshotBtn.disabled = false;
    updateStageRatioFromVideo();
    await populateCameras();
    tracker.reset();
    if (state.frozen && state.frozen.source === 'snapshot') unfreeze();
    if (state.mode === 'continuous' && !state.frozen) {
      loop.start();
      setStatus('Camera on — analysing continuously.');
    } else {
      setStatus('Camera on — snapshot mode. Click “Analyze snapshot”.');
    }
    const track = stream.getVideoTracks()[0];
    if (track) track.addEventListener('ended', () => stopCamera('The camera stream ended.'));
  } catch (err) {
    el.startBtn.disabled = false;
    const message = cameraErrorMessage(err);
    setBanner(el.banners.camera, true, message);
    setStatus(message, 'error');
  }
}

function releaseStream() {
  if (state.stream) {
    for (const track of state.stream.getTracks()) track.stop();
  }
  state.stream = null;
  el.video.srcObject = null;
}

function stopCamera(reason) {
  loop.stop();
  releaseStream();
  el.stage.classList.remove('live');
  el.startBtn.disabled = !state.secure;
  el.stopBtn.disabled = true;
  el.snapshotBtn.disabled = true;
  if (!state.frozen) {
    clearOverlay();
    renderCards([]);
  }
  setStatus(reason || 'Camera stopped.');
}

async function populateCameras() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  let devices = [];
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch (_) {
    return;
  }
  const cameras = devices.filter((d) => d.kind === 'videoinput');
  const currentTrack = state.stream && state.stream.getVideoTracks()[0];
  const currentId = currentTrack ? currentTrack.getSettings().deviceId : '';
  el.cameraSelect.textContent = '';
  if (!cameras.length) {
    el.cameraSelect.append(new Option('Default camera', ''));
  }
  cameras.forEach((cam, i) => {
    const option = new Option(cam.label || `Camera ${i + 1}`, cam.deviceId);
    if (cam.deviceId && cam.deviceId === currentId) option.selected = true;
    el.cameraSelect.append(option);
  });
  el.cameraSelect.disabled = cameras.length < 2;
}

function onCameraChange() {
  const deviceId = el.cameraSelect.value;
  if (!state.stream || !deviceId) return;
  loop.stop();
  startCamera(deviceId);
}

// ---------------------------------------------------------------------------
// Consent dialog
// ---------------------------------------------------------------------------

let consentReturnFocus = null;

function openConsent() {
  consentReturnFocus = document.activeElement;
  const dialog = el.consentDialog;
  if (typeof dialog.showModal === 'function') {
    dialog.showModal();
  } else {
    dialog.setAttribute('open', '');
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');
  }
  el.consentAccept.focus();
}

function closeConsent() {
  const dialog = el.consentDialog;
  if (dialog.open && typeof dialog.close === 'function') dialog.close();
  else dialog.removeAttribute('open');
  if (consentReturnFocus && typeof consentReturnFocus.focus === 'function') consentReturnFocus.focus();
}

function onConsentAccept() {
  state.consented = true;
  closeConsent();
  startCamera(el.cameraSelect.value || undefined);
}

function onStartClick() {
  if (state.consented) startCamera(el.cameraSelect.value || undefined);
  else openConsent();
}

// ---------------------------------------------------------------------------
// Stage geometry + overlay drawing
// ---------------------------------------------------------------------------

const overlayCtx = el.overlay.getContext('2d');

function updateStageRatioFromVideo() {
  const w = el.video.videoWidth;
  const h = el.video.videoHeight;
  if (w && h && !state.frozen) el.stage.style.setProperty('--stage-ratio', `${w} / ${h}`);
}

function isMirrored() {
  if (state.frozen) return state.frozen.mirror;
  return el.mirrorToggle.checked;
}

/** Match the canvas buffer to its CSS size (and device pixel ratio). Returns {w, h, changed}. */
function syncCanvasSize() {
  const w = Math.max(1, Math.round(el.stage.clientWidth));
  const h = Math.max(1, Math.round(el.stage.clientHeight));
  const dpr = window.devicePixelRatio || 1;
  const bw = Math.round(w * dpr);
  const bh = Math.round(h * dpr);
  let changed = false;
  if (el.overlay.width !== bw || el.overlay.height !== bh) {
    el.overlay.width = bw;
    el.overlay.height = bh;
    changed = true;
  }
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h, changed };
}

function clearOverlay() {
  const { w, h } = syncCanvasSize();
  overlayCtx.clearRect(0, 0, w, h);
  state.lastResult = null;
}

function drawFrozenFrame(w, h) {
  const frozen = state.frozen;
  if (!frozen) return;
  overlayCtx.save();
  if (frozen.mirror) {
    overlayCtx.translate(w, 0);
    overlayCtx.scale(-1, 1);
  }
  overlayCtx.drawImage(frozen.image, 0, 0, w, h);
  overlayCtx.restore();
}

function overlayLabel(face) {
  const agePart = state.ageAvailable === false ? 'Age n/a' : `Age ${formatAge(face.smoothedAge)}`;
  const riskPart = state.survivalAvailable ? `Risk ${formatRisk(face.smoothedRisk)}` : 'Risk n/a';
  return `${agePart} · ${riskPart}`;
}

function drawOverlay(faces, imageSize) {
  const { w, h } = syncCanvasSize();
  overlayCtx.clearRect(0, 0, w, h);
  drawFrozenFrame(w, h);
  if (!imageSize || !imageSize.width || !imageSize.height) return;

  const sx = w / imageSize.width;
  const sy = h / imageSize.height;
  const mirror = isMirrored();
  const mx = (x) => (mirror ? w - x * sx : x * sx);
  const ctx = overlayCtx;
  const fontPx = Math.max(12, Math.round(w / 48));
  ctx.lineWidth = Math.max(2, Math.round(w / 320));
  ctx.font = `600 ${fontPx}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  ctx.textBaseline = 'middle';

  for (const face of faces) {
    const color = trackColor(face.id);
    const b = face.box;
    const left = Math.min(mx(b.x1), mx(b.x2));
    const right = Math.max(mx(b.x1), mx(b.x2));
    const top = b.y1 * sy;
    const bottom = b.y2 * sy;

    ctx.strokeStyle = color;
    ctx.strokeRect(left, top, right - left, bottom - top);

    if (el.landmarksToggle.checked && face.landmarks) {
      ctx.fillStyle = color;
      for (const point of Object.values(face.landmarks)) {
        if (!Array.isArray(point) || point.length < 2) continue;
        ctx.beginPath();
        ctx.arc(mx(point[0]), point[1] * sy, Math.max(2.5, ctx.lineWidth * 1.2), 0, Math.PI * 2);
        ctx.fill();
      }
    }

    const label = overlayLabel(face);
    const padX = 6;
    const textW = ctx.measureText(label).width;
    const boxH = Math.round(fontPx * 1.6);
    const boxW = textW + padX * 2;
    let lx = left;
    if (lx + boxW > w) lx = Math.max(0, w - boxW);
    let ly = top - boxH - 2;
    if (ly < 0) ly = top + 2;
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly, boxW, boxH);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, lx + padX, ly + boxH / 2);
  }
}

function redrawOverlay() {
  if (state.lastResult) drawOverlay(state.lastResult.faces, state.lastResult.imageSize);
  else if (state.frozen) {
    const { w, h } = syncCanvasSize();
    overlayCtx.clearRect(0, 0, w, h);
    drawFrozenFrame(w, h);
  } else clearOverlay();
}

// ---------------------------------------------------------------------------
// Frame capture
// ---------------------------------------------------------------------------

const captureCanvas = document.createElement('canvas');

function canvasToBlob(canvas, type, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('Could not encode frame'))), type, quality);
  });
}

/** Draw `source` (video or image) scaled to at most maxWidth onto the capture canvas. */
function drawScaled(source, srcW, srcH, maxWidth) {
  const scale = Math.min(1, maxWidth / srcW);
  const w = Math.max(1, Math.round(srcW * scale));
  const h = Math.max(1, Math.round(srcH * scale));
  captureCanvas.width = w;
  captureCanvas.height = h;
  const ctx = captureCanvas.getContext('2d');
  ctx.drawImage(source, 0, 0, w, h);
  return { width: w, height: h };
}

async function captureVideoFrame() {
  const video = el.video;
  if (!state.stream || video.readyState < 2 || !video.videoWidth || !video.videoHeight) return null;
  drawScaled(video, video.videoWidth, video.videoHeight, CAPTURE_MAX_WIDTH);
  return canvasToBlob(captureCanvas, 'image/jpeg', CAPTURE_JPEG_QUALITY);
}

/** Copy the current video frame into a standalone bitmap for the frozen stage. */
function snapshotBitmap() {
  const video = el.video;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas;
}

// ---------------------------------------------------------------------------
// Results → overlay + cards
// ---------------------------------------------------------------------------

function handleResult(result, latencyMs, source) {
  applyResponseModels(result.models);
  setWarnings(result.warnings);
  const faces = tracker.update(Array.isArray(result.faces) ? result.faces : []);
  const imageSize = result.image || null;
  state.lastResult = { faces, imageSize };
  drawOverlay(faces, imageSize);
  renderCards(faces, tracker.alive());
  el.faceList.dataset.source = source;
  el.latency.textContent = `${Math.round(latencyMs)} ms`;
  const n = faces.length;
  const prefix = source === 'live' ? 'Live: ' : source === 'snapshot' ? 'Snapshot: ' : 'Upload: ';
  setStatus(`${prefix}${n === 0 ? 'no face detected' : n === 1 ? '1 face detected' : `${n} faces detected`}.`);
}

function handleLoopError(err, delayMs) {
  const seconds = delayMs >= 1000 ? `${delayMs / 1000} s` : `${delayMs} ms`;
  setStatus(`Analysis failed — ${err.message}. Retrying in ${seconds}.`, 'error');
}

const cardsById = new Map();

function renderCards(currentFaces, aliveTracks = currentFaces) {
  const currentIds = new Set(currentFaces.map((f) => f.id));
  const ordered = [
    ...currentFaces.slice().sort((a, b) => a.index - b.index),
    ...aliveTracks.filter((t) => !currentIds.has(t.id)),
  ];
  const keep = new Set(ordered.map((t) => t.id));

  for (const [id, card] of cardsById) {
    if (!keep.has(id)) {
      card.remove();
      cardsById.delete(id);
    }
  }

  ordered.forEach((track, position) => {
    let card = cardsById.get(track.id);
    if (!card) {
      card = el.cardTemplate.content.firstElementChild.cloneNode(true);
      cardsById.set(track.id, card);
    }
    fillCard(card, track, currentIds.has(track.id));
    if (el.faceList.children[position] !== card) el.faceList.insertBefore(card, el.faceList.children[position] || null);
  });

  el.panelEmpty.hidden = ordered.length > 0;
}

function fillCard(card, track, seenNow) {
  card.dataset.trackId = String(track.id);
  card.dataset.seen = seenNow ? 'true' : 'false';
  card.style.setProperty('--track-color', trackColor(track.id));
  card.querySelector('.face-index').textContent = `Face ${track.index + 1}`;
  card.querySelector('.face-status').textContent = seenNow ? `track #${track.id + 1}` : 'last seen <1 s ago';

  const ageEl = card.querySelector('.face-age');
  if (state.ageAvailable === false) {
    ageEl.textContent = 'not available';
    ageEl.classList.add('unavailable');
    ageEl.title = 'The FAHR-FaceAge checkpoint is not loaded on the server.';
  } else {
    ageEl.textContent = formatAge(track.smoothedAge);
    ageEl.classList.toggle('unavailable', !isNum(track.smoothedAge));
    ageEl.title = `Smoothed (EMA α=${EMA_ALPHA}); latest raw value: ${formatAge(track.rawAge)}`;
  }

  const riskEl = card.querySelector('.face-risk');
  const riskNote = card.querySelector('.face-risk-note');
  if (!state.survivalAvailable) {
    riskEl.textContent = 'not available';
    riskEl.classList.add('unavailable');
    riskEl.title = state.survivalReason || SURVIVAL_MISSING_NOTE;
    riskNote.textContent = SURVIVAL_MISSING_NOTE;
  } else {
    riskEl.textContent = formatRisk(track.smoothedRisk);
    riskEl.classList.toggle('unavailable', !isNum(track.smoothedRisk));
    riskEl.title = `${RISK_EXPLANATION} Latest raw value: ${formatRisk(track.rawRisk)}`;
    riskNote.textContent = RISK_EXPLANATION;
  }

  card.querySelector('.face-score').textContent = formatPercent(track.score);

  const figure = card.querySelector('.face-figure');
  const img = card.querySelector('.face-crop');
  if (track.crop) {
    figure.hidden = false;
    if (img.src !== track.crop) img.src = track.crop;
    img.alt = `Aligned 112 by 112 crop of face ${track.index + 1}`;
  } else {
    figure.hidden = true;
  }
}

// ---------------------------------------------------------------------------
// Modes: continuous / snapshot / upload
// ---------------------------------------------------------------------------

const loop = new AnalysisLoop({
  capture: captureVideoFrame,
  send: (blob) => postAnalyze(blob),
  onResult: (result, latency) => handleResult(result, latency, 'live'),
  onError: handleLoopError,
});

function setMode(mode) {
  state.mode = mode;
  const snapshot = mode === 'snapshot';
  el.modeToggle.dataset.mode = mode;
  el.modeToggle.textContent = snapshot ? 'Snapshot' : 'Continuous';
  el.modeToggle.setAttribute('aria-pressed', snapshot ? 'true' : 'false');
  el.modeToggle.title = snapshot ? 'Snapshot mode: switch to continuous analysis' : 'Continuous mode: switch to single snapshots';
  if (snapshot) {
    loop.stop();
    if (state.stream && !state.frozen) setStatus('Snapshot mode — click “Analyze snapshot” to analyse one frame.');
  } else {
    if (state.frozen && state.frozen.source === 'snapshot') unfreeze();
    if (state.stream && !state.frozen) {
      loop.start();
      setStatus('Camera on — analysing continuously.');
    }
  }
}

function toggleMode() {
  setMode(state.mode === 'continuous' ? 'snapshot' : 'continuous');
}

function freeze(image, width, height, mirror, source) {
  loop.stop();
  state.frozen = { image, width, height, mirror, source };
  el.stage.dataset.frozen = source;
  el.stage.style.setProperty('--stage-ratio', `${width} / ${height}`);
  el.backToLiveBtn.hidden = false;
  state.lastResult = null;
  redrawOverlay();
}

function unfreeze() {
  state.frozen = null;
  el.stage.dataset.frozen = '';
  el.backToLiveBtn.hidden = true;
  clearOverlay();
  updateStageRatioFromVideo();
}

async function analyzeOnce(blob, source) {
  setStatus(source === 'snapshot' ? 'Analysing snapshot…' : 'Analysing uploaded image…');
  const t0 = performance.now();
  try {
    const result = await enqueueRequest(() => postAnalyze(blob));
    if (state.frozen && state.frozen.source === source) handleResult(result, performance.now() - t0, source);
  } catch (err) {
    setStatus(`Analysis failed — ${err.message}`, 'error');
  }
}

async function analyzeSnapshot() {
  if (!state.stream || el.video.readyState < 2) {
    setStatus('The camera is not delivering frames yet.', 'warn');
    return;
  }
  if (state.mode !== 'snapshot') setMode('snapshot');
  const blob = await captureVideoFrame();
  if (!blob) return;
  freeze(snapshotBitmap(), el.video.videoWidth, el.video.videoHeight, el.mirrorToggle.checked, 'snapshot');
  await analyzeOnce(blob, 'snapshot');
}

function backToLive() {
  const wasUpload = state.frozen && state.frozen.source === 'upload';
  unfreeze();
  renderCards([]);
  if (wasUpload) tracker.reset();
  el.uploadInput.value = '';
  if (!state.stream) {
    setStatus(state.secure ? 'Camera is off. Click “Start camera” or upload another photo.' : 'Upload another photo to analyse it.');
    return;
  }
  if (state.mode === 'continuous') {
    loop.start();
    setStatus('Camera on — analysing continuously.');
  } else {
    setStatus('Snapshot mode — click “Analyze snapshot” to analyse one frame.');
  }
}

function loadImageFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve({ img, url });
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('The selected file is not a readable image.'));
    };
    img.src = url;
  });
}

/** Original bytes when within the server limit; otherwise re-encode at ≤1600 px. */
async function blobForUpload(file, img) {
  if (file.size <= state.maxUploadBytes) return file;
  const maxSide = Math.max(img.naturalWidth, img.naturalHeight);
  const scale = Math.min(1, UPLOAD_REENCODE_MAX_PX / maxSide);
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
  canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
  return canvasToBlob(canvas, 'image/jpeg', 0.9);
}

async function onUpload(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  let loaded;
  try {
    loaded = await loadImageFile(file);
  } catch (err) {
    setStatus(err.message, 'error');
    return;
  }
  const { img, url } = loaded;
  tracker.reset();
  renderCards([]);
  freeze(img, img.naturalWidth, img.naturalHeight, false, 'upload');
  try {
    const blob = await blobForUpload(file, img);
    await analyzeOnce(blob, 'upload');
  } finally {
    URL.revokeObjectURL(url);
  }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function applyMirror() {
  el.stage.classList.toggle('mirrored', el.mirrorToggle.checked);
  if (state.frozen && state.frozen.source === 'snapshot') state.frozen.mirror = el.mirrorToggle.checked;
  redrawOverlay();
}

function checkSecureContext() {
  const supported = Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  state.secure = Boolean(window.isSecureContext && supported);
  if (!state.secure) {
    setBanner(el.banners.insecure, true);
    el.startBtn.disabled = true;
    el.startBtn.title = 'Camera needs a secure context (localhost or HTTPS).';
  }
}

function init() {
  const origin = location.origin;
  byId('server-origin').textContent = origin;
  for (const node of document.querySelectorAll('.consent-origin')) node.textContent = origin;

  checkSecureContext();
  loadHealth();

  el.startBtn.addEventListener('click', onStartClick);
  el.stopBtn.addEventListener('click', () => stopCamera());
  el.cameraSelect.addEventListener('change', onCameraChange);
  el.modeToggle.addEventListener('click', toggleMode);
  el.snapshotBtn.addEventListener('click', analyzeSnapshot);
  el.backToLiveBtn.addEventListener('click', backToLive);
  el.uploadInput.addEventListener('change', onUpload);
  el.mirrorToggle.addEventListener('change', applyMirror);
  el.landmarksToggle.addEventListener('change', redrawOverlay);

  el.consentAccept.addEventListener('click', onConsentAccept);
  el.consentCancel.addEventListener('click', () => {
    closeConsent();
    setStatus('Camera not started. You can still upload a photo.');
  });
  el.consentDialog.addEventListener('cancel', (event) => {
    // Escape key: let the dialog close, restore focus, keep state consistent.
    event.preventDefault();
    closeConsent();
    setStatus('Camera not started. You can still upload a photo.');
  });
  el.consentDialog.addEventListener('click', (event) => {
    if (event.target === el.consentDialog) closeConsent(); // backdrop click
  });

  el.video.addEventListener('loadedmetadata', () => {
    updateStageRatioFromVideo();
    redrawOverlay();
  });
  if (typeof ResizeObserver === 'function') {
    new ResizeObserver(() => redrawOverlay()).observe(el.stage);
  } else {
    window.addEventListener('resize', redrawOverlay);
  }
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', populateCameras);
  }
  window.addEventListener('pagehide', releaseStream);

  applyMirror();
  setMode('continuous');
}

init();

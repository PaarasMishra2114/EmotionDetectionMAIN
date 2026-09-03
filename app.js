const emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'];
const colors = { Angry: '#ed6a50', Disgusted: '#4c966d', Fearful: '#8170a7', Happy: '#f2c14e', Neutral: '#4f82a7', Sad: '#147d78', Surprised: '#c46680' };
const state = { stream: null, running: false, startedAt: null, samples: [], history: [], analyzing: false, face: null, captureWidth: 640, captureHeight: 480 };
const $ = (id) => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const plot = $('plot');
const context = overlay.getContext('2d');
const plotContext = plot.getContext('2d');
const savedTheme = localStorage.getItem('emotion-theme');
if (savedTheme === 'dark') document.documentElement.dataset.theme = 'dark';

function renderDistribution(values = {}) {
  $('distribution').innerHTML = emotions.map((emotion) => `<div class="emotion-row"><span>${emotion}</span><div class="emotion-bar"><span style="width:${Math.round((values[emotion] || 0) * 100)}%;background:${colors[emotion]}"></span></div><strong>${Math.round((values[emotion] || 0) * 100)}%</strong></div>`).join('');
}
function setSignal(emotion, confidence, source = 'demo signal') {
  $('signalLabel').textContent = source;
  $('signalEmotion').textContent = emotion;
  $('signalEmotion').style.color = colors[emotion];
  $('confidenceValue').textContent = `${Math.round(confidence * 100)}%`;
  $('confidenceBar').style.width = `${Math.round(confidence * 100)}%`;
  $('signalTime').textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
async function getPrediction(image) {
  try {
    const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image }) });
    if (!response.ok) throw new Error(await response.text());
    return await response.json();
  } catch (error) { throw new Error(`CNN unavailable: ${error.message}`); }
}
function frameDataUrl() {
  const sourceWidth = video.videoWidth || 640; const sourceHeight = video.videoHeight || 480;
  const scale = Math.min(1, 960 / Math.max(sourceWidth, sourceHeight));
  state.captureWidth = Math.round(sourceWidth * scale); state.captureHeight = Math.round(sourceHeight * scale);
  const frame = document.createElement('canvas'); frame.width = state.captureWidth; frame.height = state.captureHeight;
  frame.getContext('2d').drawImage(video, 0, 0, frame.width, frame.height);
  return frame.toDataURL('image/jpeg', .9);
}
function drawFaceMarker(face = state.face) {
  const width = overlay.clientWidth; const height = overlay.clientHeight;
  overlay.width = width; overlay.height = height;
  context.clearRect(0, 0, width, height);
  if (!state.running || !face) return;
  const scaleX = width / state.captureWidth; const scaleY = height / state.captureHeight; const x = face.x * scaleX; const y = face.y * scaleY; const boxWidth = face.width * scaleX; const boxHeight = face.height * scaleY;
  context.strokeStyle = '#f2c14e'; context.lineWidth = 2; context.setLineDash([8, 6]); context.strokeRect(x, y, boxWidth, boxHeight); context.setLineDash([]);
  context.fillStyle = '#f2c14e'; context.font = '11px monospace'; context.fillText('FACE / TRACKING', x, y - 10);
}
function drawPlot() {
  const width = plot.clientWidth; const height = plot.clientHeight; const ratio = window.devicePixelRatio || 1;
  plot.width = width * ratio; plot.height = height * ratio; plotContext.setTransform(ratio, 0, 0, ratio, 0, 0); plotContext.clearRect(0, 0, width, height);
  if (!state.samples.length) return;
  emotions.forEach((emotion) => { plotContext.beginPath(); plotContext.strokeStyle = colors[emotion]; plotContext.lineWidth = emotion === state.samples[state.samples.length - 1].emotion ? 2.4 : 1.15; state.samples.forEach((sample, index) => { const x = (index / Math.max(state.samples.length - 1, 1)) * (width - 20) + 10; const y = height - ((sample.values[emotion] || 0) * (height - 24)) - 12; index ? plotContext.lineTo(x, y) : plotContext.moveTo(x, y); }); plotContext.stroke(); });
}
function addHistory(prediction) {
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  state.history.unshift({ ...prediction, time }); state.history = state.history.slice(0, 6);
  $('historyCount').textContent = `${state.history.length} events`;
  $('historyList').innerHTML = state.history.map((item) => `<div class="history-item"><span class="history-time">${item.time}</span><div><div class="history-emotion" style="color:${colors[item.emotion]}">${item.emotion}</div><div class="history-bar"><span style="width:${Math.round(item.confidence * 100)}%;background:${colors[item.emotion]}"></span></div></div><span class="history-confidence">${Math.round(item.confidence * 100)}%</span></div>`).join('');
}
function updateSession() { if (!state.startedAt) return; const seconds = Math.floor((Date.now() - state.startedAt) / 1000); $('sessionDuration').textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`; }
async function captureAndAnalyze() {
  if (!state.running || state.analyzing) return;
  state.analyzing = true; $('captureButton').disabled = true; $('captureButton').innerHTML = '<span class="button-icon">&#8230;</span> Analyzing'; $('signalLabel').textContent = 'CNN analyzing';
  try {
    const prediction = await getPrediction(frameDataUrl());
    const values = prediction.emotions || Object.fromEntries(emotions.map((emotion) => [emotion, emotion === prediction.emotion ? prediction.confidence : (1 - prediction.confidence) / 6]));
    state.face = prediction.face || null; drawFaceMarker(); setSignal(prediction.emotion, prediction.confidence, 'CNN + vision signal'); renderDistribution(values); state.samples.push({ emotion: prediction.emotion, values }); state.samples = state.samples.slice(-45); $('sampleCount').textContent = state.samples.length; $('plotEmpty').style.display = 'none'; drawPlot(); addHistory(prediction);
  } catch (error) { $('signalLabel').textContent = error.message.includes('No face') ? 'no face detected' : 'analysis failed'; $('signalEmotion').textContent = 'Try again'; $('confidenceValue').textContent = '0%'; $('confidenceBar').style.width = '0%'; console.error(error); }
  finally { state.analyzing = false; $('captureButton').disabled = !state.running; $('captureButton').innerHTML = '<span class="button-icon">&#9673;</span> Capture &amp; analyze'; }
}
async function analyzeUploadedImage(file) {
  if (state.analyzing) return;
  state.analyzing = true; $('signalLabel').textContent = 'CNN analyzing upload'; $('uploadStatus').textContent = `Analyzing ${file.name}`;
  try {
    const image = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject; reader.readAsDataURL(file); });
    const prediction = await getPrediction(image); const values = prediction.emotions || {};
    setSignal(prediction.emotion, prediction.confidence, 'CNN + vision signal'); renderDistribution(values); state.samples.push({ emotion: prediction.emotion, values }); state.samples = state.samples.slice(-45); $('sampleCount').textContent = state.samples.length; $('plotEmpty').style.display = 'none'; drawPlot(); addHistory({ ...prediction, source: file.name });
  } catch (error) { $('signalLabel').textContent = error.message.includes('No face') ? 'no face detected' : 'upload analysis failed'; $('signalEmotion').textContent = 'Try another image'; console.error(error); }
  finally { state.analyzing = false; $('uploadStatus').textContent = 'Choose a clear face image for CNN analysis'; }
}
async function startCamera() {
  try { state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 1280 }, facingMode: 'user', aspectRatio: { ideal: 1 } }, audio: false }); video.srcObject = state.stream; await video.play(); state.running = true; state.face = null; state.startedAt = Date.now(); $('cameraFrame').classList.add('is-live'); $('placeholder').style.display = 'none'; $('cameraStatus').textContent = 'camera live'; $('cameraStatus').className = 'status-pill live'; $('startButton').disabled = true; $('captureButton').disabled = false; document.documentElement.style.setProperty('--camera-ratio', `${video.videoWidth} / ${video.videoHeight}`); drawFaceMarker(); } catch (error) { $('cameraStatus').textContent = 'camera blocked'; $('signalLabel').textContent = 'permission needed'; $('placeholder').querySelector('p').textContent = 'Camera access was not granted'; console.error(error); }
}
function reset() { if (state.stream) state.stream.getTracks().forEach((track) => track.stop()); state.stream = null; state.running = false; state.analyzing = false; state.face = null; state.startedAt = null; state.samples = []; state.history = []; $('cameraFrame').classList.remove('is-live'); $('placeholder').style.display = 'grid'; $('cameraStatus').textContent = 'camera idle'; $('cameraStatus').className = 'status-pill idle'; $('startButton').disabled = false; $('captureButton').disabled = true; $('signalLabel').textContent = 'Waiting for input'; $('signalEmotion').textContent = 'Neutral'; $('signalEmotion').style.color = colors.Neutral; $('confidenceValue').textContent = '0%'; $('confidenceBar').style.width = '0%'; $('sampleCount').textContent = '0'; $('sessionDuration').textContent = '00:00'; $('historyCount').textContent = '0 events'; $('historyList').innerHTML = '<div class="empty-history">No readings yet. Your session will appear here.</div>'; $('plotEmpty').style.display = 'grid'; renderDistribution(); context.clearRect(0, 0, overlay.width, overlay.height); drawPlot(); }
$('startButton').addEventListener('click', startCamera); $('captureButton').addEventListener('click', captureAndAnalyze); $('resetButton').addEventListener('click', reset); setInterval(updateSession, 1000); window.addEventListener('resize', () => { drawPlot(); drawFaceMarker(); }); renderDistribution(); drawPlot();
$('imageInput').addEventListener('change', (event) => { const [file] = event.target.files; if (file) analyzeUploadedImage(file); event.target.value = ''; });
$('themeToggle').addEventListener('click', () => { const dark = document.documentElement.dataset.theme !== 'dark'; document.documentElement.dataset.theme = dark ? 'dark' : 'light'; localStorage.setItem('emotion-theme', dark ? 'dark' : 'light'); $('themeLabel').textContent = dark ? 'dark' : 'light'; $('themeToggle').setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme'); }); $('themeLabel').textContent = document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';

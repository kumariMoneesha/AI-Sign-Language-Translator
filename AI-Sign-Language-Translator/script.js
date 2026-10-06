// SIGNAL demo: detects a small set of static hand poses in the browser.
const video = document.getElementById('webcam');
const canvas = document.getElementById('landmark-canvas');
const context = canvas.getContext('2d');
const startButton = document.getElementById('start-camera');
const stopButton = document.getElementById('stop-camera');
const clearButton = document.getElementById('clear-text');
const speakButton = document.getElementById('speak-text');
const statusLabel = document.getElementById('detection-status');
const cameraState = document.getElementById('camera-state');
const cameraStateText = document.getElementById('camera-state-text');
const cameraFrame = document.getElementById('camera-frame');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const detectedSign = document.getElementById('detected-sign');
const signCaption = document.getElementById('sign-caption');
const translatedText = document.getElementById('translated-text');
const wordCount = document.getElementById('word-count');
const confidenceFill = document.getElementById('confidence-fill');
const confidenceValue = document.getElementById('confidence-value');
const toast = document.getElementById('toast');

let handModel;
let cameraStream;
let animationFrame;
let isProcessingFrame = false;
let sentence = [];
let activeSign = '';
let toastTimer;

function setStatus(message, state = 'normal') {
  statusLabel.classList.toggle('is-detecting', state === 'detecting');
  statusLabel.classList.toggle('is-error', state === 'error');
  statusLabel.lastChild.textContent = ` ${message}`;
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('is-visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('is-visible'), 3600);
}

function setCameraActive(active) {
  cameraState.classList.toggle('is-active', active);
  cameraStateText.textContent = active ? 'CAMERA ACTIVE' : 'OFFLINE';
  cameraPlaceholder.hidden = active;
  video.style.display = active ? 'block' : 'none';
  cameraFrame.classList.toggle('is-live', active);
  startButton.disabled = active;
  stopButton.disabled = !active;
}

function updateTranscript() {
  const text = sentence.join(' ');
  translatedText.textContent = text || 'Your translated sentence will appear here.';
  translatedText.classList.toggle('is-empty', !text);
  wordCount.textContent = `${sentence.length} ${sentence.length === 1 ? 'WORD' : 'WORDS'}`;
}

function addSignToSentence(sign) {
  sentence.push(sign);
  updateTranscript();
}

// Classify finger extension using the standard MediaPipe 21-point hand layout.
function getFingerState(landmarks) {
  const isExtended = (tipIndex, jointIndex) => landmarks[tipIndex].y < landmarks[jointIndex].y - 0.015;
  const thumbTip = landmarks[4];
  const thumbBase = landmarks[2];
  const thumbJoint = landmarks[3];
  const thumbExtended = Math.abs(thumbTip.x - thumbBase.x) > Math.abs(thumbJoint.x - thumbBase.x) * 1.15;

  return {
    thumb: thumbExtended,
    index: isExtended(8, 6),
    middle: isExtended(12, 10),
    ring: isExtended(16, 14),
    pinky: isExtended(20, 18),
  };
}

// Returns a demo label for a recognizable static pose, or null when unmatched.
function recognizeSign(landmarks) {
  if (!landmarks || landmarks.length < 21) return null;

  const fingers = getFingerState(landmarks);
  const extendedCount = Object.values(fingers).filter(Boolean).length;
  const handScale = Math.hypot(landmarks[0].x - landmarks[9].x, landmarks[0].y - landmarks[9].y);
  const pinchDistance = Math.hypot(landmarks[4].x - landmarks[8].x, landmarks[4].y - landmarks[8].y);

  if (pinchDistance < handScale * 0.38) return 'Thank You';
  if (fingers.thumb && !fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky) return 'Please';
  if (extendedCount === 5) return 'Hello';
  if (!fingers.thumb && fingers.index && fingers.middle && fingers.ring && fingers.pinky) return 'Stop';
  if (!fingers.thumb && !fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky) return 'Yes';
  if (!fingers.thumb && fingers.index && fingers.middle && !fingers.ring && !fingers.pinky) return 'No';
  return null;
}

function updateRecognition(landmarks) {
  if (!landmarks || landmarks.length === 0) {
    activeSign = '';
    detectedSign.textContent = '—';
    signCaption.textContent = 'Keep one hand in view of the camera';
    confidenceFill.style.width = '0';
    confidenceValue.textContent = 'WAITING';
    setStatus('No Hand Detected');
    return;
  }

  const sign = recognizeSign(landmarks[0]);
  if (!sign) {
    activeSign = '';
    detectedSign.textContent = '—';
    signCaption.textContent = 'Hand found · pose not in the demo set';
    confidenceFill.style.width = '18%';
    confidenceValue.textContent = 'UNMATCHED';
    setStatus('Detecting Hand...', 'detecting');
    return;
  }

  detectedSign.textContent = sign;
  signCaption.textContent = 'Demo pose recognized';
  confidenceFill.style.width = '100%';
  confidenceValue.textContent = 'POSE MATCH';
  setStatus('Sign Detected', 'detecting');
  if (sign !== activeSign) addSignToSentence(sign);
  activeSign = sign;
}

function drawResults(results) {
  context.save();
  context.clearRect(0, 0, canvas.width, canvas.height);

  const allHands = results.multiHandLandmarks || [];
  allHands.forEach((landmarks) => {
    window.drawConnectors(context, landmarks, window.HAND_CONNECTIONS, {
      color: '#6be0d1',
      lineWidth: Math.max(2, canvas.width / 320),
    });
    window.drawLandmarks(context, landmarks, {
      color: '#b9aaff',
      fillColor: '#f0ecff',
      lineWidth: 1,
      radius: Math.max(2, canvas.width / 180),
    });
  });

  context.restore();
  updateRecognition(allHands);
}

async function processVideoFrame() {
  if (!cameraStream || !video.videoWidth) return;
  if (!isProcessingFrame) {
    isProcessingFrame = true;
    try {
      await handModel.send({ image: video });
    } catch (error) {
      console.error('Hand detection failed:', error);
      setStatus('Hand detection failed', 'error');
      showToast('MediaPipe could not process this camera frame. Try restarting the camera.');
    } finally {
      isProcessingFrame = false;
    }
  }
  if (cameraStream) animationFrame = requestAnimationFrame(processVideoFrame);
}

function createHandModel() {
  if (typeof window.Hands !== 'function' || typeof window.drawConnectors !== 'function') {
    throw new Error('MediaPipe did not load. Check your internet connection and refresh the page.');
  }

  const model = new window.Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });
  model.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.65,
    minTrackingConfidence: 0.55,
  });
  model.onResults(drawResults);
  return model;
}

function getCameraErrorMessage(error) {
  if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
    return 'Camera permission was denied. Allow camera access in your browser settings, then try again.';
  }
  if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
    return 'No camera was found. Connect a webcam and try again.';
  }
  if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
    return 'The camera is already in use by another app. Close it and try again.';
  }
  if (error.name === 'OverconstrainedError') {
    return 'The camera could not meet the requested video settings.';
  }
  return error.message || 'Unable to start the camera. Check browser permissions and try again.';
}

async function startCamera() {
  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
    const message = 'Camera access needs a secure context. Open this page with VS Code Live Server (localhost or HTTPS).';
    setStatus('Secure connection required', 'error');
    showToast(message);
    return;
  }

  startButton.disabled = true;
  setStatus('Loading hand detector...', 'detecting');
  try {
    if (!handModel) handModel = createHandModel();
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user', width: { ideal: 960 }, height: { ideal: 720 } },
    });
    video.srcObject = cameraStream;
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    setCameraActive(true);
    setStatus('Camera Started', 'detecting');
    animationFrame = requestAnimationFrame(processVideoFrame);
  } catch (error) {
    console.error('Could not start camera:', error);
    if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
    video.srcObject = null;
    setCameraActive(false);
    const message = error.message.includes('MediaPipe') ? error.message : getCameraErrorMessage(error);
    setStatus('Camera unavailable', 'error');
    showToast(message);
    startButton.disabled = false;
  }
}

function stopCamera() {
  if (animationFrame) cancelAnimationFrame(animationFrame);
  animationFrame = null;
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  video.pause();
  video.srcObject = null;
  context.clearRect(0, 0, canvas.width, canvas.height);
  setCameraActive(false);
  setStatus('Camera Ready');
  detectedSign.textContent = '—';
  signCaption.textContent = 'Show a supported demo pose to begin';
  confidenceFill.style.width = '0';
  confidenceValue.textContent = 'WAITING';
  activeSign = '';
}

startButton.addEventListener('click', startCamera);
stopButton.addEventListener('click', stopCamera);
clearButton.addEventListener('click', () => {
  sentence = [];
  updateTranscript();
  if ('speechSynthesis' in window) window.speechSynthesis.cancel();
});
speakButton.addEventListener('click', () => {
  const text = sentence.join(' ');
  if (!text) {
    showToast('Detect a sign first, then the translated text can be spoken.');
    return;
  }
  if (!('speechSynthesis' in window)) {
    showToast('Text-to-speech is not supported by this browser.');
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = document.documentElement.lang || 'en-US';
  utterance.rate = 0.92;
  window.speechSynthesis.speak(utterance);
});

window.addEventListener('beforeunload', () => {
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  if ('speechSynthesis' in window) window.speechSynthesis.cancel();
});

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const placeholder = document.getElementById("camera-placeholder");
const scanIndicator = document.getElementById("scan-indicator");
const resultOverlay = document.getElementById("result-overlay");
const resultBadge = document.getElementById("result-badge");
const resultName = document.getElementById("result-name");
const resultMsg = document.getElementById("result-msg");
const presentList = document.getElementById("present-list");
const todayDate = document.getElementById("today-date");

const cooldownMs = window.SMART_ATTENDANCE_BOOTSTRAP.cooldown_minutes * 60 * 1000;
const STORAGE_KEY = "smart_attendance_cards";

let stream = null;
let scanning = false;
let lastStatus = null;
let hideResultTimer = null;
let nextScanTimer = null;

// name → timerId (for auto-removal)
const activeTimers = new Map();

todayDate.textContent = new Date().toLocaleDateString("en-GB", {
  weekday: "long", day: "numeric", month: "long",
});

// ── LocalStorage helpers ──────────────────────────────────────────────────────

function loadStoredCards() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

function saveStoredCards(cards) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(cards));
}

function removeStoredCard(name) {
  const cards = loadStoredCards();
  delete cards[name];
  saveStoredCards(cards);
}

// ── Card rendering ────────────────────────────────────────────────────────────

function renderCard(name, time, confidence) {
  const card = document.createElement("div");
  card.className = "present-card";
  card.dataset.name = name;
  card.innerHTML = `
    <div class="present-icon">✓</div>
    <div class="present-info">
      <div class="present-top">
        <strong>${name}</strong>
        <span class="present-time">${time}</span>
      </div>
      <span class="present-conf">${confidence}</span>
    </div>
  `;
  const emptyState = presentList.querySelector(".empty-state");
  if (emptyState) emptyState.remove();
  presentList.prepend(card);
  return card;
}

function scheduleCardRemoval(name, remainingMs) {
  if (activeTimers.has(name)) clearTimeout(activeTimers.get(name));

  const timerId = setTimeout(() => {
    const card = presentList.querySelector(`[data-name="${name}"]`);
    if (card) card.remove();
    activeTimers.delete(name);
    removeStoredCard(name);
    if (presentList.children.length === 0) {
      presentList.innerHTML = '<div class="empty-state">No one checked in yet.</div>';
    }
  }, remainingMs);

  activeTimers.set(name, timerId);
}

function addPresenceCard(name, time, confidence) {
  // Remove existing card for this person if re-checking in
  const existing = presentList.querySelector(`[data-name="${name}"]`);
  if (existing) existing.remove();

  renderCard(name, time, confidence);

  // Persist to localStorage
  const cards = loadStoredCards();
  cards[name] = { time, confidence, addedAt: Date.now() };
  saveStoredCards(cards);

  scheduleCardRemoval(name, cooldownMs);
}

// ── Restore cards after page refresh ─────────────────────────────────────────

function restoreCards() {
  const cards = loadStoredCards();
  const now = Date.now();
  let restored = false;

  for (const [name, entry] of Object.entries(cards)) {
    const elapsed = now - entry.addedAt;
    const remaining = cooldownMs - elapsed;

    if (remaining <= 0) {
      removeStoredCard(name);
      continue;
    }

    renderCard(name, entry.time, entry.confidence);
    scheduleCardRemoval(name, remaining);
    restored = true;
  }

  return restored;
}

// ── Camera & scanning ─────────────────────────────────────────────────────────

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    video.hidden = false;
    placeholder.hidden = true;
    scheduleNextScan(1500);
  } catch {
    placeholder.querySelector("p").textContent = "Camera access denied. Allow camera permissions and reload.";
  }
}

function scheduleNextScan(delay) {
  clearTimeout(nextScanTimer);
  nextScanTimer = setTimeout(autoScan, delay);
}

async function autoScan() {
  if (!stream || scanning) return;
  scanning = true;
  scanIndicator.hidden = false;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageData = canvas.toDataURL("image/jpeg", 0.92);

  try {
    const response = await fetch("/api/recognize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_data: imageData }),
    });
    const data = await response.json();
    if (response.ok) {
      handleResult(data.result);
    } else {
      showOverlay(null);
    }
  } catch {
    showOverlay(null);
  }

  scanIndicator.hidden = true;
  scanning = false;

  const delay = lastStatus === "already_marked" ? 20000
              : lastStatus === "marked_present"  ? 8000
              : 4000;
  scheduleNextScan(delay);
}

function handleResult(result) {
  if (!result) { showOverlay(null); return; }

  lastStatus = result.status;

  if (result.status === "marked_present") {
    const time = new Date().toTimeString().slice(0, 8);
    const conf = `${Math.round(result.confidence * 100)}%`;
    addPresenceCard(result.name, time, conf);
  }

  showOverlay(result);
}

function showOverlay(result) {
  clearTimeout(hideResultTimer);

  if (!result) {
    resultOverlay.hidden = true;
    return;
  }

  if (result.status === "marked_present") {
    resultBadge.className = "result-badge success";
    resultBadge.textContent = "Present";
    resultName.textContent = result.name;
    resultMsg.textContent = `${Math.round(result.confidence * 100)}% confidence`;
  } else if (result.status === "already_marked") {
    resultBadge.className = "result-badge warning";
    resultBadge.textContent = "Already marked";
    resultName.textContent = result.name;
    resultMsg.textContent = "Already logged this session";
  } else if (result.status === "unknown") {
    resultBadge.className = "result-badge unknown";
    resultBadge.textContent = "Unknown";
    resultName.textContent = "";
    resultMsg.textContent = result.best_guess
      ? `Best guess: ${result.best_guess} (${Math.round(result.confidence * 100)}%)`
      : "Face not recognized";
  } else {
    resultOverlay.hidden = true;
    return;
  }

  resultOverlay.hidden = false;
  hideResultTimer = setTimeout(() => { resultOverlay.hidden = true; }, 5000);
}

restoreCards();
startCamera();

const EXAMPLE_QUERIES = [
  "Patienten mit ARDS, die spontan gegen das Beatmungsgerät atmen und dadurch einen Druckanstieg verursachen",
  "Abfall der Sauerstoffsättigung (SpO2) unter 90 Prozent",
  "COPD-Patient mit Air Trapping und steigendem endexspiratorischem Druck",
];

const form = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const modalitySelect = document.getElementById("modality");
const topKSelect = document.getElementById("top_k");
const searchBtn = document.getElementById("search-btn");
const examplesEl = document.getElementById("examples");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const answerSection = document.getElementById("answer-section");
const answerText = document.getElementById("answer-text");
const answerSources = document.getElementById("answer-sources");
const answerTimer = document.getElementById("answer-timer");

let timerInterval = null;
let timerStart = 0;

function startTimer() {
  timerStart = performance.now();
  answerTimer.hidden = false;
  answerTimer.className = "timer timer--live";
  answerSection.hidden = false;
  answerText.textContent = "";
  answerSources.innerHTML = "";
  const tick = () => {
    const elapsed = (performance.now() - timerStart) / 1000;
    answerTimer.textContent = `⏱ ${elapsed.toFixed(1)}s`;
  };
  tick();
  timerInterval = setInterval(tick, 100);
}

function stopTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  const elapsed = (performance.now() - timerStart) / 1000;
  answerTimer.className = "timer";
  answerTimer.textContent = `⏱ ${elapsed.toFixed(1)}s`;
  return elapsed;
}

function renderExamples() {
  examplesEl.innerHTML = "";
  for (const example of EXAMPLE_QUERIES) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "example-chip";
    chip.textContent = example;
    chip.addEventListener("click", () => {
      queryInput.value = example;
      form.requestSubmit();
    });
    examplesEl.appendChild(chip);
  }
}

function setStatus(message, kind) {
  if (!message) {
    statusEl.hidden = true;
    statusEl.textContent = "";
    statusEl.className = "status";
    return;
  }
  statusEl.hidden = false;
  statusEl.textContent = message;
  statusEl.className = `status status--${kind}`;
}

function renderAnswer(answer, sources) {
  if (!answer) {
    answerSection.hidden = true;
    answerTimer.hidden = true;
    return;
  }
  answerSection.hidden = false;
  answerText.textContent = answer;
  answerSources.innerHTML = "";
  for (const source of sources || []) {
    const chip = document.createElement("span");
    chip.className = "source-chip";
    chip.textContent = source;
    answerSources.appendChild(chip);
  }
}

function renderHits(hits) {
  resultsEl.innerHTML = "";
  if (!hits.length) {
    resultsEl.innerHTML = '<p class="empty">No matching waveform epochs found.</p>';
    return;
  }
  for (const hit of hits) {
    const card = document.createElement("article");
    card.className = "hit";

    if (hit.plot_url) {
      const img = document.createElement("img");
      img.className = "hit__plot";
      img.loading = "lazy";
      img.src = hit.plot_url;
      img.alt = `Waveform plot for ${hit.epoch_id}`;
      card.appendChild(img);
    }

    const body = document.createElement("div");
    body.className = "hit__body";

    const meta = document.createElement("div");
    meta.className = "hit__meta";
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = hit.modality;
    meta.appendChild(badge);
    if (typeof hit.score === "number") {
      const score = document.createElement("span");
      score.className = "hit__score";
      score.textContent = `score ${hit.score.toFixed(3)}`;
      meta.appendChild(score);
    }
    body.appendChild(meta);

    const idLine = document.createElement("p");
    idLine.className = "hit__id";
    idLine.textContent = `${hit.record_id} · ${hit.epoch_id}`;
    body.appendChild(idLine);

    if (hit.text) {
      const text = document.createElement("p");
      text.className = "hit__text";
      text.textContent = hit.text;
      body.appendChild(text);
    }

    card.appendChild(body);
    resultsEl.appendChild(card);
  }
}

async function refreshHealth() {
  const storeDot = document.getElementById("health-store");
  const llmDot = document.getElementById("health-llm");
  try {
    const res = await fetch("/health");
    const data = await res.json();
    storeDot.className = `dot ${data.status === "ok" ? "dot--ok" : "dot--down"}`;
    llmDot.className = `dot ${data.llm ? "dot--ok" : "dot--down"}`;
  } catch (err) {
    storeDot.className = "dot dot--down";
    llmDot.className = "dot dot--down";
  }
}

async function runSearch(event) {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  searchBtn.disabled = true;
  setStatus("Searching waveform epochs and synthesizing an answer…", "loading");
  resultsEl.innerHTML = "";
  startTimer();

  try {
    const res = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: Number(topKSelect.value),
        modality: modalitySelect.value || null,
        synthesize: true,
      }),
    });

    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    stopTimer();
    setStatus(null);
    renderAnswer(data.answer, data.sources);
    renderHits(data.hits || []);
  } catch (err) {
    stopTimer();
    answerTimer.hidden = true;
    setStatus(err.message || "Search failed", "error");
    answerSection.hidden = true;
    resultsEl.innerHTML = "";
  } finally {
    searchBtn.disabled = false;
  }
}

form.addEventListener("submit", runSearch);
renderExamples();
refreshHealth();

/*
 * app.js — BAGIAN 4: Client web real-time BISINDO.
 *
 * Alur:
 *   webcam (getUserMedia) -> ambil frame tiap ~200 ms (5 fps) -> JPEG base64
 *   -> POST /predict ke FastAPI -> simpan ke jendela voting -> tampilkan huruf stabil
 *
 * Anti-flicker:
 *   - Hanya satu request berjalan pada satu waktu (tidak menumpuk bila server lambat).
 *   - Majority voting atas N prediksi terakhir; huruf baru tampil bila menang
 *     > 50% suara. Prediksi di bawah ambang confidence dihitung sebagai "ragu".
 *   - Huruf masuk ke teks tersusun bila stabil selama X detik (hold-to-commit).
 */
(() => {
  "use strict";

  // ------------------------------------------------------------------ config
  const cfg = {
    apiUrl: defaultApiUrl(),
    fps: 5,
    window: 7,
    threshold: 0.6,
    holdSec: 1.0,
    captureWidth: 320,   // sesuai proposal: frame kecil (lebar 320 px) -> hemat bandwidth saat di-hosting
    jpegQuality: 0.8,
  };

  function defaultApiUrl() {
    // Jika halaman disajikan oleh FastAPI (http://localhost:8000/app/), pakai origin yang sama.
    if (location.protocol.startsWith("http") && location.pathname.startsWith("/app")) return location.origin;
    return "http://localhost:8000";
  }

  // Koneksi antar landmark MediaPipe Hands (21 titik)
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],          // ibu jari
    [0, 5], [5, 6], [6, 7], [7, 8],          // telunjuk
    [5, 9], [9, 10], [10, 11], [11, 12],     // jari tengah
    [9, 13], [13, 14], [14, 15], [15, 16],   // jari manis
    [13, 17], [17, 18], [18, 19], [19, 20],  // kelingking
    [0, 17],                                  // pangkal telapak
  ];
  const HAND_COLORS = ["#DA1212", "#1E6BFF"];

  // ------------------------------------------------------------------- DOM
  const $ = (id) => document.getElementById(id);
  const video = $("video"), overlay = $("overlay"), capture = $("capture");
  const octx = overlay.getContext("2d"), cctx = capture.getContext("2d");
  const ui = {
    stage: $("stage"), stageMsg: $("stageMsg"), noHand: $("noHand"),
    letter: $("letter"), confFill: $("confFill"), confText: $("confText"), voteText: $("voteText"),
    topk: $("topk"), sentence: $("sentence"), holdFill: $("holdFill"), history: $("history"),
    server: $("serverStatus"), fps: $("statFps"), latency: $("statLatency"),
    hands: $("statHands"), model: $("statModel"),
    btnStart: $("btnStart"), btnStop: $("btnStop"),
  };

  // ----------------------------------------------------------------- state
  const st = {
    stream: null, running: false, inFlight: false, timer: null,
    votes: [],                 // {label: "A" | "?" | null, conf}
    shown: null,               // huruf yang sedang ditampilkan
    streak: { label: null, start: 0, committed: false },
    sentence: "", history: [],
    reqTimes: [],              // untuk hitung prediksi/detik
  };

  // ============================================================ server health
  async function checkHealth() {
    try {
      const r = await fetch(`${cfg.apiUrl}/health`, { cache: "no-store" });
      const j = await r.json();
      setServer(j.status === "ok", j.status === "ok" ? `Server OK · ${j.model}` : "Server memuat model…");
      ui.model.textContent = j.model || "–";
    } catch {
      setServer(false, "Server tidak terhubung", true);
    }
  }
  function setServer(ok, text, err = false) {
    ui.server.classList.toggle("ok", ok);
    ui.server.classList.toggle("err", err);
    ui.server.querySelector(".label").textContent = text;
  }

  // ================================================================== camera
  async function startCamera() {
    try {
      st.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
    } catch (e) {
      ui.stageMsg.hidden = false;
      ui.stageMsg.innerHTML = "<span>" + (e.name === "NotAllowedError"
        ? "Akses kamera ditolak. Izinkan kamera di pengaturan browser lalu coba lagi."
        : `Kamera tidak bisa dibuka (${e.name}). Pastikan halaman dibuka via <b>http://localhost</b>.`) + "</span>";
      return;
    }
    video.srcObject = st.stream;
    await video.play();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    ui.stageMsg.hidden = true;
    st.running = true;
    ui.btnStart.disabled = true;
    ui.btnStop.disabled = false;
    scheduleNext(0);
  }

  function stopCamera() {
    st.running = false;
    clearTimeout(st.timer);
    st.stream?.getTracks().forEach((t) => t.stop());
    st.stream = null;
    video.srcObject = null;
    octx.clearRect(0, 0, overlay.width, overlay.height);
    st.votes = [];
    resetStreak();
    renderLetter(null, 0, 0);
    ui.noHand.hidden = true;
    ui.stageMsg.hidden = false;
    ui.stageMsg.innerHTML = "<span>Kamera berhenti.<br />Klik <b>Mulai Kamera</b> untuk melanjutkan.</span>";
    ui.btnStart.disabled = false;
    ui.btnStop.disabled = true;
    ui.fps.textContent = "0";
  }

  // ============================================================ capture loop
  function scheduleNext(delay = 1000 / cfg.fps) {
    clearTimeout(st.timer);
    if (st.running) st.timer = setTimeout(tick, delay);
  }

  async function tick() {
    if (!st.running || st.inFlight || !video.videoWidth) return scheduleNext();
    const t0 = performance.now();
    st.inFlight = true;
    try {
      const res = await sendFrame(grabFrame());
      handleResult(res, performance.now() - t0);
      if (!ui.server.classList.contains("ok")) checkHealth();
    } catch (e) {
      setServer(false, "Gagal menghubungi server", true);
      console.warn(e);
    } finally {
      st.inFlight = false;
      // jaga interval tetap ~1/fps walau request memakan waktu
      scheduleNext(Math.max(0, 1000 / cfg.fps - (performance.now() - t0)));
    }
  }

  function grabFrame() {
    // Frame TIDAK di-mirror: digambar apa adanya dari kamera.
    const scale = cfg.captureWidth / video.videoWidth;
    capture.width = cfg.captureWidth;
    capture.height = Math.round(video.videoHeight * scale);
    cctx.drawImage(video, 0, 0, capture.width, capture.height);
    return capture.toDataURL("image/jpeg", cfg.jpegQuality);
  }

  async function sendFrame(dataUrl) {
    const r = await fetch(`${cfg.apiUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  // ======================================================= hasil & smoothing
  function handleResult(res, latencyMs) {
    // statistik
    const now = performance.now();
    st.reqTimes.push(now);
    st.reqTimes = st.reqTimes.filter((t) => now - t < 2000);
    ui.fps.textContent = (st.reqTimes.length / 2).toFixed(1);
    ui.latency.textContent = Math.round(latencyMs);
    ui.hands.textContent = res.num_hands;

    drawLandmarks(res.landmarks || []);
    renderTopK(res.top_k || []);

    // masukkan ke jendela voting
    let label = null;
    if (res.hand_detected) label = res.confidence >= cfg.threshold ? res.prediction : "?";
    st.votes.push({ label, conf: res.confidence || 0 });
    while (st.votes.length > cfg.window) st.votes.shift();

    const { winner, count, avgConf } = majority(st.votes);
    const needed = Math.floor(cfg.window / 2) + 1;
    const stable = count >= needed ? winner : undefined;   // undefined = belum ada mayoritas

    ui.noHand.hidden = stable !== null;
    if (stable === undefined) {
      // tidak ada mayoritas -> pertahankan tampilan sebelumnya (anti-flicker)
      ui.voteText.textContent = `voting ${count}/${st.votes.length}`;
      return;   // streak tidak di-reset karena ambiguitas sesaat
    }
    if (stable === null || stable === "?") {
      renderLetter(stable === "?" ? "?" : null, 0, count);
      return updateStreak(null);
    }
    renderLetter(stable, avgConf, count);
    updateStreak(stable);
  }

  function majority(votes) {
    const tally = new Map();
    for (const v of votes) {
      const k = v.label;
      const e = tally.get(k) || { n: 0, conf: 0 };
      e.n += 1; e.conf += v.conf;
      tally.set(k, e);
    }
    let winner, best = { n: 0, conf: 0 };
    for (const [k, e] of tally) if (e.n > best.n) { winner = k; best = e; }
    return { winner, count: best.n, avgConf: best.n ? best.conf / best.n : 0 };
  }

  // ===================================================== hold-to-commit huruf
  function updateStreak(label) {
    const now = performance.now();
    if (label !== st.streak.label) {
      st.streak = { label, start: now, committed: false };
    }
    if (!label) { ui.holdFill.style.width = "0"; return; }

    const progress = Math.min(1, (now - st.streak.start) / (cfg.holdSec * 1000));
    ui.holdFill.style.width = st.streak.committed ? "0" : `${progress * 100}%`;
    if (progress >= 1 && !st.streak.committed) {
      st.streak.committed = true;     // huruf sama perlu dilepas dulu untuk diulang
      commitLetter(label);
    }
  }
  function resetStreak() { st.streak = { label: null, start: 0, committed: false }; ui.holdFill.style.width = "0"; }

  function commitLetter(ch) {
    st.sentence += ch;
    const conf = st.votes.filter((v) => v.label === ch).reduce((a, v) => a + v.conf, 0) /
                 Math.max(1, st.votes.filter((v) => v.label === ch).length);
    st.history.unshift({ ch, conf, time: new Date() });
    st.history = st.history.slice(0, 30);
    renderSentence();
    renderHistory();
  }

  // ================================================================= render
  function renderLetter(ch, conf, votes) {
    const changed = ch !== st.shown;
    st.shown = ch;
    ui.letter.textContent = ch ?? "–";
    ui.letter.classList.toggle("idle", !ch || ch === "?");
    if (changed && ch && ch !== "?") {
      ui.letter.classList.add("pop");
      setTimeout(() => ui.letter.classList.remove("pop"), 150);
    }
    ui.confFill.style.width = `${(conf || 0) * 100}%`;
    ui.confText.textContent = ch && ch !== "?" ? `confidence ${(conf * 100).toFixed(1)}%`
                            : ch === "?" ? "model ragu" : "confidence –";
    ui.voteText.textContent = `voting ${votes}/${st.votes.length}`;
  }

  function renderTopK(topk) {
    ui.topk.innerHTML = topk.map((t) => `
      <li><b>${t.label}</b>
        <div class="bar"><div style="width:${(t.confidence * 100).toFixed(1)}%"></div></div>
        <em>${(t.confidence * 100).toFixed(0)}%</em></li>`).join("");
  }

  function renderSentence() {
    ui.sentence.innerHTML = st.sentence
      ? `${escapeHtml(st.sentence).replace(/ /g, "&nbsp;")}<span class="cursor"></span>`
      : `<span class="placeholder">Tahan satu huruf ± ${cfg.holdSec.toFixed(1)} detik untuk menambahkannya…</span>`;
  }

  function renderHistory() {
    ui.history.innerHTML = st.history.map((h) =>
      `<span title="${h.time.toLocaleTimeString()}">${h.ch}<small>${(h.conf * 100).toFixed(0)}%</small></span>`).join("");
  }

  function drawLandmarks(hands) {
    octx.clearRect(0, 0, overlay.width, overlay.height);
    if (!$("chkLandmarks").checked) return;
    const W = overlay.width, H = overlay.height;
    hands.forEach((pts, i) => {
      octx.strokeStyle = HAND_COLORS[i % 2];
      octx.fillStyle = HAND_COLORS[i % 2];
      octx.lineWidth = 3;
      for (const [a, b] of HAND_CONNECTIONS) {
        octx.beginPath();
        octx.moveTo(pts[a][0] * W, pts[a][1] * H);
        octx.lineTo(pts[b][0] * W, pts[b][1] * H);
        octx.stroke();
      }
      for (const [x, y] of pts) {
        octx.beginPath();
        octx.arc(x * W, y * H, 4, 0, Math.PI * 2);
        octx.fill();
      }
    });
  }

  const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ================================================================ events
  ui.btnStart.onclick = startCamera;
  ui.btnStop.onclick = stopCamera;
  $("chkMirror").onchange = (e) => ui.stage.classList.toggle("mirror", e.target.checked);
  ui.stage.classList.toggle("mirror", $("chkMirror").checked);

  $("btnSpace").onclick = () => { st.sentence += " "; renderSentence(); };
  $("btnBack").onclick = () => { st.sentence = st.sentence.slice(0, -1); renderSentence(); };
  $("btnClear").onclick = () => { st.sentence = ""; st.history = []; renderSentence(); renderHistory(); };

  const apiInput = $("apiUrl");
  apiInput.value = cfg.apiUrl;
  apiInput.onchange = () => { cfg.apiUrl = apiInput.value.replace(/\/+$/, ""); checkHealth(); };

  bindRange("fpsRange", "fpsVal", (v) => (cfg.fps = +v), (v) => v);
  bindRange("winRange", "winVal", (v) => { cfg.window = +v; st.votes = st.votes.slice(-cfg.window); }, (v) => v);
  bindRange("thrRange", "thrVal", (v) => (cfg.threshold = +v), (v) => (+v).toFixed(2));
  bindRange("holdRange", "holdVal", (v) => { cfg.holdSec = +v; renderSentence(); }, (v) => (+v).toFixed(1));

  function bindRange(id, labelId, apply, fmt) {
    const el = $(id);
    const update = () => { apply(el.value); $(labelId).textContent = fmt(el.value); };
    el.oninput = update;
    update();
  }

  document.addEventListener("visibilitychange", () => {
    // hemat resource saat tab tidak terlihat
    if (document.hidden && st.running) clearTimeout(st.timer);
    else if (!document.hidden && st.running) scheduleNext(0);
  });

  checkHealth();
  setInterval(checkHealth, 10000);
  renderSentence();
})();

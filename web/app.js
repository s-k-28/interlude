const API_BASE = "http://localhost:8000";

/* ---------------- helpers ---------------- */

function escapeHtml(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmtDuration(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n) => (n < 10 ? "0" + n : String(n));
  if (h > 0) return h + ":" + pad(m) + ":" + pad(s);
  return m + ":" + pad(s);
}

function fmtMs(ms) {
  const v = Number(ms) || 0;
  if (v >= 1000) return (v / 1000).toFixed(1) + "s";
  return Math.round(v) + "ms";
}

function fmtNum(n) {
  const v = Number(n) || 0;
  return v.toLocaleString();
}

function fmtDate(iso) {
  if (iso === null || iso === undefined || iso === "") return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "";
  return d.toLocaleString();
}

function pct(ratio) {
  const v = Number(ratio) || 0;
  return (v * 100).toFixed(1) + "%";
}

function $(id) {
  return document.getElementById(id);
}

function show(el) {
  if (el) el.classList.remove("hidden");
}

function hide(el) {
  if (el) el.classList.add("hidden");
}

function pillHtml(status) {
  const s = String(status || "pending");
  const known = ["pending", "running", "partial", "complete", "failed"];
  const cls = known.indexOf(s) >= 0 ? s : "pending";
  return '<span class="pill ' + cls + '">' + escapeHtml(s) + "</span>";
}

/* ---------------- state ---------------- */

const state = { jobs: [], selectedId: null, detail: null, polling: false, healthTimer: null, jobsTimer: null };

/* ---------------- fetch helper ---------------- */

async function api(path, options) {
  let res;
  try {
    res = await fetch(API_BASE + path, options);
  } catch (e) {
    const err = new Error("network error");
    err.status = 0;
    throw err;
  }
  let body = null;
  const text = await res.text();
  if (text) {
    try {
      body = JSON.parse(text);
    } catch (e) {
      body = null;
    }
  }
  if (!res.ok) {
    const detail = body && body.detail ? body.detail : "request failed (" + res.status + ")";
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return body;
}

/* ---------------- health ---------------- */

function setHealth(cls, msg) {
  const dot = $("health-dot");
  const text = $("health-text");
  if (dot) {
    dot.classList.remove("ok", "warn", "error");
    dot.classList.add(cls);
  }
  if (text) text.textContent = msg;
}

async function pollHealth() {
  try {
    const h = await api("/health");
    const missing = (h && h.providers_missing) || [];
    if (missing.length > 0) {
      setHealth("warn", "degraded: " + missing.join(", ") + " missing");
    } else if (h && h.b2_ok) {
      setHealth("ok", "operational");
    } else {
      setHealth("warn", "degraded: storage unavailable");
    }
  } catch (e) {
    setHealth("error", "offline");
  } finally {
    clearTimeout(state.healthTimer);
    state.healthTimer = setTimeout(pollHealth, 30000);
  }
}

/* ---------------- job list ---------------- */

function renderJobs(jobs) {
  const tbody = $("job-tbody");
  if (!tbody) return;
  let html = "";
  for (const j of jobs) {
    const name = j.title ? j.title : j.job_id;
    const sel = j.job_id === state.selectedId ? " selected" : "";
    html +=
      '<tr class="job-row' + sel + '" data-job-id="' + escapeHtml(j.job_id) + '">' +
      "<td>" + escapeHtml(name) + "</td>" +
      "<td>" + pillHtml(j.status) + "</td>" +
      "<td>" + escapeHtml(String(j.segments_rendered != null ? j.segments_rendered : 0)) + "/" +
        escapeHtml(String(j.segments_described != null ? j.segments_described : 0)) + " described</td>" +
      '<td class="num">' + escapeHtml(fmtMs(j.duration_ms)) + "</td>" +
      "</tr>";
  }
  tbody.innerHTML = html;
  tbody.querySelectorAll("tr.job-row").forEach((row) => {
    row.addEventListener("click", () => {
      selectJob(row.getAttribute("data-job-id"));
    });
  });
}

function selectJob(jobId) {
  state.selectedId = jobId;
  const tbody = $("job-tbody");
  if (tbody) {
    tbody.querySelectorAll("tr.job-row").forEach((r) => {
      if (r.getAttribute("data-job-id") === jobId) r.classList.add("selected");
      else r.classList.remove("selected");
    });
  }
  loadDetail(jobId);
}

function scheduleJobsPoll(jobs) {
  clearTimeout(state.jobsTimer);
  const active = jobs.some((j) => j.status === "pending" || j.status === "running");
  state.polling = active;
  if (active) {
    state.jobsTimer = setTimeout(() => {
      loadJobs();
      if (state.selectedId) loadDetail(state.selectedId);
    }, 4000);
  }
}

async function loadJobs() {
  const table = $("job-table");
  const empty = $("job-empty");
  const error = $("job-error");
  const tbody = $("job-tbody");
  try {
    const data = await api("/jobs");
    const jobs = (data && data.jobs) || [];
    state.jobs = jobs;
    hide(error);
    if (jobs.length === 0) {
      show(empty);
      hide(table);
      if (tbody) tbody.innerHTML = "";
    } else {
      hide(empty);
      show(table);
      renderJobs(jobs);
    }
    scheduleJobsPoll(jobs);
  } catch (e) {
    // keep the last good data on screen; only surface the error strip
    show(error);
  }
}

/* ---------------- submit ---------------- */

function showSubmitError(msg) {
  const errEl = $("submit-error");
  if (!errEl) return;
  errEl.textContent = msg;
  show(errEl);
}

async function onSubmit(ev) {
  ev.preventDefault();
  const urlInput = $("input-url");
  const titleInput = $("input-title");
  const errEl = $("submit-error");
  const btn = $("btn-submit");
  const url = urlInput ? urlInput.value.trim() : "";
  const title = titleInput ? titleInput.value.trim() : "";

  if (!url) {
    showSubmitError("Enter a video URL.");
    return;
  }
  if (!/^https?:\/\//i.test(url)) {
    showSubmitError("URL must start with http:// or https://");
    return;
  }

  if (btn) btn.disabled = true;
  try {
    const body = { source_url: url };
    if (title) body.title = title;
    const res = await api("/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    if (urlInput) urlInput.value = "";
    if (titleInput) titleInput.value = "";
    if (errEl) {
      errEl.textContent = "";
      hide(errEl);
    }
    if (res && res.job_id) state.selectedId = res.job_id;
    await loadJobs();
    if (res && res.job_id) selectJob(res.job_id);
  } catch (e) {
    showSubmitError(e && e.message ? e.message : "Submission failed.");
  } finally {
    if (btn) btn.disabled = false;
  }
}

/* ---------------- detail ---------------- */

const STAGE_ORDER = ["transcribe", "detect_gaps", "describe", "synthesize", "manifest"];

function renderWaveform(segments) {
  const wf = $("waveform");
  const legend = $("waveform-legend");
  if (wf) wf.innerHTML = "";
  if (!segments || segments.length === 0) {
    if (legend) legend.textContent = "No describable gaps found";
    return;
  }
  const total = Math.max.apply(null, segments.map((s) => Number(s.end) || 0).concat([0])) || 1;
  if (wf) {
    for (const s of segments) {
      const div = document.createElement("div");
      div.className = "wf-segment" + (s.accepted === false ? " truncated" : "");
      const start = Number(s.start) || 0;
      const end = Number(s.end) || 0;
      div.style.left = (start / total) * 100 + "%";
      div.style.width = ((end - start) / total) * 100 + "%";
      const text = s.text ? String(s.text) : "";
      div.textContent = text;
      const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
      div.title = text + " \u2014 " + wordCount + " words / budget " + (s.word_budget != null ? s.word_budget : 0);
      wf.appendChild(div);
    }
  }
  if (legend) legend.textContent = segments.length + " description blocks over " + fmtDuration(total);
}

function renderSegments(segments) {
  const tbody = $("segments-tbody");
  if (!tbody) return;
  let html = "";
  for (const s of segments || []) {
    const audio =
      typeof s.audio_url === "string" && s.audio_url
        ? '<audio controls preload="none" src="' + escapeHtml(s.audio_url) + '"></audio>'
        : "\u2014";
    html +=
      "<tr>" +
      '<td class="num">' + escapeHtml(fmtDuration(s.start)) + "</td>" +
      '<td class="num">' + escapeHtml(fmtDuration(s.end)) + "</td>" +
      '<td class="num">' + escapeHtml((Number(s.duration) || 0).toFixed(2)) + "s</td>" +
      '<td class="num">' + escapeHtml(String(s.word_budget != null ? s.word_budget : 0)) + "</td>" +
      '<td class="num">' + escapeHtml(String(s.attempts != null ? s.attempts : 0)) + "</td>" +
      "<td>" + (s.accepted === false ? "truncated" : "within budget") + "</td>" +
      "<td>" + escapeHtml(s.text || "") + "</td>" +
      "<td>" + audio + "</td>" +
      "</tr>";
  }
  tbody.innerHTML = html;
}

function renderTokens(tokens) {
  const tbody = $("tokens-tbody");
  if (!tbody) return;
  const rows = [
    ["calls", fmtNum(tokens.calls)],
    ["prompt tokens (uncompressed)", fmtNum(tokens.prompt_tokens_uncompressed)],
    ["prompt tokens (compressed)", fmtNum(tokens.prompt_tokens_compressed)],
    ["completion tokens", fmtNum(tokens.completion_tokens)],
    ["tokens saved", fmtNum(tokens.tokens_saved)],
    ["reduction", tokens.calls === 0 ? "not measured" : pct(tokens.reduction_ratio)]
  ];
  let html = "";
  for (const r of rows) {
    html += "<tr><td>" + escapeHtml(r[0]) + '</td><td class="num">' + escapeHtml(r[1]) + "</td></tr>";
  }
  tbody.innerHTML = html;
}

function renderManifest(d) {
  const hashEl = $("manifest-hash");
  const full = d.manifest_hash ? String(d.manifest_hash) : "";
  if (hashEl) {
    hashEl.textContent = full ? full.slice(0, 16) + "\u2026" : "\u2014";
    hashEl.title = full;
  }
  const link = $("link-manifest");
  if (link) {
    link.href = API_BASE + "/jobs/" + encodeURIComponent(d.job_id) + "/manifest";
    link.target = "_blank";
    link.rel = "noopener";
  }
}

function renderDetail(d) {
  const segments = d.segments || [];
  const stages = d.stages || [];
  const tokens = d.tokens || {};

  const titleEl = $("detail-title");
  if (titleEl) titleEl.textContent = d.title ? d.title : d.job_id;
  const statusEl = $("detail-status");
  if (statusEl) statusEl.innerHTML = pillHtml(d.status);
  const idEl = $("detail-jobid");
  if (idEl) idEl.textContent = d.job_id || "";
  const createdEl = $("detail-created");
  if (createdEl) createdEl.textContent = fmtDate(d.created_at);

  const banner = $("detail-banner");
  if (banner) {
    banner.classList.remove("banner", "partial", "failed");
    if (d.status === "partial") {
      banner.classList.add("banner", "partial");
      banner.textContent =
        (d.segments_rendered != null ? d.segments_rendered : 0) +
        " of " +
        (d.segments_described != null ? d.segments_described : 0) +
        " descriptions rendered. The rest failed synthesis and were skipped \u2014 the remainder are still usable.";
      show(banner);
    } else if (d.status === "failed") {
      banner.classList.add("banner", "failed");
      banner.textContent = "This job failed. Use Resume to retry from the last completed stage.";
      show(banner);
    } else {
      banner.textContent = "";
      hide(banner);
    }
  }

  const strip = $("stage-strip");
  if (strip) {
    let html = "";
    for (const name of STAGE_ORDER) {
      const st = stages.find((s) => s && s.stage === name);
      let cls, label;
      if (!st) {
        cls = "chip pending";
        label = escapeHtml(name);
      } else if (st.ok) {
        cls = "chip done";
        label = "\u2713 " + escapeHtml(name);
      } else {
        cls = "chip failed";
        label = escapeHtml(name);
      }
      const t = st && st.detail ? ' title="' + escapeHtml(st.detail) + '"' : "";
      html += '<div class="' + cls + '"' + t + ">" + label;
      if (st) html += ' <span class="num">' + escapeHtml(fmtMs(st.duration_ms)) + "</span>";
      html += "</div>";
    }
    strip.innerHTML = html;
  }

  renderWaveform(segments);
  renderSegments(segments);
  renderTokens(tokens);
  renderManifest(d);

  const resume = $("btn-resume");
  if (resume) resume.disabled = !(d.status === "failed" || d.status === "partial");
}

async function loadDetail(jobId) {
  if (!jobId) return;
  const placeholder = $("detail-placeholder");
  const content = $("detail-content");
  try {
    const d = await api("/jobs/" + encodeURIComponent(jobId));
    state.detail = d;
    hide(placeholder);
    show(content);
    renderDetail(d);
  } catch (e) {
    if (e && e.status === 404) {
      state.detail = null;
      if (placeholder) {
        placeholder.textContent = "Job not found: " + jobId;
        show(placeholder);
      }
      hide(content);
    }
    // other errors: keep the last good detail rendered
  }
}

/* ---------------- actions ---------------- */

async function onCopyHash() {
  const btn = $("btn-copy-hash");
  const full = state.detail && state.detail.manifest_hash ? String(state.detail.manifest_hash) : "";
  if (!full || !btn) return;
  const original = btn.textContent;
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(full);
    }
    btn.textContent = "copied";
  } catch (e) {
    btn.textContent = "copy failed";
  } finally {
    setTimeout(() => {
      btn.textContent = original;
    }, 1200);
  }
}

async function onResume() {
  const btn = $("btn-resume");
  const id = state.selectedId;
  if (!id) return;
  if (btn) btn.disabled = true;
  try {
    await api("/jobs/" + encodeURIComponent(id) + "/resume", { method: "POST" });
    await loadDetail(id);
    await loadJobs();
  } catch (e) {
    const banner = $("detail-banner");
    if (banner) {
      banner.classList.add("banner", "failed");
      banner.textContent = "Resume failed: " + (e && e.message ? e.message : "unknown error");
      show(banner);
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

/* ---------------- init ---------------- */

function init() {
  const form = $("submit-form");
  if (form) form.addEventListener("submit", onSubmit);

  const urlInput = $("input-url");
  if (urlInput) {
    urlInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (form) form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
      }
    });
  }

  const retry = $("btn-retry-jobs");
  if (retry) {
    retry.addEventListener("click", () => {
      hide($("job-error"));
      loadJobs();
    });
  }

  const copy = $("btn-copy-hash");
  if (copy) copy.addEventListener("click", onCopyHash);

  const resume = $("btn-resume");
  if (resume) {
    resume.disabled = true;
    resume.addEventListener("click", onResume);
  }

  hide($("detail-content"));
  pollHealth();
  loadJobs();
}

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", init);
}

/* ---------------- selfcheck ---------------- */

if (typeof window !== "undefined" && window.__SELFCHECK__) {
  (function () {
    function assert(cond, msg) {
      if (!cond) throw new Error("selfcheck failed: " + msg);
    }
    const esc = escapeHtml("<a href='x'>&");
    assert(esc.indexOf("<") === -1 && esc.indexOf(">") === -1, "escapeHtml angle brackets");
    assert(esc.indexOf("'") === -1, "escapeHtml single quote");
    assert(esc.indexOf("&lt;") !== -1 && esc.indexOf("&gt;") !== -1, "escapeHtml lt/gt entities");
    assert(esc.indexOf("&#39;") !== -1, "escapeHtml apostrophe entity");
    assert(esc.indexOf("&amp;") !== -1, "escapeHtml ampersand entity");

    assert(fmtDuration(3661) === "1:01:01", "fmtDuration(3661) got " + fmtDuration(3661));
    assert(fmtDuration(125) === "2:05", "fmtDuration(125) got " + fmtDuration(125));
    assert(fmtDuration(7) === "0:07", "fmtDuration(7) got " + fmtDuration(7));

    assert(fmtMs(1234) === "1.2s", "fmtMs(1234) got " + fmtMs(1234));
    assert(fmtMs(340) === "340ms", "fmtMs(340) got " + fmtMs(340));

    assert(pct(0.7413) === "74.1%", "pct(0.7413) got " + pct(0.7413));

    const seg = { start: 5, end: 10 };
    const total = 20;
    const left = (seg.start / total) * 100 + "%";
    const width = ((seg.end - seg.start) / total) * 100 + "%";
    assert(left === "25%", "waveform left got " + left);
    assert(width === "25%", "waveform width got " + width);

    assert(fmtDate(null) === "" && fmtDate(undefined) === "", "fmtDate nullish");

    console.log("selfcheck OK");
  })();
}

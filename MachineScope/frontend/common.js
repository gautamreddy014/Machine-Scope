/* MachineScope — shared frontend helpers (API, formatting, toasts, chart theme). */
(function () {
  "use strict";

  const C = {
    primary: "#2563EB", lightBlue: "#DBEAFE", teal: "#0D9488", text: "#172033", text2: "#64748B",
    text3: "#94A3B8", border: "#E2E8F0", grid: "#EEF2F7", success: "#16A34A", warning: "#D97706",
    failure: "#DC2626", violet: "#7C3AED", normal: "#2563EB", failed: "#DC2626",
    types: { L: "#2563EB", M: "#0D9488", H: "#D97706" },
  };

  // ------------------------------------------------------------------ API
  async function api(path, opts = {}) {
    let res;
    try {
      res = await fetch(path, opts);
    } catch (e) {
      throw new Error("Cannot reach the MachineScope server. Is `python backend/app.py` running?");
    }
    let body = null;
    try { body = await res.json(); } catch (_) { /* non-JSON */ }
    if (!res.ok || (body && body.ok === false)) {
      const err = new Error((body && body.error) || `Request failed (${res.status})`);
      err.details = body && body.errors;
      err.status = res.status;
      throw err;
    }
    return body;
  }
  const qs = (obj) => {
    const p = new URLSearchParams();
    Object.entries(obj).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") p.set(k, v); });
    const s = p.toString();
    return s ? "?" + s : "";
  };

  // ------------------------------------------------------------------ formatting
  const nf = new Intl.NumberFormat("en-US");
  const fmt = {
    int: (v) => (v == null ? "—" : nf.format(Math.round(v))),
    pct: (v, d = 2) => (v == null || isNaN(v) ? "—" : (v * 100).toFixed(d) + "%"),
    num: (v, d = 2) => (v == null || isNaN(v) ? "—" : Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })),
    auto: (v) => {
      if (v == null || isNaN(v)) return "—";
      const a = Math.abs(v);
      const d = a >= 1000 ? 0 : a >= 100 ? 1 : a >= 1 ? 2 : 3;
      return fmt.num(v, d);
    },
    time: (iso) => {
      if (!iso) return "—";
      const d = new Date(iso);
      return d.toLocaleString(undefined, { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
    },
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ------------------------------------------------------------------ feature labels
  const FEATURES = {
    Air_Temperature: { label: "Air temperature", unit: "K" },
    Process_Temperature: { label: "Process temperature", unit: "K" },
    Rotational_Speed: { label: "Rotational speed", unit: "rpm" },
    Torque: { label: "Torque", unit: "Nm" },
    Tool_Wear: { label: "Tool wear", unit: "min" },
    Temp_Difference: { label: "Temperature difference", unit: "K" },
    Power: { label: "Mechanical power", unit: "W" },
    Wear_Torque: { label: "Tool wear × torque", unit: "min·Nm" },
    Machine_Failure: { label: "Machine failure (actual)", unit: "" },
    Type: { label: "Machine type", unit: "" },
  };
  const SENSORS = ["Air_Temperature", "Process_Temperature", "Rotational_Speed", "Torque", "Tool_Wear"];
  const DERIVED = ["Temp_Difference", "Power", "Wear_Torque"];
  const label = (f, withUnit = true) => {
    const m = FEATURES[f];
    if (!m) return f;
    return withUnit && m.unit ? `${m.label} (${m.unit})` : m.label;
  };

  // ------------------------------------------------------------------ toasts
  function toast(type, title, msg, ms = 4500) {
    let box = document.querySelector(".toasts");
    if (!box) { box = document.createElement("div"); box.className = "toasts"; box.setAttribute("role", "status"); box.setAttribute("aria-live", "polite"); document.body.appendChild(box); }
    const icon = { success: "circle-check", error: "circle-alert", warn: "triangle-alert", info: "info" }[type] || "info";
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.innerHTML = `${MSIcons.svg(icon)}<div><b>${esc(title)}</b>${msg ? esc(msg) : ""}</div>`;
    box.appendChild(el);
    setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 220); }, ms);
  }

  function overlay(show, title = "Working…", sub = "") {
    let o = document.getElementById("overlay");
    if (!o) {
      o = document.createElement("div"); o.id = "overlay"; o.className = "overlay";
      o.innerHTML = `<div class="overlay-card" role="alertdialog" aria-live="assertive"><span class="spinner"></span><div><h4></h4><p></p></div></div>`;
      document.body.appendChild(o);
    }
    o.querySelector("h4").textContent = title;
    o.querySelector("p").textContent = sub;
    o.classList.toggle("show", show);
  }

  function setLoading(el, on) {
    if (typeof el === "string") el = document.getElementById(el);
    if (el) el.classList.toggle("is-loading", !!on);
  }

  function emptyState(el, title, msg, icon = "database") {
    if (typeof el === "string") el = document.getElementById(el);
    if (!el) return;
    try { if (window.Plotly) Plotly.purge(el); } catch (_) { /* not a plot */ }
    el.innerHTML = `<div class="empty-state">${MSIcons.svg(icon)}<h4>${esc(title)}</h4><p>${esc(msg || "")}</p></div>`;
  }

  // ------------------------------------------------------------------ Plotly theme
  const baseLayout = (extra = {}) => {
    const l = {
      font: { family: "Inter, system-ui, sans-serif", size: 12, color: C.text2 },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      margin: { l: 56, r: 16, t: 12, b: 48 },
      hoverlabel: { bgcolor: "#fff", bordercolor: C.border, font: { family: "Inter, sans-serif", color: C.text, size: 12 } },
      legend: { orientation: "h", y: 1.1, x: 0, font: { size: 12, color: C.text2 }, bgcolor: "rgba(0,0,0,0)" },
      xaxis: { gridcolor: C.grid, zerolinecolor: C.border, linecolor: C.border, tickcolor: C.border, automargin: true, title: { font: { size: 12, color: C.text2 } } },
      yaxis: { gridcolor: C.grid, zerolinecolor: C.border, linecolor: C.border, tickcolor: C.border, automargin: true, title: { font: { size: 12, color: C.text2 } } },
      bargap: 0.08,
    };
    return deepMerge(l, extra);
  };
  function deepMerge(a, b) {
    const out = Array.isArray(a) ? [...a] : { ...a };
    Object.entries(b || {}).forEach(([k, v]) => {
      out[k] = v && typeof v === "object" && !Array.isArray(v) && a[k] && typeof a[k] === "object" ? deepMerge(a[k], v) : v;
    });
    return out;
  }
  const plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"] };
  function plot(el, data, layout, cfg) {
    if (typeof el === "string") el = document.getElementById(el);
    if (!el) return;
    if (!el.classList.contains("js-plotly-plot")) el.innerHTML = "";
    return Plotly.react(el, data, baseLayout(layout), { ...plotConfig, ...(cfg || {}) });
  }
  // keep charts sized after layout changes / tab switches
  let rt;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => document.querySelectorAll(".js-plotly-plot").forEach((p) => { try { Plotly.Plots.resize(p); } catch (_) {} }), 150);
  });

  // ------------------------------------------------------------------ header
  async function refreshStatus() {
    const dot = document.getElementById("statusDot");
    const txt = document.getElementById("statusText");
    const last = document.getElementById("lastAnalysis");
    try {
      const s = await api("/api/status");
      if (dot) dot.className = "dot ok";
      if (txt) txt.innerHTML = `<strong>${esc(s.dataset)}</strong> · ${fmt.int(s.rows)} rows`;
      if (last) last.textContent = fmt.time(s.last_analysis);
      return s;
    } catch (e) {
      if (dot) dot.className = "dot err";
      if (txt) txt.textContent = e.status ? "Dataset error" : "Server offline";
      if (last) last.textContent = "—";
      throw e;
    }
  }

  function initHeader() {
    const header = document.querySelector(".app-header");
    const btn = document.getElementById("menuBtn");
    if (btn && header) {
      btn.addEventListener("click", () => {
        const open = header.classList.toggle("open");
        btn.setAttribute("aria-expanded", String(open));
      });
    }
    MSIcons.render();
  }

  function download(path) {
    const a = document.createElement("a");
    a.href = path; a.download = ""; document.body.appendChild(a); a.click(); a.remove();
    toast("success", "Export started", "Your CSV download should begin shortly.", 2500);
  }

  function debounce(fn, ms = 250) {
    let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  }

  window.MS = { C, api, qs, fmt, esc, FEATURES, SENSORS, DERIVED, label, toast, overlay, setLoading, emptyState, plot, baseLayout, refreshStatus, initHeader, download, debounce };
})();

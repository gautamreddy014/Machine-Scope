/* MachineScope — Overview dashboard logic. */
(function () {
  "use strict";
  const { C, api, qs, fmt, esc, label, toast, overlay, setLoading, emptyState, plot } = MS;
  const $ = (id) => document.getElementById(id);

  let PERF = null; // model-performance payload
  const state = {
    mt: { page: 1, sort: "priority", order: "asc" },
    reviewed: loadReviewed(),
  };

  // ================================================================ boot
  document.addEventListener("DOMContentLoaded", async () => {
    MS.initHeader();
    wireActions();
    await loadAll();
  });

  async function loadAll() {
    try {
      await MS.refreshStatus();
    } catch (e) {
      showFatal(e);
      return;
    }
    await Promise.allSettled([loadOverview(), loadModel(), loadAnomalies(), loadMaintenance()]);
  }

  function showFatal(e) {
    const kv = $("datasetKv");
    kv.innerHTML = `<dt>Status</dt><dd style="color:var(--failure)">Unavailable</dd>`;
    $("datasetNote").textContent = e.message;
    toast("error", "Could not load data", e.message, 9000);
    ["cardDist", "cardModes", "cardTypes", "cardSensors", "cardModelMini", "cardFlagged"].forEach((id) => setLoading(id, false));
  }

  // ================================================================ overview
  async function loadOverview() {
    const cards = ["cardDist", "cardModes", "cardTypes", "cardSensors", "cardModelMini", "cardFlagged"];
    cards.forEach((c) => setLoading(c, true));
    try {
      const o = await api("/api/overview");
      renderDataset(o.dataset, o.model);
      renderKpis(o);
      renderDist(o.label_distribution);
      renderModes(o.failure_modes, o.failure_mode_note);
      renderTypes(o.machine_types);
      renderSensorTable(o.sensor_summary);
      renderFlagged(o.flagged);
    } catch (e) {
      toast("error", "Overview failed", e.message);
    } finally {
      cards.forEach((c) => setLoading(c, false));
    }
  }

  function renderDataset(d, m) {
    $("datasetKv").innerHTML = `
      <dt>Name</dt><dd>${esc(d.name)}</dd>
      <dt>Records · columns</dt><dd class="num">${fmt.int(d.rows)} · ${d.columns.length}</dd>
      <dt>Source</dt><dd><code>${esc(d.source)}</code></dd>
      <dt>Missing values</dt><dd class="num">${fmt.int(d.total_missing)}</dd>
      <dt>Model trained</dt><dd>${m.valid ? fmt.time(m.trained_at) : '<span style="color:var(--failure)">unavailable</span>'}</dd>`;
    $("datasetNote").textContent = (d.notes || []).join(" ");
  }

  function renderKpis(o) {
    $("kTotal").textContent = fmt.int(o.kpis.total_machines);
    $("kFails").textContent = fmt.int(o.kpis.actual_failures);
    $("kRate").textContent = fmt.pct(o.kpis.actual_failure_rate);
    const m = o.model;
    if (m.valid) {
      $("kAcc").textContent = fmt.pct(m.accuracy);
      $("kPrec").textContent = fmt.pct(m.precision);
      $("kRec").textContent = fmt.pct(m.recall);
      $("kF1").textContent = fmt.pct(m.f1);
      $("kTestTag").innerHTML = `${MSIcons.svg("target")}Held-out test set · n = ${fmt.int(m.n_test)}`;
      $("kAccCap").textContent = `majority-class baseline: ${fmt.pct(m.majority_baseline_accuracy)}`;
    } else {
      ["kAcc", "kPrec", "kRec", "kF1"].forEach((k) => ($(k).textContent = "n/a"));
      $("kAccCap").textContent = m.message || "evaluation unavailable";
    }
  }

  function renderDist(d) {
    const total = d.normal + d.failed;
    plot("chDist", [{
      type: "pie", hole: 0.62, values: [d.normal, d.failed], labels: ["No failure", "Failure"],
      marker: { colors: [C.primary, C.failure], line: { color: "#fff", width: 2 } },
      textinfo: "none", sort: false, direction: "clockwise",
      hovertemplate: "<b>%{label}</b><br>%{value:,} records (%{percent})<extra></extra>",
    }], {
      margin: { l: 10, r: 10, t: 6, b: 6 }, showlegend: true,
      legend: { orientation: "h", x: 0.5, xanchor: "center", y: -0.04 },
      annotations: [{ text: `<b style="font-size:20px;color:${C.text}">${fmt.pct(d.failed / total)}</b><br><span style="font-size:11px">actual failure rate</span>`, showarrow: false, x: 0.5, y: 0.5, xref: "paper", yref: "paper" }],
    });
  }

  function renderModes(modes, note) {
    if (!modes || !modes.length) { emptyState("chModes", "No failure-mode columns", "This dataset has no TWF/HDF/PWF/OSF/RNF flags."); return; }
    const sorted = [...modes].sort((a, b) => a.count - b.count);
    plot("chModes", [{
      type: "bar", orientation: "h", x: sorted.map((m) => m.count), y: sorted.map((m) => m.code),
      text: sorted.map((m) => m.count), textposition: "outside", cliponaxis: false,
      marker: { color: C.failure, opacity: 0.85 },
      customdata: sorted.map((m) => [m.label, m.with_label]),
      hovertemplate: "<b>%{customdata[0]}</b> (%{y})<br>%{x} flagged records<br>%{customdata[1]} also labelled as machine failure<extra></extra>",
    }], { margin: { l: 44, r: 30, t: 6, b: 30 }, xaxis: { title: { text: "Records" } }, yaxis: { gridcolor: "rgba(0,0,0,0)" } });
    $("modesNote").textContent = "TWF tool wear · HDF heat dissipation · PWF power · OSF overstrain · RNF random. " + (note || "");
  }

  function renderTypes(types) {
    plot("chTypes", [
      { type: "bar", name: "Records", x: types.map((t) => t.type), y: types.map((t) => t.machines), marker: { color: C.lightBlue, line: { color: C.primary, width: 1 } },
        customdata: types.map((t) => t.label), hovertemplate: "<b>Type %{x}</b> · %{customdata}<br>%{y:,} records<extra></extra>" },
      { type: "scatter", mode: "lines+markers+text", name: "Failure rate", yaxis: "y2", x: types.map((t) => t.type), y: types.map((t) => t.failure_rate),
        text: types.map((t) => fmt.pct(t.failure_rate, 1)), textposition: "top center", textfont: { color: C.failure, size: 11 },
        line: { color: C.failure, width: 2 }, marker: { size: 8, color: C.failure },
        customdata: types.map((t) => t.failures), hovertemplate: "<b>Type %{x}</b><br>Failure rate %{y:.2%}<br>%{customdata} actual failures<extra></extra>" },
    ], {
      margin: { l: 48, r: 48, t: 30, b: 30 },
      yaxis: { title: { text: "Records" }, range: [0, Math.max(...types.map((t) => t.machines)) * 1.12] },
      yaxis2: { overlaying: "y", side: "right", tickformat: ".1%", range: [0, Math.max(...types.map((t) => t.failure_rate)) * 1.35], showgrid: false, title: { text: "Failure rate" } },
      legend: { y: 1.18 },
    });
  }

  function renderSensorTable(rows) {
    const max = Math.max(...rows.map((r) => Math.abs(r.effect_size || 0)), 0.01);
    $("sensorTable").querySelector("tbody").innerHTML = rows.map((r) => {
      const d = r.effect_size;
      const w = d == null ? 0 : Math.abs(d) / max * 100;
      const col = d == null ? C.text3 : d > 0 ? C.failure : C.primary;
      return `<tr>
        <td><b>${esc(r.label)}</b> <span class="muted small">${esc(r.unit)}</span>${r.derived ? ' <span class="tag tag-neutral">derived</span>' : ""}</td>
        <td class="r">${fmt.auto(r.mean)}</td>
        <td class="r">${fmt.auto(r.mean_normal)}</td><td class="r" style="color:#B91C1C">${fmt.auto(r.mean_failed)}</td>
        <td><div style="display:flex;align-items:center;gap:8px;min-width:110px"><div class="bar" style="flex:1"><span style="width:${w}%;background:${col}"></span></div><span class="num" style="width:44px;text-align:right">${d == null ? "—" : (d > 0 ? "+" : "") + d.toFixed(2)}</span></div></td>
      </tr>`;
    }).join("");
  }

  function badgePriority(p) { return `<span class="badge b-${p.toLowerCase()}">${p}</span>`; }
  function badgeLabel(v) { return v === 1 ? '<span class="badge b-fail">Failed</span>' : '<span class="badge b-ok">Normal</span>'; }

  function renderFlagged(rows) {
    const tb = $("flaggedTable").querySelector("tbody");
    if (!rows.length) { tb.innerHTML = `<tr><td colspan="8"><div class="empty-state"><h4>No records flagged</h4></div></td></tr>`; return; }
    tb.innerHTML = rows.map((r) => `<tr>
      <td class="num">${r.UDI}</td><td class="mono">${esc(r.Product_ID)}</td><td><span class="badge b-type">${r.Type}</span></td>
      <td>${badgePriority(r.priority)}</td><td class="r">${r.model_score == null ? "—" : r.model_score.toFixed(2)}</td>
      <td class="r">${r.n_anomalous}</td><td>${badgeLabel(r.Machine_Failure)}</td><td class="wrap">${esc(r.basis)}</td></tr>`).join("");
  }

  // ================================================================ model
  async function loadModel() {
    ["cardMetrics", "cardCM", "cardROC", "cardPR", "cardCompare", "cardModelMini"].forEach((c) => setLoading(c, true));
    try {
      PERF = await api("/api/model-performance");
      if (!PERF.valid) {
        $("modelGrid").style.display = "none";
        $("modelInvalid").innerHTML = `<div class="notice err">${MSIcons.svg("circle-alert")}<div><b>A valid evaluation cannot be performed.</b> ${esc(PERF.message || "")}</div></div>`;
        emptyState("chMini", "Model unavailable", PERF.message, "circle-alert");
        return;
      }
      $("modelGrid").style.display = "";
      $("modelInvalid").innerHTML = "";
      renderMetrics(PERF);
      renderCM(PERF.metrics.confusion_matrix);
      renderCurves(PERF);
      renderCompare(PERF);
      renderMethod(PERF);
      renderMini(PERF);
      setPredictHints(PERF.train_ranges);
      if (!$("fAir").value) fillExample("normal");
    } catch (e) {
      toast("error", "Model evaluation failed", e.message);
    } finally {
      ["cardMetrics", "cardCM", "cardROC", "cardPR", "cardCompare", "cardModelMini"].forEach((c) => setLoading(c, false));
    }
  }

  function renderMetrics(p) {
    const m = p.metrics;
    $("metricsSub").textContent = `Random Forest · ${fmt.int(m.n_test)} test records (${m.n_test_failures} actual failures) · threshold ${m.threshold}`;
    const tiles = [
      ["Accuracy", m.accuracy, "correct / total"],
      ["Precision", m.precision, "TP / (TP + FP)"],
      ["Recall", m.recall, "TP / (TP + FN)"],
      ["F1-score", m.f1, "2PR / (P + R)"],
      ["ROC-AUC", m.roc_auc, "ranking quality"],
      ["Balanced accuracy", m.balanced_accuracy, "mean of class recalls"],
      ["Specificity", m.specificity, "TN / (TN + FP)"],
      ["Avg precision", m.average_precision, "area under PR curve"],
      ["CV F1 (train)", p.cv.f1.mean, `± ${fmt.pct(p.cv.f1.std)} over 5 folds`],
      ["Majority baseline", p.majority_baseline_accuracy, "accuracy if always 'no failure'"],
    ];
    $("metricTiles").innerHTML = tiles.map(([k, v, s]) => `<div class="mt"><div class="k">${k}</div><div class="v">${fmt.pct(v)}</div><div class="s">${s}</div></div>`).join("");
    $("accNote").querySelector("div").innerHTML =
      `<b>Read accuracy with care.</b> Only ${fmt.pct(m.n_test_failures / m.n_test)} of test records are failures, so a model that never predicts a failure would already score ${fmt.pct(p.majority_baseline_accuracy)} accuracy. Precision, recall and F1 describe failure detection far better.`;
  }

  function renderCM(cm) {
    const tot = cm.tn + cm.fp + cm.fn + cm.tp;
    const cell = (cls, v, name) => `<div class="cell ${cls}"><b>${fmt.int(v)}</b><span>${name} · ${fmt.pct(v / tot, 1)}</span></div>`;
    $("cmGrid").innerHTML = `
      <div></div><div class="h">Pred. normal</div><div class="h">Pred. failure</div>
      <div class="rh">Actual<br>normal</div>${cell("tn", cm.tn, "TN")}${cell("fp", cm.fp, "FP")}
      <div class="rh">Actual<br>failure</div>${cell("fn", cm.fn, "FN")}${cell("tp", cm.tp, "TP")}`;
  }

  function renderCurves(p) {
    const m = p.metrics, b = p.baseline_metrics;
    $("rocSub").textContent = `AUC: RF ${fmt.pct(m.roc_auc)} · LR ${fmt.pct(b.roc_auc)}`;
    $("prSub").textContent = `Average precision: RF ${fmt.pct(m.average_precision)} · LR ${fmt.pct(b.average_precision)}`;
    const ax = (t) => ({ title: { text: t }, range: [0, 1.02], tickformat: ".0%" });
    plot("chROC", [
      { x: [0, 1], y: [0, 1], mode: "lines", line: { dash: "dot", color: C.text3, width: 1 }, name: "Chance", hoverinfo: "skip" },
      { x: b.roc_curve.x, y: b.roc_curve.y, mode: "lines", name: "Logistic Reg.", line: { color: C.teal, width: 2 }, hovertemplate: "FPR %{x:.1%}<br>TPR %{y:.1%}<extra>LR</extra>" },
      { x: m.roc_curve.x, y: m.roc_curve.y, mode: "lines", name: "Random Forest", line: { color: C.primary, width: 2.5 }, hovertemplate: "FPR %{x:.1%}<br>TPR %{y:.1%}<extra>RF</extra>" },
    ], { xaxis: ax("False positive rate"), yaxis: ax("True positive rate"), legend: { x: 1, xanchor: "right", y: 0.05, orientation: "v" }, margin: { t: 6 } });
    const prev = m.n_test_failures / m.n_test;
    plot("chPR", [
      { x: [0, 1], y: [prev, prev], mode: "lines", line: { dash: "dot", color: C.text3, width: 1 }, name: "Chance", hoverinfo: "skip" },
      { x: b.pr_curve.x, y: b.pr_curve.y, mode: "lines", name: "Logistic Reg.", line: { color: C.teal, width: 2 }, hovertemplate: "Recall %{x:.1%}<br>Precision %{y:.1%}<extra>LR</extra>" },
      { x: m.pr_curve.x, y: m.pr_curve.y, mode: "lines", name: "Random Forest", line: { color: C.primary, width: 2.5 }, hovertemplate: "Recall %{x:.1%}<br>Precision %{y:.1%}<extra>RF</extra>" },
    ], { xaxis: ax("Recall"), yaxis: ax("Precision"), legend: { x: 0, y: 0.05, orientation: "v" }, margin: { t: 6 } });
  }

  function renderCompare(p) {
    const row = (name, m, primary) => `<tr><td><b>${name}</b>${primary ? ' <span class="tag tag-model">primary</span>' : ' <span class="tag tag-neutral">baseline</span>'}</td>
      ${["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"].map((k) => `<td class="r">${fmt.pct(m[k])}</td>`).join("")}</tr>`;
    $("compareTable").querySelector("tbody").innerHTML =
      row("Random Forest", p.metrics, true) + row("Logistic Regression", p.baseline_metrics, false) +
      `<tr><td><b>Always "no failure"</b> <span class="tag tag-neutral">reference</span></td><td class="r">${fmt.pct(p.majority_baseline_accuracy)}</td><td class="r">—</td><td class="r">0.00%</td><td class="r">0.00%</td><td class="r">50.00%</td><td class="r">—</td></tr>`;
    const cv = p.cv;
    $("cvNote").innerHTML = `5-fold stratified cross-validation on the training portion (Random Forest): accuracy ${fmt.pct(cv.accuracy.mean)} ± ${fmt.pct(cv.accuracy.std)}, recall ${fmt.pct(cv.recall.mean)} ± ${fmt.pct(cv.recall.std)}, F1 ${fmt.pct(cv.f1.mean)} ± ${fmt.pct(cv.f1.std)}. The test-set numbers are consistent with these, which suggests they are not a lucky split.`;
  }

  function renderMethod(p) {
    $("methodList").innerHTML = `
      <li><b>Target:</b> <code>Machine failure</code> (0/1). <b>Inputs:</b> ${p.inputs.map((f) => label(f, false)).join(", ")}.</li>
      <li><b>Excluded to prevent leakage:</b> ${Object.keys(p.excluded_columns).map(esc).join(", ")} — the failure-mode flags are parts of the label itself.</li>
      <li><b>Split:</b> stratified ${100 - p.test_size * 100}/${p.test_size * 100} train/test, <code>random_state=${p.random_state}</code> → ${fmt.int(p.train_size)} train (${p.train_failures} failures), ${fmt.int(p.test_size_n)} test (${p.test_failures} failures).</li>
      <li><b>Pipeline:</b> engineered ΔT, power and wear×torque → median imputation → one-hot <code>Type</code> (+ scaling for LR) → classifier. Fitted on training rows only.</li>
      <li><b>Random Forest:</b> 300 trees, <code>min_samples_leaf=2</code>, <code>class_weight="balanced"</code>; decision threshold ${p.threshold}.</li>
      <li><b>Test set</b> used once for the final metrics; CV on the training portion checks stability.</li>
      <li>Trained ${fmt.time(p.trained_at)} in ${p.training_seconds}s with scikit-learn ${esc(p.sklearn_version)}.</li>`;
  }

  function renderMini(p) {
    const keys = [["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"], ["roc_auc", "ROC-AUC"]];
    plot("chMini", [
      { type: "bar", orientation: "h", name: "Logistic Reg.", y: keys.map((k) => k[1]), x: keys.map((k) => p.baseline_metrics[k[0]]), marker: { color: "#99D5CF" }, hovertemplate: "%{y}: %{x:.2%}<extra>Logistic Regression</extra>" },
      { type: "bar", orientation: "h", name: "Random Forest", y: keys.map((k) => k[1]), x: keys.map((k) => p.metrics[k[0]]), marker: { color: C.primary }, hovertemplate: "%{y}: %{x:.2%}<extra>Random Forest</extra>" },
    ], { barmode: "group", bargap: 0.25, margin: { l: 70, r: 10, t: 30, b: 30 }, xaxis: { range: [0, 1], tickformat: ".0%" }, yaxis: { autorange: "reversed", gridcolor: "rgba(0,0,0,0)" }, legend: { y: 1.16 } });
    const cm = p.metrics.confusion_matrix;
    $("miniFoot").innerHTML = `On ${fmt.int(p.metrics.n_test)} unseen records the forest caught <b>${cm.tp}</b> of ${cm.tp + cm.fn} real failures and raised <b>${cm.fp}</b> false alarms.`;
  }

  // ================================================================ prediction
  const EXAMPLES = {
    normal: { Type: "M", Air_Temperature: 300.0, Process_Temperature: 310.0, Rotational_Speed: 1500, Torque: 40.0, Tool_Wear: 100 },
    strain: { Type: "L", Air_Temperature: 300.5, Process_Temperature: 310.5, Rotational_Speed: 1340, Torque: 62.0, Tool_Wear: 215 },
    heat: { Type: "L", Air_Temperature: 302.8, Process_Temperature: 310.9, Rotational_Speed: 1300, Torque: 55.0, Tool_Wear: 60 },
  };
  const FIELDS = { Type: "fType", Air_Temperature: "fAir", Process_Temperature: "fProc", Rotational_Speed: "fSpeed", Torque: "fTorque", Tool_Wear: "fWear" };
  const LIMITS = { Air_Temperature: [250, 350], Process_Temperature: [250, 400], Rotational_Speed: [0, 5000], Torque: [0, 150], Tool_Wear: [0, 500] };

  function setPredictHints(ranges) {
    document.querySelectorAll("[data-range]").forEach((el) => {
      const r = ranges[el.dataset.range];
      if (r) el.textContent = `Training range ${fmt.auto(r.min)} – ${fmt.auto(r.max)}`;
    });
  }
  function fillExample(name) {
    const ex = EXAMPLES[name];
    Object.entries(FIELDS).forEach(([k, id]) => { $(id).value = ex[k]; });
    clearErrors();
  }
  function clearErrors() {
    document.querySelectorAll("#predictForm .input").forEach((i) => { i.classList.remove("invalid"); i.removeAttribute("aria-invalid"); });
    document.querySelectorAll("#predictForm .err-msg").forEach((e) => (e.textContent = ""));
  }
  function validateForm() {
    clearErrors();
    let ok = true;
    const body = { Type: $("fType").value };
    Object.entries(LIMITS).forEach(([k, [lo, hi]]) => {
      const inp = $(FIELDS[k]);
      const err = inp.parentElement.querySelector(".err-msg");
      const raw = inp.value.trim();
      let msg = "";
      if (raw === "") msg = "Required";
      else if (isNaN(Number(raw))) msg = "Must be a number";
      else if (Number(raw) < lo || Number(raw) > hi) msg = `Must be between ${lo} and ${hi}`;
      if (msg) { ok = false; inp.classList.add("invalid"); inp.setAttribute("aria-invalid", "true"); err.textContent = msg; }
      else body[k] = Number(raw);
    });
    return ok ? body : null;
  }

  async function submitPrediction(ev) {
    ev.preventDefault();
    const body = validateForm();
    if (!body) { toast("warn", "Check the inputs", "Some readings are missing or out of range."); return; }
    const btn = $("predictBtn");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>Predicting…`;
    setLoading("predictResult", true);
    try {
      const r = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      renderPrediction(r);
    } catch (e) {
      const list = e.details ? `<ul class="method-list">${e.details.map((d) => `<li>${esc(d)}</li>`).join("")}</ul>` : "";
      $("predictResult").innerHTML = `<div class="notice err">${MSIcons.svg("circle-alert")}<div><b>${esc(e.message)}</b>${list}</div></div>`;
      toast("error", "Prediction failed", e.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = `${MSIcons.svg("zap")}Predict`;
      setLoading("predictResult", false);
    }
  }

  function renderPrediction(r) {
    const fail = r.predicted_class === 1;
    const s = r.failure_score;
    const warn = r.warnings.length ? `<div class="notice warn">${MSIcons.svg("triangle-alert")}<div>${r.warnings.map(esc).join("<br>")}</div></div>` : "";
    $("predictResult").innerHTML = `
      <div class="verdict ${fail ? "fail" : "ok"}">
        <div class="verdict-ico">${MSIcons.svg(fail ? "triangle-alert" : "circle-check")}</div>
        <div><span class="tag tag-model">${MSIcons.svg("brain")}Model prediction</span><h3 style="margin-top:6px">${esc(r.predicted_label)}</h3>
        <p class="small muted">Predicted class: <b>${r.predicted_class}</b> (${fail ? "failure" : "no failure"}) at threshold ${r.threshold}</p></div>
      </div>
      ${s == null ? "" : `<div>
        <div style="display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:8px"><span class="muted">Failure score (forest vote share)</span><b class="num">${fmt.pct(s, 1)}</b></div>
        <div class="gauge-bar"><span class="thr" style="left:${r.threshold * 100}%"></span><span class="mark" style="left:${Math.max(0, Math.min(100, s * 100))}%"></span></div>
        <div class="gauge-scale"><span>0%</span><span>threshold ${fmt.pct(r.threshold, 0)}</span><span>100%</span></div>
      </div>`}
      <div class="stat-grid">
        <div class="stat"><div class="k">ΔT (proc − air)</div><div class="v">${fmt.num(r.derived.Temp_Difference, 1)} K</div></div>
        <div class="stat"><div class="k">Power</div><div class="v">${fmt.int(r.derived.Power)} W</div></div>
        <div class="stat"><div class="k">LR baseline</div><div class="v">${r.baseline_prediction === 1 ? "Failure" : "No failure"}</div></div>
      </div>
      ${warn}
      <p class="small muted">${esc(r.probability_note)} This is an estimate from historical patterns — <b>not confirmation of a physical machine failure.</b></p>`;
  }

  // ================================================================ anomalies
  async function loadAnomalies() {
    const params = { feature: $("anFeature").value, k: $("anK").value, type: $("anType").value, failure: $("anFail").value };
    setLoading("anChartWrap", true);
    try {
      const r = await api("/api/anomalies" + qs(params));
      renderAnomalies(r);
    } catch (e) {
      toast("error", "Anomaly detection failed", e.message);
    } finally {
      setLoading("anChartWrap", false);
    }
  }

  function renderAnomalies(r) {
    const single = r.feature !== "all";
    const pf = single ? r.per_feature[0] : null;
    const stat = (k, v, sub, cls = "") => `<div class="kpi ${cls}"><div class="kpi-label">${k}</div><div class="kpi-value" style="font-size:22px">${v}</div><div class="kpi-caption">${sub}</div></div>`;
    $("anStats").innerHTML =
      stat(`<span class="kpi-ico ico-violet">${MSIcons.svg("scan-search")}</span>Flagged readings`, fmt.int(r.count), `of ${fmt.int(r.n)} records in view`) +
      stat(`<span class="kpi-ico ico-amber">${MSIcons.svg("activity")}</span>Flagged share`, fmt.pct(r.pct), `IQR multiplier k = ${r.k}`) +
      (single
        ? stat(`<span class="kpi-ico ico-blue">${MSIcons.svg("arrow-down")}</span>Lower fence`, fmt.auto(pf.lower), `${label(pf.feature)}`) +
          stat(`<span class="kpi-ico ico-blue">${MSIcons.svg("arrow-up")}</span>Upper fence`, fmt.auto(pf.upper), `Q1 ${fmt.auto(pf.q1)} · Q3 ${fmt.auto(pf.q3)}`)
        : stat(`<span class="kpi-ico ico-blue">${MSIcons.svg("layers")}</span>Sensors checked`, r.per_feature.length, "each with its own fences") +
          stat(`<span class="kpi-ico ico-red">${MSIcons.svg("flag")}</span>Also actual failures`, fmt.int(r.crosstab.anomaly_and_failure), "anomalous AND labelled failed"));

    const x = r.crosstab;
    $("anCross").querySelector("div").innerHTML = r.n
      ? `<b>Anomaly ≠ failure.</b> Of ${fmt.int(r.count)} flagged records, <b>${fmt.int(x.anomaly_and_failure)}</b> carry an actual failure label and <b>${fmt.int(x.anomaly_no_failure)}</b> do not. Meanwhile <b>${fmt.int(x.failure_no_anomaly)}</b> actual failures show no anomalous reading at all.`
      : "No records match the current filters.";

    if (!r.n) { emptyState("chAnom", "No records", "Adjust the filters to see anomalies."); }
    else if (single) {
      const p = r.plot;
      plot("chAnom", [
        { type: "scattergl", mode: "markers", name: "Within fences (sample)", x: p.normal_udi, y: p.normal_val, marker: { size: 4, color: "rgba(37,99,235,.28)" }, hovertemplate: "UDI %{x}<br>%{y}<extra>within fences</extra>" },
        { type: "scattergl", mode: "markers", name: "Statistical anomaly", x: p.anom_udi, y: p.anom_val, marker: { size: 6, color: C.violet, line: { color: "#fff", width: 0.5 } }, hovertemplate: "UDI %{x}<br>%{y}<extra>anomaly</extra>" },
      ], {
        xaxis: { title: { text: "Record (UDI)" } }, yaxis: { title: { text: label(pf.feature) } },
        shapes: [pf.lower, pf.upper].map((v) => ({ type: "line", xref: "paper", x0: 0, x1: 1, y0: v, y1: v, line: { color: C.violet, dash: "dash", width: 1.5 } })),
        annotations: [{ xref: "paper", x: 1, y: pf.upper, text: "upper fence", showarrow: false, yshift: 9, xanchor: "right", font: { size: 11, color: C.violet } },
          { xref: "paper", x: 1, y: pf.lower, text: "lower fence", showarrow: false, yshift: -9, xanchor: "right", font: { size: 11, color: C.violet } }],
      });
    } else {
      const pfs = [...r.per_feature].sort((a, b) => a.count - b.count);
      plot("chAnom", [{ type: "bar", orientation: "h", x: pfs.map((f) => f.count), y: pfs.map((f) => f.label), marker: { color: C.violet, opacity: 0.85 },
        text: pfs.map((f) => `${f.count} (${fmt.pct(f.pct, 1)})`), textposition: "outside", cliponaxis: false,
        hovertemplate: "%{y}: %{x} flagged<extra></extra>" }], { margin: { l: 150, r: 70 }, xaxis: { title: { text: "Flagged readings" } }, yaxis: { gridcolor: "rgba(0,0,0,0)" } });
    }

    const cols = single ? ["UDI", "Product_ID", "Type", r.feature, "direction", "Machine_Failure"] : ["UDI", "Product_ID", "Type", "n_anomalous", "anomalous_sensors", "Machine_Failure"];
    const heads = { UDI: "UDI", Product_ID: "Product", Type: "Type", direction: "Side", n_anomalous: "#", anomalous_sensors: "Anomalous sensors", Machine_Failure: "Actual label" };
    $("anTable").querySelector("thead").innerHTML = `<tr>${cols.map((c) => `<th${["n_anomalous", r.feature].includes(c) ? ' class="r"' : ""}>${heads[c] || label(c, false)}</th>`).join("")}</tr>`;
    const tb = $("anTable").querySelector("tbody");
    if (!r.records.length) tb.innerHTML = `<tr><td colspan="${cols.length}"><div class="empty-state"><h4>No anomalies found</h4><p>No readings fall outside the fences for this selection.</p></div></td></tr>`;
    else tb.innerHTML = r.records.slice(0, 200).map((rec) => `<tr>${cols.map((c) => {
      const v = rec[c];
      if (c === "Machine_Failure") return `<td>${badgeLabel(v)}</td>`;
      if (c === "Type") return `<td><span class="badge b-type">${v}</span></td>`;
      if (c === "direction") return `<td><span class="badge b-anom">${v}</span></td>`;
      if (c === "anomalous_sensors") return `<td class="wrap">${esc(v)}</td>`;
      if (c === "Product_ID") return `<td class="mono">${esc(v)}</td>`;
      if (c === "UDI") return `<td class="num">${v}</td>`;
      return `<td class="r">${typeof v === "number" ? fmt.auto(v) : esc(v)}</td>`;
    }).join("")}</tr>`).join("");
    $("anTableNote").textContent = r.count > 200 ? `Showing 200 of ${fmt.int(r.count)} flagged records — export the CSV for the full list.` : `${fmt.int(r.count)} flagged record(s).`;
  }

  // ================================================================ maintenance
  function loadReviewed() {
    try { return new Set(JSON.parse(sessionStorage.getItem("ms_reviewed") || "[]")); } catch (_) { return new Set(); }
  }
  function saveReviewed() {
    try { sessionStorage.setItem("ms_reviewed", JSON.stringify([...state.reviewed])); } catch (_) { /* storage blocked: keep in memory */ }
    $("mtReviewedCount").textContent = `${state.reviewed.size} reviewed this session`;
  }
  function mtParams(extra = {}) {
    return {
      priority: $("mtPriority").value, type: $("mtType").value, failure: $("mtFail").value, search: $("mtSearch").value.trim(),
      sort: state.mt.sort, order: state.mt.order,
      exclude: $("mtHideReviewed").checked ? [...state.reviewed].join(",") : "",
      ...extra,
    };
  }

  async function loadMaintenance() {
    setLoading("mtCard", true);
    try {
      const r = await api("/api/maintenance" + qs(mtParams({ page: state.mt.page, page_size: $("mtPageSize").value })));
      renderMaintenance(r);
    } catch (e) {
      toast("error", "Maintenance planner failed", e.message);
    } finally {
      setLoading("mtCard", false);
    }
  }

  function renderMaintenance(r) {
    const total = r.counts.High + r.counts.Medium + r.counts.Low;
    $("mtRules").innerHTML = r.rules.map((x) => `<div class="kpi">
        <div style="display:flex;justify-content:space-between;align-items:center">${badgePriority(x.priority)}<b class="num">${fmt.int(r.counts[x.priority])}</b></div>
        <div class="bar" style="margin:6px 0"><span style="width:${(r.counts[x.priority] / Math.max(total, 1)) * 100}%;background:${{ High: C.failure, Medium: C.warning, Low: C.teal }[x.priority]}"></span></div>
        <p class="small muted">${esc(x.rule)}</p></div>`).join("");
    $("mtNote").textContent = r.note;
    const pages = Math.max(1, Math.ceil(r.total / r.page_size));
    if (state.mt.page > pages) { state.mt.page = pages; return loadMaintenance(); }
    $("mtPage").textContent = `${r.page} / ${pages}`;
    $("mtPrev").disabled = r.page <= 1;
    $("mtNext").disabled = r.page >= pages;
    $("mtCount").textContent = r.total ? `Showing ${fmt.int((r.page - 1) * r.page_size + 1)}–${fmt.int(Math.min(r.page * r.page_size, r.total))} of ${fmt.int(r.total)} records` : "No records";
    document.querySelectorAll("#mtTable th.sortable").forEach((th) => {
      const on = th.dataset.sort === state.mt.sort;
      th.innerHTML = th.textContent.trim() + (on ? MSIcons.svg(state.mt.order === "asc" ? "arrow-up" : "arrow-down") : "");
      th.setAttribute("aria-sort", on ? (state.mt.order === "asc" ? "ascending" : "descending") : "none");
    });
    const tb = $("mtTable").querySelector("tbody");
    if (!r.records.length) {
      tb.innerHTML = `<tr><td colspan="12"><div class="empty-state">${MSIcons.svg("search")}<h4>No matching records</h4><p>Try clearing the search or filters.</p></div></td></tr>`;
      return;
    }
    tb.innerHTML = r.records.map((x) => {
      const rev = state.reviewed.has(String(x.UDI));
      return `<tr class="${rev ? "reviewed" : ""}">
        <td><input type="checkbox" class="review-box" data-udi="${x.UDI}" ${rev ? "checked" : ""} aria-label="Mark record ${x.UDI} as reviewed"></td>
        <td class="num">${x.UDI}</td><td class="mono">${esc(x.Product_ID)}</td><td><span class="badge b-type">${x.Type}</span></td>
        <td>${badgePriority(x.priority)}</td>
        <td class="r">${x.model_score == null ? "—" : x.model_score.toFixed(3)}</td>
        <td class="r">${x.n_anomalous}</td>
        <td class="r">${fmt.num(x.Torque, 1)}</td><td class="r">${fmt.int(x.Rotational_Speed)}</td><td class="r">${fmt.int(x.Tool_Wear)}</td>
        <td>${badgeLabel(x.Machine_Failure)}</td>
        <td class="wrap">${esc(x.basis)}</td></tr>`;
    }).join("");
  }

  // ================================================================ actions
  function wireActions() {
    $("runBtn").addEventListener("click", runAnalysis);
    $("printBtn").addEventListener("click", () => {
      $("printStamp").textContent = `Generated ${new Date().toLocaleString()} · dataset ${$("statusText").textContent}`;
      window.print();
    });
    $("uploadInput").addEventListener("change", uploadCsv);
    $("resetDataBtn").addEventListener("click", resetDataset);

    // jump nav
    document.querySelectorAll("#jumpNav button").forEach((b) => b.addEventListener("click", () => {
      document.getElementById(b.dataset.target).scrollIntoView({ behavior: "smooth" });
    }));
    const io = new IntersectionObserver((ents) => {
      ents.forEach((e) => {
        if (e.isIntersecting) document.querySelectorAll("#jumpNav button").forEach((b) => b.classList.toggle("on", b.dataset.target === e.target.id));
      });
    }, { rootMargin: "-40% 0px -55% 0px" });
    ["kpis", "model", "predict", "anomalies", "maintenance"].forEach((id) => io.observe($(id)));

    // prediction
    $("predictForm").addEventListener("submit", submitPrediction);
    $("resetPredict").addEventListener("click", () => {
      $("predictForm").reset();
      clearErrors();
      $("predictResult").innerHTML = `<div class="result-empty"><div>${MSIcons.svg("scan-search")}<h4 style="color:var(--text)">Inputs cleared</h4><p class="small">Enter new readings or pick an example.</p></div></div>`;
      toast("info", "Prediction inputs reset", "", 2000);
    });
    document.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", () => fillExample(b.dataset.example)));
    document.querySelectorAll("#predictForm .input").forEach((i) => i.addEventListener("input", () => { i.classList.remove("invalid"); i.parentElement.querySelector(".err-msg").textContent = ""; }));

    // anomalies
    ["anFeature", "anK", "anType", "anFail"].forEach((id) => $(id).addEventListener("change", loadAnomalies));
    $("anExport").addEventListener("click", () => MS.download("/api/export/anomalies" + qs({ feature: $("anFeature").value, k: $("anK").value, type: $("anType").value, failure: $("anFail").value })));

    // maintenance
    const reload = () => { state.mt.page = 1; loadMaintenance(); };
    ["mtPriority", "mtType", "mtFail", "mtPageSize", "mtHideReviewed"].forEach((id) => $(id).addEventListener("change", reload));
    $("mtSearch").addEventListener("input", MS.debounce(reload, 300));
    $("mtPrev").addEventListener("click", () => { state.mt.page--; loadMaintenance(); });
    $("mtNext").addEventListener("click", () => { state.mt.page++; loadMaintenance(); });
    document.querySelectorAll("#mtTable th.sortable").forEach((th) => th.addEventListener("click", () => {
      const k = th.dataset.sort;
      if (state.mt.sort === k) state.mt.order = state.mt.order === "asc" ? "desc" : "asc";
      else { state.mt.sort = k; state.mt.order = ["model_score", "n_anomalous", "Torque", "Tool_Wear", "Rotational_Speed"].includes(k) ? "desc" : "asc"; }
      reload();
    }));
    $("mtTable").addEventListener("change", (e) => {
      if (!e.target.classList.contains("review-box")) return;
      const udi = e.target.dataset.udi;
      if (e.target.checked) state.reviewed.add(udi); else state.reviewed.delete(udi);
      e.target.closest("tr").classList.toggle("reviewed", e.target.checked);
      saveReviewed();
      if ($("mtHideReviewed").checked) loadMaintenance();
    });
    $("mtClearReviewed").addEventListener("click", () => { state.reviewed.clear(); saveReviewed(); loadMaintenance(); toast("info", "Reviewed marks cleared", "", 2000); });
    $("mtExport").addEventListener("click", () => MS.download("/api/export/maintenance" + qs({ ...mtParams(), reviewed: [...state.reviewed].join(",") })));
    saveReviewed();
  }

  async function runAnalysis() {
    const retrain = $("retrainChk").checked;
    const btn = $("runBtn");
    btn.disabled = true;
    overlay(true, retrain ? "Retraining models…" : "Running analysis…", retrain ? "Fitting Random Forest, baseline and cross-validation. This can take 30–60 seconds." : "Reloading the dataset and recomputing every chart.");
    try {
      const r = await api("/api/run-analysis", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ retrain }) });
      await loadAll();
      toast("success", "Analysis complete", retrain ? `Model retrained at ${fmt.time(r.model_trained_at)}.` : "All charts and metrics were recomputed.");
    } catch (e) {
      toast("error", "Analysis failed", e.message, 8000);
    } finally {
      overlay(false);
      btn.disabled = false;
    }
  }

  async function uploadCsv(ev) {
    const file = ev.target.files[0];
    ev.target.value = "";
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) { toast("error", "Unsupported file", "Please choose a .csv file."); return; }
    const fd = new FormData();
    fd.append("file", file);
    overlay(true, `Loading ${file.name}…`, "Validating the schema and training a model on this dataset.");
    try {
      const r = await api("/api/dataset/upload", { method: "POST", body: fd });
      state.reviewed.clear(); saveReviewed();
      await loadAll();
      toast(r.model_valid ? "success" : "warn", "Dataset loaded", `${r.dataset.name}: ${fmt.int(r.dataset.rows)} rows.${r.model_valid ? "" : " " + (r.message || "")}`, 6000);
    } catch (e) {
      toast("error", "Upload rejected", e.message, 9000);
    } finally {
      overlay(false);
    }
  }

  async function resetDataset() {
    overlay(true, "Restoring bundled dataset…", "Loading data/ai4i2020.csv and the saved model.");
    try {
      await api("/api/dataset/reset", { method: "POST" });
      state.reviewed.clear(); saveReviewed();
      await loadAll();
      toast("success", "Bundled dataset restored", "AI4I 2020 is active again.");
    } catch (e) {
      toast("error", "Reset failed", e.message);
    } finally {
      overlay(false);
    }
  }
})();

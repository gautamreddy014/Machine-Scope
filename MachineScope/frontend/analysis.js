/* MachineScope — Comprehensive Data Analysis page (univariate · bivariate · multivariate). */
(function () {
  "use strict";
  const { C, api, qs, fmt, esc, label, toast, setLoading, emptyState, plot, SENSORS, DERIVED } = MS;
  const $ = (id) => document.getElementById(id);
  const NUMERIC = [...SENSORS, ...DERIVED];
  const S = { uHistMode: "all", bColor: "failure", method: "pearson", lastU: null, lastB: null, lastM: null };

  const filters = () => ({ type: $("gType").value, failure: $("gFail").value });

  document.addEventListener("DOMContentLoaded", async () => {
    MS.initHeader();
    buildSelectors();
    wire();
    try {
      await MS.refreshStatus();
    } catch (e) {
      toast("error", "Could not load data", e.message, 9000);
      document.querySelectorAll(".loading-block").forEach((el) => setLoading(el, false));
      return;
    }
    await refreshAll();
  });

  function opt(v, t, sel) { return `<option value="${v}"${sel ? " selected" : ""}>${esc(t)}</option>`; }

  function buildSelectors() {
    $("uFeature").innerHTML =
      `<optgroup label="Sensors">${SENSORS.map((f) => opt(f, label(f), f === "Torque")).join("")}</optgroup>` +
      `<optgroup label="Derived (from sensors)">${DERIVED.map((f) => opt(f, label(f))).join("")}</optgroup>` +
      `<optgroup label="Categorical">${opt("Type", "Machine type (L / M / H)")}</optgroup>`;
    const bOpts = (sel) =>
      `<optgroup label="Sensors">${SENSORS.map((f) => opt(f, label(f), f === sel)).join("")}</optgroup>` +
      `<optgroup label="Derived">${DERIVED.map((f) => opt(f, label(f), f === sel)).join("")}</optgroup>` +
      `<optgroup label="Categorical">${opt("Machine_Failure", "Machine failure (actual label)", sel === "Machine_Failure")}${opt("Type", "Machine type", sel === "Type")}</optgroup>`;
    $("bX").innerHTML = bOpts("Air_Temperature");
    $("bY").innerHTML = bOpts("Process_Temperature");
    $("bGroupFeature").innerHTML = NUMERIC.map((f) => opt(f, label(f), f === "Torque")).join("");
    $("mFeatures").innerHTML = NUMERIC.map((f) => `<label class="chip-check"><input type="checkbox" value="${f}" ${SENSORS.includes(f) ? "checked" : ""}><span>${esc(label(f, false))}</span></label>`).join("");
    const mOpts = (sel) => NUMERIC.map((f) => opt(f, label(f), f === sel)).join("");
    $("mX").innerHTML = mOpts("Rotational_Speed");
    $("mY").innerHTML = mOpts("Torque");
    $("mC").innerHTML = mOpts("Tool_Wear");
  }

  async function refreshAll() {
    const f = filters();
    try {
      const info = await api("/api/univariate" + qs({ feature: "Type", ...f }));
      $("gCount").innerHTML = `<b class="num">${fmt.int(info.n)}</b>&nbsp;records in view`;
    } catch (_) { /* shown elsewhere */ }
    await Promise.allSettled([loadUni(), loadBi(), loadGroup(), loadComparison(), loadMulti()]);
  }

  // ================================================================== UNIVARIATE
  async function loadUni() {
    const cards = ["uStatsCard", "uHistCard", "uDensCard", "uBoxCard", "uOutCard", "uMissCard", "uCatCard"];
    cards.forEach((c) => setLoading(c, true));
    try {
      const r = await api("/api/univariate" + qs({ feature: $("uFeature").value, bins: $("uBins").value, ...filters() }));
      S.lastU = r;
      renderUni(r);
    } catch (e) {
      toast("error", "Univariate analysis failed", e.message);
    } finally {
      cards.forEach((c) => setLoading(c, false));
    }
  }

  function renderUni(r) {
    const cat = r.categorical;
    $("uNumeric").style.display = cat ? "none" : "";
    $("uCategorical").style.display = cat ? "" : "none";
    if (cat) {
      renderTypeBar("chUTypeBig", r.type_frequency, false);
      renderLabelBar("chULabelBig", r.label_distribution, false);
      return;
    }
    renderMissing(r.missing);
    renderTypeBar("chUType", r.type_frequency, true);
    renderLabelBar("chULabel", r.label_distribution, true);
    if (r.empty || !r.n) {
      ["chUHist", "chUDens", "chUBox"].forEach((id) => emptyState(id, "No records", "No records match the current filters."));
      $("uStats").innerHTML = ""; $("uOutStats").innerHTML = "";
      return;
    }
    const s = r.stats, u = r.meta.unit;
    $("uStatsSub").textContent = `${label(r.feature)} · n = ${fmt.int(s.count)}`;
    const tiles = [["Mean", s.mean], ["Median", s.median], ["Std dev", s.std], ["Minimum", s.min], ["Q1 (25%)", s.q1], ["Q3 (75%)", s.q3], ["Maximum", s.max], ["IQR", s.iqr]];
    $("uStats").innerHTML = tiles.map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${fmt.auto(v)}</div></div>`).join("");
    const sk = s.skew;
    const shape = Math.abs(sk) < 0.5 ? "approximately symmetric" : sk > 0 ? (sk > 1 ? "strongly right-skewed (long upper tail)" : "moderately right-skewed") : (sk < -1 ? "strongly left-skewed" : "moderately left-skewed");
    $("uShape").textContent = `The distribution is ${shape}. Mean ${s.mean > s.median ? "exceeds" : "is below"} the median by ${fmt.auto(Math.abs(s.mean - s.median))} ${u}. Skewness ${fmt.num(s.skew, 3)}, excess kurtosis ${fmt.num(s.kurtosis, 3)}. Missing values: ${s.missing}.`;

    // histogram
    const xt = { title: { text: label(r.feature) } };
    if (S.uHistMode === "all") {
      const e = r.histogram.edges;
      const mids = e.slice(0, -1).map((v, i) => (v + e[i + 1]) / 2);
      plot("chUHist", [{ type: "bar", x: mids, y: r.histogram.counts, width: e[1] - e[0], marker: { color: C.primary, opacity: 0.85, line: { color: "#fff", width: 0.5 } },
        customdata: e.slice(0, -1).map((v, i) => [v, e[i + 1]]), hovertemplate: "%{customdata[0]:.2f} – %{customdata[1]:.2f}<br><b>%{y:,}</b> records<extra></extra>" }],
      { xaxis: xt, yaxis: { title: { text: "Records" } }, bargap: 0.02,
        shapes: [{ type: "line", x0: s.mean, x1: s.mean, yref: "paper", y0: 0, y1: 1, line: { color: C.failure, dash: "dash", width: 1.5 } },
          { type: "line", x0: s.median, x1: s.median, yref: "paper", y0: 0, y1: 1, line: { color: C.teal, dash: "dot", width: 1.5 } }],
        annotations: [{ x: s.mean, yref: "paper", y: 1.02, text: "mean", showarrow: false, font: { size: 10, color: C.failure }, xanchor: "left" },
          { x: s.median, yref: "paper", y: 1.02, text: "median", showarrow: false, font: { size: 10, color: C.teal }, xanchor: "right" }] });
    } else {
      const nb = Number($("uBins").value);
      const e = r.histogram.edges;
      const xbins = { start: e[0], end: e[e.length - 1], size: (e[e.length - 1] - e[0]) / nb };
      plot("chUHist", [
        { type: "histogram", name: "Normal", x: r.values_by_label.normal, xbins, histnorm: "percent", marker: { color: C.primary, opacity: 0.55 }, hovertemplate: "%{x}<br>%{y:.2f}% of normal<extra></extra>" },
        { type: "histogram", name: "Failed (actual)", x: r.values_by_label.failed, xbins, histnorm: "percent", marker: { color: C.failure, opacity: 0.6 }, hovertemplate: "%{x}<br>%{y:.2f}% of failed<extra></extra>" },
      ], { barmode: "overlay", xaxis: xt, yaxis: { title: { text: "% of group" } } });
    }

    // density
    const d = r.density;
    const traces = [{ x: d.x, y: d.all, name: "All records", mode: "lines", line: { color: C.text3, width: 1.5, dash: "dot" }, hovertemplate: "%{x:.2f}<extra>all</extra>" }];
    if (r.label_distribution.normal) traces.push({ x: d.x, y: d.normal, name: "Normal", mode: "lines", fill: "tozeroy", fillcolor: "rgba(37,99,235,.10)", line: { color: C.primary, width: 2 }, hovertemplate: "%{x:.2f}<extra>normal</extra>" });
    if (r.label_distribution.failed > 1) traces.push({ x: d.x, y: d.failed, name: "Failed (actual)", mode: "lines", fill: "tozeroy", fillcolor: "rgba(220,38,38,.10)", line: { color: C.failure, width: 2 }, hovertemplate: "%{x:.2f}<extra>failed</extra>" });
    plot("chUDens", traces, { xaxis: xt, yaxis: { title: { text: "Density" }, showticklabels: false } });

    // box
    const boxes = [];
    if (r.values_by_label.normal.length) boxes.push({ type: "box", name: "Normal", y: r.values_by_label.normal, marker: { color: C.primary, size: 3, opacity: 0.5 }, line: { width: 1.5 }, boxpoints: "outliers", boxmean: true });
    if (r.values_by_label.failed.length) boxes.push({ type: "box", name: "Failed (actual)", y: r.values_by_label.failed, marker: { color: C.failure, size: 3, opacity: 0.6 }, line: { width: 1.5 }, boxpoints: "outliers", boxmean: true });
    boxes.unshift({ type: "box", name: "All", y: r.values, marker: { color: C.teal, size: 3, opacity: 0.5 }, line: { width: 1.5 }, boxpoints: "outliers", boxmean: true });
    plot("chUBox", boxes, { yaxis: { title: { text: label(r.feature) } }, showlegend: false });

    // outliers
    const o = r.outliers;
    $("uOutSub").textContent = `IQR rule: outside [${fmt.auto(o.lower)}, ${fmt.auto(o.upper)}] ${u}`;
    $("uOutStats").innerHTML = [["Outliers", fmt.int(o.count)], ["Share", fmt.pct(o.pct)], ["Below fence", fmt.int(o.low)], ["Above fence", fmt.int(o.high)], ["Also failed", fmt.int(o.actual_failures_among)]]
      .map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
    $("uOutTable").querySelector("thead").innerHTML = `<tr><th>UDI</th><th>Product</th><th>Type</th><th class="r">${esc(r.meta.label)}</th><th>Actual label</th></tr>`;
    $("uOutTable").querySelector("tbody").innerHTML = o.records.length
      ? o.records.map((x) => `<tr><td class="num">${x.UDI}</td><td class="mono">${esc(x.Product_ID)}</td><td><span class="badge b-type">${x.Type}</span></td><td class="r">${fmt.auto(x[r.feature])}</td><td>${x.Machine_Failure ? '<span class="badge b-fail">Failed</span>' : '<span class="badge b-ok">Normal</span>'}</td></tr>`).join("")
      : `<tr><td colspan="5"><div class="empty-state"><h4>No outliers</h4><p>All readings fall within the IQR fences.</p></div></td></tr>`;
  }

  function renderMissing(rows) {
    const total = rows.reduce((a, r) => a + r.missing, 0);
    $("uMissing").innerHTML = (total === 0 ? `<div class="notice neutral" style="margin-bottom:6px">${MSIcons.svg("circle-check")}<div>No missing values in any column.</div></div>` : "") +
      rows.slice(0, 7).map((r) => `<div class="barrow" style="grid-template-columns:minmax(0,1fr) 60px 46px"><span class="lbl small">${esc(label(r.column, false))}</span><div class="bar"><span style="width:${r.pct * 100}%;background:${C.warning}"></span></div><span class="val small">${r.missing}</span></div>`).join("");
  }

  function renderTypeBar(id, freq, compact) {
    if (!freq.length) { emptyState(id, "No records"); return; }
    plot(id, [{ type: "bar", x: freq.map((f) => f.type), y: freq.map((f) => f.count), marker: { color: freq.map((f) => C.types[f.type] || C.primary) },
      text: freq.map((f) => fmt.pct(f.pct, 1)), textposition: "outside", cliponaxis: false,
      customdata: freq.map((f) => f.label), hovertemplate: "<b>Type %{x}</b> · %{customdata}<br>%{y:,} records<extra></extra>" }],
    { margin: compact ? { l: 36, r: 6, t: 14, b: 22 } : { t: 20 }, yaxis: { title: { text: compact ? "" : "Records" } }, xaxis: { title: { text: compact ? "" : "Machine type" } } });
  }
  function renderLabelBar(id, d, compact) {
    plot(id, [{ type: "bar", orientation: "h", y: ["Normal", "Failed"], x: [d.normal, d.failed], marker: { color: [C.primary, C.failure] },
      text: [d.normal, d.failed].map((v) => fmt.int(v)), textposition: "outside", cliponaxis: false, hovertemplate: "%{y}: %{x:,}<extra></extra>" }],
    { margin: compact ? { l: 56, r: 40, t: 4, b: 20 } : { l: 80, r: 50 }, xaxis: { title: { text: compact ? "" : "Records" } }, yaxis: { gridcolor: "rgba(0,0,0,0)" } });
  }

  // ================================================================== BIVARIATE
  async function loadBi() {
    ["bMainCard", "bStatsCard", "bRateCard"].forEach((c) => setLoading(c, true));
    try {
      const r = await api("/api/bivariate" + qs({ x: $("bX").value, y: $("bY").value, ...filters() }));
      S.lastB = r;
      renderBi(r);
    } catch (e) {
      toast("error", "Bivariate analysis failed", e.message);
    } finally {
      ["bMainCard", "bStatsCard", "bRateCard"].forEach((c) => setLoading(c, false));
    }
  }

  function statTiles(list) { return list.map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join(""); }

  function renderBi(r) {
    if (r.empty) { ["chBMain", "chBRate"].forEach((id) => emptyState(id, "No records", "No records match the current filters.")); $("bStats").innerHTML = ""; return; }
    const xl = label(r.x), yl = label(r.y);
    $("bColor").style.display = r.kind === "numeric-numeric" ? "" : "none";
    $("bStats").className = "stat-grid c2";

    if (r.kind === "numeric-numeric") {
      $("bMainTitle").innerHTML = `${MSIcons.svg("chart-scatter")}Scatter plot`;
      $("bMainSub").textContent = `${yl} vs ${xl}${r.sampled ? " (sampled; all failures kept)" : ""}`;
      const p = r.points;
      const groups = S.bColor === "failure"
        ? [{ name: "Normal", f: (i) => p.failure[i] === 0, color: "rgba(37,99,235,.35)", size: 4 }, { name: "Failed (actual)", f: (i) => p.failure[i] === 1, color: C.failure, size: 6 }]
        : ["L", "M", "H"].map((t) => ({ name: `Type ${t}`, f: (i) => p.type[i] === t, color: C.types[t], size: 4, opacity: 0.5 }));
      const traces = groups.map((g) => {
        const idx = p.x.map((_, i) => i).filter(g.f);
        return { type: "scattergl", mode: "markers", name: g.name, x: idx.map((i) => p.x[i]), y: idx.map((i) => p.y[i]), customdata: idx.map((i) => p.udi[i]),
          marker: { color: g.color, size: g.size, opacity: g.opacity || 1, line: { width: 0 } }, hovertemplate: `UDI %{customdata}<br>${esc(xl)}: %{x}<br>${esc(yl)}: %{y}<extra>${g.name}</extra>` };
      }).filter((t) => t.x.length);
      if (r.trend) {
        const xs = [Math.min(...p.x), Math.max(...p.x)];
        traces.push({ type: "scatter", mode: "lines", name: "Linear fit", x: xs, y: xs.map((v) => r.trend.slope * v + r.trend.intercept), line: { color: C.text, width: 1.5, dash: "dash" }, hoverinfo: "skip" });
      }
      plot("chBMain", traces, { xaxis: { title: { text: xl } }, yaxis: { title: { text: yl } } });
      $("bStatsSub").textContent = `${fmt.int(r.n)} records`;
      $("bStats").innerHTML = statTiles([["Pearson r", fmt.num(r.pearson, 3)], ["Spearman ρ", fmt.num(r.spearman, 3)], ["r²", fmt.num(r.pearson * r.pearson, 3)], ["Slope", r.trend ? fmt.auto(r.trend.slope) : "—"]]);

      $("bInterp").innerHTML = `Linear association: <b>${esc(r.strength)}</b>. Pearson measures linear association; Spearman measures monotonic (rank) association and is robust to outliers.`;
    } else if (r.kind === "numeric-categorical") {
      $("bMainTitle").innerHTML = `${MSIcons.svg("box")}Grouped distribution`;
      const nl = label(r.numeric), isFail = r.category === "Machine_Failure";
      $("bMainSub").textContent = `${nl} by ${isFail ? "actual failure status" : "machine type"}`;
      const colors = { Normal: C.primary, Failed: C.failure, "Type L": C.types.L, "Type M": C.types.M, "Type H": C.types.H };
      plot("chBMain", Object.entries(r.groups).map(([g, v]) => ({ type: "violin", name: g, y: v.values, box: { visible: true }, meanline: { visible: true }, points: false,
        line: { color: colors[g] || C.primary, width: 1.5 }, fillcolor: (colors[g] || C.primary) + "22", hoverinfo: "y+name" })),
      { yaxis: { title: { text: nl } }, showlegend: false, violinmode: "group" });
      $("bStatsSub").textContent = `${fmt.int(r.n)} records`;
      const gs = Object.entries(r.groups);
      $("bStats").innerHTML = statTiles([
        ...gs.map(([g, v]) => [`${g} mean`, fmt.auto(v.stats.mean)]),
        ...(isFail ? [["Effect size d", fmt.num(r.effect_size, 2)], ["Point-biserial r", fmt.num(r.point_biserial, 3)]] : []),
      ]);
      $("bInterp").innerHTML = isFail
        ? `Effect size d is the difference in means between actually-failed and normal records, in pooled standard deviations (|d| ≈ 0.2 small, 0.5 medium, 0.8 large).`
        : `Compare the medians and spreads of each machine type. Overlapping boxes mean the types behave similarly on this sensor.`;
    } else {
      $("bMainTitle").innerHTML = `${MSIcons.svg("chart-column")}Failure rate by machine type`;
      $("bMainSub").textContent = "Cross-tabulation of machine type and actual failure label";
      const types = Object.keys(r.crosstab).sort((a, b) => "LMH".indexOf(a) - "LMH".indexOf(b));
      const n0 = types.map((t) => r.crosstab[t]["0"] || 0), n1 = types.map((t) => r.crosstab[t]["1"] || 0);
      plot("chBMain", [
        { type: "bar", name: "Normal", x: types.map((t) => "Type " + t), y: n0, marker: { color: C.lightBlue, line: { color: C.primary, width: 1 } }, hovertemplate: "%{x}: %{y:,} normal<extra></extra>" },
        { type: "bar", name: "Failed (actual)", x: types.map((t) => "Type " + t), y: n1, marker: { color: C.failure }, hovertemplate: "%{x}: %{y:,} failed<extra></extra>" },
      ], { barmode: "stack", yaxis: { title: { text: "Records" } } });
      $("bStatsSub").textContent = "Contingency table";
      $("bStats").innerHTML = statTiles(types.map((t, i) => [`Type ${t} failure rate`, fmt.pct(n1[i] / Math.max(n0[i] + n1[i], 1))]));
      $("bInterp").textContent = "Differences in failure rate between types are descriptive; type is also correlated with other operating conditions.";
    }

    if (r.failure_rate_by_bin && r.failure_rate_by_bin.length) {
      const b = r.failure_rate_by_bin;
      $("bRateSub").textContent = `Records grouped into deciles of ${label(r.x, false)}`;
      plot("chBRate", [
        { type: "bar", name: "Records", x: b.map((d) => d.bin), y: b.map((d) => d.count), yaxis: "y2", marker: { color: "rgba(148,163,184,.25)" }, hovertemplate: "%{x}<br>%{y:,} records<extra></extra>" },
        { type: "scatter", mode: "lines+markers", name: "Actual failure rate", x: b.map((d) => d.bin), y: b.map((d) => d.rate), line: { color: C.failure, width: 2.5 }, marker: { size: 7 },
          customdata: b.map((d) => d.failures), hovertemplate: "%{x}<br>Failure rate %{y:.2%}<br>%{customdata} failures<extra></extra>" },
      ], { xaxis: { title: { text: label(r.x) }, tickangle: -30, type: "category" }, yaxis: { title: { text: "Failure rate" }, tickformat: ".0%", rangemode: "tozero" },
        yaxis2: { overlaying: "y", side: "right", showgrid: false, showticklabels: false }, margin: { b: 80 } });
    } else {
      emptyState("chBRate", "Not available", r.x === "Machine_Failure" || r.x === "Type" ? "Choose a numeric X feature to see failure rate across its range." : "Both actual classes are needed — relax the failure-status filter.", "trending-up");
    }
  }

  async function loadGroup() {
    setLoading("bGroupCard", true);
    const g = $("bGroupFeature").value;
    try {
      const r = await api("/api/bivariate" + qs({ x: "Machine_Failure", y: g, group: g, type: $("gType").value, failure: $("gFail").value }));
      if (r.empty || !r.grouped) { emptyState("chBGroup", "No records"); return; }
      $("bGroupSub").textContent = `${label(g)} by machine type, split by actual label`;
      const d = r.grouped.data;
      const tr = [["Normal", C.primary], ["Failed", C.failure]].filter(([k]) => d[k].values.length).map(([k, c]) => ({
        type: "box", name: k === "Failed" ? "Failed (actual)" : "Normal", x: d[k].type.map((t) => "Type " + t), y: d[k].values,
        marker: { color: c, size: 3, opacity: 0.5 }, line: { width: 1.5 }, boxpoints: "outliers",
      }));
      plot("chBGroup", tr, { boxmode: "group", yaxis: { title: { text: label(g) } }, xaxis: { categoryorder: "array", categoryarray: ["Type L", "Type M", "Type H"] } });
    } catch (e) {
      toast("error", "Grouped box plot failed", e.message);
    } finally {
      setLoading("bGroupCard", false);
    }
  }

  async function loadComparison() {
    setLoading("bCompCard", true);
    try {
      const r = await api("/api/failure-comparison" + qs({ type: $("gType").value }));
      $("bCompSub").textContent = `${fmt.int(r.n)} records, ${fmt.int(r.failures)} actual failures${$("gType").value !== "all" ? " · type " + $("gType").value : ""} (failure-status filter does not apply here)`;
      $("bCompTable").querySelector("tbody").innerHTML = r.rows.map((x) => `<tr>
        <td><b>${esc(x.label)}</b> <span class="muted small">${esc(x.unit)}</span></td>
        <td class="r">${fmt.auto(x.normal_mean)}</td><td class="r" style="color:#B91C1C">${fmt.auto(x.failed_mean)}</td>
        <td class="r">${fmt.auto(x.normal_median)}</td><td class="r">${fmt.auto(x.failed_median)}</td>
        <td class="r">${x.diff_pct == null ? "—" : (x.diff_pct > 0 ? "+" : "") + x.diff_pct.toFixed(1) + "%"}</td>
        <td class="r"><b>${x.effect_size == null ? "—" : (x.effect_size > 0 ? "+" : "") + x.effect_size.toFixed(2)}</b></td></tr>`).join("");
      const rows = r.rows.filter((x) => x.effect_size != null).sort((a, b) => Math.abs(a.effect_size) - Math.abs(b.effect_size));
      if (!rows.length) { emptyState("chBComp", "Not enough failures", "At least two actual failures are needed."); return; }
      plot("chBComp", [{ type: "bar", orientation: "h", y: rows.map((x) => x.label), x: rows.map((x) => x.effect_size),
        marker: { color: rows.map((x) => (x.effect_size > 0 ? C.failure : C.primary)) }, hovertemplate: "%{y}<br>d = %{x:.2f}<extra></extra>" }],
      { margin: { l: 150, t: 6 }, xaxis: { title: { text: "Effect size d (failed − normal)" }, zeroline: true }, yaxis: { gridcolor: "rgba(0,0,0,0)" } });
    } catch (e) {
      toast("error", "Comparison failed", e.message);
    } finally {
      setLoading("bCompCard", false);
    }
  }

  // ================================================================== MULTIVARIATE
  function selectedFeatures() { return [...document.querySelectorAll("#mFeatures input:checked")].map((i) => i.value); }

  async function loadMulti() {
    const feats = selectedFeatures();
    const cards = ["mHeatCard", "mPcaCard", "mSplomCard", "mMultiCard", "mProfCard", "mImpCard"];
    if (feats.length < 2) { toast("warn", "Select at least two features", "Multivariate analysis needs two or more variables."); return; }
    cards.forEach((c) => setLoading(c, true));
    try {
      const r = await api("/api/multivariate" + qs({ features: feats.join(","), method: S.method, ...filters() }));
      S.lastM = r;
      renderMulti(r);
    } catch (e) {
      toast("error", "Multivariate analysis failed", e.message);
    } finally {
      cards.forEach((c) => setLoading(c, false));
    }
  }

  function renderMulti(r) {
    if (r.empty) { ["chMHeat", "chMPca", "chMSplom", "chMMulti"].forEach((id) => emptyState(id, "Not enough records", "Relax the filters to run multivariate analysis.")); return; }
    // heatmap
    const labs = r.correlation.labels.map((l) => label(l, false).replace(" (actual)", ""));
    const z = r.correlation.matrix;
    $("mHeatSub").textContent = `${r.method === "pearson" ? "Pearson (linear)" : "Spearman (rank)"} correlation · ${fmt.int(r.n)} records · includes actual failure label`;
    plot("chMHeat", [{ type: "heatmap", z, x: labs, y: labs, zmin: -1, zmax: 1, xgap: 2, ygap: 2,
      colorscale: [[0, "#1D4ED8"], [0.25, "#93C5FD"], [0.5, "#F8FAFC"], [0.75, "#FCA5A5"], [1, "#B91C1C"]],
      text: z.map((row) => row.map((v) => v.toFixed(2))), texttemplate: "%{text}", textfont: { size: 11 },
      hovertemplate: "%{y} × %{x}<br>r = %{z:.3f}<extra></extra>", colorbar: { thickness: 10, outlinewidth: 0, len: 0.8 } }],
    { margin: { l: 150, r: 10, t: 10, b: 120 }, xaxis: { tickangle: -40, gridcolor: "rgba(0,0,0,0)" }, yaxis: { autorange: "reversed", gridcolor: "rgba(0,0,0,0)" } });

    // PCA
    if (r.pca) {
      const ev = r.pca.explained_variance;
      $("mPcaSub").textContent = `Standardised features · PC1 ${fmt.pct(ev[0], 1)} + PC2 ${fmt.pct(ev[1], 1)} = ${fmt.pct(ev[0] + ev[1], 1)} of variance`;
      const p = r.pca.points;
      const idxN = [], idxF = [];
      p.failure.forEach((f, i) => (f ? idxF : idxN).push(i));
      plot("chMPca", [
        { type: "scattergl", mode: "markers", name: "Normal", x: idxN.map((i) => p.pc1[i]), y: idxN.map((i) => p.pc2[i]), marker: { size: 4, color: "rgba(37,99,235,.3)" }, hovertemplate: "PC1 %{x:.2f}<br>PC2 %{y:.2f}<extra>normal</extra>" },
        { type: "scattergl", mode: "markers", name: "Failed (actual)", x: idxF.map((i) => p.pc1[i]), y: idxF.map((i) => p.pc2[i]), marker: { size: 6, color: C.failure }, hovertemplate: "PC1 %{x:.2f}<br>PC2 %{y:.2f}<extra>failed</extra>" },
      ], { xaxis: { title: { text: `PC1 (${fmt.pct(ev[0], 1)})` } }, yaxis: { title: { text: `PC2 (${fmt.pct(ev[1], 1)})` } } });
      const L = r.pca.loadings;
      const top = (pc) => [...L].sort((a, b) => Math.abs(b[pc]) - Math.abs(a[pc])).slice(0, 2).map((l) => `${label(l.feature, false)} (${l[pc] > 0 ? "+" : ""}${l[pc].toFixed(2)})`).join(" and ");
      $("mPcaExplain").innerHTML = `Each feature is standardised (mean 0, SD 1) so units don't dominate. Principal components are uncorrelated weighted combinations of the features, ordered by the variance they capture. <b>PC1</b> is driven mostly by ${top("PC1")}; <b>PC2</b> by ${top("PC2")}. Labels are <i>not</i> used to compute PCA — colouring by failure only shows where failures fall.`;
      const pcs = Object.keys(L[0]).filter((k) => k.startsWith("PC"));
      $("mLoadings").querySelector("thead").innerHTML = `<tr><th>Feature</th>${pcs.map((k, i) => `<th class="r">${k} <span class="muted">${fmt.pct(ev[i], 0)}</span></th>`).join("")}</tr>`;
      $("mLoadings").querySelector("tbody").innerHTML = L.map((l) => `<tr><td>${esc(label(l.feature, false))}</td>${pcs.map((k) => `<td class="r" style="color:${Math.abs(l[k]) >= 0.4 ? C.text : C.text3};font-weight:${Math.abs(l[k]) >= 0.4 ? 650 : 400}">${l[k].toFixed(3)}</td>`).join("")}</tr>`).join("");
    }

    // SPLOM
    const sp = r.splom, feats = r.features.slice(0, 5);
    $("mSplomSub").textContent = `Scatter-plot matrix of ${feats.length}${r.features.length > 5 ? ` (first 5 of ${r.features.length})` : ""} selected features (${fmt.int(sp.failure.length)} sampled records, all failures kept), coloured by actual label`;
    const dims = feats.map((f) => ({ label: label(f, false), values: sp[f] }));
    const failIdx = sp.failure.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
    const normIdx = sp.failure.map((v, i) => (v ? -1 : i)).filter((i) => i >= 0);
    const pick = (idx) => feats.map((f) => ({ label: label(f, false), values: idx.map((i) => sp[f][i]) }));
    plot("chMSplom", [
      { type: "splom", name: "Normal", dimensions: pick(normIdx), marker: { size: 2.5, color: "rgba(37,99,235,.28)" }, diagonal: { visible: false }, showupperhalf: false },
      { type: "splom", name: "Failed (actual)", dimensions: pick(failIdx), marker: { size: 4, color: C.failure, opacity: 0.85 }, diagonal: { visible: false }, showupperhalf: false },
    ], Object.assign({ margin: { l: 70, r: 10, t: 30, b: 60 }, hovermode: "closest", dragmode: "select" },
      ...feats.map((_, i) => ({ [`xaxis${i ? i + 1 : ""}`]: { gridcolor: C.grid, linecolor: C.border, tickfont: { size: 10 } }, [`yaxis${i ? i + 1 : ""}`]: { gridcolor: C.grid, linecolor: C.border, tickfont: { size: 10 } } }))));
    if (!dims.length) emptyState("chMSplom", "Select features");

    renderMultiScatter();

    // failure-mode profiles
    const pr = r.failure_profiles;
    if (pr) {
      plot("chMProf", [{ type: "heatmap", z: pr.z, x: pr.features.map((f) => label(f, false)), y: pr.modes.map((m, i) => `${m.split(" · ")[0]} (n=${pr.counts[i]})`),
        zmid: 0, colorscale: [[0, "#1D4ED8"], [0.5, "#F8FAFC"], [1, "#B91C1C"]], xgap: 2, ygap: 2,
        text: pr.z.map((row) => row.map((v) => (v == null ? "" : v.toFixed(1)))), texttemplate: "%{text}", textfont: { size: 11 },
        customdata: pr.modes.map((m) => pr.features.map(() => m)), hovertemplate: "%{customdata}<br>%{x}: z = %{z:.2f}<extra></extra>", colorbar: { thickness: 10, outlinewidth: 0, title: { text: "z", side: "top" } } }],
      { margin: { l: 90, r: 10, t: 10, b: 110 }, xaxis: { tickangle: -40, gridcolor: "rgba(0,0,0,0)" }, yaxis: { autorange: "reversed", gridcolor: "rgba(0,0,0,0)" } });
    } else emptyState("chMProf", "No failure-mode columns", "This dataset has no TWF/HDF/PWF/OSF/RNF flags.");

    // importance
    const imp = r.importance;
    if (imp) {
      const pm = [...imp.permutation].sort((a, b) => a.mean - b.mean);
      plot("chMPerm", [{ type: "bar", orientation: "h", y: pm.map((d) => label(d.feature, false)), x: pm.map((d) => d.mean), error_x: { type: "data", array: pm.map((d) => d.std), color: C.text3, thickness: 1 },
        marker: { color: C.primary }, hovertemplate: "%{y}<br>drop in avg precision: %{x:.3f}<extra></extra>" }],
      { margin: { l: 140, t: 6 }, xaxis: { title: { text: "Drop in average precision when shuffled" } }, yaxis: { gridcolor: "rgba(0,0,0,0)" } });
      $("mPermNote").textContent = `Each raw input is shuffled 10 times on the held-out test set; the bar is the mean drop in ${imp.scoring.split(" on ")[0]} (± SD). Computed after training, unaffected by the page filters.`;
      const gi = [...imp.impurity].sort((a, b) => a.importance - b.importance);
      const nice = (f) => (f.startsWith("Type_") ? `Type = ${f.slice(5)}` : label(f, false));
      plot("chMGini", [{ type: "bar", orientation: "h", y: gi.map((d) => nice(d.feature)), x: gi.map((d) => d.importance),
        marker: { color: gi.map((d) => (["Temp_Difference", "Power", "Wear_Torque"].includes(d.feature) ? C.teal : "#93B4F5")) }, hovertemplate: "%{y}<br>%{x:.3f}<extra></extra>" }],
      { margin: { l: 150, t: 6 }, xaxis: { title: { text: "Mean decrease in impurity" } }, yaxis: { gridcolor: "rgba(0,0,0,0)" } });
    } else {
      emptyState("chMPerm", "Model unavailable", "Feature importance requires a valid trained model.");
      emptyState("chMGini", "Model unavailable", "");
    }
  }

  function renderMultiScatter() {
    const r = S.lastM;
    if (!r || r.empty) return;
    const sp = r.splom, fx = $("mX").value, fy = $("mY").value, fc = $("mC").value;
    const n = sp.failure.length;
    const lineW = sp.failure.map((f) => (f ? 1.6 : 0));
    plot("chMMulti", [{
      type: "scattergl", mode: "markers", x: sp[fx], y: sp[fy],
      marker: { color: sp[fc], colorscale: [[0, "#DBEAFE"], [0.5, "#2563EB"], [1, "#172033"]], size: sp.failure.map((f) => (f ? 8 : 5)), opacity: 0.75,
        line: { color: C.failure, width: lineW }, colorbar: { title: { text: label(fc), side: "right" }, thickness: 10, outlinewidth: 0 } },
      customdata: sp.failure.map((f, i) => [f ? "Failed (actual)" : "Normal", sp[fc][i]]),
      hovertemplate: `${esc(label(fx, false))}: %{x}<br>${esc(label(fy, false))}: %{y}<br>${esc(label(fc, false))}: %{customdata[1]}<br>%{customdata[0]}<extra></extra>`,
    }], { xaxis: { title: { text: label(fx) } }, yaxis: { title: { text: label(fy) } }, margin: { r: 10 } });
    if (!n) emptyState("chMMulti", "No records");
  }

  // ================================================================== wiring
  function wire() {
    // section tabs
    const tabs = document.querySelectorAll(".tab");
    tabs.forEach((t) => t.addEventListener("click", () => {
      document.getElementById(t.dataset.target).scrollIntoView({ behavior: "smooth" });
      setTab(t.dataset.target);
    }));
    const io = new IntersectionObserver((ents) => ents.forEach((e) => { if (e.isIntersecting) setTab(e.target.id); }), { rootMargin: "-35% 0px -60% 0px" });
    ["univariate", "bivariate", "multivariate"].forEach((id) => io.observe($(id)));
    if (location.hash && ["#univariate", "#bivariate", "#multivariate"].includes(location.hash)) setTab(location.hash.slice(1));

    ["gType", "gFail"].forEach((id) => $(id).addEventListener("change", refreshAll));
    $("exportFiltered").addEventListener("click", () => MS.download("/api/export/records" + qs(filters())));
    $("printBtn").addEventListener("click", () => {
      $("printStamp").textContent = `Generated ${new Date().toLocaleString()} · ${$("statusText").textContent} · filters: type=${$("gType").value}, status=${$("gFail").value}`;
      window.print();
    });

    // univariate
    $("uFeature").addEventListener("change", loadUni);
    $("uBins").addEventListener("change", () => (S.uHistMode === "label" ? S.lastU && renderUni(S.lastU) : loadUniBins()));
    segment("uHistMode", "mode", (v) => { S.uHistMode = v; if (S.lastU) renderUni(S.lastU); });

    // bivariate
    ["bX", "bY"].forEach((id) => $(id).addEventListener("change", () => { syncGroup(); loadBi(); }));
    document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
      const [x, y] = b.dataset.preset.split(",");
      $("bX").value = x; $("bY").value = y;
      syncGroup(); loadBi();
    }));
    segment("bColor", "color", (v) => { S.bColor = v; if (S.lastB) renderBi(S.lastB); });
    $("bGroupFeature").addEventListener("change", loadGroup);

    // multivariate
    segment("mMethod", "method", (v) => { S.method = v; loadMulti(); });
    $("mApply").addEventListener("click", loadMulti);
    $("mFeatures").addEventListener("change", MS.debounce(loadMulti, 500));
    ["mX", "mY", "mC"].forEach((id) => $(id).addEventListener("change", renderMultiScatter));
  }

  async function loadUniBins() {
    // bins are computed server-side for the "all" histogram
    setLoading("uHistCard", true);
    try {
      const r = await api("/api/univariate" + qs({ feature: $("uFeature").value, bins: $("uBins").value, ...filters() }));
      S.lastU = r; renderUni(r);
    } catch (e) { toast("error", "Histogram failed", e.message); } finally { setLoading("uHistCard", false); }
  }

  function syncGroup() {
    const y = $("bY").value, x = $("bX").value;
    const pick = NUMERIC.includes(y) ? y : NUMERIC.includes(x) ? x : null;
    if (pick && $("bGroupFeature").value !== pick) { $("bGroupFeature").value = pick; loadGroup(); }
  }

  function segment(id, attr, cb) {
    $(id).querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      $(id).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      cb(b.dataset[attr]);
    }));
  }

  function setTab(id) {
    document.querySelectorAll(".tab").forEach((t) => {
      const on = t.dataset.target === id;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", String(on));
    });
  }
})();

# MachineScope — Industrial Failure Intelligence

A working web application that analyses industrial machine sensor data, predicts machine
failures with machine learning, evaluates the model honestly on held-out data, screens for
statistical anomalies and builds a maintenance review queue.

* **Frontend:** HTML5, CSS3, vanilla JavaScript, Plotly.js (interactive charts), Lucide icons, Inter font — all vendored in `frontend/vendor/`, so the app works **offline**.
* **Backend:** Python, Flask, pandas, NumPy, scikit-learn.
* **Dataset:** AI4I 2020 Predictive Maintenance Dataset (bundled, 10,000 records).

---

## 1. Run it on Windows (PowerShell)

Requires **Python 3.10 or newer** (`py --version`).

```powershell
cd MachineScope
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python backend\app.py
```

Then open **http://127.0.0.1:5000** in your browser. Stop the server with `Ctrl + C`.

* If PowerShell blocks `Activate.ps1`, run once:
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
* The first start may print `saved model does not match … retraining` — this is normal when your
  scikit-learn version differs from the one used to build the bundled model. Training takes about
  30–60 seconds and only happens once.
* Different port: `$env:PORT = 8080; python backend\app.py`
* Different dataset: `$env:MACHINESCOPE_DATA = "C:\path\to\data.csv"; python backend\app.py`

Retrain the model manually and print the metrics:

```powershell
python backend\model.py
```

Run the automated backend tests:

```powershell
python -m unittest discover -s backend/tests -v
```

macOS / Linux: same steps with `python3 -m venv .venv` and `source .venv/bin/activate`.

---

## 2. What the app contains

### Page 1 — Overview dashboard (`index.html`)
| Area | What it shows |
|---|---|
| Header | Logo, navigation, dataset status, last-analysis timestamp, mobile menu |
| Welcome | **Run Analysis** (optionally *Retrain model*), **Explore Analytics**, dataset card with **Load local CSV**, **Bundled dataset**, **Print summary** |
| KPIs | Two clearly separated groups: *Dataset – actual labels* (machines, failures, failure rate) and *Random Forest – held-out test set* (accuracy, precision, recall, F1) |
| Failure analysis | Actual failure vs non-failure, failures by mode (TWF/HDF/PWF/OSF/RNF), machine-type distribution + failure rate, sensor summary (normal vs failed, effect size), compact model comparison, machines flagged for inspection |
| Model evaluation | 10 metric tiles, confusion matrix, ROC and precision–recall curves (RF vs LR), model comparison incl. the "always no failure" reference, 5-fold CV, methodology |
| Quick prediction | Type + 5 sensor inputs, client **and** server validation, loading state, example presets, reset, class + failure score (vote share), training-range warnings |
| Anomaly detection | IQR detector per sensor or across all sensors, adjustable k, filters, counts and %, fence chart, affected records, cross-tab *anomaly vs actual failure*, CSV export |
| Maintenance planner | High / Medium / Low priorities with the rule shown for each, search, filters, sorting, pagination, mark-as-reviewed (session), hide reviewed, CSV export |

### Page 2 — Comprehensive Data Analysis (`analysis.html`)
One page, three sections with sticky tab navigation and **global filters** (machine type, actual failure status) plus *Export filtered records* and *Print summary*.

* **A · Univariate** — feature selector (sensors, derived features, categorical Type); mean, median, SD, min, Q1, Q3, max, IQR, skewness, kurtosis; histogram (adjustable bins, overall or split by label); KDE distribution; box plots; IQR outlier detection with record table; missing-value summary; categorical frequencies; failure-label distribution.
* **B · Bivariate** — X / Y selectors and quick views (air vs process temperature, torque vs speed, tool wear vs failure, type vs failure); scatter with linear fit coloured by label or type; Pearson, Spearman, r², slope; violin plots for numeric-vs-label; failure rate across deciles of X; grouped box plots (type × label); failed-vs-normal comparison table with Cohen's d. A *correlation ≠ causation* notice is always visible.
* **C · Multivariate** — feature-selection chips; Pearson/Spearman correlation heatmap (includes the label); PCA on standardised features with explained variance, loadings and a plain-language explanation; scatter-plot matrix; multivariate scatter (x, y, colour = third variable, failures outlined); failure-mode sensor profiles (z-scores); permutation importance (raw inputs, test set) and impurity importance, with a note that importance is not causation.

---

## 3. Architecture

```
MachineScope/
├── backend/
│   ├── app.py            Flask app: serves frontend + JSON/CSV API, app state, error handling
│   ├── data_loader.py    CSV loading, schema validation, cleaning, derived features, filters
│   ├── ml_features.py    PhysicsFeatures transformer (ΔT, power, wear×torque) used inside the pipeline
│   ├── model.py          training, evaluation, persistence, prediction + input validation
│   ├── analysis.py       univariate / bivariate / multivariate stats, anomalies, maintenance rules
│   └── tests/            unittest suite + optional Playwright UI test
├── frontend/
│   ├── index.html  app.js        Overview dashboard
│   ├── analysis.html analysis.js Comprehensive analysis page
│   ├── common.js                 API client, formatting, toasts, chart theme
│   ├── style.css                 light-theme design system (responsive + print styles)
│   └── vendor/                   plotly.min.js, icons.js (Lucide subset), Inter font + licences
├── data/   ai4i2020.csv, README.md
├── models/ machinescope_model.joblib, README.md
└── requirements.txt
```

The browser only ever receives JSON or CSV. All data processing and model inference run in Python;
the pickle is never sent to the browser.

### API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/status` | dataset / model status, last-analysis time |
| GET | `/api/dataset-info` | schema, rows, missing values, provenance notes |
| GET | `/api/overview` | KPIs, label/mode/type distributions, sensor summary, flagged machines |
| GET | `/api/univariate?feature=&bins=&type=&failure=` | statistics, histogram, KDE, outliers |
| GET | `/api/bivariate?x=&y=&group=&type=&failure=` | scatter / grouped data, correlations, failure rate by bin |
| GET | `/api/failure-comparison?type=` | failed vs normal sensor comparison |
| GET | `/api/multivariate?features=&method=&type=&failure=` | correlation matrix, PCA, SPLOM data, mode profiles, importances |
| GET | `/api/model-performance` | all evaluation results and methodology metadata |
| POST | `/api/predict` | JSON sensor readings → class, failure score, warnings (422 on invalid input) |
| GET | `/api/anomalies?feature=&k=&type=&failure=` | IQR anomalies |
| GET | `/api/maintenance?priority=&search=&type=&failure=&sort=&order=&page=&page_size=&exclude=` | review queue |
| GET | `/api/export/records` · `/api/export/anomalies` · `/api/export/maintenance` | CSV downloads |
| POST | `/api/run-analysis` `{ "retrain": bool }` | recompute everything (optionally retrain) |
| POST | `/api/dataset/upload` (multipart `file`) · `/api/dataset/reset` | switch dataset |

Errors are returned as `{"ok": false, "error": "..."}` with status 400 / 404 / 405 / 413 / 422 / 500 / 503.

---

## 4. Machine-learning methodology

| Step | Choice |
|---|---|
| Target | `Machine failure` (0/1); 339 of 10,000 records = 3.39 % |
| Inputs | `Type`, air temperature, process temperature, rotational speed, torque, tool wear |
| Excluded (leakage) | `UDI`, `Product ID` (identifiers); `TWF`, `HDF`, `PWF`, `OSF`, `RNF` — failure-mode flags are components of the label |
| Engineered (inside the pipeline) | ΔT = process − air temperature, power = torque × speed × 2π/60, tool wear × torque — computed from inputs only |
| Preprocessing | median imputation, one-hot `Type`; standard scaling for Logistic Regression only — all fitted on training rows only |
| Split | single **stratified 80/20** split, `random_state = 42` → 8,000 train (271 failures) / 2,000 test (68 failures) |
| Primary model | Random Forest, 300 trees, `min_samples_leaf = 2`, `class_weight = "balanced"`, threshold 0.5 |
| Baseline | Logistic Regression, `class_weight = "balanced"` |
| Final metrics | computed once on the 2,000-record held-out test set |
| Stability check | 5-fold stratified CV on the training portion only |
| Maintenance scores | 5-fold **out-of-fold** predictions over all records (no record is scored by a model that saw its label) |

**Results on the held-out test set** (reproduced exactly on every retrain with seed 42, scikit-learn 1.9.1):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 98.95 % | 84.06 % | 85.29 % | 84.67 % | 97.00 % |
| Logistic Regression | 85.85 % | 17.91 % | 88.24 % | 29.78 % | 93.85 % |
| Always "no failure" | 96.60 % | — | 0 % | 0 % | 50 % |

Confusion matrix (RF): TN 1,921 · FP 11 · FN 10 · TP 58. CV on the training data: F1 84.00 % ± 2.29 %.
Because only ~3.4 % of records fail, accuracy alone is misleading — the dashboard states this
next to the metrics. Numbers can differ slightly with other scikit-learn versions; the app always
displays the values it actually computed.

**What the failure score means.** The prediction panel shows the forest's vote share for the
failure class. It ranks risk but is not a calibrated real-world probability, and a prediction is an
estimate from historical patterns — not confirmation that a physical machine will fail.

### Three kinds of flags — kept separate everywhere
* **Actual failure** — the recorded `Machine failure` label (red "Actual" tags).
* **Model prediction** — Random Forest output (blue "Model" tags).
* **Statistical anomaly** — a reading outside `[Q1 − k·IQR, Q3 + k·IQR]` (purple "Statistical" tags). With k = 1.5 across all sensors, 470 records are flagged, of which only 83 are actual failures, and 256 actual failures have no anomalous reading.

### Maintenance priority rules (transparent heuristics, not probabilities)
* **High** — out-of-fold model score ≥ 0.50
* **Medium** — score 0.20–0.49, or readings outside the IQR fences on 2+ sensors
* **Low** — score 0.05–0.19, or one sensor outside its fence

---

## 5. Tests performed

All of these were executed during development (Linux, Python 3.13, scikit-learn 1.9.1, Flask 3.1, pandas 3.0):

* **Python syntax** — `py_compile` on every backend module; `node --check` on every JS file.
* **Backend unit/API tests** — `python -m unittest discover -s backend/tests`: **24 tests, all passing**. They cover dataset loading and schema rejection, missing-file handling, filters, absence of leakage columns, split size/stratification, metrics re-computed from the stored test predictions (accuracy = (TP+TN)/n etc.), test accuracy ≠ training accuracy, single-class and untrainable-dataset disclosure, prediction validation (bad types, ranges, missing values, extrapolation warnings), IQR fences, correlation, PCA variance, maintenance rule consistency, every GET endpoint, JSON error responses, CSV exports and rejected uploads.
* **End-to-end UI test** — `backend/tests/ui_test_playwright.py` in headless Chromium: **42 checks, all passing** — KPIs, charts, predictions (failure and no-failure), client and server validation, reset, anomaly selectors and CSV export, maintenance filters/search/review/hide/sort/pagination/CSV, Run Analysis, rejected and accepted CSV uploads, dataset reset, page navigation, global filters, every analysis control and preset, Spearman toggle, feature chips, filtered-records CSV, no JavaScript errors, no horizontal scrolling at 390 px and a working mobile menu.
* **Reproducibility** — retraining from scratch reproduced identical test metrics.

Not tested: the Windows PowerShell commands themselves (written for Windows but executed here on Linux), Safari/Firefox, and printing to paper (print styles were written but only checked in the browser).

## 6. Limitations

* AI4I 2020 is synthetic; real plant data will be noisier and may need different features and thresholds.
* The decision threshold is fixed at 0.5 and the failure score is uncalibrated.
* Each record is a snapshot; there is no time series, so the planner ranks records, not machine histories.
* Uploaded datasets must follow the AI4I schema; a model trained on an upload lives in memory only.
* "Reviewed" marks are kept for the current browser session only.
* Feature importance and correlations describe the data and model, not physical causes.
* Educational prototype — do not use as the sole basis for safety, shutdown or maintenance decisions.

Third-party assets: Plotly.js (MIT), Lucide icons (ISC), Inter (SIL OFL) — licences in `frontend/vendor/`.

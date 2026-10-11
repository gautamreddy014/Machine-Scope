"""
MachineScope — Flask backend.

Serves the static frontend and a JSON API. All data processing and model
inference happen here in Python; the browser only receives JSON / CSV.

Run:  python backend/app.py      (then open http://127.0.0.1:5000)
"""

from __future__ import annotations

import io
import os
import sys
import threading
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analysis as an  # noqa: E402
import model as ml  # noqa: E402
from data_loader import (  # noqa: E402
    PROJECT_ROOT,
    TARGET,
    DatasetError,
    dataset_info,
    filter_frame,
    load_dataset,
)

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
MAX_UPLOAD_MB = 20


class AppState:
    """Holds the loaded dataset, model artifact and cached derived tables."""

    def __init__(self):
        self.lock = threading.RLock()
        self.ds = None
        self.art = None
        self.maint = None
        self.error = None
        self.last_analysis = None

    def load(self, path=None, content=None, name=None, retrain=False):
        with self.lock:
            ds = load_dataset(path, content=content, name=name)
            art = ml.load_or_train(ds, force=retrain)
            self.ds, self.art = ds, art
            self.maint = an.maintenance_table(ds, art)
            self.error = None
            self.last_analysis = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def require(self):
        if self.ds is None:
            raise DatasetError(self.error or "No dataset is loaded.")
        return self.ds, self.art


STATE = AppState()


def create_app(load_on_start: bool = True) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    app.json.sort_keys = False

    if load_on_start and STATE.ds is None:
        try:
            STATE.load()
        except Exception as exc:  # keep serving so the UI can show the error
            STATE.error = str(exc)
            print(f"[startup] {exc}")

    # ------------------------------------------------------------------ errors
    @app.errorhandler(DatasetError)
    def _bad_request(exc):
        return jsonify({"ok": False, "error": str(exc)}), 400

    @app.errorhandler(ml.ModelError)
    def _model_error(exc):
        return jsonify({"ok": False, "error": str(exc)}), 503

    @app.errorhandler(413)
    def _too_large(_):
        return jsonify({"ok": False, "error": f"File too large (max {MAX_UPLOAD_MB} MB)."}), 413

    @app.errorhandler(404)
    def _not_found(_):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": f"Unknown endpoint {request.path}"}), 404
        return send_from_directory(FRONTEND_DIR, "index.html"), 404

    @app.errorhandler(405)
    def _method(_):
        return jsonify({"ok": False, "error": f"{request.method} is not allowed on {request.path}"}), 405

    @app.errorhandler(Exception)
    def _internal(exc):
        app.logger.exception(exc)
        return jsonify({"ok": False, "error": f"Internal error: {exc}"}), 500

    # ---------------------------------------------------------------- helpers
    def filtered():
        ds, _ = STATE.require()
        return filter_frame(ds.df, request.args.get("type"), request.args.get("failure"))

    def csv_response(frame, filename):
        buf = io.StringIO()
        frame.to_csv(buf, index=False)
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    def stamp():
        return datetime.now().strftime("%Y%m%d_%H%M")

    # ---------------------------------------------------------------- frontend
    @app.get("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.get("/<path:path>")
    def static_files(path):
        return send_from_directory(FRONTEND_DIR, path)

    # --------------------------------------------------------------------- API
    @app.get("/api/status")
    def status():
        ok = STATE.ds is not None
        return jsonify({
            "ok": ok,
            "error": STATE.error,
            "dataset": STATE.ds.name if ok else None,
            "rows": int(len(STATE.ds.df)) if ok else 0,
            "model_valid": bool(STATE.art and STATE.art.get("valid")),
            "model_trained_at": STATE.art.get("trained_at") if STATE.art else None,
            "last_analysis": STATE.last_analysis,
        })

    @app.get("/api/dataset-info")
    def api_dataset_info():
        ds, _ = STATE.require()
        return jsonify({"ok": True, **dataset_info(ds)})

    @app.get("/api/overview")
    def api_overview():
        ds, art = STATE.require()
        return jsonify({"ok": True, "last_analysis": STATE.last_analysis, "dataset": dataset_info(ds),
                        **an.overview(ds, art, STATE.maint)})

    @app.get("/api/univariate")
    def api_univariate():
        feature = request.args.get("feature", "Torque")
        try:
            bins = int(request.args.get("bins", 40))
        except ValueError:
            raise DatasetError("bins must be an integer")
        if not 5 <= bins <= 200:
            raise DatasetError("bins must be between 5 and 200")
        return jsonify({"ok": True, **an.univariate(filtered(), feature, bins)})

    @app.get("/api/bivariate")
    def api_bivariate():
        x = request.args.get("x", "Air_Temperature")
        y = request.args.get("y", "Process_Temperature")
        return jsonify({"ok": True, **an.bivariate(filtered(), x, y, request.args.get("group") or None)})

    @app.get("/api/failure-comparison")
    def api_failure_comparison():
        df = filter_frame(STATE.require()[0].df, request.args.get("type"), None)
        return jsonify({"ok": True, "n": int(len(df)), "failures": int(df[TARGET].sum()),
                        "rows": an.failure_comparison(df)})

    @app.get("/api/multivariate")
    def api_multivariate():
        ds, art = STATE.require()
        feats = [f for f in request.args.get("features", "").split(",") if f]
        method = request.args.get("method", "pearson")
        return jsonify({"ok": True, **an.multivariate(filtered(), feats, method, art, ds.has_failure_modes)})

    @app.get("/api/model-performance")
    def api_model_performance():
        _, art = STATE.require()
        if not art.get("valid"):
            return jsonify({"ok": True, "valid": False, "message": art.get("message")})
        keys = ["trained_at", "dataset_name", "inputs", "excluded_columns", "random_state", "test_size",
                "threshold", "train_size", "test_size_n", "train_failures", "test_failures", "metrics",
                "baseline_metrics", "majority_baseline_accuracy", "cv", "impurity_importance",
                "permutation_importance", "training_seconds", "sklearn_version", "train_ranges", "types_seen"]
        return jsonify({"ok": True, "valid": True, **{k: art.get(k) for k in keys}})

    @app.post("/api/predict")
    def api_predict():
        _, art = STATE.require()
        payload = request.get_json(silent=True)
        if payload is None:
            raise DatasetError("Send the sensor readings as a JSON body.")
        values, errors, warns = ml.validate_prediction_input(payload, art)
        if errors:
            return jsonify({"ok": False, "error": "Invalid input.", "errors": errors}), 422
        return jsonify({"ok": True, "input": values, "warnings": warns, **ml.predict(art, values)})

    @app.get("/api/anomalies")
    def api_anomalies():
        ds, _ = STATE.require()
        try:
            k = float(request.args.get("k", 1.5))
        except ValueError:
            raise DatasetError("k must be a number")
        result, _ = an.anomalies(filtered(), ds.df, request.args.get("feature", "Rotational_Speed"), k)
        return jsonify({"ok": True, **result})

    @app.get("/api/maintenance")
    def api_maintenance():
        STATE.require()
        a = request.args
        tbl = an.filter_maintenance(STATE.maint, a.get("priority"), a.get("search"), a.get("type"),
                                    a.get("failure"), a.get("sort", "priority"), a.get("order", "asc"),
                                    a.get("exclude"))
        try:
            page = max(int(a.get("page", 1)), 1)
            size = min(max(int(a.get("page_size", 25)), 1), 500)
        except ValueError:
            raise DatasetError("page and page_size must be integers")
        start = (page - 1) * size
        counts = STATE.maint["priority"].value_counts().to_dict()
        return jsonify({
            "ok": True, "total": int(len(tbl)), "page": page, "page_size": size,
            "rules": an.PRIORITY_RULES, "counts": {k: int(counts.get(k, 0)) for k in ("High", "Medium", "Low")},
            "records": an._records(tbl.drop(columns=["priority_rank"]).iloc[start:start + size]),
            "note": ("Priorities are transparent screening rules, not calibrated failure probabilities. "
                     "Model scores are out-of-fold: each record was scored by a model that never saw its label."),
        })

    # ---------------------------------------------------------------- exports
    @app.get("/api/export/records")
    def export_records():
        df = filtered()
        cols = [c for c in df.columns if c not in ("Wear_Torque",)]
        return csv_response(df[cols], f"machinescope_records_{stamp()}.csv")

    @app.get("/api/export/anomalies")
    def export_anomalies():
        ds, _ = STATE.require()
        k = float(request.args.get("k", 1.5))
        _, frame = an.anomalies(filtered(), ds.df, request.args.get("feature", "Rotational_Speed"), k)
        frame = frame.copy()
        frame.insert(len(frame.columns), "note", "statistical anomaly (IQR) - not a confirmed failure")
        return csv_response(frame, f"machinescope_anomalies_{stamp()}.csv")

    @app.get("/api/export/maintenance")
    def export_maintenance():
        STATE.require()
        a = request.args
        tbl = an.filter_maintenance(STATE.maint, a.get("priority"), a.get("search"), a.get("type"),
                                    a.get("failure"), a.get("sort", "priority"), a.get("order", "asc"),
                                    a.get("exclude"))
        tbl = tbl.drop(columns=["priority_rank"]).copy()
        reviewed = {s for s in a.get("reviewed", "").split(",") if s.strip()}
        tbl["reviewed_this_session"] = tbl["UDI"].astype(str).isin(reviewed).map({True: "yes", False: "no"})
        tbl = tbl.rename(columns={TARGET: "actual_failure_label", "model_score": "model_score_out_of_fold"})
        return csv_response(tbl, f"machinescope_maintenance_{stamp()}.csv")

    # ------------------------------------------------------------- actions
    @app.post("/api/run-analysis")
    def run_analysis():
        body = request.get_json(silent=True) or {}
        retrain = bool(body.get("retrain", False))
        ds = STATE.ds
        if ds is not None and ds.source == "uploaded file":
            # re-run on the uploaded data held in memory
            with STATE.lock:
                art = ml.train(ds, save=False) if retrain else STATE.art
                STATE.art = art
                STATE.maint = an.maintenance_table(ds, art)
                STATE.last_analysis = datetime.now(timezone.utc).isoformat(timespec="seconds")
        else:
            STATE.load(retrain=retrain)
        return jsonify({"ok": True, "retrained": retrain, "last_analysis": STATE.last_analysis,
                        "model_trained_at": STATE.art.get("trained_at")})

    @app.post("/api/dataset/upload")
    def upload_dataset():
        f = request.files.get("file")
        if f is None or not f.filename:
            raise DatasetError("Choose a CSV file to upload.")
        if not f.filename.lower().endswith(".csv"):
            raise DatasetError("Only .csv files are supported.")
        content = f.read()
        ds = load_dataset(content=content, name=f.filename)  # validates schema
        with STATE.lock:
            art = ml.train(ds, save=False)  # never overwrite the bundled model with uploaded data
            STATE.ds, STATE.art = ds, art
            STATE.maint = an.maintenance_table(ds, art)
            STATE.last_analysis = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return jsonify({"ok": True, "dataset": dataset_info(ds), "model_valid": art.get("valid"),
                        "message": art.get("message")})

    @app.post("/api/dataset/reset")
    def reset_dataset():
        STATE.load()
        return jsonify({"ok": True, "dataset": dataset_info(STATE.ds)})

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    application = create_app()
    print(f"\n  MachineScope running at http://127.0.0.1:{port}\n")
    application.run(host="127.0.0.1", port=port, debug=False, threaded=True)

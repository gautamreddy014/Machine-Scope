"""
Backend tests for MachineScope (standard-library unittest, no extra packages).

Run from the project root:
    python -m unittest discover -s backend/tests -v
"""

import csv
import io
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import analysis as an  # noqa: E402
import model as ml  # noqa: E402
from app import STATE, create_app  # noqa: E402
from data_loader import FAILURE_MODES, DatasetError, filter_frame, load_dataset  # noqa: E402
from ml_features import ENGINEERED, MODEL_INPUTS  # noqa: E402

VALID = {"Type": "M", "Air_Temperature": 300.0, "Process_Temperature": 310.0,
         "Rotational_Speed": 1500, "Torque": 40.0, "Tool_Wear": 100}


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_dataset()

    def test_loads_real_ai4i(self):
        df = self.ds.df
        self.assertEqual(len(df), 10000)
        self.assertEqual(int(df["Machine_Failure"].sum()), 339)
        self.assertTrue(self.ds.has_failure_modes)
        self.assertEqual(sorted(df["Type"].unique()), ["H", "L", "M"])

    def test_rejects_wrong_schema(self):
        with self.assertRaises(DatasetError):
            load_dataset(content=b"a,b,c\n1,2,3\n", name="bad.csv")

    def test_rejects_non_binary_target(self):
        rows = ["Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min],Machine failure"]
        rows += [f"L,300,310,1500,40,{i},{i % 3}" for i in range(60)]
        with self.assertRaises(DatasetError):
            load_dataset(content="\n".join(rows).encode(), name="x.csv")

    def test_missing_file_is_reported_not_faked(self):
        with self.assertRaises(DatasetError):
            load_dataset("does/not/exist.csv")

    def test_filters(self):
        df = self.ds.df
        self.assertEqual(len(filter_frame(df, "H", None)), 1003)
        self.assertEqual(len(filter_frame(df, None, "failed")), 339)
        self.assertEqual(len(filter_frame(df, "L", "normal")), int(((df.Type == "L") & (df.Machine_Failure == 0)).sum()))
        with self.assertRaises(DatasetError):
            filter_frame(df, None, "maybe")


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_dataset()
        cls.art = ml.load_or_train(cls.ds, verbose=False)

    def test_no_leakage_columns_in_inputs(self):
        for col in FAILURE_MODES + ["UDI", "Product_ID", "Machine_Failure"]:
            self.assertNotIn(col, MODEL_INPUTS)
        self.assertEqual(list(self.art["model"].feature_names_in_), MODEL_INPUTS)
        prep_inputs = set(self.art["model"].named_steps["prep"].feature_names_in_)
        self.assertTrue(prep_inputs <= set(MODEL_INPUTS + ENGINEERED))

    def test_split_is_stratified_and_held_out(self):
        a = self.art
        self.assertEqual(a["train_size"] + a["test_size_n"], 10000)
        self.assertEqual(a["test_size_n"], 2000)
        self.assertEqual(len(set(a["test_index"])), 2000)
        # stratified: failure share in test ~ overall share
        y = self.ds.df["Machine_Failure"].values
        self.assertAlmostEqual(y[a["test_index"]].mean(), y.mean(), places=2)

    def test_metrics_match_recomputation_from_test_set(self):
        a = self.art
        df = self.ds.df.iloc[a["test_index"]]
        proba = a["model"].predict_proba(df[MODEL_INPUTS])[:, 1]
        np.testing.assert_allclose(proba, a["test_proba"], atol=1e-9)
        y = df["Machine_Failure"].values
        pred = (proba >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        m = a["metrics"]
        self.assertEqual(m["confusion_matrix"], {"tn": tn, "fp": fp, "fn": fn, "tp": tp})
        self.assertAlmostEqual(m["accuracy"], (tp + tn) / len(y))
        self.assertAlmostEqual(m["precision"], tp / (tp + fp))
        self.assertAlmostEqual(m["recall"], tp / (tp + fn))
        self.assertAlmostEqual(m["f1"], 2 * tp / (2 * tp + fp + fn))

    def test_test_metrics_are_not_training_metrics(self):
        # training-set accuracy of a forest is near-perfect; reported test accuracy must differ
        a = self.art
        train_idx = np.setdiff1d(np.arange(10000), a["test_index"])
        df = self.ds.df.iloc[train_idx]
        train_acc = (a["model"].predict(df[MODEL_INPUTS]) == df["Machine_Failure"].values).mean()
        self.assertGreater(train_acc, a["metrics"]["accuracy"])

    def test_oof_scores_cover_every_record(self):
        oof = np.array(self.art["oof_proba"])
        self.assertEqual(len(oof), 10000)
        self.assertTrue(((oof >= 0) & (oof <= 1)).all())

    def test_evaluate_handles_single_class(self):
        out = ml.evaluate(np.zeros(10), np.linspace(0, 0.4, 10))
        self.assertIsNone(out["roc_auc"])
        self.assertEqual(out["accuracy"], 1.0)

    def test_validation(self):
        _, errs, _ = ml.validate_prediction_input(VALID, self.art)
        self.assertEqual(errs, [])
        bad = dict(VALID, Torque="abc", Tool_Wear=None, Type="Q", Rotational_Speed=-5)
        _, errs, _ = ml.validate_prediction_input(bad, self.art)
        self.assertEqual(len(errs), 4)
        _, errs, warns = ml.validate_prediction_input(dict(VALID, Torque=120), self.art)
        self.assertEqual(errs, [])
        self.assertTrue(any("outside the training range" in w for w in warns))

    def test_untrainable_dataset_disclosed(self):
        ds = load_dataset()
        ds.df = ds.df[ds.df["Machine_Failure"] == 0].head(200).copy()
        art = ml.train(ds, save=False, verbose=False)
        self.assertFalse(art["valid"])
        self.assertIn("cannot be performed", art["message"])


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_dataset()
        cls.art = ml.load_or_train(cls.ds, verbose=False)

    def test_iqr_fences(self):
        s = self.ds.df["Torque"]
        f = an.iqr_fences(s)
        q1, q3 = np.percentile(s, [25, 75])
        self.assertAlmostEqual(f["lower"], q1 - 1.5 * (q3 - q1))
        self.assertAlmostEqual(f["upper"], q3 + 1.5 * (q3 - q1))

    def test_univariate_stats(self):
        r = an.univariate(self.ds.df, "Tool_Wear")
        self.assertAlmostEqual(r["stats"]["mean"], round(self.ds.df["Tool_Wear"].mean(), 4))
        self.assertEqual(sum(r["histogram"]["counts"]), 10000)
        with self.assertRaises(DatasetError):
            an.univariate(self.ds.df, "nope")

    def test_bivariate_correlation(self):
        r = an.bivariate(self.ds.df, "Rotational_Speed", "Torque")
        expected = self.ds.df["Rotational_Speed"].corr(self.ds.df["Torque"])
        self.assertAlmostEqual(r["pearson"], round(expected, 4))

    def test_maintenance_rules(self):
        t = an.maintenance_table(self.ds, self.art)
        high = t[t.priority == "High"]
        self.assertTrue((high.model_score >= 0.5).all())
        low = t[t.priority == "Low"]
        self.assertTrue(((low.model_score < 0.2) & (low.n_anomalous <= 1)).all())

    def test_multivariate_pca(self):
        r = an.multivariate(self.ds.df, ["Torque", "Rotational_Speed", "Tool_Wear"], "pearson", self.art, True)
        self.assertAlmostEqual(sum(r["pca"]["explained_variance"]), 1.0, places=4)
        self.assertEqual(r["correlation"]["matrix"][0][0], 1.0)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app().test_client()

    def test_get_endpoints(self):
        for url in ["/api/status", "/api/dataset-info", "/api/overview", "/api/univariate?feature=Torque",
                    "/api/bivariate?x=Torque&y=Rotational_Speed", "/api/bivariate?x=Machine_Failure&y=Tool_Wear",
                    "/api/multivariate?method=spearman", "/api/model-performance", "/api/anomalies?feature=all",
                    "/api/maintenance?priority=High", "/api/failure-comparison?type=L", "/"]:
            r = self.client.get(url)
            self.assertEqual(r.status_code, 200, url)

    def test_overview_kpis(self):
        o = self.client.get("/api/overview").get_json()
        self.assertEqual(o["kpis"]["total_machines"], 10000)
        self.assertAlmostEqual(o["kpis"]["actual_failure_rate"], 0.0339)

    def test_bad_params_return_json_errors(self):
        for url in ["/api/univariate?feature=zzz", "/api/anomalies?k=99", "/api/maintenance?sort=zzz",
                    "/api/univariate?bins=1", "/api/univariate?failure=maybe"]:
            r = self.client.get(url)
            self.assertEqual(r.status_code, 400, url)
            self.assertFalse(r.get_json()["ok"])
        self.assertEqual(self.client.get("/api/nope").status_code, 404)

    def test_predict(self):
        r = self.client.post("/api/predict", json=VALID)
        body = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertIn(body["predicted_class"], (0, 1))
        self.assertTrue(0 <= body["failure_score"] <= 1)
        r = self.client.post("/api/predict", json={"Type": "M"})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(self.client.post("/api/predict", data="x").status_code, 400)
        self.assertIn(self.client.get("/api/predict").status_code, (404, 405))

    def test_csv_exports(self):
        r = self.client.get("/api/export/records?type=H&failure=failed")
        rows = list(csv.DictReader(io.StringIO(r.data.decode())))
        self.assertEqual(r.mimetype, "text/csv")
        self.assertTrue(rows and all(x["Type"] == "H" and x["Machine_Failure"] == "1" for x in rows))
        r = self.client.get("/api/export/anomalies?feature=Torque")
        rows = list(csv.DictReader(io.StringIO(r.data.decode())))
        self.assertEqual(len(rows), 69)
        r = self.client.get("/api/export/maintenance?priority=High&reviewed=8399")
        rows = list(csv.DictReader(io.StringIO(r.data.decode())))
        self.assertEqual([x["reviewed_this_session"] for x in rows if x["UDI"] == "8399"], ["yes"])

    def test_upload_validation(self):
        data = {"file": (io.BytesIO(b"a,b\n1,2\n"), "bad.csv")}
        r = self.client.post("/api/dataset/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(STATE.ds.name, "ai4i2020.csv")  # unchanged after rejected upload


if __name__ == "__main__":
    unittest.main()

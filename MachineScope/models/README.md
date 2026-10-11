# Models

`machinescope_model.joblib` is produced by `python backend/model.py` (or automatically when the
server starts). It is a dictionary containing:

* `model` — fitted scikit-learn `Pipeline` (PhysicsFeatures → ColumnTransformer → RandomForestClassifier)
* `baseline` — fitted Logistic Regression pipeline
* held-out test indices, test-set scores and all evaluation metrics
* 5-fold CV summary, impurity and permutation importances
* out-of-fold scores for every record (used by the maintenance planner)
* metadata: dataset SHA-256, scikit-learn version, random seed, training ranges

On start-up the server reuses this file only if the dataset hash **and** the installed
scikit-learn version match; otherwise it retrains (≈30–60 s) and overwrites it. Pickles are
only loaded by the Python backend — the browser never touches this file.

The original project's `machine_failure_model.pkl` was **not** reused: its train/test split was
not recorded, so no honest held-out evaluation could be reported for it. The new model keeps the
same inputs (Type + 5 sensors) and the same Random-Forest family/settings the original code used.

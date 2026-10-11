"""
Feature engineering transformer used inside the scikit-learn pipeline.

Keeping it in its own module means the saved pipeline can be unpickled by
both the Flask app and the training script (they import the same module).

All engineered features are computed ONLY from the model's input sensors,
so they are available at prediction time and cannot leak target information.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MODEL_INPUTS = [
    "Type",
    "Air_Temperature",
    "Process_Temperature",
    "Rotational_Speed",
    "Torque",
    "Tool_Wear",
]
NUMERIC_INPUTS = MODEL_INPUTS[1:]
ENGINEERED = ["Temp_Difference", "Power", "Wear_Torque"]


class PhysicsFeatures(BaseEstimator, TransformerMixin):
    """Adds temperature difference, mechanical power and wear x torque."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(list(X.columns) if isinstance(X, pd.DataFrame) else MODEL_INPUTS, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=MODEL_INPUTS) if not isinstance(X, pd.DataFrame) else X.copy()
        X["Temp_Difference"] = X["Process_Temperature"] - X["Air_Temperature"]
        X["Power"] = X["Torque"] * X["Rotational_Speed"] * 2 * np.pi / 60.0
        X["Wear_Torque"] = X["Tool_Wear"] * X["Torque"]
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(MODEL_INPUTS + ENGINEERED, dtype=object)

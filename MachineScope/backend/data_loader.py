"""
Dataset loading and validation for MachineScope.

The application is built around the AI4I 2020 Predictive Maintenance Dataset
(UCI Machine Learning Repository, id 601). The bundled copy lives in
``data/ai4i2020.csv``. A different local CSV can be used if it follows the same
column schema (see REQUIRED_COLUMNS) - either by setting the environment
variable ``MACHINESCOPE_DATA`` or by uploading it from the dashboard.

Nothing in this module fabricates or substitutes data: if the file is missing
or malformed, a DatasetError is raised and the API reports it.
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ai4i2020.csv")

# Original AI4I column name -> internal snake-ish name
RENAME_MAP = {
    "UDI": "UDI",
    "Product ID": "Product_ID",
    "Type": "Type",
    "Air temperature [K]": "Air_Temperature",
    "Process temperature [K]": "Process_Temperature",
    "Rotational speed [rpm]": "Rotational_Speed",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Tool_Wear",
    "Machine failure": "Machine_Failure",
}

# Columns that MUST be present (after renaming) for the app to work.
REQUIRED_COLUMNS = [
    "Type",
    "Air_Temperature",
    "Process_Temperature",
    "Rotational_Speed",
    "Torque",
    "Tool_Wear",
    "Machine_Failure",
]

SENSOR_FEATURES = [
    "Air_Temperature",
    "Process_Temperature",
    "Rotational_Speed",
    "Torque",
    "Tool_Wear",
]
CATEGORICAL_FEATURES = ["Type"]
TARGET = "Machine_Failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
ID_COLUMNS = ["UDI", "Product_ID"]

FEATURE_META = {
    "Air_Temperature": {"label": "Air temperature", "unit": "K"},
    "Process_Temperature": {"label": "Process temperature", "unit": "K"},
    "Rotational_Speed": {"label": "Rotational speed", "unit": "rpm"},
    "Torque": {"label": "Torque", "unit": "Nm"},
    "Tool_Wear": {"label": "Tool wear", "unit": "min"},
    "Temp_Difference": {"label": "Temperature difference", "unit": "K"},
    "Power": {"label": "Mechanical power", "unit": "W"},
    "Wear_Torque": {"label": "Tool wear x torque", "unit": "min·Nm"},
    "Machine_Failure": {"label": "Machine failure (actual)", "unit": ""},
}

FAILURE_MODE_META = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    "RNF": "Random failure",
}

TYPE_META = {"L": "Low quality", "M": "Medium quality", "H": "High quality"}

# Derived analysis-only columns (computed from sensors; never from labels).
DERIVED_FEATURES = ["Temp_Difference", "Power", "Wear_Torque"]


class DatasetError(ValueError):
    """Raised when a dataset cannot be loaded or does not fit the schema."""


@dataclass
class Dataset:
    df: pd.DataFrame
    name: str
    source: str
    sha256: str
    loaded_at: str
    has_failure_modes: bool
    notes: list = field(default_factory=list)

    @property
    def analysis_features(self) -> list:
        return SENSOR_FEATURES + DERIVED_FEATURES


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physically motivated features computed only from sensor inputs."""
    out = df.copy()
    out["Temp_Difference"] = out["Process_Temperature"] - out["Air_Temperature"]
    out["Power"] = out["Torque"] * out["Rotational_Speed"] * 2 * np.pi / 60.0
    out["Wear_Torque"] = out["Tool_Wear"] * out["Torque"]
    return out


def _validate_and_clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    notes = []
    df = raw.copy()
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(
            "The CSV does not match the AI4I 2020 schema. Missing columns: "
            + ", ".join(missing)
            + ". Expected original headers such as 'Air temperature [K]', "
            "'Torque [Nm]', 'Type' and 'Machine failure'."
        )
    if len(df) < 50:
        raise DatasetError(f"The dataset has only {len(df)} rows; at least 50 are required.")

    for col in SENSOR_FEATURES + [TARGET] + [m for m in FAILURE_MODES if m in df.columns]:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        coerced = int(df[col].isna().sum() - before)
        if coerced:
            notes.append(f"{coerced} non-numeric value(s) in '{col}' were treated as missing.")

    df["Type"] = df["Type"].astype("string").str.strip().str.upper()

    # Target must be present and binary; rows without a label cannot be used.
    n_missing_target = int(df[TARGET].isna().sum())
    if n_missing_target:
        df = df[df[TARGET].notna()].copy()
        notes.append(f"{n_missing_target} row(s) without a 'Machine failure' label were dropped.")
    labels = set(pd.unique(df[TARGET]))
    if not labels.issubset({0, 1}):
        raise DatasetError(f"'Machine failure' must contain only 0/1 values; found {sorted(labels)}.")
    df[TARGET] = df[TARGET].astype(int)
    if len(labels) < 2:
        notes.append("Only one class is present in 'Machine failure'; model evaluation will be unavailable.")

    if "UDI" not in df.columns:
        df.insert(0, "UDI", np.arange(1, len(df) + 1))
        notes.append("No 'UDI' column found; a sequential record ID was generated.")
    if "Product_ID" not in df.columns:
        df["Product_ID"] = df["UDI"].astype(str)

    df = df.reset_index(drop=True)
    df = add_derived_features(df)
    return df, notes


def load_dataset(path: str | None = None, *, content: bytes | None = None, name: str | None = None) -> Dataset:
    """Load either a CSV path or raw CSV bytes (for uploads)."""
    if content is None:
        path = path or os.environ.get("MACHINESCOPE_DATA") or DEFAULT_DATA_PATH
        if not os.path.isfile(path):
            raise DatasetError(
                f"Dataset file not found: {path}. Place the AI4I 2020 CSV at data/ai4i2020.csv "
                "or set MACHINESCOPE_DATA to a local CSV path."
            )
        with open(path, "rb") as fh:
            content = fh.read()
        source = os.path.relpath(path, PROJECT_ROOT) if path.startswith(PROJECT_ROOT) else path
        name = name or os.path.basename(path)
    else:
        source = "uploaded file"
        name = name or "uploaded.csv"

    try:
        raw = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    except Exception as exc:  # pandas raises many types
        raise DatasetError(f"Could not parse CSV: {exc}") from exc

    df, notes = _validate_and_clean(raw)
    is_ai4i = name.lower().startswith("ai4i2020")
    return Dataset(
        df=df,
        name=name,
        source=source,
        sha256=hashlib.sha256(content).hexdigest(),
        loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        has_failure_modes=all(m in df.columns for m in FAILURE_MODES),
        notes=notes + (
            ["AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020; UCI ML Repository id 601). "
             "It is a synthetic dataset designed to reflect real industrial predictive-maintenance data."]
            if is_ai4i else []
        ),
    )


def dataset_info(ds: Dataset) -> dict:
    df = ds.df
    missing = {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS}
    return {
        "name": ds.name,
        "source": ds.source,
        "sha256": ds.sha256[:12],
        "loaded_at": ds.loaded_at,
        "rows": int(len(df)),
        "columns": [c for c in df.columns if c not in DERIVED_FEATURES],
        "derived_columns": DERIVED_FEATURES,
        "sensor_features": SENSOR_FEATURES,
        "target": TARGET,
        "failure_modes": FAILURE_MODES if ds.has_failure_modes else [],
        "failure_mode_labels": FAILURE_MODE_META,
        "machine_types": sorted(df["Type"].dropna().unique().tolist()),
        "type_labels": TYPE_META,
        "feature_meta": FEATURE_META,
        "missing_values": missing,
        "total_missing": int(sum(missing.values())),
        "notes": ds.notes,
        "is_ai4i": ds.name.lower().startswith("ai4i2020"),
    }


def filter_frame(df: pd.DataFrame, machine_type: str | None = None, failure: str | None = None) -> pd.DataFrame:
    """Apply the shared dashboard filters (machine type, actual failure status)."""
    out = df
    if machine_type and machine_type.upper() != "ALL":
        types = [t.strip().upper() for t in machine_type.split(",") if t.strip()]
        out = out[out["Type"].isin(types)]
    if failure is not None and str(failure).lower() not in ("", "all"):
        f = str(failure).lower()
        if f in ("1", "failed", "failure", "true"):
            out = out[out[TARGET] == 1]
        elif f in ("0", "normal", "false", "ok"):
            out = out[out[TARGET] == 0]
        else:
            raise DatasetError("failure filter must be one of: all, failed, normal")
    return out

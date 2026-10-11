"""
Exploratory analysis, anomaly screening and maintenance prioritisation.

Every number here is computed from the loaded dataset (and, where stated,
from the evaluation results of the trained model). Three kinds of "flag"
are kept strictly separate:

* actual failure      - the 'Machine failure' label recorded in the dataset
* model prediction    - output of the Random Forest (out-of-fold scores here)
* statistical anomaly - a reading outside the IQR fences of its sensor
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_loader import (
    DERIVED_FEATURES,
    FAILURE_MODE_META,
    FAILURE_MODES,
    FEATURE_META,
    SENSOR_FEATURES,
    TARGET,
    TYPE_META,
    Dataset,
    DatasetError,
)

NUMERIC_ANALYSIS = SENSOR_FEATURES + DERIVED_FEATURES
ANOMALY_FEATURES = SENSOR_FEATURES + ["Temp_Difference", "Power"]
MAX_POINTS = 10000  # scatter payload cap (dataset is 10k rows)


def _r(x, n=4):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(x), n)


def _check_feature(feature: str, allowed=None) -> str:
    allowed = allowed or NUMERIC_ANALYSIS
    if feature not in allowed:
        raise DatasetError(f"Unknown feature '{feature}'. Choose one of: {', '.join(allowed)}.")
    return feature


def _sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Random sample that always keeps every failure record (they are rare)."""
    if len(df) <= n:
        return df
    fails = df[df[TARGET] == 1]
    rest = df[df[TARGET] == 0]
    k = max(n - len(fails), 0)
    return pd.concat([rest.sample(n=min(k, len(rest)), random_state=seed), fails]).sort_index()


def iqr_fences(series: pd.Series, k: float = 1.5) -> dict:
    s = series.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return {"q1": float(q1), "q3": float(q3), "iqr": float(iqr), "lower": float(q1 - k * iqr), "upper": float(q3 + k * iqr), "k": k}


def describe(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"count": 0}
    return {
        "count": int(s.count()),
        "missing": int(series.isna().sum()),
        "mean": _r(s.mean()),
        "median": _r(s.median()),
        "std": _r(s.std()),
        "min": _r(s.min()),
        "q1": _r(s.quantile(0.25)),
        "q3": _r(s.quantile(0.75)),
        "max": _r(s.max()),
        "iqr": _r(s.quantile(0.75) - s.quantile(0.25)),
        "skew": _r(s.skew()),
        "kurtosis": _r(s.kurt()),
    }


def kde(values: np.ndarray, grid: np.ndarray) -> list:
    """Gaussian kernel density estimate with Silverman's bandwidth (numpy only)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2 or np.std(v) == 0:
        return [0.0] * len(grid)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    sigma = min(np.std(v, ddof=1), iqr / 1.34) if iqr > 0 else np.std(v, ddof=1)
    h = 0.9 * sigma * len(v) ** (-0.2)
    if len(v) > 4000:  # subsample for speed; density shape is unchanged
        v = np.random.default_rng(0).choice(v, 4000, replace=False)
    z = (grid[:, None] - v[None, :]) / h
    dens = np.exp(-0.5 * z * z).sum(axis=1) / (len(v) * h * np.sqrt(2 * np.pi))
    return np.round(dens, 8).tolist()


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def overview(ds: Dataset, art: dict, maintenance_rows: pd.DataFrame | None) -> dict:
    df = ds.df
    n = len(df)
    fails = int(df[TARGET].sum())
    by_type = (
        df.groupby("Type")[TARGET].agg(["count", "sum"]).rename(columns={"count": "machines", "sum": "failures"})
    )
    by_type["rate"] = by_type["failures"] / by_type["machines"]
    type_order = [t for t in ["L", "M", "H"] if t in by_type.index] + [t for t in by_type.index if t not in ("L", "M", "H")]

    modes = []
    if ds.has_failure_modes:
        for m in FAILURE_MODES:
            modes.append({"code": m, "label": FAILURE_MODE_META[m], "count": int(df[m].sum()),
                          "with_label": int(df.loc[df[m] == 1, TARGET].sum())})

    sensor_summary = []
    for f in NUMERIC_ANALYSIS:
        normal = df.loc[df[TARGET] == 0, f]
        failed = df.loc[df[TARGET] == 1, f]
        pooled = np.sqrt((normal.var() + failed.var()) / 2) if fails else np.nan
        sensor_summary.append({
            "feature": f, **FEATURE_META[f], "derived": f in DERIVED_FEATURES,
            "mean": _r(df[f].mean()), "std": _r(df[f].std()),
            "mean_normal": _r(normal.mean()), "mean_failed": _r(failed.mean()) if fails else None,
            "effect_size": _r((failed.mean() - normal.mean()) / pooled) if fails and pooled else None,
        })

    model_block = {"valid": bool(art.get("valid")), "message": art.get("message")}
    if art.get("valid"):
        m = art["metrics"]
        model_block.update({
            "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "roc_auc": m["roc_auc"], "confusion_matrix": m["confusion_matrix"], "n_test": m["n_test"],
            "majority_baseline_accuracy": art["majority_baseline_accuracy"], "trained_at": art["trained_at"],
        })

    flagged = []
    priority_counts = {"High": 0, "Medium": 0, "Low": 0}
    if maintenance_rows is not None and len(maintenance_rows):
        priority_counts.update(maintenance_rows["priority"].value_counts().to_dict())
        top = maintenance_rows.sort_values(["priority_rank", "model_score"], ascending=[True, False]).head(8)
        flagged = _records(top)

    return {
        "kpis": {
            "total_machines": n,
            "actual_failures": fails,
            "actual_failure_rate": fails / n if n else None,
            "unique_products": int(df["Product_ID"].nunique()),
        },
        "label_distribution": {"normal": n - fails, "failed": fails},
        "failure_modes": modes,
        "failure_mode_note": (
            "One record can have several failure modes; a few records are labelled as failures "
            "without a mode flag, and some RNF flags occur without a failure label."
        ) if modes else None,
        "machine_types": [
            {"type": t, "label": TYPE_META.get(t, t), "machines": int(by_type.loc[t, "machines"]),
             "failures": int(by_type.loc[t, "failures"]), "failure_rate": float(by_type.loc[t, "rate"])}
            for t in type_order
        ],
        "sensor_summary": sensor_summary,
        "model": model_block,
        "flagged": flagged,
        "priority_counts": {k: int(v) for k, v in priority_counts.items()},
    }


# --------------------------------------------------------------------------- #
# Univariate
# --------------------------------------------------------------------------- #
def univariate(df: pd.DataFrame, feature: str, bins: int = 40) -> dict:
    missing = [{"column": c, "missing": int(df[c].isna().sum()), "pct": float(df[c].isna().mean()) if len(df) else 0}
               for c in ["Type"] + NUMERIC_ANALYSIS + [TARGET]]
    type_freq = df["Type"].value_counts().reindex(["L", "M", "H"]).dropna()
    label_freq = df[TARGET].value_counts()
    base = {
        "n": int(len(df)),
        "missing": missing,
        "type_frequency": [{"type": t, "label": TYPE_META.get(t, t), "count": int(c), "pct": float(c / len(df))}
                           for t, c in type_freq.items()] if len(df) else [],
        "label_distribution": {"normal": int(label_freq.get(0, 0)), "failed": int(label_freq.get(1, 0))},
    }
    if feature == "Type":
        base["feature"] = "Type"
        base["categorical"] = True
        return base
    _check_feature(feature)
    if df.empty:
        return {**base, "feature": feature, "categorical": False, "empty": True}

    s = df[feature]
    stats = describe(s)
    fences = iqr_fences(s)
    out_mask = (s < fences["lower"]) | (s > fences["upper"])
    vmin, vmax = float(s.min()), float(s.max())
    pad = (vmax - vmin) * 0.05 or 1.0
    grid = np.linspace(vmin - pad, vmax + pad, 160)
    edges = np.histogram_bin_edges(s.dropna(), bins=bins)
    counts, _ = np.histogram(s.dropna(), bins=edges)
    normal = s[df[TARGET] == 0].dropna().values
    failed = s[df[TARGET] == 1].dropna().values
    outliers = df.loc[out_mask, ["UDI", "Product_ID", "Type", feature, TARGET]].sort_values(feature)

    return {
        **base,
        "feature": feature,
        "meta": FEATURE_META[feature],
        "categorical": False,
        "stats": stats,
        "histogram": {"edges": np.round(edges, 5).tolist(), "counts": counts.tolist()},
        "values": np.round(s.dropna().values, 4).tolist(),
        "values_by_label": {"normal": np.round(normal, 4).tolist(), "failed": np.round(failed, 4).tolist()},
        "density": {"x": np.round(grid, 5).tolist(), "all": kde(s.values, grid),
                    "normal": kde(normal, grid), "failed": kde(failed, grid)},
        "outliers": {
            **{k: _r(v) for k, v in fences.items()},
            "count": int(out_mask.sum()),
            "pct": float(out_mask.mean()),
            "low": int((s < fences["lower"]).sum()),
            "high": int((s > fences["upper"]).sum()),
            "actual_failures_among": int(df.loc[out_mask, TARGET].sum()),
            "records": _records(outliers.head(200)),
        },
    }


# --------------------------------------------------------------------------- #
# Bivariate
# --------------------------------------------------------------------------- #
def _failure_rate_by_bin(df: pd.DataFrame, feature: str, q: int = 10) -> list:
    s = df[feature]
    try:
        bins = pd.qcut(s, q=q, duplicates="drop")
    except ValueError:
        return []
    g = df.groupby(bins, observed=True)[TARGET].agg(["count", "sum"])
    return [{"bin": f"{iv.left:.1f}–{iv.right:.1f}", "mid": float(iv.mid), "count": int(r["count"]),
             "failures": int(r["sum"]), "rate": float(r["sum"] / r["count"]) if r["count"] else 0.0}
            for iv, r in g.iterrows()]


def bivariate(df: pd.DataFrame, x: str, y: str, group_feature: str | None = None) -> dict:
    allowed = NUMERIC_ANALYSIS + [TARGET, "Type"]
    _check_feature(x, allowed)
    _check_feature(y, allowed)
    if df.empty:
        return {"x": x, "y": y, "empty": True}

    out = {"x": x, "y": y, "n": int(len(df)),
           "x_meta": FEATURE_META.get(x, {"label": x, "unit": ""}),
           "y_meta": FEATURE_META.get(y, {"label": y, "unit": ""})}
    x_num = x in NUMERIC_ANALYSIS
    y_num = y in NUMERIC_ANALYSIS

    if x_num and y_num:
        sub = df[[x, y]].dropna()
        pearson = sub[x].corr(sub[y]) if len(sub) > 2 else None
        spearman = sub[x].rank().corr(sub[y].rank()) if len(sub) > 2 else None
        slope = intercept = None
        if len(sub) > 2 and sub[x].std() > 0:
            slope, intercept = np.polyfit(sub[x], sub[y], 1)
        pts = _sample(df, MAX_POINTS)
        out.update({
            "kind": "numeric-numeric",
            "pearson": _r(pearson), "spearman": _r(spearman),
            "strength": _strength(pearson),
            "trend": {"slope": _r(slope, 6), "intercept": _r(intercept, 6)} if slope is not None else None,
            "points": {
                "x": np.round(pts[x].values, 4).tolist(), "y": np.round(pts[y].values, 4).tolist(),
                "failure": pts[TARGET].astype(int).tolist(), "type": pts["Type"].astype(str).tolist(),
                "udi": pts["UDI"].astype(int).tolist(),
            },
            "sampled": len(pts) < len(df),
        })
    else:
        num = x if x_num else (y if y_num else None)
        cat = y if num == x else x
        if num is None:
            ct = pd.crosstab(df["Type"], df[TARGET])
            out.update({"kind": "categorical-categorical",
                        "crosstab": {str(t): {str(k): int(v) for k, v in row.items()} for t, row in ct.iterrows()}})
            return out
        groups = {}
        keys = sorted(df[cat].dropna().unique().tolist(), key=lambda v: (["L", "M", "H"].index(v) if v in ["L", "M", "H"] else v))
        for k in keys:
            vals = df.loc[df[cat] == k, num].dropna()
            label = ("Failed" if k == 1 else "Normal") if cat == TARGET else f"Type {k}"
            groups[label] = {"values": np.round(vals.values, 4).tolist(), "stats": describe(vals)}
        out.update({"kind": "numeric-categorical", "numeric": num, "category": cat, "groups": groups})
        if cat == TARGET:
            normal = df.loc[df[TARGET] == 0, num]
            failed = df.loc[df[TARGET] == 1, num]
            pooled = np.sqrt((normal.var() + failed.var()) / 2)
            out["effect_size"] = _r((failed.mean() - normal.mean()) / pooled) if len(failed) > 1 and pooled else None
            out["point_biserial"] = _r(df[num].corr(df[TARGET])) if df[TARGET].nunique() == 2 else None

    # Failure rate along the x feature (historical labels) - useful for every numeric x
    if x_num and df[TARGET].nunique() == 2:
        out["failure_rate_by_bin"] = _failure_rate_by_bin(df, x)

    # Grouped box plot data: a numeric feature split by machine type x actual label
    gf = group_feature or (y if y_num else x if x_num else None)
    if gf:
        _check_feature(gf)
        grouped = {}
        for lbl, name in ((0, "Normal"), (1, "Failed")):
            part = df[df[TARGET] == lbl]
            grouped[name] = {"type": part["Type"].astype(str).tolist(),
                             "values": np.round(part[gf].values, 4).tolist()}
        out["grouped"] = {"feature": gf, "meta": FEATURE_META[gf], "data": grouped}
    return out


def _strength(r):
    if r is None or pd.isna(r):
        return "n/a"
    a = abs(r)
    word = "very weak" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong" if a < 0.8 else "very strong"
    return f"{word} {'positive' if r > 0 else 'negative'}" if a >= 0.1 else word


def failure_comparison(df: pd.DataFrame) -> list:
    """Sensor readings for actual failed vs normal machines (historical labels)."""
    rows = []
    normal_df, failed_df = df[df[TARGET] == 0], df[df[TARGET] == 1]
    for f in NUMERIC_ANALYSIS:
        n, fl = normal_df[f], failed_df[f]
        pooled = np.sqrt((n.var() + fl.var()) / 2) if len(fl) > 1 else np.nan
        rows.append({
            "feature": f, **FEATURE_META[f],
            "normal_mean": _r(n.mean()), "failed_mean": _r(fl.mean()),
            "normal_median": _r(n.median()), "failed_median": _r(fl.median()),
            "diff_pct": _r((fl.mean() - n.mean()) / abs(n.mean()) * 100 if n.mean() else None, 2),
            "effect_size": _r((fl.mean() - n.mean()) / pooled) if pooled and not pd.isna(pooled) else None,
        })
    return rows


# --------------------------------------------------------------------------- #
# Multivariate
# --------------------------------------------------------------------------- #
def multivariate(df: pd.DataFrame, features: list, method: str, art: dict, has_modes: bool) -> dict:
    features = [f for f in features if f in NUMERIC_ANALYSIS] or SENSOR_FEATURES
    if method not in ("pearson", "spearman"):
        raise DatasetError("method must be 'pearson' or 'spearman'")
    if len(df) < 10:
        return {"empty": True, "features": features}

    cols = features + [TARGET]
    sub = df[cols].dropna()
    if method == "spearman":
        corr = sub.rank().corr()
    else:
        corr = sub.corr()
    corr = corr.fillna(0)

    # PCA on standardised features
    pca_block = None
    if len(features) >= 2:
        Xs = StandardScaler().fit_transform(sub[features].values)
        n_comp = min(len(features), 6)
        pca = PCA(n_components=n_comp, random_state=42).fit(Xs)
        scores = pca.transform(Xs)[:, :2]
        pts_idx = _sample(sub.assign(_i=np.arange(len(sub))), 4000)["_i"].values
        pca_block = {
            "explained_variance": np.round(pca.explained_variance_ratio_, 5).tolist(),
            "loadings": [{"feature": f, **{f"PC{i + 1}": _r(pca.components_[i, j]) for i in range(n_comp)}}
                         for j, f in enumerate(features)],
            "points": {"pc1": np.round(scores[pts_idx, 0], 4).tolist(), "pc2": np.round(scores[pts_idx, 1], 4).tolist(),
                       "failure": sub[TARGET].values[pts_idx].astype(int).tolist()},
        }

    splom = _sample(df, 2500)
    splom_block = {f: np.round(splom[f].values, 3).tolist() for f in NUMERIC_ANALYSIS}
    splom_block["failure"] = splom[TARGET].astype(int).tolist()
    splom_block["type"] = splom["Type"].astype(str).tolist()

    profiles = None
    if has_modes:
        normal = df[df[TARGET] == 0]
        mu, sd = normal[NUMERIC_ANALYSIS].mean(), normal[NUMERIC_ANALYSIS].std().replace(0, np.nan)
        profiles = {"modes": [], "features": NUMERIC_ANALYSIS, "z": [], "counts": []}
        for m in FAILURE_MODES:
            g = df[df[m] == 1]
            profiles["modes"].append(f"{m} · {FAILURE_MODE_META[m]}")
            profiles["counts"].append(int(len(g)))
            profiles["z"].append([_r((g[f].mean() - mu[f]) / sd[f], 3) if len(g) else None for f in NUMERIC_ANALYSIS])

    importance = None
    if art.get("valid"):
        importance = {"impurity": art["impurity_importance"], "permutation": art["permutation_importance"],
                      "scoring": "average precision on the held-out test set (10 repeats)"}

    return {
        "features": features,
        "method": method,
        "n": int(len(sub)),
        "correlation": {"labels": cols, "matrix": np.round(corr.loc[cols, cols].values, 4).tolist()},
        "pca": pca_block,
        "splom": splom_block,
        "failure_profiles": profiles,
        "importance": importance,
    }


# --------------------------------------------------------------------------- #
# Anomalies (statistical - not failures)
# --------------------------------------------------------------------------- #
def anomaly_frame(df: pd.DataFrame, features: list, k: float = 1.5, fence_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Fences are computed on fence_df (the full dataset) so filtering doesn't move them."""
    fence_df = df if fence_df is None else fence_df
    fences = {f: iqr_fences(fence_df[f], k) for f in features}
    flags = pd.DataFrame({f: (df[f] < fences[f]["lower"]) | (df[f] > fences[f]["upper"]) for f in features}, index=df.index)
    return flags, fences


def anomalies(df: pd.DataFrame, full_df: pd.DataFrame, feature: str, k: float = 1.5) -> tuple[dict, pd.DataFrame]:
    features = ANOMALY_FEATURES if feature == "all" else [_check_feature(feature, ANOMALY_FEATURES)]
    if not 0.5 <= k <= 5:
        raise DatasetError("IQR multiplier k must be between 0.5 and 5.")
    if df.empty:
        return {"feature": feature, "k": k, "count": 0, "pct": 0, "n": 0, "records": [], "per_feature": []}, df
    flags, fences = anomaly_frame(df, features, k, full_df)
    any_flag = flags.any(axis=1)
    flagged = df.loc[any_flag].copy()
    flagged["anomalous_sensors"] = flags.loc[any_flag].apply(
        lambda r: ", ".join(FEATURE_META[f]["label"] for f in features if r[f]), axis=1)
    flagged["n_anomalous"] = flags.loc[any_flag].sum(axis=1).astype(int)
    if len(features) == 1:
        f = features[0]
        med = full_df[f].median()
        flagged["direction"] = np.where(flagged[f] > med, "high", "low")
        flagged = flagged.reindex(flagged[f].sub(med).abs().sort_values(ascending=False).index)

    per_feature = [{"feature": f, **FEATURE_META[f], **{kk: _r(v) for kk, v in fences[f].items()},
                    "count": int(flags[f].sum()), "pct": float(flags[f].mean())} for f in features]
    cross = {
        "anomaly_and_failure": int((any_flag & (df[TARGET] == 1)).sum()),
        "anomaly_no_failure": int((any_flag & (df[TARGET] == 0)).sum()),
        "failure_no_anomaly": int((~any_flag & (df[TARGET] == 1)).sum()),
    }
    cols = ["UDI", "Product_ID", "Type"] + features + (["direction"] if len(features) == 1 else ["n_anomalous", "anomalous_sensors"]) + [TARGET]
    if len(features) > 1:
        flagged = flagged.sort_values("n_anomalous", ascending=False)
    result = {
        "feature": feature, "k": k, "n": int(len(df)),
        "count": int(any_flag.sum()), "pct": float(any_flag.mean()),
        "per_feature": per_feature, "crosstab": cross,
        "records": _records(flagged[cols].head(500)),
        "truncated": bool(len(flagged) > 500),
    }
    if len(features) == 1:
        f = features[0]
        normal_pts = df.loc[~any_flag]
        samp = normal_pts.sample(n=min(2500, len(normal_pts)), random_state=1) if len(normal_pts) else normal_pts
        result["plot"] = {
            "normal_udi": samp["UDI"].astype(int).tolist(), "normal_val": np.round(samp[f].values, 3).tolist(),
            "anom_udi": flagged["UDI"].astype(int).tolist(), "anom_val": np.round(flagged[f].values, 3).tolist(),
        }
    return result, flagged[cols]


# --------------------------------------------------------------------------- #
# Maintenance planner
# --------------------------------------------------------------------------- #
PRIORITY_RULES = [
    {"priority": "High", "rule": "Out-of-fold model failure score ≥ 0.50 (the model would predict a failure)."},
    {"priority": "Medium", "rule": "Model score 0.20–0.49, or readings outside the IQR fences on 2+ sensors."},
    {"priority": "Low", "rule": "Model score 0.05–0.19, or one sensor outside its IQR fence."},
]


def maintenance_table(ds: Dataset, art: dict) -> pd.DataFrame:
    df = ds.df
    flags, _ = anomaly_frame(df, ANOMALY_FEATURES, 1.5)
    n_anom = flags.sum(axis=1).astype(int)
    anom_names = flags.apply(lambda r: ", ".join(FEATURE_META[f]["label"] for f in ANOMALY_FEATURES if r[f]), axis=1)
    score = pd.Series(art["oof_proba"], index=df.index) if art.get("valid") else pd.Series(np.nan, index=df.index)

    pr = pd.Series("", index=df.index)
    has_score = score.notna()
    high = has_score & (score >= 0.5)
    med = ~high & ((has_score & (score >= 0.2)) | (n_anom >= 2))
    low = ~high & ~med & ((has_score & (score >= 0.05)) | (n_anom == 1))
    pr[high], pr[med], pr[low] = "High", "Medium", "Low"

    def basis(i):
        parts = []
        s = score[i]
        if not pd.isna(s) and s >= 0.05:
            parts.append(f"model score {s:.3f}")
        if n_anom[i]:
            parts.append(f"{n_anom[i]} anomalous sensor{'s' if n_anom[i] > 1 else ''} ({anom_names[i]})")
        return "; ".join(parts)

    keep = pr != ""
    out = df.loc[keep, ["UDI", "Product_ID", "Type"] + SENSOR_FEATURES + [TARGET]].copy()
    out["model_score"] = score[keep].round(4)
    out["n_anomalous"] = n_anom[keep]
    out["priority"] = pr[keep]
    out["priority_rank"] = out["priority"].map({"High": 0, "Medium": 1, "Low": 2})
    out["basis"] = [basis(i) for i in out.index]
    return out


def filter_maintenance(tbl: pd.DataFrame, priority=None, search=None, machine_type=None, failure=None,
                       sort="priority", order="asc", exclude=None) -> pd.DataFrame:
    out = tbl
    if exclude:
        ids = {s.strip() for s in str(exclude).split(",") if s.strip()}
        out = out[~out["UDI"].astype(str).isin(ids)]
    if priority and priority.lower() != "all":
        out = out[out["priority"].str.lower() == priority.lower()]
    if machine_type and machine_type.upper() != "ALL":
        out = out[out["Type"] == machine_type.upper()]
    if failure and failure.lower() != "all":
        out = out[out[TARGET] == (1 if failure.lower() in ("failed", "1") else 0)]
    if search:
        q = str(search).strip().lower()
        out = out[out["Product_ID"].str.lower().str.contains(q, regex=False)
                  | out["UDI"].astype(str).str.contains(q, regex=False)
                  | out["basis"].str.lower().str.contains(q, regex=False)]
    asc = order != "desc"
    if sort == "priority":
        out = out.sort_values(["priority_rank", "model_score"], ascending=[asc, not asc])
    elif sort in out.columns:
        out = out.sort_values(sort, ascending=asc)
    else:
        raise DatasetError(f"Cannot sort by '{sort}'.")
    return out


# --------------------------------------------------------------------------- #
def _records(frame: pd.DataFrame) -> list:
    recs = []
    for row in frame.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating, float)):
                v = None if pd.isna(v) else round(float(v), 4)
            elif v is pd.NA:
                v = None
            clean[k] = v
        recs.append(clean)
    return recs

# Data

`ai4i2020.csv` — **AI4I 2020 Predictive Maintenance Dataset** (10,000 rows, 14 columns).

* Source: S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications",
  AI4I 2020; UCI Machine Learning Repository, dataset id 601
  (https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), CC BY 4.0.
* It is a *synthetic* dataset built to reflect real industrial predictive-maintenance data.
* This copy was taken from the original MachineScope project ZIP; it is unchanged.

| Column | Meaning | Used by model? |
|---|---|---|
| UDI | record id | No (identifier) |
| Product ID | serial number; first letter = Type | No (identifier) |
| Type | product quality L / M / H | Yes |
| Air temperature [K] | ambient temperature | Yes |
| Process temperature [K] | process temperature | Yes |
| Rotational speed [rpm] | spindle speed | Yes |
| Torque [Nm] | torque | Yes |
| Tool wear [min] | accumulated tool wear | Yes |
| Machine failure | **target** (0/1) | Target |
| TWF, HDF, PWF, OSF, RNF | failure-mode flags | No — they are parts of the label (leakage) |

## Using a different local CSV

The file must contain the same headers (at minimum `Type`, the five sensor columns and
`Machine failure`). Either

* click **Load local CSV** on the dashboard (validated, then a model is trained on it in memory), or
* start the server with `MACHINESCOPE_DATA` pointing at the file:
  `$env:MACHINESCOPE_DATA = "C:\path\to\file.csv"; python backend\app.py`

If the file is missing or malformed the app reports an error — it never substitutes generated data.

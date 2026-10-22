# UAV Flight Memory

**KNN-Based Historical Flight Similarity and Anomaly Diagnosis**

This project turns UAV-SEAD telemetry into a leakage-safe KNN experiment and a web demo that explains both anomaly classification and historical flight similarity. The core question is not only "which class is this window?", but also "which historical UAV behavior does this window resemble?"

## Dataset

The project uses only **UAV-SEAD: State Estimation Anomaly Dataset for UAVs**. Place `mapping.json` at:

```text
data/raw/mapping.json
```

CSV files are cached under:

```text
data/raw/csv_files_annotated/
```

Current `mapping.json` inspection found:

| Class | Annotation count |
|---|---:|
| Normal | 901 |
| External Position | 198 |
| Uncategorized | 141 |
| Altitude | 78 |
| Mechanical | 47 |
| Global Position | 42 |

`Uncategorized` is reported but excluded from the main classification experiments.

## Problem Definition

Each flight is divided into fixed telemetry windows. For each test window, the system predicts the anomaly class with KNN and retrieves the nearest historical training windows with their `flight_id`, `window_id`, class, and distance.

KNN is intentionally the only ML model. Its role is:

1. anomaly classification,
2. historical similarity search.

## Data Pipeline

The pipeline is modular:

```text
src/uav_flight_memory/
  data/          mapping parsing, CSV download/cache, data quality
  labeling/      interval union and overlap calculations
  features/      windowing and feature extraction
  experiments/   flight-level split and experiment orchestration
  models/        KNN pipeline and Flight Memory neighbors
  web/           FastAPI app and static UI
```

Run commands from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m uav_flight_memory.cli analyze-mapping
python -m uav_flight_memory.cli select-flights --normal-limit 80 --external-position-limit 120
python -m uav_flight_memory.cli download-selected
python -m uav_flight_memory.cli build-windows
python -m uav_flight_memory.cli run-experiments
uvicorn uav_flight_memory.web.app:app --reload --port 8001
```

For a small smoke run:

```powershell
python -m uav_flight_memory.cli download-selected --limit 20
```

The downloader skips already cached CSV files.

## Windowing

The default is **20-second non-overlapping windows**. This is long enough to summarize local position and velocity behavior without creating highly correlated overlapping samples, while still short enough to capture localized anomalies. The window size is configurable, but the baseline experiments should keep it fixed for fair comparison.

Every generated row keeps:

- `flight_id`
- `window_id`
- absolute start/end timestamps
- label and label status
- extracted feature values

## Labeling Strategy

Annotation `timestamp` is not used as event time. It is metadata from annotation creation. Anomaly timing is taken from:

```text
annotations[].ranges
```

For each class, overlapping intervals are unioned before measuring window coverage, so overlapping ranges from the same class are not double-counted.

Default decision rule:

- no anomaly overlap: `Normal`
- exactly one anomaly class and coverage >= `0.10`: that anomaly class
- low coverage: `Uncertain`
- multiple anomaly classes in the same window: `Uncertain`

`Uncertain` windows are saved and reported, but excluded from the main single-label KNN training/evaluation. The `0.10` threshold means at least 2 seconds of a 20-second window must overlap a verified anomaly. This is a domain-motivated baseline intended to avoid losing short events; it is not tuned against test accuracy.

## Feature Engineering

The baseline feature set is intentionally compact and interpretable:

| Source columns | Statistics |
|---|---|
| `vehicle_local_position.x` | mean, std, min, max |
| `vehicle_local_position.y` | mean, std, min, max |
| `vehicle_local_position.z` | mean, std, min, max |
| `vehicle_local_position.vx` | mean, std, min, max |
| `vehicle_local_position.vy` | mean, std, min, max |
| `vehicle_local_position.vz` | mean, std, min, max |

This yields 24 features. Missing values are not filled with zero. Train-fitted median imputation is applied inside the sklearn pipeline before scaling.

## Data Quality

For each CSV, the pipeline checks:

- timestamp order,
- duplicate timestamps,
- sampling gaps,
- duration,
- required telemetry columns,
- missingness in required columns.

Quality results are saved to:

```text
data/processed/data_quality.csv
```

## Train/Test Strategy

Splits are flight-level:

```python
set(train_flight_ids).isdisjoint(test_flight_ids)
```

This assertion is enforced in code. Windows from the same flight are never split across train and test.

The required comparisons are:

- 80/20 train/test,
- 90/10 train/test.

## KNN and Scaling

The sklearn pipeline is:

```text
SimpleImputer(median) -> StandardScaler -> KNeighborsClassifier
```

The imputer and scaler are fit on training data only. Cross-validation uses grouped folds so preprocessing is refit inside each train fold.

Compared K values:

```text
1, 3, 5, 7, 9
```

Compared distance metrics:

```text
Euclidean, Manhattan
```

Minkowski is available in the web API for exploration, but the main grid keeps Euclidean and Manhattan as the required fair comparison.

## Experiments and Results

The experiment grid reports:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- class-level precision/recall/F1
- confusion matrix

Artifacts are saved under:

```text
results/experiments/
  mapping_summary.json
  windowing_summary.json
  experiment_metrics.csv
  latest_summary.json
```

Model selection is based on train-only grouped CV macro F1. Test metrics are reported for comparison, not for selecting parameters.

Current reproducible local run (`seed=42`, 362 selected flights, 2,093 windows) produced:

| Item | Value |
|---|---:|
| Usable single-label windows | 2,026 |
| Uncertain partial windows | 63 |
| Uncertain mixed-anomaly windows | 4 |
| Selected-by-CV configuration | 90/10, k=1, Euclidean |
| Selected-by-CV macro F1 | 0.425 |
| 80/20, k=1, Euclidean test accuracy | 0.739 |
| 80/20, k=1, Euclidean test macro F1 | 0.509 |

The higher or lower test values in the grid are reported transparently, but the final model-choice field in `latest_summary.json` is based on grouped cross-validation on training data only.

## Flight Memory

For a selected test window, the demo:

1. applies the train-fitted preprocessing pipeline,
2. predicts with KNN,
3. retrieves nearest neighbors only from training windows,
4. shows each neighbor's `flight_id`, `window_id`, class, and distance.

Distances are feature-space distances after preprocessing. They are not calibrated confidence probabilities.

## Web Application

Start the web app:

```powershell
uvicorn uav_flight_memory.web.app:app --reload --port 8001
```

Open:

```text
http://127.0.0.1:8001
```

The UI includes:

- experiment controls for split ratio, k, and distance,
- metrics and confusion matrix,
- comparison charts,
- Flight Memory demo.

The UI does not fabricate results. If `windows.csv` or experiment artifacts are missing, it reports that data preparation is required.

## Tests

Run:

```powershell
python -m unittest discover tests
```

Covered areas:

- mapping range parsing,
- timestamp/range domain matching on a real sample,
- interval union,
- anomaly overlap calculation,
- window labeling,
- feature extraction,
- flight-level leakage prevention,
- train-only preprocessing fit,
- KNN prediction.

## Limitations

- UAV-SEAD is class-imbalanced.
- Rare anomaly classes may have few usable windows after window-level labeling.
- A 20-second fixed window can dilute very short anomalies.
- `Uncertain` windows are excluded from the main single-label experiment, so results describe clearly labeled windows only.
- Local-position-only features may underrepresent actuator, magnetometer, or battery-specific anomalies.
- KNN stores training windows and can become slower as the memory grows.
- If only a subset of UAV-SEAD CSV files is downloaded, results must be described as selected-dataset results.

## Future Work

- Add a controlled second feature set for actuator, battery, and magnetometer signals after missingness and sampling analysis.
- Compare window-size sensitivity with one or two predeclared alternatives.
- Add approximate nearest-neighbor indexing if the historical memory grows large.
- Add a separate multilabel analysis for mixed anomaly windows.


# Technical Documentation
## Interoperable Healthcare AI Agent for Parkinson's Speech Screening

**Version:** 2.0.0 (Hackathon Edition) | **Python:** 3.10+ | **Last Updated:** May 2026
*Built for Agents Assemble: The Healthcare AI Endgame Challenge (Prompt Opinion)*

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Dataset](#3-dataset)
4. [Module Reference](#4-module-reference)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Model Training](#6-model-training)
7. [Explainability (SHAP)](#7-explainability-shap)
8. [API Reference](#8-api-reference)
9. [Frontend](#9-frontend)
10. [Monitoring & Drift Detection](#10-monitoring--drift-detection)
11. [Artifact Reference](#11-artifact-reference)
12. [DVC Pipeline](#12-dvc-pipeline)
13. [Docker](#13-docker)
14. [CI/CD (Jenkins)](#14-cicd-jenkins)
15. [Environment Variables](#15-environment-variables)
16. [Error Handling](#16-error-handling)
17. [Model Results](#17-model-results)

---

## 1. System Overview

Three-layer MLOps pipeline for Parkinson's disease detection from speech biomarkers:

```
Layer 1 — Training     src/train.py → models/ + artifacts/
Layer 2 — Analysis     src/explain.py + src/learning_curve.py → static/
Layer 3 — Serving      api/main.py → HTTP endpoints + UI
```

All layers share configuration through `src/config.py`, which resolves all file paths relative to the project root using `pathlib.Path`. This ensures scripts run correctly from any working directory — locally, inside Docker, or in CI.

The production model is **XGBoost**, chosen not for raw performance (SVM and KNN score higher on some metrics) but because SHAP `TreeExplainer` provides fast, exact feature attribution — critical for a medical application where clinicians need to understand *why* a prediction was made.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                           │
│  pd_speech_features.csv  ──►  DVC  ──►  DagsHub Remote     │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    TRAINING PIPELINE                        │
│  SelectFromModel(RF)  ──►  StandardScaler  ──►  ImbPipeline │
│  6 Models × RandomizedSearchCV × StratifiedKFold(5)        │
│  LR │ RF │ SVM │ KNN │ DT │ XGBoost                        │
│  Best Model (XGBoost) ──► MLflow Registry ──► DagsHub      │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    EXPLAINABILITY                           │
│  SHAP TreeExplainer ──► feature_importance.png              │
│  Per-prediction SHAP ──► shap_bar.png + top 10 impacts      │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    AGENT / SERVING LAYER                    │
│  FastAPI  ──►  /agent/screen (A2A Protocol + SHARP Context) │
│             │──► Speech Screening (XGBoost + SHAP)          │
│             │──► Clinical Summary (LLM)                     │
│             │──► FHIR Formatting (HL7 R4 DiagnosticReport)  │
│  FastAPI  ──►  /predict (Legacy ML pipeline)                │
│  7-tab UI including AI Agent workflows and Drift Monitoring │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset

| Property | Value |
|---|---|
| File | `data/pd_speech_features.csv` |
| Versioning | DVC tracked, remote on DagsHub |
| Rows | 756 |
| Raw columns | 755 (`id` + `class` + 753 features) |
| Selected features | 100 (by Random Forest importance) |
| Target | Binary — `1` = Parkinson's, `0` = Healthy |
| Class distribution | ~74.6% Parkinson's / 25.4% Healthy (imbalanced) |
| CSV header | Row index 1 (row 0 is subject metadata, not column names) |

### Feature Groups

| Group | Description |
|---|---|
| `PPE`, `DFA`, `RPDE` | Nonlinear dynamical complexity measures |
| `numPulses`, `numPeriodsPulses` | Glottal pulse counts |
| `locPctJitter`, `locAbsJitter` | Jitter — frequency variation |
| `localShimmer`, `localdbShimmer` | Shimmer — amplitude variation |
| `mean_MFCC_*` | Mel-frequency cepstral coefficients (0–12) |
| `std_*_delta`, `mean_*_delta` | Delta and delta-delta MFCC features |
| `tqwt_TKEO_*_dec_N` | TKEO energy/std per TQWT sub-band (36 levels) |
| `tqwt_entropy_log_dec_N` | Log entropy per TQWT sub-band |
| `tqwt_kurtosisValue_dec_N` | Kurtosis per TQWT sub-band |
| `f1`–`f4`, `b1`–`b4` | Formant frequencies and bandwidths |
| `GNE_*`, `VFER_*`, `IMF_*` | Glottal noise excitation, vocal fold energy ratio |

---

## 4. Module Reference

### 4.1 `src/config.py`

Central configuration. All other modules import paths and constants from here. Never hardcode paths elsewhere.

#### Key Constants

| Constant | Value | Description |
|---|---|---|
| `ROOT` | `Path(__file__).parent.parent` | Project root |
| `DATA_FILE` | `ROOT/data/pd_speech_features.csv` | Dataset path |
| `CSV_HEADER_ROW` | `1` | Row used as column names (UCI 2-row header) |
| `TARGET_COLUMN` | `"class"` | Binary target |
| `DROP_COLUMNS` | `["id"]` | Dropped before training |
| `EXPECTED_RAW_FEATURES` | `753` | Feature count after dropping id/class |
| `MODEL_PATH` | `ROOT/models/model.pkl` | Production XGBoost |
| `SCALER_PATH` | `ROOT/models/scaler.pkl` | Fitted StandardScaler |
| `SELECTOR_PATH` | `ROOT/models/selector.pkl` | Fitted SelectFromModel |
| `FEATURE_NAMES_PATH` | `ROOT/models/feature_names.pkl` | 100 selected feature names |
| `MODEL_METRICS_PATH` | `ROOT/artifacts/model_metrics.json` | Per-model metrics |
| `FEATURE_IMPORTANCE_PNG` | `ROOT/static/feature_importance.png` | SHAP global chart |
| `LEARNING_CURVE_PNG` | `ROOT/static/learning_curve.png` | Bias-variance plot |

#### `load_dataset() → tuple[pd.DataFrame, pd.Series]`

Loads and validates the dataset. Guards:
- Raises `FileNotFoundError` if CSV missing → run `dvc pull`
- Raises `ValueError` if `class` column absent → wrong `CSV_HEADER_ROW`
- Raises `ValueError` if feature count ≠ 753 → wrong `DROP_COLUMNS`

### 4.2 `src/train.py`

Full training pipeline. Trains 6 models, logs all runs to MLflow/DagsHub, saves production artifacts.

#### Execution Flow

```
1.  Load .env credentials (python-dotenv)
2.  Initialize DagsHub + MLflow remote tracking
3.  load_dataset() → X (753 features), y (binary)
4.  train_test_split (80/20, stratified, random_state=42)
5.  Feature selection:
      SMOTE on X_train → X_train_fs (balanced)
      SelectFromModel(RF, max_features=100).fit(X_train_fs)
      selector.transform(X_train) → X_train_sel (604 × 100)
      selector.transform(X_test)  → X_test_sel  (152 × 100)
6.  Save monitoring/baseline_data.csv (X_train_sel, unscaled)
7.  Scaling: StandardScaler.fit(X_train_sel) → X_train_sel_scaled
8.  Train 6 models with RandomizedSearchCV + ImbPipeline
9.  Log each run to MLflow (params, metrics, model artifact)
10. Persist artifacts/model_metrics.json with selection flags
11. Save models/*.pkl (model, scaler, selector, feature_names, column_order)
12. Generate static/feature_medians.json (all 753 column medians)
```

#### Why SMOTE Before Feature Selection

SMOTE is applied to `X_train` before fitting the RF selector so the selector learns features that distinguish the minority class. The selector is then applied to the original (non-SMOTE) splits — no leakage.

#### Preprocessing Order

```
select → scale   (NOT scale → select)
```

The scaler is fitted on `X_train_sel` (100 features). This order must be identical in `train.py` and `api/main.py`.

#### `compute_metrics(y_true, y_pred, y_prob) → dict`

Returns all metrics macro-averaged:
```python
{ "accuracy", "precision", "recall", "macro_f1", "roc_auc" }
```

#### Production Model Selection

XGBoost is **always** saved as the production model regardless of leaderboard rank. All 6 models are still logged to MLflow for comparison. Rationale: SHAP `TreeExplainer` provides fast, exact attribution for tree models. KNN/SVM require slow `KernelExplainer` approximations — unacceptable for clinical transparency.

#### MLflow Registration

```python
mlflow.sklearn.log_model(
    best_tuned_xgb,
    name="model",
    registered_model_name="parkinson_detection_model",
)
```

This atomically logs and registers the model in the MLflow Model Registry (MLflow 3.x pattern).

---

### 4.3 `src/explain.py`

Generates the global SHAP feature importance chart.

#### Flow

```
1. Load model, scaler, selector
2. load_dataset() → sample 50 rows
3. selector.transform → scaler.transform
4. Assert selector output shape == scaler.n_features_in_
5. TreeExplainer (or KernelExplainer fallback)
6. extract_shap_for_class1() → mean |SHAP| per feature
7. Plot top 20 → static/feature_importance.png
```

#### `extract_shap_for_class1(raw) → np.ndarray`

Normalises SHAP output to `(n_samples, n_features)` for class 1:

| Input | Source | Handling |
|---|---|---|
| `list` of arrays | sklearn RF, old SHAP | `raw[1]` |
| `(n, f, c)` ndarray | some XGBoost configs | `[:, :, 1]` |
| `(n, f)` ndarray | XGBoost binary, new SHAP | use as-is |
| `(f,)` ndarray | single-sample | `newaxis` → `(1, f)` |

---

### 4.4 `src/learning_curve.py`

Generates the bias-variance learning curve with zero data leakage.

The full preprocessing pipeline is wrapped in `ImbPipeline` and passed to `sklearn.learning_curve`. SMOTE, feature selection, and scaling are **all refitted inside each CV fold** — the validation fold never influences preprocessing.

```python
pipeline = ImbPipeline([
    ("smote",    SMOTE(random_state=42)),
    ("selector", SelectFromModel(RF(n_estimators=100), max_features=100)),
    ("scaler",   StandardScaler()),
    ("model",    model),
])
```

Plots train vs. validation macro F1 with ±1 std bands. Annotates the gap at the largest training size. Prints a warning if gap > 0.15 (potential overfitting).

---

### 4.5 `src/model_selection.py`

#### `apply_selection_flags(rows: list) → None`

Mutates rows in-place. Sets `selected: True` on the best model using:

1. Best **interpretable** model by composite score: `0.6 × roc_auc + 0.4 × macro_f1`
2. Fallback to overall best `macro_f1` if no interpretable model exists

Interpretable models: `XGBoost`, `XGBoost_tuned`, `Random Forest`, `RandomForest`, `Decision Tree`

---

### 4.6 `src/mlflow_comparison.py`

#### `fetch_model_comparison_from_mlflow() → list[dict]`

Two-strategy approach:

1. **Fast path:** Read `artifacts/model_metrics.json` (no network call)
2. **Fallback:** Query MLflow tracking server, keep most recent run per model name

Returns: `[{ model, accuracy, macro_f1, roc_auc, selected }]`

---

### 4.7 `api/main.py`

FastAPI application. All artifacts loaded at startup with consistency assertions.

#### Startup Assertions

```python
assert len(column_order) == 753           # column_order integrity
assert selector.n_features_in_ == 753     # selector expects 753 raw features
assert scaler.n_features_in_ == len(feature_names)  # scaler expects 100 features
```

Fails with `RuntimeError` before accepting any requests if artifacts are inconsistent.

#### `build_explainer(model, selector, scaler)`

- Tree models → `shap.TreeExplainer(model)` (fast, exact)
- Other models → `shap.KernelExplainer` with k-means background (10 clusters, 50 samples)
- Last-resort fallback → zero-vector background if sampling fails

#### `extract_shap_for_class1(raw) → np.ndarray`

Same logic as `src/explain.py` but returns 1-D array (single prediction).

#### `FeatureInput` (Pydantic model)

```python
features: Annotated[List[float], Field(min_length=753, max_length=753)]
```

Wrong length → `422 Unprocessable Entity` before any ML code runs.

#### Column Alignment

```python
arr = pd.DataFrame([data.features], columns=column_order).values
```

Prevents silent wrong-feature selection if input columns arrive in a different order.

#### Side Effects of `/predict`

1. Appends unscaled selected features to `monitoring/current_data.csv`
2. Overwrites `static/shap_bar.png` with a new dark-themed bar chart
3. Logs prediction + probability to local `mlruns/` via MLflow

---

## 5. Preprocessing Pipeline

### Data Flow

```
pd_speech_features.csv (756 × 755)
    │ drop "id"
(756 × 754)
    │ drop "class"
X: (756 × 753)    y: (756,)
    │ train_test_split(stratify=y, test_size=0.2, random_state=42)
X_train: (604 × 753)    X_test: (152 × 753)
    │ SMOTE on X_train (for selector only, not for CV)
X_train_fs: (~1128 × 753)
    │ SelectFromModel(RF, max_features=100).fit(X_train_fs)
    │   .transform(X_train) → X_train_sel: (604 × 100)
    │   .transform(X_test)  → X_test_sel:  (152 × 100)
    │ StandardScaler.fit(X_train_sel)
    │   .transform(X_train_sel) → X_train_sel_scaled: (604 × 100)
    │   .transform(X_test_sel)  → X_test_sel_scaled:  (152 × 100)
    │ ImbPipeline([SMOTE, model]).fit(X_train_sel, y_train)
      (SMOTE applied inside each CV fold — validation folds are real data)
```

### Training vs. Serving Consistency

| Step | Training (`train.py`) | Serving (`api/main.py`) |
|---|---|---|
| Input | 753 raw features | 753 raw features (column-aligned via `column_order`) |
| Step 1 | `selector.transform(X_train)` | `selector.transform(arr)` |
| Step 2 | `scaler.transform(X_train_sel)` | `scaler.transform(arr_selected)` |
| Step 3 | `model.predict(X_scaled)` | `model.predict(arr_scaled)` |

---

## 6. Model Training

### Hyperparameter Search Spaces

**Logistic Regression** — scaled input, 20 iterations
```
C:      Uniform(0.01, 10.01)
solver: ["lbfgs", "saga"]
```

**Random Forest** — unscaled input, 20 iterations
```
n_estimators:      randint(100, 400)
max_depth:         [None, 10, 20, 30]
max_features:      ["sqrt", "log2"]
min_samples_split: randint(2, 10)
```

**SVM** — scaled input, 15 iterations
```
C:      Uniform(0.1, 10.1)
gamma:  ["scale", "auto"]
kernel: ["rbf", "poly"]
```

**KNN** — scaled input, 15 iterations
```
n_neighbors: randint(3, 20)
weights:     ["uniform", "distance"]
metric:      ["euclidean", "manhattan"]
```

**Decision Tree** — unscaled input, 20 iterations
```
max_depth:         [None, 5, 10, 15, 20]
min_samples_split: randint(2, 20)
min_samples_leaf:  randint(1, 10)
criterion:         ["gini", "entropy"]
```

**XGBoost** — unscaled input, 30 iterations
```
max_depth:        randint(3, 8)
min_child_weight: randint(1, 6)
gamma:            Uniform(0, 0.5)
subsample:        Uniform(0.7, 1.0)
colsample_bytree: Uniform(0.7, 1.0)
n_estimators:     randint(100, 400)
learning_rate:    0.05 (fixed)
```

### MLflow Logging Per Run

- **Params:** model name, hyperparameters, `num_features=100`
- **Metrics:** `accuracy`, `precision`, `recall`, `macro_f1`, `roc_auc`
- **Artifact:** serialised model (sklearn flavor)

### Best XGBoost Configuration (current run)

```
max_depth: 5          colsample_bytree: 0.8
min_child_weight: 1   n_estimators: 300
gamma: 0              learning_rate: 0.05
subsample: 0.8
```

---

## 7. Explainability (SHAP)

### Global Feature Importance (`src/explain.py`)

- `TreeExplainer` computes SHAP values on 50 sampled rows
- Mean absolute SHAP values computed per feature
- Top 20 features plotted → `static/feature_importance.png`

### Per-Prediction Explanation (`api/main.py` `/predict`)

Every prediction returns:
- Top 10 SHAP feature contributions with actual feature names and impact values
- A server-generated `shap_bar.png` (dark-themed horizontal bar chart)
- Color coding: red = pushes toward Parkinson's, green = pushes toward Healthy

```json
{
  "feature_index": 42,
  "feature_name": "tqwt_kurtosisValue_dec_5",
  "impact": 0.2341
}
```

### SHAP Output Normalisation

SHAP output format varies by model type and SHAP version. Both `explain.py` and `api/main.py` use `extract_shap_for_class1()` to handle all known formats:

```python
if isinstance(raw, list):       # sklearn RF, old SHAP → [class0, class1]
    arr = np.array(raw[1])
else:
    arr = np.array(raw)

if arr.ndim == 3:               # (n_samples, n_features, n_classes)
    arr = arr[:, :, 1]

if arr.ndim == 1:               # single sample shortcut
    arr = arr[np.newaxis, :]
```

### KernelExplainer Fallback

Used when the model is not a tree type (e.g., SVM, KNN). Background dataset: k-means clustering of 50 training samples into 10 clusters. Falls back to a zero-vector background if clustering fails.

---

## 8. API Reference

### Agent Endpoints (A2A Protocol)

- **`POST /agent/screen`**: Orchestrates the 3-agent pipeline. Returns JSON with prediction, SHAP contributions, clinical summary, and FHIR report. Supports **SHARP Extension Context** for dynamic patient linking via the Prompt Opinion Platform.
- **`GET /agent/report/{session_id}`**: Retrieves the FHIR DiagnosticReport (`Content-Type: application/fhir+json`).
- **`GET /agent/health`**: Returns subsystem liveness.
- **`GET /agent/schema`**: Returns the machine-readable Pydantic JSON schema.
- **`GET /.well-known/agent-card.json`**: A2A v1.0 Agent Card detailing `supportedInterfaces` and capabilities.

---

### `GET /`
Returns the main HTML interface (Jinja2 template `templates/index.html`).

---

### `GET /health`

```json
{
  "status": "ok",
  "model": "XGBClassifier",
  "explainer": "TreeExplainer",
  "model_loaded": true
}
```

---

### `POST /predict`

**Request:**
```json
{ "features": [753 floats in training column order] }
```

Pydantic enforces exactly 753 values. Wrong count → `422 Unprocessable Entity`.

**Response:**
```json
{
  "prediction": 1,
  "label": "Parkinson's Detected",
  "probability": 0.923,
  "top_contributions": [
    { "feature_index": 42, "feature_name": "tqwt_kurtosisValue_dec_5", "impact": 0.2341 }
  ],
  "shap_bar_url": "/static/shap_bar.png"
}
```

**Side effects:**
- Appends unscaled selected features to `monitoring/current_data.csv`
- Overwrites `static/shap_bar.png`
- Logs to local `mlruns/` (non-fatal if MLflow unavailable)

---

### `GET /feature-defaults`

Returns top 5 SHAP-ranked features with dataset statistics for the prediction form, plus all 753 column medians for filling the full feature vector.

**Response:**
```json
{
  "top5": [
    {
      "name": "maxIntensity",
      "label": "Max Intensity (maxIntensity)",
      "tooltip": "Maximum vocal intensity...",
      "median": 78.5814,
      "min": 44.1335,
      "max": 86.3162
    }
  ],
  "columns": ["gender", "PPE", ...],
  "medians": { "gender": 1.0, "PPE": 0.5234, ... }
}
```

Top 5 computed by SHAP importance (100-sample background). Falls back to `feature_names[:5]` if SHAP fails.

---

### `GET /model-comparison`

Returns model leaderboard from `artifacts/model_metrics.json` (or MLflow fallback), sorted by ROC AUC.

**Response:**
```json
{
  "models": [
    { "model": "XGBoost", "accuracy": 0.88, "macro_f1": 0.836, "roc_auc": 0.943, "selected": true },
    { "model": "SVM",     "accuracy": 0.89, "macro_f1": 0.854, "roc_auc": 0.964, "selected": false }
  ]
}
```

---

### `GET /top-features`

Returns top 5 globally important features by mean absolute SHAP value.

**Response:**
```json
{
  "top_features": [
    { "rank": 1, "name": "tqwt_kurtosisValue_dec_5", "importance": 0.1823 }
  ]
}
```

---

### `GET /drift-status`

Returns the latest drift check results from `monitoring/drift_summary.txt` and `monitoring/drift_feature_details.csv`. Run `python monitoring/drift_check.py` to refresh these files before calling this endpoint.

**Response:**
```json
{
  "summary": {
    "total_features": 100,
    "drifted_count": 42,
    "drift_pct": 42.0,
    "generated_at": "2026-04-05 02:28:52",
    "status": "No significant dataset drift",
    "note": "Simulated data used — near-zero drift expected."
  },
  "features": [
    { "feature": "std_10th_delta_delta", "p_value": 0.0,    "drifted": true  },
    { "feature": "std_Log_energy",       "p_value": 0.9929, "drifted": false }
  ]
}
```

`features` is sorted ascending by p-value (most drifted first). `note` is only present when simulated data was used (fewer than 50 real predictions logged).

**Error:** `404` if drift files don't exist yet — run `python monitoring/drift_check.py` first.

---

### Error Responses

| Condition | Status | Detail |
|---|---|---|
| Wrong feature count | `422` | Pydantic validation error |
| Non-numeric values | `422` | Pydantic type error |
| ML pipeline failure | `500` | `"Prediction failed: <message>"` |
| `/feature-defaults` missing JSON | `404` | `"Run feature_medians generation first"` |
| `/model-comparison` no data | `404` | `"No model runs found in MLflow"` |
| MLflow unreachable | `503` | `"Could not load metrics from MLflow"` |
| `/drift-status` files missing | `404` | `"Drift data not found. Run drift_check.py first"` |
| `/drift-status` parse error | `500` | `"Failed to parse drift summary: <message>"` |

---

## 9. Frontend

Dark-themed single-page application served by FastAPI + Jinja2. No build step — vanilla HTML/CSS/JS with Chart.js from CDN.

### Tabs

| Tab | ID | Loaded by | Content |
|---|---|---|---|
| Feature Importance | `importance` | `loadTopFeatures()` | SHAP global chart + top 5 ranked list from `/top-features` |
| Learning Curve | `bias` | static image | Bias-variance plot + legend |
| Prediction | `prediction` | `loadPredictionFields()` | 5-input form + result card + SHAP chart + SHAP PNG |
| Model Comparison | `comparison` | `loadModelComparison()` | Live leaderboard from `/model-comparison` |
| Feature Insights | `insights` | `loadInsights()` | Biomarker analysis cards from `static/feature_insights.json` |
| Drift Monitor | `drift` | `loadDriftStatus()` | Status banner, drifted features chart, full feature table from `/drift-status` |

### Drift Monitor Tab (`loadDriftStatus()`)

```
1. GET /drift-status
2. Render status banner — green (no drift) or red (>50% drift)
   - Shows drift %, drifted/total count, progress gauge
   - Shows last-checked timestamp
3. If simulated data was used → blue info note
4. Bar chart of top 15 most drifted features (p-values, Chart.js)
5. Full feature table with filter buttons: All / Drifted only / Stable only
   - Each row: feature name, p-value, colored badge (Drifted / Stable)
6. Bottom callout explaining how drift monitoring works
```

The tab loads lazily on first click. It does not auto-refresh — re-run `drift_check.py` and reload the tab to update.

### Prediction Flow (`static/script.js`)

```
1. loadPredictionFields() → GET /feature-defaults
   - Stores _allColumns (753 column names) and _allMedians (753 medians)
   - Renders 5 input cards for top SHAP features

2. predict() on button click:
   - Builds featureMap from _allMedians (all 753 defaults)
   - Overrides with user-edited values from the 5 input cards
   - Builds ordered array: _allColumns.map(col => featureMap[col] ?? 0)
   - POST /predict with { features: [753 floats] }
   - renderResult() → result card with probability bar
   - renderShapChart() → Chart.js horizontal bar (canvas)
   - renderShapImage() → server-generated shap_bar.png
   - renderTopInfluencing() → top 5 feature list
```

### Key Design Decisions

- All 753 features are always sent — user only edits the top 5, rest use dataset medians
- `shap_bar.png` is cache-busted with `?t=Date.now()` on every prediction
- Model comparison table highlights best ROC AUC and best Macro F1 cells independently
- `escapeHtml()` used on all dynamic content to prevent XSS

---

## 10. Monitoring & Drift Detection

### How It Works

The API logs every prediction input to `monitoring/current_data.csv`. The drift check compares this against `monitoring/baseline_data.csv` (saved from `X_train_sel` after each training run).

Both files contain **unscaled selected features** (post-selector, pre-scaler) so distributions are directly comparable.

### `monitoring/drift_check.py`

```
1. Load baseline_data.csv (reference distribution)
2. Load current_data.csv (production inputs)
   - If missing or < 50 rows: simulate with baseline + σ=0.01 noise
3. Align columns (fill missing with 0, drop extra)
4. Run Evidently DataDriftPreset
5. Save drift_report.html (interactive)
6. Save drift_summary.txt (plain text)
7. Save drift_feature_details.csv (per-feature p-values)
8. Print console summary with ASCII histogram
```

### Drift Threshold

`DRIFT_THRESHOLD = 0.50` — flags if >50% of features drift (Kolmogorov-Smirnov p < 0.05).

### Outputs

| File | Description |
|---|---|
| `monitoring/drift_report.html` | Interactive Evidently HTML report |
| `monitoring/drift_summary.txt` | Plain-text summary with top 5 drifted features |
| `monitoring/drift_feature_details.csv` | Per-feature p-values, sorted ascending |

### Jenkins Integration

The `Drift Detection` stage runs after every build and archives `drift_report.html` as a build artifact, accessible from the Jenkins UI.

### Local MLflow Prediction Tracking

Every `/predict` call logs to `mlruns/` (local filesystem, no server required):
- **Param:** `model_type` (e.g., `XGBClassifier`)
- **Metrics:** `prediction` (0 or 1), `probability`

View with: `mlflow ui` → `http://localhost:5000`

---

## 11. Artifact Reference

### `models/`

| File | Created by | Used by | Description |
|---|---|---|---|
| `model.pkl` | `train.py` | `api/main.py`, `explain.py`, `learning_curve.py` | Production XGBoost |
| `scaler.pkl` | `train.py` | `api/main.py`, `explain.py` | StandardScaler (100 features) |
| `selector.pkl` | `train.py` | `api/main.py`, `explain.py` | SelectFromModel (753 → 100) |
| `feature_names.pkl` | `train.py` | `api/main.py`, `explain.py` | List of 100 selected feature names |
| `column_order.pkl` | `train.py` | `api/main.py` | List of 753 column names in training order |

### `artifacts/`

| File | Created by | Used by | Description |
|---|---|---|---|
| `model_metrics.json` | `train.py` | `/model-comparison` | All 6 model metrics with selection flags |
| `feature_config.json` | `train.py` | (legacy) | Top 5 features + all 753 column means |

### `static/`

| File | Created by | Used by | Description |
|---|---|---|---|
| `feature_importance.png` | `explain.py` | Feature Importance tab | Global SHAP bar chart |
| `learning_curve.png` | `learning_curve.py` | Learning Curve tab | Bias-variance plot |
| `shap_bar.png` | `/predict` endpoint | Prediction tab | Per-prediction SHAP chart (overwritten each call) |
| `feature_medians.json` | `train.py` | `/feature-defaults` | All 753 column medians |
| `feature_insights.json` | manual/notebook | Feature Insights tab | Biomarker analysis data |

### `monitoring/`

| File | Created by | Description |
|---|---|---|
| `baseline_data.csv` | `train.py` | X_train_sel (unscaled) — reference distribution |
| `current_data.csv` | `/predict` endpoint | Production inputs (appended per prediction) |
| `drift_report.html` | `drift_check.py` | Interactive Evidently report |
| `drift_summary.txt` | `drift_check.py` | Plain-text drift summary |
| `drift_feature_details.csv` | `drift_check.py` | Per-feature p-values |

---

## 12. DVC Pipeline

```yaml
stages:
  train:
    cmd: python src/train.py
    deps: [src/train.py, data/pd_speech_features.csv]
    outs: [models/model.pkl, models/scaler.pkl, models/selector.pkl,
           models/feature_names.pkl, models/column_order.pkl]

  explain:
    cmd: python src/explain.py
    deps: [src/explain.py, models/model.pkl, models/scaler.pkl, models/selector.pkl]
    outs: [static/feature_importance.png]

  learning_curve:
    cmd: python src/learning_curve.py
    deps: [src/learning_curve.py, models/model.pkl, data/pd_speech_features.csv]
    outs: [static/learning_curve.png]
```

Run with `dvc repro`. DVC tracks file hashes and only re-runs stages whose dependencies have changed.

DVC remote is configured in `.dvc/config` pointing to DagsHub. Credentials are set at runtime:

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user $DAGSHUB_USERNAME
dvc remote modify origin --local password $DAGSHUB_TOKEN
dvc pull data/pd_speech_features.csv.dvc --force
```

---

## 13. Cloud Run & Docker Deployment

The system is deployed on Google Cloud Run for serverless auto-scaling.

### Cloud Run Deployment
```bash
gcloud run deploy neurolynk-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

### Docker
Multi-stage build. Stage 1 installs all API dependencies; Stage 2 copies only what the API needs at runtime.

### Runtime Image Contents

```
api/main.py
src/config.py
src/__init__.py
src/mlflow_comparison.py
src/model_selection.py
models/          (all 5 pkl files)
static/          (charts + feature_medians.json + JS/CSS)
templates/index.html
```

### Excluded from Runtime Image

`src/train.py`, `src/explain.py`, `src/learning_curve.py`, `data/`, `notebooks/`, `mlruns/`, `monitoring/`

### Commands

```bash
docker build -t parkinson-ml .
docker run -p 8000:8000 parkinson-ml
# Open http://localhost:8000
```

### Dependency Split

| File | Used in | Purpose |
|---|---|---|
| `requirements.txt` | Dev / training / CI | Full stack including training deps |
| `requirements-api.txt` | Docker runtime | Minimal API-only deps (no DVC, dagshub, etc.) |

---

## 14. CI/CD (Jenkins)

7-stage pipeline. Runs on Windows agents — all steps use `bat` commands.

| Stage | Command | Purpose |
|---|---|---|
| Install Dependencies | `pip install -r requirements.txt` | Install all packages |
| Pull Data (DVC) | `dvc pull data/pd_speech_features.csv.dvc --force` | Fetch dataset from DagsHub |
| Train Model | `python src/train.py` | Full training run + MLflow logging |
| Generate Explanations | `python src/explain.py` + `python src/learning_curve.py` | Generate charts |
| Smoke Test | `curl -f http://localhost:8000/health` | Verify API starts correctly |
| Build Docker Image | `docker build -t parkinson-api .` | Create container image |
| Drift Detection | `python monitoring/drift_check.py` | Check for data drift |

Credentials (`DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`) are injected via Jenkins credentials store (`dagshub-username`, `dagshub-token`). Never hardcoded in source.

The `Drift Detection` stage always runs (even on failure) and archives `monitoring/drift_report.html` as a build artifact.

---

## 15. Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | No | `"mock"` | Options: `mock`, `openai`, `gemini` |
| `LLM_API_KEY` | Yes (if not mock) | `""` | API key for LLM generation |
| `LLM_MODEL` | No | `"gpt-4o-mini"` | Model used for summary |
| `AGENT_VERSION` | No | `"1.0.0-hackathon"`| Agent card version |
| `DAGSHUB_USERNAME` | Yes (training/DVC) | `"nishnarudkar"` | DagsHub account username |
| `DAGSHUB_TOKEN` | Yes (training/DVC) | `""` | DagsHub personal access token |
| `MLFLOW_TRACKING_URI` | No | DagsHub URI | Override MLflow tracking server |
| `MLFLOW_EXPERIMENT_NAME` | No | `"parkinson_detection"` | MLflow experiment name |

Set via `.env` file (gitignored). Copy `.env.example` to get started:

```bash
copy .env.example .env   # Windows
cp .env.example .env     # Linux/macOS
```

---

## 16. Error Handling

### Startup Errors

| Condition | Error | Resolution |
|---|---|---|
| `models/*.pkl` missing | `RuntimeError: Model artifact not found` | Run `python src/train.py` |
| Artifact shape mismatch | `RuntimeError: Artifact consistency check failed` | Re-run `python src/train.py` (artifacts from different runs) |

### Training Errors

| Condition | Error | Resolution |
|---|---|---|
| Dataset missing | `FileNotFoundError` | Run `dvc pull` |
| Wrong CSV structure | `ValueError: Column 'class' not found` | Check `CSV_HEADER_ROW` in `config.py` |
| Wrong feature count | `ValueError: Expected 753 features` | Check `DROP_COLUMNS` in `config.py` |
| DagsHub auth failure | MLflow connection error | Set `DAGSHUB_TOKEN` in `.env` |

### API Request Errors

| Condition | Status | Detail |
|---|---|---|
| Wrong feature count | `422` | `"List should have at most/at least 753 items"` |
| Non-numeric values | `422` | Pydantic type validation error |
| ML pipeline failure | `500` | `"Prediction failed: <error message>"` |
| `/feature-defaults` missing JSON | `404` | `"Run feature_medians generation first"` |
| `/model-comparison` no data | `404` | `"No model runs found in MLflow"` |
| MLflow unreachable | `503` | `"Could not load metrics from MLflow"` |

---

## 17. Model Results

Results on held-out test set (152 samples, stratified split):

| Model | Accuracy | Macro F1 | ROC AUC | Selected |
|---|---|---|---|---|
| SVM | 0.895 | 0.854 | 0.964 | |
| KNN | 0.882 | 0.858 | 0.950 | |
| **XGBoost** | **0.882** | **0.836** | **0.943** | **✅ Production** |
| Random Forest | 0.862 | 0.817 | 0.919 | |
| Decision Tree | 0.849 | 0.809 | 0.818 | |
| Logistic Regression | 0.829 | 0.783 | 0.830 | |

XGBoost is selected over SVM and KNN despite lower raw metrics because SHAP `TreeExplainer` provides fast, exact feature attribution. SVM and KNN require slow `KernelExplainer` approximations — unacceptable for a medical application requiring clinical transparency.

Selection score formula: `0.6 × roc_auc + 0.4 × macro_f1`

---

*This system is for research and educational purposes only. It is not a validated medical diagnostic tool. Do not use predictions for clinical decision-making.*

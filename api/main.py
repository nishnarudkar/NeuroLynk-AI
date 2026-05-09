from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Annotated, Any, Dict, List
import json
import joblib
import numpy as np
import pandas as pd
import shap
import sys
import logging
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server use
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger("uvicorn.error")

# Resolve project root so config imports work from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    load_dataset,
    MODEL_PATH, SCALER_PATH, SELECTOR_PATH, FEATURE_NAMES_PATH,
    TEMPLATES_DIR, STATIC_DIR, EXPECTED_RAW_FEATURES, MODELS_DIR,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

COLUMN_ORDER_PATH = MODELS_DIR / "column_order.pkl"
TREE_MODELS = (RandomForestClassifier, GradientBoostingClassifier,
               DecisionTreeClassifier, XGBClassifier)

app = FastAPI(title="Parkinson Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class FeatureInput(BaseModel):
    """Exactly EXPECTED_RAW_FEATURES float values in training column order."""
    features: Annotated[
        List[float],
        Field(
            min_length=EXPECTED_RAW_FEATURES,
            max_length=EXPECTED_RAW_FEATURES,
            description=f"Exactly {EXPECTED_RAW_FEATURES} numeric speech feature values",
        ),
    ]


# ── SHAP normalisation ────────────────────────────────────────────────────────
def extract_shap_for_class1(raw: object) -> np.ndarray:
    """
    Robustly extract a 1-D SHAP array for the positive class (class 1)
    regardless of how the installed SHAP version returns values.

    Handles all known output shapes:
      - list of arrays  → [class0_arr, class1_arr]  (sklearn RF, old SHAP)
      - 3-D ndarray     → (n_samples, n_features, n_classes)  (some XGBoost)
      - 2-D ndarray     → (n_samples, n_features)  (XGBoost binary, new SHAP)
      - 1-D ndarray     → (n_features,)  (single-sample shortcut)
    """
    if isinstance(raw, list):
        # list output: take class-1 slice, then squeeze to 1-D
        arr = np.array(raw[1])
    else:
        arr = np.array(raw)

    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) → class-1 slice
        arr = arr[:, :, 1]

    # Now arr is (n_samples, n_features) or (n_features,)
    if arr.ndim == 2:
        arr = arr[0]   # single prediction → (n_features,)

    if arr.ndim != 1:
        raise ValueError(
            f"Unexpected SHAP output shape after normalisation: {arr.shape}. "
            "Please check your SHAP version."
        )
    return arr


# ── Explainer factory ─────────────────────────────────────────────────────────
def build_explainer(model, selector, scaler):
    """
    Build the appropriate SHAP explainer.
    For non-tree models, falls back to KernelExplainer with a safe
    background dataset — handles small datasets and sampling failures.
    """
    if isinstance(model, TREE_MODELS):
        logger.info("Using SHAP TreeExplainer")
        return shap.TreeExplainer(model)

    logger.info(f"Model is {type(model).__name__} — building KernelExplainer background")
    try:
        X_all, _ = load_dataset()
        n_bg = min(50, len(X_all))          # safe even on tiny datasets
        X_bg = X_all.sample(n_bg, random_state=42)
        X_bg_sel    = selector.transform(X_bg)
        X_bg_scaled = scaler.transform(X_bg_sel)

        n_clusters = min(10, n_bg)          # kmeans needs k ≤ n_samples
        background  = shap.kmeans(X_bg_scaled, n_clusters)
        logger.info(f"KernelExplainer background: {n_bg} samples → {n_clusters} clusters")
    except Exception as e:
        # Last-resort fallback: use the feature-wise mean as a single background point
        logger.warning(
            f"Background sampling failed ({e}). "
            "Falling back to zero-vector background for KernelExplainer."
        )
        n_features  = scaler.n_features_in_
        background  = np.zeros((1, n_features))

    return shap.KernelExplainer(model.predict_proba, background)


# ── Load artifacts at startup ─────────────────────────────────────────────────
try:
    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    selector      = joblib.load(SELECTOR_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    column_order  = joblib.load(COLUMN_ORDER_PATH)
    COLUMN_ORDER_SET = frozenset(column_order)

    # Validate the preprocessing chain is internally consistent
    assert len(column_order) == EXPECTED_RAW_FEATURES, (
        f"column_order has {len(column_order)} entries, expected {EXPECTED_RAW_FEATURES}"
    )
    assert selector.n_features_in_ == EXPECTED_RAW_FEATURES, (
        f"selector expects {selector.n_features_in_} features, expected {EXPECTED_RAW_FEATURES}"
    )
    assert scaler.n_features_in_ == len(feature_names), (
        f"scaler expects {scaler.n_features_in_} features but feature_names has {len(feature_names)}"
    )

    explainer = build_explainer(model, selector, scaler)
    logger.info(f"Loaded: {type(model).__name__} + {type(explainer).__name__}")

except FileNotFoundError as e:
    raise RuntimeError(
        f"Model artifact not found: {e}. Run `python src/train.py` first."
    )
except AssertionError as e:
    raise RuntimeError(f"Artifact consistency check failed: {e}")
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict")
def predict(data: FeatureInput):
    """Accept a list of 753 features in column_order."""
    try:
        arr = pd.DataFrame([data.features], columns=column_order).values

        # Preprocessing: select → scale  (matches train.py order)
        arr_selected = selector.transform(arr)        # 753 → 100 features
        arr_scaled   = scaler.transform(arr_selected) # standardise

        # ── Log input to monitoring/current_data.csv ──────────────────────────
        # Log UNSCALED selected features (same scale as baseline_data.csv)
        # baseline_data.csv = X_train_sel (unscaled), so we log arr_selected (unscaled)
        try:
            _monitoring_csv = Path(__file__).resolve().parent.parent / "monitoring" / "current_data.csv"
            _log_df = pd.DataFrame(arr_selected, columns=feature_names)
            _write_header = not _monitoring_csv.exists() or _monitoring_csv.stat().st_size == 0
            _log_df.to_csv(_monitoring_csv, mode="a", header=_write_header, index=False)
        except Exception as _log_err:
            logger.warning(f"Prediction logging failed (non-fatal): {_log_err}")
        # ─────────────────────────────────────────────────────────────────────

        # Prediction
        prediction = int(model.predict(arr_scaled)[0])
        prob       = float(model.predict_proba(arr_scaled)[0][1])

        # SHAP — robust extraction regardless of SHAP version / model type
        raw_shap  = explainer.shap_values(arr_scaled)
        shap_vals = extract_shap_for_class1(raw_shap)   # guaranteed 1-D

        top_indices = np.argsort(np.abs(shap_vals))[-10:][::-1]
        explanation = [
            {
                "feature_index": int(i),
                "feature_name":  feature_names[i],
                "impact":        float(shap_vals[i]),
            }
            for i in top_indices
        ]

        # ── Generate SHAP bar chart PNG ───────────────────────────────────────
        names   = [feature_names[i] for i in top_indices]
        impacts = [float(shap_vals[i]) for i in top_indices]
        colors  = ["#f87171" if v >= 0 else "#34d399" for v in impacts]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(names[::-1], impacts[::-1], color=colors[::-1])
        ax.axvline(0, color="#8892b0", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP Value (impact on prediction)", color="#e2e8f0")
        ax.set_title(
            f"Top Biomarkers — {'Parkinson' if prediction == 1 else 'Healthy'} "
            f"({prob*100:.1f}% confidence)",
            color="#e2e8f0", fontsize=11,
        )
        fig.patch.set_facecolor("#1a1d27")
        ax.set_facecolor("#22263a")
        ax.tick_params(colors="#e2e8f0")
        ax.spines[:].set_color("#2e3250")
        plt.tight_layout()

        shap_bar_path = STATIC_DIR / "shap_bar.png"
        plt.savefig(shap_bar_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        # ─────────────────────────────────────────────────────────────────────

        # ── Lightweight local MLflow tracking (no server required) ──────────
        try:
            import mlflow as _mlflow
            _mlflow.set_tracking_uri("mlruns")   # local filesystem only
            with _mlflow.start_run(run_name="prediction", nested=True):
                _mlflow.log_param("model_type", type(model).__name__)
                _mlflow.log_metric("prediction", float(prediction))
                _mlflow.log_metric("probability", prob)
        except Exception as _mf_err:
            logger.debug(f"MLflow local tracking skipped: {_mf_err}")
        # ─────────────────────────────────────────────────────────────────────

        return {
            "prediction":        prediction,
            "label":             "Parkinson's Detected" if prediction == 1 else "Healthy",
            "probability":       prob,
            "top_contributions": explanation,
            "shap_bar_url":      "/static/shap_bar.png",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/feature-defaults")
def feature_defaults():
    """Return the top 5 features by SHAP importance with medians, min, max + all 753 median defaults."""
    import json
    defaults_path = STATIC_DIR / "feature_medians.json"
    if not defaults_path.exists():
        raise HTTPException(status_code=404, detail="Run feature_medians generation first.")
    with open(defaults_path) as f:
        data = json.load(f)

    # ── Compute top 5 by SHAP importance (same logic as /top-features) ───────
    try:
        import shap as _shap
        X_all, _ = load_dataset()
        X_s  = selector.transform(X_all.sample(100, random_state=42))
        X_sc = scaler.transform(X_s)
        sv   = _shap.TreeExplainer(model).shap_values(X_sc)
        if isinstance(sv, list):
            sv = sv[1]
        elif hasattr(sv, "ndim") and sv.ndim == 3:
            sv = sv[:, :, 1]
        importance  = np.abs(sv).mean(axis=0)
        top5_idx    = np.argsort(importance)[::-1][:5]
        top5_names  = [feature_names[i] for i in top5_idx]
    except Exception:
        # Fallback to first 5 selected features if SHAP fails
        top5_names = feature_names[:5]
    # ─────────────────────────────────────────────────────────────────────────

    # Compute min/max from the dataset for the top 5 features
    try:
        X_all2, _ = load_dataset()
        stats = X_all2[top5_names].agg(["min", "max"]).to_dict()
    except Exception:
        stats = {}

    friendly = {
        "maxIntensity":       {"label": "Max Intensity (maxIntensity)",             "tooltip": "Maximum vocal intensity (loudness) of the speech signal"},
        "f2":                 {"label": "Formant F2 Hz (f2)",                       "tooltip": "Second formant frequency — related to tongue position during speech"},
        "mean_MFCC_2nd_coef": {"label": "MFCC Coefficient 2 (mean_MFCC_2nd_coef)", "tooltip": "2nd Mel-frequency cepstral coefficient — captures spectral shape of voice"},
        "mean_MFCC_3rd_coef": {"label": "MFCC Coefficient 3 (mean_MFCC_3rd_coef)", "tooltip": "3rd MFCC — reflects fine spectral detail of vocal tract"},
        "mean_MFCC_6th_coef": {"label": "MFCC Coefficient 6 (mean_MFCC_6th_coef)", "tooltip": "6th MFCC — captures higher-order spectral variation in speech"},
    }

    top5 = []
    for name in top5_names:
        if name not in data["medians"]:
            continue
        feat_stats = stats.get(name, {})
        top5.append({
            "name":    name,
            "label":   friendly.get(name, {}).get("label", f"{name} ({name})"),
            "tooltip": friendly.get(name, {}).get("tooltip", f"Speech biomarker: {name}"),
            "median":  round(data["medians"][name], 4),
            "min":     round(float(feat_stats.get("min", 0)), 4),
            "max":     round(float(feat_stats.get("max", 0)), 4),
        })

    return {
        "top5":    top5,
        "columns": data["columns"],
        "medians": data["medians"],
    }


@app.get("/model-comparison")
def model_comparison():
    """
    Latest metrics per model from MLflow (newest run per model name), sorted by
    roc_auc descending. ``selected`` follows ``src.model_selection``: prefer
    interpretable models (XGBoost, Random Forest, Decision Tree) ranked by
    0.6*roc_auc + 0.4*macro_f1, else fall back with a penalty on non-interpretable models.
    """
    from src.mlflow_comparison import fetch_model_comparison_from_mlflow

    try:
        models = fetch_model_comparison_from_mlflow()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception("MLflow model comparison failed")
        raise HTTPException(
            status_code=503,
            detail=f"Could not load metrics from MLflow: {e}",
        ) from e
    if not models:
        raise HTTPException(
            status_code=404,
            detail=(
                "No model runs with accuracy, macro_f1, and roc_auc found in MLflow. "
                "Run training to log metrics."
            ),
        )
    return {"models": models}


@app.get("/drift-status")
def drift_status():
    """
    Return drift summary and per-feature details from the last drift_check.py run.
    Reads monitoring/drift_summary.txt and monitoring/drift_feature_details.csv.
    """
    import csv
    monitoring_dir = Path(__file__).resolve().parent.parent / "monitoring"
    summary_path = monitoring_dir / "drift_summary.txt"
    details_path = monitoring_dir / "drift_feature_details.csv"

    logger.info(f"Drift summary path: {summary_path} exists={summary_path.exists()}")
    logger.info(f"Drift details path: {details_path} exists={details_path.exists()}")

    if not summary_path.exists() or not details_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Drift data not found at {monitoring_dir}. Run `python monitoring/drift_check.py` first.",
        )

    # Parse summary
    summary = {}
    try:
        import re
        text = summary_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            m = re.match(r"Total Features\s*:\s*(\d+)", line)
            if m:
                summary["total_features"] = int(m.group(1)); continue
            m = re.match(r"Drifted Features\s*:\s*(\d+)", line)
            if m:
                summary["drifted_count"] = int(m.group(1)); continue
            m = re.match(r"Drift Percentage\s*:\s*([\d.]+)", line)
            if m:
                summary["drift_pct"] = float(m.group(1)); continue
            m = re.match(r"Drift Severity\s*:\s*(.+)", line)
            if m:
                summary["severity"] = m.group(1).strip(); continue
            m = re.match(r"Retraining Recommended\s*:\s*(.+)", line)
            if m:
                summary["retrain"] = m.group(1).strip(); continue
            m = re.match(r"Retraining Reason\s*:\s*(.+)", line)
            if m:
                summary["retrain_reason"] = m.group(1).strip(); continue
            m = re.match(r"Generated\s*:\s*(.+)", line)
            if m:
                summary["generated_at"] = m.group(1).strip(); continue
            if "[NOTE]" in line:
                summary["note"] = line.replace("[NOTE]", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse drift summary: {e}")

    # Parse feature details
    features = []
    try:
        with open(details_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                features.append({
                    "feature":   row["feature"],
                    "ks_stat":   float(row.get("ks_stat", 0)),
                    "p_value":   float(row["p_value"]),
                    "p_display": row.get("p_display", str(row["p_value"])),
                    "drifted":   row["drifted"].strip().lower() == "true",
                    "important": row.get("important", "false").strip().lower() == "true",
                })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse drift details: {e}")

    return {
        "summary": summary,
        "features": features,
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model":        type(model).__name__,
        "explainer":    type(explainer).__name__,
        "model_loaded": model is not None,
    }


@app.get("/top-features")
def top_features():
    """Return the top 5 globally important features from the saved SHAP analysis."""
    try:
        import shap as _shap
        import joblib as _joblib
        from src.config import load_dataset as _load

        _model    = _joblib.load(MODEL_PATH)
        _scaler   = _joblib.load(SCALER_PATH)
        _selector = _joblib.load(SELECTOR_PATH)
        _fnames   = _joblib.load(FEATURE_NAMES_PATH)

        X_all, _ = _load()
        X_s   = _selector.transform(X_all.sample(100, random_state=42))
        X_sc  = _scaler.transform(X_s)

        _exp  = _shap.TreeExplainer(_model)
        sv    = _exp.shap_values(X_sc)
        if isinstance(sv, list):
            sv = sv[1]
        elif hasattr(sv, "ndim") and sv.ndim == 3:
            sv = sv[:, :, 1]

        importance = np.abs(sv).mean(axis=0)
        top5_idx   = np.argsort(importance)[::-1][:5]
        return {
            "top_features": [
                {"rank": int(r+1), "name": _fnames[i], "importance": float(importance[i])}
                for r, i in enumerate(top5_idx)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── A2A Agent Card ────────────────────────────────────────────────────────────
# Standard A2A discovery endpoint required by Prompt Opinion Platform
# Served at /.well-known/agent-card.json

@app.get("/.well-known/agent-card.json", include_in_schema=False)
async def agent_card(request: Request):
    base_url = str(request.base_url).rstrip("/")
    return {
        "name": "NeuraLynk_AI",
        "description": (
            "Interoperable healthcare AI agent for explainable Parkinson's disease "
            "screening from speech biomarkers. Produces predictions, SHAP explanations, "
            "LLM clinical summaries, and FHIR R4 DiagnosticReports in a single API call."
        ),
        "version": "1.0.0-hackathon",
        "url": base_url,
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json", "application/fhir+json"],
        "skills": [
            {
                "id": "parkinson-speech-screening",
                "name": "Parkinson's Speech Screening",
                "description": (
                    "Accepts 753 vocal biomarker features and runs a three-agent pipeline: "
                    "XGBoost prediction + SHAP explanation, LLM clinical summary, "
                    "and FHIR R4 DiagnosticReport generation."
                ),
                "tags": [
                    "healthcare", "parkinson", "speech", "FHIR", "SHAP",
                    "explainability", "clinical-ai"
                ],
                "examples": [
                    "Screen a patient's speech biomarkers for Parkinson's disease indicators",
                    "Generate a FHIR DiagnosticReport from vocal feature analysis",
                    "Explain which speech biomarkers contributed to the screening result",
                ],
                "inputModes": ["application/json"],
                "outputModes": ["application/json", "application/fhir+json"],
            }
        ],
        "endpoints": {
            "screen": f"{base_url}/agent/screen",
            "report": f"{base_url}/agent/report/{{session_id}}",
            "health": f"{base_url}/agent/health",
            "schema": f"{base_url}/agent/schema",
        },
    }


# ── Healthcare AI Agent integration ──────────────────────────────────────────
# These two lines are the only change to main.py required by the agent extension.
# The agent router is mounted AFTER all existing routes so nothing is shadowed.
from api.agent import agent_router, init_agent
init_agent(model, scaler, selector, feature_names, column_order, explainer)
app.include_router(agent_router, prefix="/agent")

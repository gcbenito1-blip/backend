from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import numpy as np
import pandas as pd
from io import StringIO, BytesIO
from scipy.stats import norm
import os
import math
import warnings
import joblib

warnings.filterwarnings('ignore')

app = Flask(__name__)
# Configure CORS properly
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Specify your frontend origin in production
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False,  # Set to True if using cookies/auth
        "max_age": 3600
    }
})

# Add favicon route to prevent 404 errors
@app.route('/favicon.ico')
def favicon():
    # Return empty response with 204 No Content
    return Response('', status=204, mimetype='image/x-icon')

# ===========================================================================
# LOAD ARTIFACT
# ===========================================================================
ARTIFACT_PATH = "best_model.joblib"

# Subject aggregation map (must match pipeline)
SUBJECT_AGGREGATE_MAP = {
    "Filipino_avg": ["Filipino 1", "Filipino 2", "Filipino 3", "Filipino 4", "Filipino 5"],
    "English_avg":  ["English 1",  "English 2",  "English 3",  "English 4",  "English 5"],
    "Math_avg":     ["Math 1",     "Math 2",     "Math 3",     "Math 4",     "Math 5"],
    "AralPan_avg":  ["Aral Pan 1", "Aral Pan 2", "Aral Pan 3", "Aral Pan 4", "Aral Pan 5"],
    "Science_avg":  ["Science 3",  "Science 4",  "Science 5"],
}

SUBJECT_AVG_COLS = list(SUBJECT_AGGREGATE_MAP.keys())


def serialize_outputs(data):
    if data is None:
        return None
    return {
        "y_true": data["y_true"].tolist(),
        "y_pred": data["y_pred"].tolist(),
        "y_true_cat": data.get("y_true_cat").tolist() if data.get("y_true_cat") is not None else None,
        "y_pred_cat": data.get("y_pred_cat").tolist() if data.get("y_pred_cat") is not None else None,
    }

def _fallback_bands():
    return [
        (90, float('inf'), 4, "Highly Proficient",  "90–100", "#22c55e"),
        (75, 90,           3, "Proficient",          "75–89",  "#84cc16"),
        (50, 75,           2, "Nearly Proficient",   "50–74",  "#f59e0b"),
        (25, 50,           1, "Low Proficient",      "25–49",  "#f97316"),
        (0,  25,           0, "Not Proficient",      "0–24",   "#ef4444"),
    ]

try:
    try:
        import shap as _shap
        artifact = joblib.load(ARTIFACT_PATH)
    except ImportError:
        import pickle

        class _ShapStub(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith("shap"):
                    return type(f"_Shap_{name}", (), {})
                return super().find_class(module, name)

        with open(ARTIFACT_PATH, "rb") as f:
            artifact = _ShapStub(f).load()

    if not isinstance(artifact, dict):
        raise ValueError("Artifact is not a dict — retrain with the updated pipeline.")

    model                = artifact["model"]
    model_name           = artifact.get("model_name", "Unknown")
    residual_std         = artifact.get("residual_std")
    metrics              = artifact.get("metrics", [])
    features             = artifact.get("features", [])
    transformed_features = artifact.get("transformed_features", [])
    feature_importance   = artifact.get("feature_importance", {})
    shap_explainer       = artifact.get("shap_explainer")
    per_model_outputs    = artifact.get("per_model_outputs", {})
    school_report_df     = artifact.get("school_report")
    test_results_df      = artifact.get("test_results")
    school_test          = artifact.get("school_test")
    learner_test         = artifact.get("learner_test")
    y_test               = artifact.get("y_test")

    # Z-score params from training
    zscore_params        = artifact.get("zscore_params", {})
    zscore_applied       = artifact.get("zscore_applied", False)

    PROFICIENCY_BANDS  = artifact.get("proficiency_bands") or _fallback_bands()
    PROFICIENCY_LABELS = artifact.get("proficiency_labels") or {b[2]: b[3] for b in PROFICIENCY_BANDS}
    PROFICIENCY_RANGES = artifact.get("proficiency_ranges") or {b[2]: b[4] for b in PROFICIENCY_BANDS}
    PROFICIENCY_COLORS = artifact.get("proficiency_colors") or {b[2]: b[5] for b in PROFICIENCY_BANDS}

    print(f"[OK] Model loaded: {model_name}")
    print(f"[OK] Z-score applied during training: {zscore_applied}")
    if zscore_applied and zscore_params:
        print(f"[OK] Z-score params loaded: {len(zscore_params)} cohort-column entries")
    if school_report_df is not None:
        print(f"[OK] School report available: {len(school_report_df)} schools")
    if test_results_df is not None:
        if "Pass_Probability" not in test_results_df.columns:
            test_results_df["Pass_Probability"] = test_results_df["Predicted_MPS"].apply(
                lambda s: get_pass_probability(s, residual_std))

except Exception as e:
    print(f"[WARN] Could not load artifact: {e} — running in degraded mode")
    model = shap_explainer = residual_std = None
    model_name           = "Unavailable"
    metrics              = []
    features             = []
    transformed_features = []
    feature_importance   = {}
    school_report_df     = None
    test_results_df      = None
    school_test          = None
    learner_test         = None
    y_test               = None
    zscore_params        = {}
    zscore_applied       = False
    PROFICIENCY_BANDS    = _fallback_bands()
    PROFICIENCY_LABELS   = {b[2]: b[3] for b in PROFICIENCY_BANDS}
    PROFICIENCY_RANGES   = {b[2]: b[4] for b in PROFICIENCY_BANDS}
    PROFICIENCY_COLORS   = {b[2]: b[5] for b in PROFICIENCY_BANDS}


# ===========================================================================
# GUARDS
# ===========================================================================
def _require_model():
    if model is None:
        return jsonify({"error": "Model not loaded. Ensure best_model.joblib exists."}), 503
    return None

def _require_shap():
    if shap_explainer is None:
        return jsonify({"error": "SHAP explainer not available in this artifact."}), 503
    return None

# ===========================================================================
# BUILD DATAFRAME WITH AGGREGATION + Z-SCORE
# ===========================================================================
def _build_df(data, feature_list):
    """
    Build a DataFrame ready for model inference:
      1. Aggregate quarterly subject cols → subject avg cols (same as pipeline)
      2. Apply z-score using train-derived params (fallback = mean across cohorts)
      3. Select only the features the model expects
    """
    rows = data if isinstance(data, list) else [data]
    df = pd.DataFrame(rows)

    # Step 1: aggregate quarterly → subject avgs
    for new_col, src_cols in SUBJECT_AGGREGATE_MAP.items():
        present = [c for c in src_cols if c in df.columns]
        if present:
            df[new_col] = df[present].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # Step 2: apply z-score with train params (fallback = mean of all cohort params)
    if zscore_applied and zscore_params:
        for col in SUBJECT_AVG_COLS:
            if col not in df.columns:
                continue
            all_means = [p["mean"] for k, p in zscore_params.items() if k[1] == col]
            all_stds  = [p["std"]  for k, p in zscore_params.items() if k[1] == col]
            if not all_means:
                continue
            mu = float(np.mean(all_means))
            sd = float(np.mean(all_stds)) if float(np.mean(all_stds)) > 0 else 1.0
            df[col] = (pd.to_numeric(df[col], errors="coerce") - mu) / sd

    # Step 3: select only model features
    if feature_list:
        model_features = [f for f in feature_list if f not in ("learnerID", "School", "Section")]
        return pd.DataFrame([
            {k: row.get(k) for k in model_features}
            for row in df.to_dict("records")
        ])

    return df.drop(columns=["learnerID", "School", "Section"], errors="ignore")


# ===========================================================================
# PROFICIENCY HELPERS
# ===========================================================================
def encode_proficiency(score):
    for lower, upper, code, *_ in PROFICIENCY_BANDS:
        if score >= lower and (upper == float('inf') or score < upper):
            return code
    return 0

def get_proficiency_meta(score):
    code = encode_proficiency(score)
    return {
        "code":  code,
        "label": PROFICIENCY_LABELS[code],
        "range": PROFICIENCY_RANGES[code],
        "color": PROFICIENCY_COLORS[code],
    }

def get_proficiency_probabilities(pred_score, std):
    """P(a <= Y < b) per band via normal CDF. Values sum to 1.0."""
    out = []
    for lower, upper, code, label, rng, color in PROFICIENCY_BANDS:
        p_upper = norm.cdf(upper, loc=pred_score, scale=std) if upper != float('inf') else 1.0
        p_lower = norm.cdf(lower, loc=pred_score, scale=std)
        out.append({
            "code":        code,
            "label":       label,
            "range":       rng,
            "color":       color,
            "probability": round(float(p_upper - p_lower), 4),
        })
    return out

def get_pass_probability(pred_score, std, threshold=75):
    """P(Y >= threshold) using normal CDF."""
    if std is None:
        return None
    return round(float(1.0 - norm.cdf(threshold, loc=pred_score, scale=std)), 4)

def _prediction_payload(p, row_data):
    """Core prediction block shared by all routes."""
    prof      = get_proficiency_meta(p)
    breakdown = get_proficiency_probabilities(p, residual_std) if residual_std else None
    top_band  = max(breakdown, key=lambda b: b["probability"]) if breakdown else None
    pass_prob = get_pass_probability(p, residual_std) if residual_std else None
    return {
        "learnerID":             row_data.get("learnerID"),
        "School":                row_data.get("School"),
        "Section":               row_data.get("Section"),
        "prediction":            round(float(p), 4),
        "proficiency":           prof,
        "top_probable_band":     top_band,
        "probability_breakdown": breakdown,
        "pass_probability":      pass_prob,
    }


# ===========================================================================
# SHAP HELPER
# ===========================================================================
def get_shap_explanation(row_df):
    """
    Per-feature SHAP values for a single preprocessed row.
    Positive = pushed score UP, negative = pushed score DOWN.
    base_value + sum(shap_values) == predicted score exactly.
    """
    prep          = model.named_steps['prep']
    X_transformed = prep.transform(row_df)
    shap_values   = shap_explainer.shap_values(X_transformed)
    values        = shap_values[0] if hasattr(shap_values[0], '__len__') else shap_values

    named = [
        {
            "feature":    transformed_features[i] if i < len(transformed_features) else f"feature_{i}",
            "shap_value": round(float(values[i]), 4),
            "direction":  "positive" if values[i] >= 0 else "negative",
        }
        for i in range(len(values))
    ]
    named.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    expected = shap_explainer.expected_value
    return {
        "base_value":  round(float(expected[0] if hasattr(expected, '__len__') else expected), 4),
        "top_drivers": named[:5],
        "features":    named,
    }


# ===========================================================================
# SCHOOL REPORT GENERATOR
# ===========================================================================
def generate_school_report(y_true_arr, y_pred_arr, school_arr):
    df = pd.DataFrame({
        "School":        school_arr,
        "Actual_MPS":    y_true_arr,
        "Predicted_MPS": y_pred_arr,
    })
    df["Difference"] = df["Predicted_MPS"] - df["Actual_MPS"]

    def get_band(mps):
        for lower, upper, code, *_ in PROFICIENCY_BANDS:
            if mps >= lower and (upper == float('inf') or mps < upper):
                return code
        return 0

    band_labels = [b[3] for b in PROFICIENCY_BANDS]

    rows = []
    for school, grp in df.groupby("School"):
        n           = len(grp)
        avg_actual  = float(grp["Actual_MPS"].mean())
        avg_pred    = float(grp["Predicted_MPS"].mean())
        bias        = float(avg_pred - avg_actual)
        mae         = float(grp["Difference"].abs().mean())

        actual_counts = grp["Actual_MPS"].apply(get_band).value_counts()
        pred_counts   = grp["Predicted_MPS"].apply(get_band).value_counts()

        row = {
            "School":            school,
            "Student_Count":     int(n),
            "Avg_Actual_MPS":    avg_actual,
            "Avg_Predicted_MPS": avg_pred,
            "Avg_Bias":          bias,
            "MAE":               mae,
        }
        for band in band_labels:
            pct = (actual_counts.get(band, 0) / n * 100) if n > 0 else 0.0
            row[f"Actual_{band}"] = round(pct, 2)
        for band in band_labels:
            pct = (pred_counts.get(band, 0) / n * 100) if n > 0 else 0.0
            row[f"Pred_{band}"] = round(pct, 2)

        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values("School").reset_index(drop=True)
    return result_df


def generate_test_results(y_true_arr, y_pred_arr, school_arr, learner_arr):
    df = pd.DataFrame({
        "learnerID":       learner_arr,
        "School":          school_arr,
        "Actual_MPS":      y_true_arr,
        "Predicted_MPS":   y_pred_arr,
        "Difference":      y_pred_arr - y_true_arr,
        "Error_Magnitude": np.abs(y_true_arr - y_pred_arr),
    })
    df["Actual_Proficiency"]    = df["Actual_MPS"].apply(
        lambda s: PROFICIENCY_LABELS.get(encode_proficiency(s), "Unknown"))
    df["Predicted_Proficiency"] = df["Predicted_MPS"].apply(
        lambda s: PROFICIENCY_LABELS.get(encode_proficiency(s), "Unknown"))
    df["Pass_Probability"] = df["Predicted_MPS"].apply(
        lambda s: get_pass_probability(s, residual_std))
    return df[[
        "learnerID", "School", "Actual_MPS", "Predicted_MPS",
        "Difference", "Actual_Proficiency", "Predicted_Proficiency", "Pass_Probability", "Error_Magnitude"
    ]].sort_values(["School", "learnerID"]).reset_index(drop=True)


# ===========================================================================
# ROUTES — INFO
# ===========================================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok" if model is not None else "degraded",
        "model":        model_name,
        "model_loaded": model is not None,
        "shap_ready":   shap_explainer is not None,
    })

@app.route("/model-predict", methods=["GET"])
def model_predict():
    return jsonify({
        "linear":       serialize_outputs(per_model_outputs.get("Linear")),
        "lasso":        serialize_outputs(per_model_outputs.get("Lasso")),
        "decisionTree": serialize_outputs(per_model_outputs.get("DecisionTree")),
        "randomForest": serialize_outputs(per_model_outputs.get("RandomForest")),
        "gradientBoost":serialize_outputs(per_model_outputs.get("GradientBoosting")),
    })

@app.route("/proficiency-labels", methods=["GET"])
def proficiency_labels_route():
    return jsonify({
        "bands": [
            {"code": b[2], "label": b[3], "range": b[4], "color": b[5]}
            for b in PROFICIENCY_BANDS
        ]
    })

@app.route("/metrics", methods=["GET"])
def get_metrics():
    if isinstance(metrics, list) and metrics:
        return jsonify(max(metrics, key=lambda m: m.get("R2", -float("inf"))))
    return jsonify(metrics)

@app.route("/all-metrics", methods=["GET"])
def get_all_metrics():
    return jsonify({"models": metrics})

@app.route("/feature-importance", methods=["GET"])
def get_fi():
    return jsonify(feature_importance)


# ===========================================================================
# ROUTES — EXPLAIN  (prediction + SHAP in one call)
# ===========================================================================
@app.route("/explain", methods=["POST"])
def explain():
    err = _require_model() or _require_shap()
    if err: return err

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty input"}), 400

        df      = _build_df(data, features)
        pred    = float(model.predict(df)[0])
        payload = _prediction_payload(pred, data)
        payload["explanation"] = get_shap_explanation(df)
        return jsonify(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/explain-batch", methods=["POST"])
def explain_batch():
    err = _require_model() or _require_shap()
    if err: return err

    try:
        data = request.get_json()
        if not isinstance(data, list):
            return jsonify({"error": "Expected a JSON array"}), 400

        df    = _build_df(data, features)
        preds = model.predict(df)

        results       = []
        label_summary = {v: 0 for v in PROFICIENCY_LABELS.values()}

        for i, p in enumerate(preds):
            payload = _prediction_payload(p, data[i])
            payload["explanation"] = get_shap_explanation(df.iloc[[i]])
            label_summary[payload["proficiency"]["label"]] += 1
            results.append(payload)

        total        = len(results)
        distribution = [
            {
                "code":       b[2],
                "label":      b[3],
                "range":      b[4],
                "color":      b[5],
                "count":      label_summary[b[3]],
                "percentage": round(label_summary[b[3]] / total * 100, 2),
            }
            for b in PROFICIENCY_BANDS
        ]

        return jsonify({"results": results, "total": total, "distribution": distribution})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===========================================================================
# ROUTES — UPLOAD / ANALYZE
# ===========================================================================
@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file     = request.files["file"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    if file_ext not in (".csv"):
        return jsonify({"error": "Only CSV files are allowed"}), 400

    try:
        df = (pd.read_csv(StringIO(file.read().decode("utf-8")))
              if file_ext == ".csv"
              else pd.read_excel(BytesIO(file.read())))

        return jsonify({
            "columns":   df.columns.tolist(),
            "row_count": len(df),
            "preview":   df.to_dict(orient="records"),
        })
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {e}"}), 400


@app.route("/api/analyze", methods=["POST"])
def analyze_data():
    try:
        records = (request.json or {}).get("data", [])
        if not records:
            return jsonify({"error": "No records provided"}), 400

        df           = pd.DataFrame(records)
        row_count    = len(df)
        column_count = len(df.columns)

        # Proficiency distribution (only when target column is present)
        proficiency_distribution = None
        target_col = "MPS"
        if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            label_counts = {v: 0 for v in PROFICIENCY_LABELS.values()}
            for score in df[target_col].dropna():
                label_counts[PROFICIENCY_LABELS[encode_proficiency(score)]] += 1

            valid_total = int(df[target_col].notna().sum())
            proficiency_distribution = [
                {
                    "code":       b[2],
                    "label":      b[3],
                    "range":      b[4],
                    "color":      b[5],
                    "count":      label_counts[b[3]],
                    "percentage": round(label_counts[b[3]] / valid_total * 100, 2) if valid_total else 0,
                }
                for b in PROFICIENCY_BANDS
            ]

        # Per-column stats
        columns_info = []
        for col in df.columns:
            info = {
                "name":            col,
                "dtype":           str(df[col].dtype),
                "non_null_count":  int(df[col].count()),
                "null_count":      int(df[col].isnull().sum()),
                "null_percentage": round(float(df[col].isnull().mean() * 100), 2),
            }
            if col.upper() == "learnerID":
                total_count   = len(df)
                unique_count  = int(df[col].nunique())
                info["unique_count"]    = unique_count
                info["duplicate_count"] = total_count - unique_count
                info["is_id_column"]    = True
            elif pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
                info["statistics"] = {
                    "mean":   round(float(df[col].mean()),           4),
                    "std":    round(float(df[col].std()),            4),
                    "min":    round(float(df[col].min()),            4),
                    "max":    round(float(df[col].max()),            4),
                    "median": round(float(df[col].median()),         4),
                    "q1":     round(float(df[col].quantile(0.25)),   4),
                    "q3":     round(float(df[col].quantile(0.75)),   4),
                }
            else:
                vc = df[col].value_counts().head(10)
                info["value_counts"]  = {str(k): int(v) for k, v in vc.items()}
                info["unique_count"]  = int(df[col].nunique())
            columns_info.append(info)

        def sanitize_nan(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            if isinstance(obj, dict):
                return {k: sanitize_nan(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize_nan(v) for v in obj]
            return obj
        # Correlation matrix
        numeric_cols       = df.select_dtypes(include=[np.number]).columns.tolist()
        correlation_matrix = {}
        if len(numeric_cols) > 1:
            corr_df            = df[numeric_cols].corr()
            correlation_matrix = {c: corr_df[c].to_dict() for c in numeric_cols}

        # Missing value summary
        total_cells   = row_count * column_count
        missing_values = {
            "total_missing":        int(df.isnull().sum().sum()),
            "total_cells":          total_cells,
            "missing_percentage":   round(float(df.isnull().sum().sum() / total_cells * 100), 2) if total_cells else 0,
            "columns_with_missing": [
                {"column": col, "missing_count": int(df[col].isnull().sum())}
                for col in df.columns if df[col].isnull().any()
            ],
        }

        return jsonify(sanitize_nan({
            "row_count":                row_count,
            "column_count":             column_count,
            "columns":                  columns_info,
            "preview":                  df.head(10).to_dict(orient="records"),
            "correlation_matrix":       correlation_matrix,
            "missing_values":           missing_values,
            "proficiency_distribution": proficiency_distribution,
        }))

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {e}"}), 500


# ===========================================================================
# ROUTES — SCHOOL-LEVEL ANALYTICS
# ===========================================================================

@app.route("/api/sample-dataset/download", methods=["GET"])
def download_sample_dataset():

    role = request.args.get("role")          # "admin", "researcher", "teacher"
    view_mode = request.args.get("viewMode") # "admin" or "teacher"

    include_section = (role == "admin") or (role == "researcher" and view_mode == "admin")

    base_row = {
        "learnerID": "L001",
        "Gender": "M",
        "Age": 11,
        "Mother Tongue": "Tagalog",
        "Nutritional Status": "Normal",
        "Filipino 1": 90, "English 1": 90, "Math 1": 87, "Aral Pan 1": 91,
        "Filipino 2": 92, "English 2": 90, "Math 2": 90, "Aral Pan 2": 91,
        "Filipino 3": 92, "English 3": 92, "Math 3": 94, "Science 3": 93, "Aral Pan 3": 93,
        "Filipino 4": 91, "English 4": 93, "Math 4": 91, "Science 4": 92, "Aral Pan 4": 91,
        "Filipino 5": 91, "English 5": 92, "Math 5": 88, "Science 5": 93, "Aral Pan 5": 91
    }

    if include_section:
        base_row = {"Section": "A", **base_row}

    df = pd.DataFrame([base_row])
    csv = df.to_csv(index=False)

    filename = "sample_dataset_admin.csv" if include_section else "sample_dataset_teacher.csv"

    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.route("/api/school-metrics", methods=["GET"])
def get_school_metrics():
    model_key = request.args.get("model", "").lower()

    if not model_key:
        if school_report_df is None:
            return jsonify({"error": "School report not available. Retrain the model."}), 404
        data = school_report_df.sort_values("School").to_dict(orient="records")
        return jsonify({"schools": data, "count": len(data)})

    model_key_map = {
        "linear":          "Linear",
        "lasso":           "Lasso",
        "decisiontree":    "DecisionTree",
        "randomforest":    "RandomForest",
        "gradientboost":   "GradientBoosting",
        "gradientboosting":"GradientBoosting",
    }
    mapped_name = model_key_map.get(model_key, model_key)

    if per_model_outputs is None or mapped_name not in per_model_outputs:
        return jsonify({"error": f"Model '{mapped_name}' not found in per_model_outputs."}), 404

    if school_test is None or y_test is None:
        return jsonify({"error": "Test metadata not available in artifact."}), 404

    y_pred    = per_model_outputs[mapped_name]["y_pred"]
    school_df = generate_school_report(y_test, y_pred, school_test)
    data      = school_df.sort_values("School").to_dict(orient="records")
    return jsonify({"schools": data, "count": len(data), "model": mapped_name})


@app.route("/api/school-proficiency", methods=["GET"])
def get_school_proficiency():
    if school_report_df is None:
        return jsonify({"error": "School report not available. Retrain the model."}), 404

    bands  = [b[3] for b in PROFICIENCY_BANDS]
    result = []
    for _, row in school_report_df.sort_values("School").iterrows():
        result.append({
            "School":    row["School"],
            "Actual":    {band: row.get(f"Actual_{band}", 0) for band in bands},
            "Predicted": {band: row.get(f"Pred_{band}", 0)   for band in bands},
        })

    return jsonify({
        "schools":          result,
        "count":            len(result),
        "proficiency_bands": bands,
    })


@app.route("/api/school-mae", methods=["GET"])
def get_school_mae():
    model_key = request.args.get("model", "").lower()

    if not model_key:
        if school_report_df is None:
            return jsonify({"error": "School report not available. Retrain the model."}), 404
        df_sorted = school_report_df.sort_values("MAE", ascending=False)
        data = [
            {
                "School":            row["School"],
                "MAE":               float(row["MAE"]),
                "Student_Count":     int(row["Student_Count"]),
                "Avg_Actual_MPS":    float(row["Avg_Actual_MPS"]),
                "Avg_Predicted_MPS": float(row["Avg_Predicted_MPS"]),
            }
            for _, row in df_sorted.iterrows()
        ]
        return jsonify({"schools": data, "count": len(data)})

    model_key_map = {
        "linear":          "Linear",
        "lasso":           "Lasso",
        "decisiontree":    "DecisionTree",
        "randomforest":    "RandomForest",
        "gradientboost":   "GradientBoosting",
        "gradientboosting":"GradientBoosting",
    }
    mapped_name = model_key_map.get(model_key, model_key)

    if per_model_outputs is None or mapped_name not in per_model_outputs:
        return jsonify({"error": f"Model '{mapped_name}' not found."}), 404

    if school_test is None or y_test is None:
        return jsonify({"error": "Test metadata not available in artifact."}), 404

    y_pred    = per_model_outputs[mapped_name]["y_pred"]
    school_df = generate_school_report(y_test, y_pred, school_test)
    df_sorted = school_df.sort_values("MAE", ascending=False)
    data = [
        {
            "School":            row["School"],
            "MAE":               float(row["MAE"]),
            "Student_Count":     int(row["Student_Count"]),
            "Avg_Actual_MPS":    float(row["Avg_Actual_MPS"]),
            "Avg_Predicted_MPS": float(row["Avg_Predicted_MPS"]),
        }
        for _, row in df_sorted.iterrows()
    ]
    return jsonify({"schools": data, "count": len(data), "model": mapped_name})


@app.route("/api/test-results", methods=["GET"])
def get_test_results():
    model_key = request.args.get("model", "").lower()

    if not model_key:
        if test_results_df is None:
            return jsonify({
                "error": "Test results not available. Retrain the model with the updated pipeline."
            }), 404
        data = test_results_df.sort_values(["School", "learnerID"]).to_dict(orient="records")
        return jsonify({
            "results": data,
            "count":   len(data),
            "columns": ["learnerID", "School", "Actual_MPS", "Predicted_MPS",
                        "Difference", "Actual_Proficiency", "Predicted_Proficiency",
                        "Pass_Probability", "Error_Magnitude"]
        })

    model_key_map = {
        "linear":          "Linear",
        "lasso":           "Lasso",
        "decisiontree":    "DecisionTree",
        "randomforest":    "RandomForest",
        "gradientboost":   "GradientBoosting",
        "gradientboosting":"GradientBoosting",
    }
    mapped_name = model_key_map.get(model_key, model_key)

    if per_model_outputs is None or mapped_name not in per_model_outputs:
        return jsonify({"error": f"Model '{mapped_name}' not found."}), 404

    if school_test is None or learner_test is None or y_test is None:
        return jsonify({"error": "Test metadata not available in artifact."}), 404

    y_pred  = per_model_outputs[mapped_name]["y_pred"]
    test_df = generate_test_results(y_test, y_pred, school_test, learner_test)
    data    = test_df.sort_values(["School", "learnerID"]).to_dict(orient="records")
    return jsonify({
        "results": data,
        "count":   len(data),
        "columns": ["learnerID", "School", "Actual_MPS", "Predicted_MPS",
                    "Difference", "Actual_Proficiency", "Predicted_Proficiency",
                    "Pass_Probability", "Error_Magnitude"],
        "model":   mapped_name,
    })


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

"""Predicción con modelo entrenado o fallback heurístico."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from .config import MODEL_METADATA_PATH, MODEL_PATH
from .features import build_feature_row, safe_float

# Columnas esperadas por el pipeline entrenado.
# Importadas desde src.train para evitar drift entre train e inferencia.
# Si el modelo se vuelve a entrenar con otro set de columnas, estas se
# sincronizan automáticamente la próxima vez que se cargue ``predict``.
try:
    from .train import CATEGORICAL_COLS as _TRAIN_CAT, NUMERIC_COLS as _TRAIN_NUM, TEXT_COL as _TRAIN_TEXT
    EXPECTED_TEXT_COL = _TRAIN_TEXT
    EXPECTED_NUMERIC_COLS: List[str] = list(_TRAIN_NUM)
    EXPECTED_CATEGORICAL_COLS: List[str] = list(_TRAIN_CAT)
except Exception:
    # Fallback estático con el mismo set anti-leakage que train.py.
    EXPECTED_TEXT_COL = "text_total"
    EXPECTED_NUMERIC_COLS: List[str] = [
        "title_len", "description_len", "text_total_word_count",
        "cta_flag", "urgency_flag", "trust_flag", "promo_flag",
        "benefit_flag", "price_flag",
        "duration_seconds",
    ]
    EXPECTED_CATEGORICAL_COLS: List[str] = ["category_id"]

# Clipping de probabilidad para evitar absolutos como 0% o 100%.
# Los modelos calibrados pueden saturar en datasets pequeños o sintéticos.
PROBABILITY_FLOOR = 0.02
PROBABILITY_CEILING = 0.98


def _clip_probability(prob: float) -> float:
    """Recorta la probabilidad al rango [PROBABILITY_FLOOR, PROBABILITY_CEILING]."""
    if prob != prob:  # NaN
        return 0.5
    return max(PROBABILITY_FLOOR, min(PROBABILITY_CEILING, float(prob)))


def _patch_legacy_sklearn_model(obj):
    """Parche defensivo para modelos .joblib creados con otra versión de scikit-learn."""
    try:
        # Pipeline
        if hasattr(obj, "named_steps"):
            for step in obj.named_steps.values():
                _patch_legacy_sklearn_model(step)

        # ColumnTransformer
        if hasattr(obj, "transformers_"):
            for _, transformer, _ in obj.transformers_:
                _patch_legacy_sklearn_model(transformer)

        # OneVsRest / wrappers
        if hasattr(obj, "estimators_"):
            for est in obj.estimators_:
                _patch_legacy_sklearn_model(est)

        if hasattr(obj, "estimator"):
            _patch_legacy_sklearn_model(obj.estimator)

        # LogisticRegression antiguo
        if obj.__class__.__name__ == "LogisticRegression":
            if not hasattr(obj, "multi_class"):
                obj.multi_class = "auto"
            if not hasattr(obj, "n_jobs"):
                obj.n_jobs = None
            if not hasattr(obj, "l1_ratio"):
                obj.l1_ratio = None
    except Exception:
        pass

    return obj


def load_model():
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            model = _patch_legacy_sklearn_model(model)
            return model
        except Exception as exc:
            print(f"[MODEL LOAD ERROR] {exc}")
            return None
    return None


def load_model_metadata() -> Dict[str, Any]:
    if MODEL_METADATA_PATH.exists():
        try:
            return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def probability_to_level(prob: float) -> str:
    if prob >= 0.85:
        return "muy_alto"
    if prob >= 0.65:
        return "alto"
    if prob >= 0.40:
        return "medio"
    return "bajo"


def heuristic_probability(features: Dict[str, Any]) -> float:
    """Fallback si no existe modelo entrenado."""
    text_power = safe_float(features.get("text_power_score", 0))
    engagement = min(safe_float(features.get("engagement_rate", 0)) / 0.10, 1.0)
    views_per_day = min(safe_float(features.get("views_per_day", 0)) / 1000.0, 1.0)
    duration_fit = safe_float(features.get("duration_fit_score", 0.5))
    ocr_coverage = min(safe_float(features.get("ocr_frame_coverage", 0)), 1.0)
    prob = (
        0.18
        + 0.30 * text_power
        + 0.22 * engagement
        + 0.18 * views_per_day
        + 0.08 * duration_fit
        + 0.04 * ocr_coverage
    )
    return round(_clip_probability(prob), 4)


def _safe_sigmoid(x: float) -> float:
    """Sigmoide numéricamente estable usada para LinearSVC sin calibrar."""
    try:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _ensure_columns(features: Dict[str, Any]) -> pd.DataFrame:
    """Devuelve un DataFrame con todas las columnas que el pipeline necesita."""
    row: Dict[str, Any] = {EXPECTED_TEXT_COL: features.get(EXPECTED_TEXT_COL, "") or ""}
    for col in EXPECTED_NUMERIC_COLS:
        row[col] = safe_float(features.get(col, 0))
    for col in EXPECTED_CATEGORICAL_COLS:
        row[col] = str(features.get(col, "unknown") or "unknown")
    return pd.DataFrame([row])


def predict_from_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Predice probabilidad y nivel a partir de la fila de features."""
    model = load_model()
    if model is None:
        prob = heuristic_probability(features)
        level = probability_to_level(prob)
        return {
            "model_available": False,
            "probability": prob,
            "level": level,
            "model_warning": "No se encontró models/best_model.joblib; se usó scoring heurístico de respaldo.",
        }
    X = _ensure_columns(features)
    try:
        model = _patch_legacy_sklearn_model(model)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)

            # Binario normal: [[p0, p1]]
            if hasattr(proba, "shape") and proba.shape[1] >= 2:
                prob = float(proba[0][1])
            else:
                prob = float(proba[0][0])

        elif hasattr(model, "decision_function"):
            score = float(model.decision_function(X)[0])
            prob = _safe_sigmoid(score)

        else:
            pred = model.predict(X)[0]
            prob = float(pred)
        prob = _clip_probability(prob)
        return {
            "model_available": True,
            "probability": round(prob, 4),
            "level": probability_to_level(prob),
            "probability_clipped": True,
            "model_warning": "",
        }
    except Exception as exc:
        prob = heuristic_probability(features)
        return {
            "model_available": False,
            "probability": prob,
            "level": probability_to_level(prob),
            "model_warning": f"Falló inferencia del modelo entrenado; se usó fallback. Error: {exc}",
        }


def predict_video_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    features = build_feature_row(**payload)
    pred = predict_from_features(features)
    return {"features": features, "prediction": pred}

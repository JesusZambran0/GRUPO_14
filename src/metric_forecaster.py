"""Predicción/estimación de métricas esperadas tras potenciación.

Este módulo separa dos ideas:
1. Si existe un modelo de regresión entrenado, estima métricas a partir del
   mismo espacio de variables del clasificador.
2. Siempre genera una proyección de pauta parametrizada por CPM, presupuesto,
   retención y engagement actuales.

No promete ROI causal ni garantiza views reales: produce escenarios esperados
para apoyar decisión humana.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib

from .config import MODELS_DIR
from .features import safe_float

METRIC_MODEL_PATH = MODELS_DIR / "metric_regressors.joblib"


def _load_metric_model():
    if METRIC_MODEL_PATH.exists():
        try:
            return joblib.load(METRIC_MODEL_PATH)
        except Exception:
            return None
    return None


def predict_expected_metrics(features: Dict[str, Any]) -> Dict[str, Any]:
    """Predice métricas relativas si existe ``metric_regressors.joblib``.

    Fallback: usa tasas actuales y heurística conservadora.
    """
    model = _load_metric_model()
    if model is None:
        return _heuristic_metric_prediction(features, warning="No se encontró modelo de métricas; se usó estimación heurística.")
    try:
        import pandas as pd
        X = pd.DataFrame([{k: features.get(k, 0) for k in [
            "text_total", "title_len", "description_len", "text_total_word_count",
            "cta_flag", "urgency_flag", "trust_flag", "promo_flag", "benefit_flag",
            "price_flag", "duration_seconds", "category_id",
        ]}])
        pred = model.predict(X)[0]
        labels = getattr(model, "target_labels_", ["views_per_day", "engagement_rate", "like_rate", "comment_rate"])
        out = {label: float(value) for label, value in zip(labels, pred)}
        return {
            "metric_model_available": True,
            "expected_views_per_day": max(out.get("views_per_day", 0.0), 0.0),
            "expected_engagement_rate": _clip_rate(out.get("engagement_rate", 0.0)),
            "expected_like_rate": _clip_rate(out.get("like_rate", 0.0)),
            "expected_comment_rate": _clip_rate(out.get("comment_rate", 0.0)),
            "warning": "",
        }
    except Exception as exc:
        return _heuristic_metric_prediction(features, warning=f"Falló el modelo de métricas; fallback heurístico. Error: {type(exc).__name__}: {exc}")


def build_boost_projection(
    features: Dict[str, Any],
    performance_probability: float,
    budget: float,
    cpm: float,
    operational_metrics: Dict[str, Any] | None = None,
    policy_risk_level: str = "bajo",
) -> Dict[str, Any]:
    """Escenario esperado de resultados si se potencia el video.

    Usa presupuesto/CPM para estimar impresiones compradas y tasas esperadas
    derivadas del modelo de métricas/fallback. Penaliza por riesgo de política.
    """
    operational_metrics = operational_metrics or {}
    cpm = max(safe_float(cpm, 5), 0.1)
    budget = max(safe_float(budget, 0), 0.0)
    impressions = (budget / cpm) * 1000.0
    metric_pred = predict_expected_metrics(features)
    engagement_rate = metric_pred.get("expected_engagement_rate", 0.01)
    like_rate = metric_pred.get("expected_like_rate", 0.008)
    comment_rate = metric_pred.get("expected_comment_rate", 0.0008)
    retention = safe_float(operational_metrics.get("retention_rate", 0.35), 0.35)
    share_rate = safe_float(operational_metrics.get("share_rate", 0.0), 0.0)

    risk_penalty = {"bajo": 1.0, "medio": 0.82, "alto": 0.45, "revisión humana": 0.25}.get(policy_risk_level, 0.75)
    quality_multiplier = 0.75 + 0.5 * max(0.0, min(1.0, performance_probability))
    retention_multiplier = 0.75 + 0.5 * max(0.0, min(1.0, retention))
    projected_views = impressions * 0.35 * quality_multiplier * retention_multiplier * risk_penalty
    projected_likes = projected_views * max(like_rate, engagement_rate * 0.65)
    projected_comments = projected_views * max(comment_rate, engagement_rate * 0.05)
    projected_shares = projected_views * max(share_rate, engagement_rate * 0.04)

    return {
        "budget_usd": round(budget, 2),
        "cpm_usd": round(cpm, 2),
        "paid_impressions_estimate": round(impressions),
        "projected_views_after_boost": round(projected_views),
        "projected_likes_after_boost": round(projected_likes),
        "projected_comments_after_boost": round(projected_comments),
        "projected_shares_after_boost": round(projected_shares),
        "risk_penalty_applied": risk_penalty,
        "performance_multiplier": round(quality_multiplier, 3),
        "retention_multiplier": round(retention_multiplier, 3),
        "metric_prediction": metric_pred,
        "methodological_note": "Escenario parametrizado; no equivale a ROI causal ni garantiza resultados reales de campaña.",
    }


def _heuristic_metric_prediction(features: Dict[str, Any], warning: str = "") -> Dict[str, Any]:
    er = _clip_rate(features.get("engagement_rate", 0.015) or 0.015)
    lr = _clip_rate(features.get("like_rate", er * 0.75) or er * 0.75)
    cr = _clip_rate(features.get("comment_rate", er * 0.06) or er * 0.06)
    vpd = max(safe_float(features.get("views_per_day", 250), 250), 0)
    return {
        "metric_model_available": False,
        "expected_views_per_day": vpd,
        "expected_engagement_rate": er,
        "expected_like_rate": lr,
        "expected_comment_rate": cr,
        "warning": warning,
    }


def _clip_rate(x: Any) -> float:
    return max(0.0, min(0.35, safe_float(x, 0.0)))

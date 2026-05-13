"""Segundo modelo: XGBoost para pauta pagada.

Contrato metodológico:
1) La regresión logística clasifica si el video parece candidato a pauta.
2) Solo si la probabilidad supera 0.51, este módulo estima rendimiento pagado
   y CPM usando un XGBoost entrenado con datasets públicos de campañas pagadas.

Advertencia: el modelo XGBoost no es causal ni reemplaza datos reales de YouTube
Ads/Meta Ads del cliente. Es una estimación de soporte para priorización.
"""
from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from .config import MODELS_DIR
from .features import safe_float

XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_paid_ads.joblib"
XGBOOST_METADATA_PATH = MODELS_DIR / "xgboost_paid_ads_metadata.json"
GATE_THRESHOLD = 0.51

NICHE_KEYWORDS = {
    "educación": ["curso", "tutorial", "aprende", "clase", "guía", "guia", "tips", "enseño"],
    "retail/ecommerce": ["compra", "tienda", "producto", "envío", "envio", "descuento", "oferta", "precio"],
    "servicios": ["agenda", "reserva", "consulta", "servicio", "cotiza", "whatsapp", "asesoría", "asesoria"],
    "entretenimiento": ["meme", "humor", "comedia", "música", "musica", "cover", "show", "podcast"],
    "tecnología": ["app", "software", "ia", "inteligencia artificial", "automatiza", "digital", "tecnología", "tecnologia"],
    "salud/bienestar": ["salud", "bienestar", "fitness", "nutrición", "nutricion", "entrenamiento"],
}


def _load_metadata() -> Dict[str, Any]:
    if XGBOOST_METADATA_PATH.exists():
        try:
            return json.loads(XGBOOST_METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


@lru_cache(maxsize=1)
def _load_model():
    if not XGBOOST_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(XGBOOST_MODEL_PATH)
    except Exception:
        return None


def infer_ad_niche(title: str = "", description: str = "", transcript_text: str = "", ocr_text: str = "") -> str:
    """Identifica el nicho dominante con reglas transparentes."""
    text = " ".join([title or "", description or "", transcript_text or "", ocr_text or ""]).lower()
    scores = {niche: sum(1 for kw in kws if kw in text) for niche, kws in NICHE_KEYWORDS.items()}
    best, value = max(scores.items(), key=lambda kv: kv[1])
    return best if value > 0 else "general/branding"


def _season_campaign(month: int) -> str:
    if month in {3, 4, 5}: return "Spring"
    if month in {6, 7, 8}: return "Summer"
    if month in {9, 10, 11}: return "Fall"
    return "Winter"


def _ad_type_from_niche(niche: str, features: Dict[str, Any]) -> str:
    if niche in {"retail/ecommerce", "servicios"}:
        if safe_float(features.get("promo_flag", 0)) or safe_float(features.get("price_flag", 0)):
            return "Discount"
        return "Collection"
    if niche == "educación":
        return "Story"
    if niche == "entretenimiento":
        return "Video"
    return "Brand"


def _channel_from_niche(niche: str) -> str:
    # El dataset disponible es social paid ads. Instagram/Facebook se usan como proxy de pauta social.
    if niche in {"retail/ecommerce", "entretenimiento"}: return "Instagram"
    if niche in {"servicios", "educación"}: return "Facebook"
    return "Facebook"


def build_paid_ad_feature_row(features: Dict[str, Any], metadata: Dict[str, Any] | None = None, ops: Dict[str, Any] | None = None) -> Dict[str, Any]:
    metadata = metadata or {}
    ops = ops or {}
    title = metadata.get("title", "") or features.get("title", "")
    description = metadata.get("description", "") or features.get("description", "")
    niche = infer_ad_niche(title, description, features.get("transcript_text", ""), features.get("ocr_text", ""))
    now = datetime.utcnow()
    month = now.month
    weekday = now.isoweekday()
    engagement = safe_float(features.get("engagement_rate", 0.0))
    retention = safe_float(ops.get("retention_rate", 0.0))
    return {
        "source_dataset": "onyx_marketing_campaign",
        "campaign": _season_campaign(month),
        "channel": _channel_from_niche(niche),
        "device": "Mobile",
        "ad_type": _ad_type_from_niche(niche, features),
        "age": "unknown",
        "gender": "unknown",
        "city": "unknown",
        "campaign_numeric_id": 0.0,
        "ad_numeric_id": 0.0,
        "interest": 0.0,
        "month": float(month),
        "weekday": float(weekday),
        "ctr_prior": max(0.0, min(0.20, engagement)),
        "cvr_prior": max(0.0, min(0.30, retention * 0.12)),
        "ad_niche": niche,
    }


def predict_paid_ad_boost(
    *,
    features: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
    operational_metrics: Dict[str, Any] | None = None,
    model_probability: float,
    budget: float = 0.0,
    manual_cpm: float = 5.0,
) -> Dict[str, Any]:
    """Predice rendimiento pagado y CPM después del gate de regresión logística."""
    probability = safe_float(model_probability, 0.0)
    row = build_paid_ad_feature_row(features, metadata, operational_metrics)
    meta = _load_metadata()
    if probability < GATE_THRESHOLD:
        return {
            "eligible_for_paid_xgboost": False,
            "gate_threshold": GATE_THRESHOLD,
            "logistic_probability": round(probability, 4),
            "ad_niche": row["ad_niche"],
            "reason": "No pasa al segundo modelo porque la probabilidad de candidatura publicitaria es menor a 51%.",
            "model_metadata": meta,
        }
    model = _load_model()
    if model is None:
        cpm = max(safe_float(manual_cpm, 5.0), 0.1)
        return {
            "eligible_for_paid_xgboost": True,
            "model_available": False,
            "gate_threshold": GATE_THRESHOLD,
            "logistic_probability": round(probability, 4),
            "ad_niche": row["ad_niche"],
            "predicted_paid_performance_score": round(probability * 100, 2),
            "predicted_cpm": round(cpm, 4),
            "estimated_impressions_per_dollar": round(1000.0 / cpm, 2),
            "estimated_budget_impressions": round((safe_float(budget, 0.0) / cpm) * 1000.0, 0),
            "warning": "No se encontró el modelo XGBoost; se usó CPM manual como fallback.",
            "model_metadata": meta,
        }
    input_cols = meta.get("feature_columns") or [k for k in row.keys() if k != "ad_niche"]
    X = pd.DataFrame([{c: row.get(c, 0) for c in input_cols}])
    try:
        pred = model.predict(X)[0]
        targets = getattr(model, "target_labels_", meta.get("targets", ["paid_performance_score", "cpm"]))
        pred_map = {str(t): float(v) for t, v in zip(targets, pred)}
        score = max(0.0, min(100.0, pred_map.get("paid_performance_score", probability * 100)))
        cpm = max(0.1, pred_map.get("cpm", manual_cpm))
        return {
            "eligible_for_paid_xgboost": True,
            "model_available": True,
            "gate_threshold": GATE_THRESHOLD,
            "logistic_probability": round(probability, 4),
            "ad_niche": row["ad_niche"],
            "paid_ad_feature_row": row,
            "predicted_paid_performance_score": round(score, 2),
            "predicted_cpm": round(cpm, 4),
            "estimated_cost_per_1000_impressions": round(cpm, 4),
            "estimated_impressions_per_dollar": round(1000.0 / cpm, 2),
            "estimated_budget_impressions": round((safe_float(budget, 0.0) / cpm) * 1000.0, 0),
            "recommendation": _paid_recommendation(score, cpm),
            "model_metadata": meta,
            "methodological_warning": "Estimación con datasets públicos de paid ads; validar con campañas reales de la cuenta antes de decisiones de alto presupuesto.",
        }
    except Exception as exc:
        cpm = max(safe_float(manual_cpm, 5.0), 0.1)
        return {
            "eligible_for_paid_xgboost": True,
            "model_available": False,
            "gate_threshold": GATE_THRESHOLD,
            "logistic_probability": round(probability, 4),
            "ad_niche": row["ad_niche"],
            "predicted_paid_performance_score": round(probability * 100, 2),
            "predicted_cpm": round(cpm, 4),
            "estimated_impressions_per_dollar": round(1000.0 / cpm, 2),
            "warning": f"Falló la inferencia XGBoost; fallback manual. Error: {type(exc).__name__}: {exc}",
            "model_metadata": meta,
        }


def _paid_recommendation(score: float, cpm: float) -> str:
    if score >= 65 and cpm <= 12:
        return "Impulso viable: buen score estimado y CPM controlado."
    if score >= 50:
        return "Impulso moderado: lanzar con presupuesto pequeño y medir CTR/CPM reales."
    return "Riesgo de eficiencia: mejorar creativo/segmentación antes de invertir."


def render_paid_ad_boost_markdown(xgb: Dict[str, Any]) -> str:
    if not xgb:
        return ""
    if not xgb.get("eligible_for_paid_xgboost"):
        return f"""
### Segundo modelo XGBoost de pauta

El video **no pasó** al segundo modelo porque la regresión logística estimó una probabilidad de **{safe_float(xgb.get('logistic_probability')):.1%}**, menor al umbral de **{safe_float(xgb.get('gate_threshold'), GATE_THRESHOLD):.0%}**.

**Nicho detectado:** {xgb.get('ad_niche', '—')}  
**Motivo:** {xgb.get('reason', 'Gate no superado.')}
""".strip()
    return f"""
### Segundo modelo XGBoost de pauta

El video pasó el gate de regresión logística (**{safe_float(xgb.get('logistic_probability')):.1%} ≥ {safe_float(xgb.get('gate_threshold'), GATE_THRESHOLD):.0%}**) y fue evaluado para impulso pagado.

| Variable | Estimación |
|---|---:|
| Nicho detectado | {xgb.get('ad_niche', '—')} |
| Score pagado estimado | {safe_float(xgb.get('predicted_paid_performance_score')):.2f}/100 |
| CPM estimado | ${safe_float(xgb.get('predicted_cpm')):.2f} |
| Impresiones estimadas por dólar | {safe_float(xgb.get('estimated_impressions_per_dollar')):.2f} |
| Impresiones estimadas con presupuesto | {safe_float(xgb.get('estimated_budget_impressions')):,.0f} |

**Lectura:** {xgb.get('recommendation', 'Validar con prueba real de bajo presupuesto.')}  
**Advertencia:** {xgb.get('methodological_warning') or xgb.get('warning') or 'Usar como estimación, no como garantía de ROI.'}
""".strip()

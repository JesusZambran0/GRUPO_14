"""Explicabilidad básica del modelo, basada en reglas.

Para explicabilidad profunda (SHAP) ver ``notebooks/03_model_explainability.ipynb``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .features import safe_float


def simple_explanation(features: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Explicación local basada en variables principales del feature row."""
    positive: List[str] = []
    negative: List[str] = []

    if safe_float(features.get("text_power_score", 0)) >= 0.5:
        positive.append("Alta potencia textual: presencia de CTA, beneficio, urgencia o confianza.")
    else:
        negative.append("Potencia textual limitada; el mensaje puede requerir mayor claridad persuasiva.")

    if safe_float(features.get("engagement_rate", 0)) >= 0.05:
        positive.append("Engagement relativo favorable frente a las visualizaciones disponibles.")
    elif safe_float(features.get("views", 0)) > 0:
        negative.append("Engagement relativo bajo frente a las visualizaciones disponibles.")

    if safe_float(features.get("duration_fit_score", 0)) >= 0.8:
        positive.append("Duración compatible con formatos publicitarios cortos o de alta retención.")
    else:
        negative.append("Duración extensa o poco ajustada para una pauta inicial eficiente.")

    if safe_float(features.get("ocr_frame_coverage", 0)) > 0:
        positive.append("Texto visible detectado por OCR sobre los frames analizados.")
    if features.get("visual_text_density") == "alta":
        negative.append("Densidad alta de texto visual: puede saturar la atención del usuario.")

    if features.get("transcript_text"):
        positive.append("Se aprovechó la transcripción del audio para reforzar el análisis textual.")

    return {
        "explicacion_local": {
            "factores_favorables": positive,
            "factores_de_riesgo": negative,
            "resumen": (
                f"La predicción fue clasificada como {prediction.get('level')} "
                f"con probabilidad {prediction.get('probability')}."
            ),
        }
    }

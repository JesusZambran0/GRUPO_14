"""Genera los archivos JSON del demo_cache/ con el contrato completo nuevo.

Uso:
    python scripts/build_demo_cache.py

Estos demos se usan cuando el usuario elige un caso precalculado en la app y
permiten que la presentación funcione aunque fallen YouTube API, OCR, Whisper o LLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEMO_CACHE_DIR, ensure_dirs
from src.features import build_feature_row, metricas_block, ocr_block, transcripcion_block
from src.llm_analyzer import rule_based_recommendation
from src.policy_evaluator import evaluate_youtube_ad_policy_risk
from src.predict import predict_from_features
from src.recommender import build_final_recommendation


def build_case(
    *, name: str, title: str, description: str, transcript: str, ocr_text: str,
    category_id: str, duration_s: float, views: int, likes: int, comments: int,
    video_type: str,
    shares: int = 0, retention_rate: float = 0.0,
    cpm: float = 5.0, budget: float = 20.0,
    extra_ocr_metrics: dict | None = None,
) -> dict:
    extra = extra_ocr_metrics or {}
    features = build_feature_row(
        title=title, description=description, transcript_text=transcript, ocr_text=ocr_text,
        category_id=category_id, duration_seconds=duration_s,
        views=views, likes=likes, comments=comments, video_type=video_type, extra=extra,
    )
    pred = predict_from_features(features)

    # Políticas
    combined = " ".join([title, description, transcript]).strip()
    policy = evaluate_youtube_ad_policy_risk(combined)

    # Métricas operativas
    share_rate = (shares / views) if views > 0 else 0.0
    op_metrics = {
        "shares": shares,
        "share_rate": round(share_rate, 6),
        "retention_rate": retention_rate,
        "hours_since_publication": 24.0,
        "views_per_hour": round(views / 24.0, 4) if views else 0,
        "average_watch_time": 0.0,
        "followers_count": 0,
        "avg_channel_reach": 0,
    }

    rec = build_final_recommendation(
        pred, features, cpm=cpm, budget=budget,
        operational_metrics=op_metrics, policy_block=policy,
    )
    llm = rule_based_recommendation(features, rec, policy_block=policy)
    metricas = metricas_block(features)
    metricas["operational"] = op_metrics

    result = {
        "prediccion_rendimiento": rec["prediccion_rendimiento"],
        "probabilidad_rendimiento": rec["probabilidad_rendimiento"],
        "recomendacion_impulso": rec["recomendacion_impulso"],
        "requiere_ajustes": rec["requiere_ajustes"],
        "nivel_prioridad": rec["nivel_prioridad"],
        "alcance_estimado_por_dolar": rec["alcance_estimado_por_dolar"],
        "alcance_estimado_total": rec["alcance_estimado_total"],
        "metricas": metricas,
        "analisis_transcripcion": transcripcion_block(features, extra_status={"source": "demo_precalculado", "warning": ""}),
        "analisis_ocr": ocr_block(features, extra_status={"engine": "demo_precalculado", "warning": ""}),
        "analisis_llm": llm,
        "analisis_politicas": policy,
        "ajustes_sugeridos": rec["ajustes_sugeridos"],
        "justificacion": rec["justificacion"],
        "accion_final": rec["accion_final"],
        "score_hibrido": rec["score_hibrido"],
        "score_hibrido_detalle": rec["score_hibrido_detalle"],
        "policy_risk_level": rec["policy_risk_level"],
        "youtube_ad_status_estimate": rec["youtube_ad_status_estimate"],
        "cpm_estimado": rec["cpm_estimado"],
        "multiplicador_potencial": rec["multiplicador_potencial"],
        "alcance_base_por_dolar": rec["alcance_base_por_dolar"],
        "nota_metodologica": rec["nota_metodologica"],
        "warnings": ["Resultado precalculado en modo demo seguro."],
    }

    return {
        "name": name,
        "transcript_text": transcript,
        "ocr_text": ocr_text,
        "result_json": result,
        "metricas": metricas,
        "analisis_transcripcion": result["analisis_transcripcion"],
        "analisis_ocr": result["analisis_ocr"],
        "analisis_llm": result["analisis_llm"],
        "analisis_politicas": result["analisis_politicas"],
        "ajustes_sugeridos": result["ajustes_sugeridos"],
        "justificacion": result["justificacion"],
        "accion_final": result["accion_final"],
        "score_hibrido": result["score_hibrido"],
        "warnings": result["warnings"],
    }


def main() -> None:
    ensure_dirs()

    # 1) Video con narrador, mensaje claro → IMPULSAR
    case_narrador = build_case(
        name="video con narrador",
        title="Mejora tu productividad en 3 pasos prácticos",
        description="Guía práctica con casos reales para mejorar tu wifi. Suscríbete y descubre cómo optimizar.",
        transcript="En este video aprenderás tres cambios concretos para mejorar tu productividad: ordenar tus tareas, reducir distracciones y medir avances. Guarda este contenido y prueba los pasos hoy.",
        ocr_text="APRENDE FACIL CONFIGURA WIFI",
        category_id="28", duration_s=180,
        views=120000, likes=8400, comments=520, video_type="con narrador",
        shares=350, retention_rate=0.62,
        extra_ocr_metrics={
            "ocr_frame_count": 14, "ocr_frames_with_text": 9,
            "ocr_frame_coverage": round(9/14, 4), "visual_text_density": "media",
            "ocr_cta_flag": 1, "ocr_promo_flag": 0, "ocr_trust_flag": 0, "ocr_urgency_flag": 0,
            "ocr_excess_text_flag": 0,
        },
    )
    (DEMO_CACHE_DIR / "video_con_narrador.json").write_text(
        json.dumps(case_narrador, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2) Video sin narrador, saturado de texto visual → AJUSTAR ANTES DE IMPULSAR
    case_sin = build_case(
        name="video sin narrador (saturado)",
        title="Oferta exclusiva limitada solo hoy",
        description="Aprovecha la oferta exclusiva de hoy. Resultados garantizados con expertos certificados.",
        transcript="Oferta exclusiva solo hoy descuento limitado aprovecha resultados garantizados.",
        ocr_text=(
            "OFERTA EXCLUSIVA SOLO HOY DESCUENTO LIMITADO APROVECHA RESULTADOS GARANTIZADOS "
            "EXPERTOS CERTIFICADOS COMPRA AHORA SUSCRIBETE GRATIS PROMOCION ULTIMOS DIAS "
            "BENEFICIO COMPROBADO MEJORA TUS RESULTADOS CALIDAD VERIFICADA APRENDE FACIL "
            "DESCUBRE MAS REGISTRATE OFERTA UNICA NO TE LO PIERDAS LLAMA YA VISITA NUESTRO SITIO"
        ),
        category_id="22", duration_s=7,
        views=85000, likes=6400, comments=520, video_type="sin narrador",
        shares=180, retention_rate=0.45,
        extra_ocr_metrics={
            "ocr_frame_count": 10, "ocr_frames_with_text": 9,
            "ocr_frame_coverage": 0.9, "visual_text_density": "alta",
            "ocr_cta_flag": 1, "ocr_promo_flag": 1, "ocr_trust_flag": 1, "ocr_urgency_flag": 1,
            "ocr_excess_text_flag": 1,
        },
    )
    (DEMO_CACHE_DIR / "video_sin_narrador.json").write_text(
        json.dumps(case_sin, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3) Caso problemático (claims de salud) → REVISIÓN HUMANA por políticas
    case_ajustes = build_case(
        name="requiere ajustes (políticas)",
        title="Pierde 10 kilos en una semana garantizado",
        description="Dieta milagrosa con resultados garantizados al 100%. Sin esfuerzo.",
        transcript="Esta dieta milagrosa funciona sin fallar, garantizado 100%. Pierde peso rápido y sin esfuerzo.",
        ocr_text="DIETA MILAGROSA RESULTADOS GARANTIZADOS",
        category_id="26", duration_s=120,
        views=2400, likes=42, comments=8, video_type="con narrador",
        shares=5, retention_rate=0.30,
        extra_ocr_metrics={
            "ocr_frame_count": 15, "ocr_frames_with_text": 8,
            "ocr_frame_coverage": round(8/15, 4), "visual_text_density": "media",
            "ocr_cta_flag": 0, "ocr_promo_flag": 1, "ocr_trust_flag": 1, "ocr_urgency_flag": 0,
            "ocr_excess_text_flag": 0,
        },
    )
    (DEMO_CACHE_DIR / "video_requiere_ajustes.json").write_text(
        json.dumps(case_ajustes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Demo cache regenerado en:", DEMO_CACHE_DIR)
    for f in sorted(DEMO_CACHE_DIR.glob("*.json")):
        with open(f) as fp:
            d = json.load(fp)
        r = d["result_json"]
        print(f"  - {f.name}: accion={r['accion_final']}, policy={r['policy_risk_level']}, score={r['score_hibrido']}")


if __name__ == "__main__":
    main()

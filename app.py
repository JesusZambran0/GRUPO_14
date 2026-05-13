"""YouTube Boost AI — Demo Gradio.

Cambios en esta versión:
- Transcripción: video→WAV (ffmpeg) → Google Speech Recognition (sin key).
- OCR + análisis de frames: SIEMPRE juntos. Los 10 frames se extraen una sola
  vez y se usan para ambos.
- Análisis de sentimiento de comentarios (YouTube API opcional o lista manual).
- Botón «Rellenar con enlace» que autocompleta el formulario con la API de YouTube.
- LLM: Gemini (GEMINI_API_KEY) · modelo open source local opcional/Qwen LoRA · rules.
- Se eliminó Groq del selector.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from src.analytics_viz import create_analysis_charts
from src.comment_sentiment import (
    analyze_comments_sentiment,
    build_sentiment_markdown,
    create_sentiment_visuals,
    fetch_and_analyze_comments,
)
from src.channel_metrics import get_channel_reach_estimate
from src.config import DEFAULT_CPM, DEMO_CACHE_DIR, RUNTIME_CACHE_DIR, YOUTUBE_API_KEY, ensure_dirs
from src.exec_summary import build_executive_summary
from src.explain import simple_explanation
from src.features import (
    build_feature_row,
    metricas_block,
    ocr_block,
    safe_float,
    transcripcion_block,
)
from src.llm_analyzer import analyze_with_llm
from src.llm_provider import generate_recommendation_with_llm, interpret_ocr_text_with_llm
from src.metric_forecaster import build_boost_projection
from src.paid_ads_xgboost import predict_paid_ad_boost, render_paid_ad_boost_markdown
from src.ocr_video import extract_ocr_from_video
from src.policy_evaluator import evaluate_youtube_ad_policy_risk
from src.predict import predict_from_features
from src.recommender import build_final_recommendation
from src.script_analyzer import analyze_video_script
from src.transcription import resolve_transcript
from src.video_processing import summarize_video
from src.visual_composition import analyze_visual_composition
from src.youtube_api import extract_video_id, fetch_youtube_metadata
from src.youtube_downloader import download_youtube_video_360p

ensure_dirs()

# ---------------------------------------------------------------------------
# CSS dark
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root { --bg-0:#0b0d12;--bg-1:#11141c;--bg-2:#161a22;--bg-3:#1c2230;
  --border:#262b36;--text-strong:#f1f5f9;--text:#cbd5e1;--muted:#8b91a3;
  --violet:#a78bfa;--cyan:#22d3ee;--green:#22c55e;--yellow:#facc15;--red:#f87171; }
body,.gradio-container,gradio-app,.main,.wrap,.app,.block,.form,.panel {
  background:var(--bg-0) !important;color:var(--text) !important;
  font-family:-apple-system,BlinkMacSystemFont,"Inter","Segoe UI",Roboto,sans-serif !important; }
.yba-hero { background:linear-gradient(135deg,rgba(167,139,250,.18),rgba(34,211,238,.08),rgba(248,113,113,.10));
  border:1px solid var(--border);border-radius:16px;padding:22px 26px;margin-bottom:12px; }
.yba-hero h1{color:var(--text-strong);margin:0 0 8px;font-size:28px;letter-spacing:-.5px}
.yba-hero p{color:var(--muted);margin:0;font-size:14px;line-height:1.5}
.yba-pill{display:inline-block;padding:3px 10px;border-radius:99px;
  background:rgba(167,139,250,.2);color:var(--violet);font-size:11px;font-weight:600;
  letter-spacing:.5px;margin-right:6px;border:1px solid rgba(167,139,250,.4)}
.block,.gr-group,.gr-form,.gr-panel,.gr-box,.gr-accordion,.gr-tab-item {
  background:var(--bg-1) !important;border:1px solid var(--border) !important;border-radius:12px !important;}
input,textarea,select,.gr-textbox textarea,.gr-textbox input,.gr-dropdown,.gr-number input {
  background:var(--bg-2) !important;color:var(--text-strong) !important;
  border:1px solid var(--border) !important;border-radius:8px !important;}
input:focus,textarea:focus{outline:2px solid var(--violet) !important;border-color:var(--violet) !important;}
label,.label-wrap,.label-wrap span{color:var(--text) !important;font-weight:500 !important;}
h1,h2,h3,h4,h5,h6{color:var(--text-strong) !important;}
.gr-markdown,.prose,.markdown{background:transparent !important;color:var(--text) !important;}
.gr-markdown table{background:var(--bg-2) !important;border-color:var(--border) !important;}
.gr-markdown th,.gr-markdown td{border-color:var(--border) !important;color:var(--text) !important;padding:8px 12px !important;}
.gr-markdown th{background:var(--bg-3) !important;color:var(--text-strong) !important;font-weight:600}
.gr-markdown blockquote{border-left:3px solid var(--violet) !important;background:var(--bg-2) !important;padding:10px 14px !important;color:var(--muted) !important;}
.gr-markdown code{background:var(--bg-3) !important;color:var(--cyan) !important;padding:1px 6px !important;border-radius:4px !important;}
.gr-button-primary,button.lg.primary{
  background:linear-gradient(135deg,#a78bfa,#7c3aed) !important;color:white !important;
  border:0 !important;font-weight:600 !important;padding:12px 24px !important;
  font-size:15px !important;border-radius:10px !important;
  box-shadow:0 4px 18px -4px rgba(124,58,237,.6) !important;}
.gr-button-primary:hover{filter:brightness(1.08) !important;}
.gr-button{background:var(--bg-2) !important;color:var(--text) !important;border:1px solid var(--border) !important;}
.gr-tab-item,button[role=tab]{background:var(--bg-1) !important;color:var(--muted) !important;border:1px solid var(--border) !important;}
button[role=tab][aria-selected="true"]{background:var(--bg-3) !important;color:var(--violet) !important;}
.gr-json,pre,code{background:var(--bg-3) !important;color:var(--cyan) !important;border-color:var(--border) !important;}
.yba-result-card{border-radius:14px;padding:18px 22px;margin:8px 0;
  background:linear-gradient(135deg,rgba(167,139,250,.08),rgba(34,211,238,.04));border:1px solid var(--border);}
.gr-gallery{background:var(--bg-1) !important;border:1px solid var(--border) !important;}
.gr-image,.gr-video,.gr-file{background:var(--bg-2) !important;border:1px solid var(--border) !important;border-radius:10px !important;}
footer{display:none !important;}
"""

# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------
DEMO_CASES = {
    "Ninguno": None,
    "Demo: video con narrador": "video_con_narrador.json",
    "Demo: video sin narrador": "video_sin_narrador.json",
    "Demo: requiere ajustes":   "video_requiere_ajustes.json",
}


def _load_demo(case: str) -> Optional[Dict[str, Any]]:
    fname = DEMO_CASES.get(case)
    if not fname:
        return None
    p = DEMO_CACHE_DIR / fname
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def _coerce_video(video_file: Any) -> Optional[str]:
    if not video_file:
        return None
    if isinstance(video_file, str):
        return video_file
    if isinstance(video_file, dict):
        return video_file.get("path") or video_file.get("name") or video_file.get("orig_name")
    if isinstance(video_file, (list, tuple)) and video_file:
        return _coerce_video(video_file[0])
    return str(video_file)


def _derive_ops(views: float, shares: float, retention: float, hours: float) -> Dict[str, Any]:
    views_v  = max(safe_float(views), 0)
    shares_v = max(safe_float(shares), 0)
    hours_v  = max(safe_float(hours), 1.0)
    return {
        "shares": shares_v,
        "share_rate": round(shares_v / views_v, 6) if views_v else 0.0,
        "retention_rate": round(max(safe_float(retention), 0.0), 4),
        "hours_since_publication": hours_v,
        "published_age_label": _format_age_from_hours(hours_v),
        "views_per_hour": round(views_v / hours_v, 4),
    }


def _result_from_demo(demo: Dict[str, Any]) -> Dict[str, Any]:
    base = dict(demo.get("result_json", {}))
    base.setdefault("metricas",              demo.get("metricas", {}))
    base.setdefault("analisis_transcripcion",demo.get("analisis_transcripcion", {}))
    base.setdefault("analisis_ocr",          demo.get("analisis_ocr", {}))
    base.setdefault("analisis_llm",          demo.get("analisis_llm", {}))
    base.setdefault("analisis_politicas",    demo.get("analisis_politicas", {
        "policy_risk_level": "bajo", "policy_risk_categories": [],
        "policy_explanation": "Demo.", "youtube_ad_status_estimate": "apto",
    }))
    base.setdefault("ajustes_sugeridos",     demo.get("ajustes_sugeridos", []))
    base.setdefault("justificacion",         demo.get("justificacion", ""))
    base.setdefault("warnings",              demo.get("warnings", ["Demo precalculado."]))
    base.setdefault("accion_final",          demo.get("accion_final", "MONITOREAR"))
    base.setdefault("score_hibrido",         demo.get("score_hibrido", 0.5))
    base.setdefault("policy_risk_level",     base["analisis_politicas"].get("policy_risk_level", "bajo"))
    base.setdefault("youtube_ad_status_estimate", base["analisis_politicas"].get("youtube_ad_status_estimate", "apto"))
    return base


# ---------------------------------------------------------------------------
# Helpers narrativos y cálculo de horas desde publicación
# ---------------------------------------------------------------------------

def _hours_since_published(published_at: Optional[str]) -> Optional[float]:
    """Calcula horas desde publicación aceptando ISO, YYYYMMDD o timestamp."""
    if not published_at:
        return None
    raw = str(published_at).strip()
    try:
        if raw.isdigit() and len(raw) == 8:
            dt = datetime.strptime(raw, "%Y%m%d").replace(tzinfo=timezone.utc)
        elif raw.replace(".", "", 1).isdigit() and len(raw) in {10, 13}:
            ts = float(raw[:10])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        hours = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600
        return round(max(hours, 1.0), 2)
    except Exception:
        return None


def _format_age_from_hours(hours: Any) -> str:
    """Traduce horas a una lectura humana: horas, días, semanas, meses o años."""
    try:
        h = max(float(hours or 0), 0.0)
    except Exception:
        return "—"
    if h < 24:
        value = max(round(h, 1), 0.1)
        unit = "hora" if abs(value - 1) < 0.05 else "horas"
        return f"{value:g} {unit}"
    days = h / 24
    if days < 7:
        value = round(days, 1)
        unit = "día" if abs(value - 1) < 0.05 else "días"
        return f"{value:g} {unit}"
    weeks = days / 7
    if weeks < 4:
        value = round(weeks, 1)
        unit = "semana" if abs(value - 1) < 0.05 else "semanas"
        return f"{value:g} {unit}"
    months = days / 30.4375
    if months < 12:
        value = round(months, 1)
        unit = "mes" if abs(value - 1) < 0.05 else "meses"
        return f"{value:g} {unit}"
    years = days / 365.25
    value = round(years, 1)
    unit = "año" if abs(value - 1) < 0.05 else "años"
    return f"{value:g} {unit}"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "—"


def _fmt_num(value: Any) -> str:
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return "—"


def _fmt_float(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "—"


def _list_md(items: Any, empty: str = "No se detectaron elementos relevantes.") -> str:
    if not items:
        return f"- {empty}"
    if not isinstance(items, list):
        return f"- {items}"
    return "\n".join(f"- {x}" for x in items[:8]) if items else f"- {empty}"


def _first_ocr_candidate(ocr_features: Dict[str, Any]) -> str:
    """Toma el primer texto legible del OCR por frame para que el LLM lo corrija/interprete."""
    per_frame = ocr_features.get("ocr_per_frame") or []
    if isinstance(per_frame, list):
        for item in per_frame:
            lines = (item or {}).get("lines") or []
            if isinstance(lines, list):
                for line in lines:
                    clean = str(line or "").strip()
                    if clean and not clean.lower().startswith("[texto detectado"):
                        return clean
    return str(ocr_features.get("ocr_text", "") or "").strip()


def _apply_llm_ocr_interpretation(
    ocr_features: Dict[str, Any],
    *,
    title: str,
    description: str,
    transcript_text: str,
    mode: str,
) -> Dict[str, Any]:
    """Corrige OCR ruidoso con LLM y guarda texto original + interpretación.

    El OCR de frames suele tener errores de lectura. Esta función usa el primer
    resultado legible como ancla y pide al LLM una reconstrucción conservadora.
    Si Gemini/local falla, deja el OCR original intacto.
    """
    raw_text = str(ocr_features.get("ocr_text", "") or "").strip()
    first_text = _first_ocr_candidate(ocr_features)
    if not raw_text and not first_text:
        return ocr_features

    interpreted = interpret_ocr_text_with_llm(
        first_ocr_text=first_text or raw_text,
        full_ocr_text=raw_text,
        title=title,
        description=description,
        transcript_text=transcript_text,
        mode=mode,
    )

    corrected = str(interpreted.get("corrected_text") or "").strip()
    if corrected:
        ocr_features["ocr_text_raw"] = raw_text
        ocr_features["ocr_first_result"] = first_text
        ocr_features["ocr_text"] = corrected
        ocr_features["ocr_word_count"] = len(corrected.split())

    ocr_features["ocr_llm_source"] = interpreted.get("source", "rules")
    ocr_features["ocr_llm_warning"] = interpreted.get("warning", "")
    ocr_features["ocr_llm_meaning"] = interpreted.get("meaning", "")
    ocr_features["ocr_llm_confidence"] = interpreted.get("confidence", 0)
    ocr_features["ocr_llm_content_intent"] = interpreted.get("content_intent", "")
    ocr_features["ocr_llm_role"] = interpreted.get("ocr_role", "")
    return ocr_features


def _metrics_markdown(metricas: Dict[str, Any]) -> str:
    ops = metricas.get("operational", {}) or {}
    views = metricas.get("views", 0)
    likes = metricas.get("likes", 0)
    comments = metricas.get("comments", 0)
    engagement = float(metricas.get("engagement_rate", 0) or 0) * 100
    share_rate = float(ops.get("share_rate", 0) or 0) * 100
    retention = float(ops.get("retention_rate", 0) or 0) * 100
    hours_source = ops.get("hours_source", "manual")
    hours_note = "calculadas automáticamente desde la fecha de publicación de YouTube" if hours_source == "youtube_published_at" else "ingresadas manualmente o estimadas"
    return f"""
### Métricas del video

Este apartado cruza métricas públicas con señales operativas para medir tracción, velocidad y eficiencia potencial antes de invertir pauta.

### Métricas públicas

| Métrica | Valor | Lectura |
|---|---:|---|
| Visualizaciones | {_fmt_num(views)} | Tamaño actual de la audiencia alcanzada. |
| Likes | {_fmt_num(likes)} | Señal de aprobación explícita. |
| Comentarios | {_fmt_num(comments)} | Señal de conversación y fricción/afinidad. |
| Engagement rate | {_fmt_pct(engagement)} | Intensidad de interacción sobre visualizaciones. |

### Métricas operativas

| Métrica | Valor | Lectura |
|---|---:|---|
| Shares | {_fmt_num(ops.get("shares"))} | Señal de distribución voluntaria. |
| Share rate | {_fmt_pct(share_rate)} | Capacidad de viralidad relativa. |
| Retención | {_fmt_pct(retention)} | Capacidad de sostener atención. |
| Tiempo desde publicación | {_format_age_from_hours(ops.get("hours_since_publication"))} | {hours_note}. |
| Views por hora | {_fmt_num(ops.get("views_per_hour"))} | Velocidad inicial de consumo. |

### Lectura estratégica

Si las visualizaciones por hora son altas y el engagement se mantiene saludable, el video tiene señales de distribución. Si el engagement o la retención son bajos, conviene ajustar hook, claridad del beneficio o duración antes de empujar presupuesto.
""".strip() + ("\n\n" + _channel_metrics_markdown(metricas.get("channel_metrics", {}) or {})) + ("\n\n" + render_paid_ad_boost_markdown(metricas.get("xgboost_pauta", {}) or {}))


def _channel_metrics_markdown(channel: Dict[str, Any]) -> str:
    """Renderiza métricas públicas del canal dentro de la pestaña Métricas."""
    if not channel:
        return ""
    if not channel.get("ok"):
        return f"""
### Métricas públicas del canal

No se pudieron obtener métricas públicas del canal.

**Detalle:** {channel.get("warning", "Sin información disponible.")}
""".strip()

    subscribers = float(channel.get("subscriber_count", 0) or 0)
    avg_views = float(channel.get("avg_views_recent", 0) or 0)
    median_views = float(channel.get("median_views_recent", 0) or 0)
    avg_eng = float(channel.get("avg_public_engagement_recent", 0) or 0) * 100
    vps = float(channel.get("views_per_subscriber_recent", 0) or 0)

    if subscribers and avg_views:
        relative = (avg_views / subscribers) * 100
        lectura = (
            "El canal tiene una base pública suficiente para comparar el video contra su rendimiento reciente. "
            f"El promedio de views recientes equivale aproximadamente al {relative:.1f}% de sus suscriptores."
        )
    else:
        lectura = (
            "La API no expone suficientes datos públicos de suscriptores o videos recientes para calcular una lectura relativa sólida."
        )

    return f"""
### Métricas públicas del canal

Estas métricas contextualizan el video frente al rendimiento público reciente del canal. El sistema no accede a YouTube Studio, por lo que el alcance real se aproxima con visualizaciones recientes.

| Métrica del canal | Valor | Lectura |
|---|---:|---|
| Canal | {channel.get("channel_title", "—")} | Fuente pública del video. |
| Suscriptores | {_fmt_num(subscribers)} | Tamaño de la base pública del canal. |
| Videos publicados | {_fmt_num(channel.get("channel_video_count", 0))} | Historial visible del canal. |
| Vistas totales del canal | {_fmt_num(channel.get("channel_view_count", 0))} | Volumen histórico acumulado. |
| Videos recientes analizados | {_fmt_num(channel.get("videos_sampled", 0))} | Muestra usada para estimar promedio. |
| Alcance promedio estimado | {_fmt_num(avg_views)} views | Promedio de views de videos recientes. |
| Mediana de views recientes | {_fmt_num(median_views)} views | Referencia menos sensible a virales aislados. |
| Engagement público promedio | {_fmt_pct(avg_eng)} | Likes + comentarios sobre views recientes. |
| Views por suscriptor | {_fmt_float(vps, 3)} | Relación entre visualizaciones recientes y base de suscriptores. |

### Interpretación del canal

{lectura}

### Advertencia metodológica

{channel.get("warning", "Este cálculo usa datos públicos de YouTube Data API, no métricas privadas de YouTube Studio.")}
""".strip()


def _llm_markdown(llm: Dict[str, Any]) -> str:
    return f"""
### Diagnóstico comunicacional

Este bloque interpreta el texto del video, título, descripción, transcripción y OCR para valorar claridad, coherencia y riesgo narrativo. No reemplaza al modelo predictivo: lo complementa.

### Evaluación rápida

| Variable | Resultado |
|---|---:|
| Claridad del mensaje | {llm.get("claridad_mensaje", "—")}/100 |
| Fuerza del CTA | {llm.get("fuerza_cta", "—")}/100 |
| Coherencia semántica | {llm.get("coherencia_semantica", "—")}/100 |
| Complejidad sintáctica | {llm.get("complejidad_sintactica", "—")} |
| Tipo de contenido | {llm.get("tipo_contenido", "—")} |
| Riesgo comunicacional | {llm.get("riesgo_comunicacional", "—")} |

### Fortalezas detectadas

{_list_md(llm.get("fortalezas"))}

### Debilidades o puntos de fricción

{_list_md(llm.get("debilidades"))}

### Justificación

{llm.get("justificacion", "No se generó justificación.")}

### Conclusión

Una pieza con buena claridad, CTA explícito y baja fricción semántica suele necesitar menos presupuesto para explicar su propuesta. Si la claridad o el CTA son bajos, conviene editar copy y primeros segundos antes de pautar.
""".strip()


def _ocr_markdown(ocr: Dict[str, Any]) -> str:
    """Render narrativo de OCR sensible al tipo de contenido.

    Ya no interpreta todo como venta/CTA. Si el video es humor, educativo o
    informativo, el OCR se evalúa como apoyo de contexto, subtítulo, setup,
    remate o explicación.
    """
    texto = str(ocr.get("ocr_text") or ocr.get("preview") or "").strip()
    preview = texto[:1100] + ("..." if len(texto) > 1100 else "")
    raw_text = str(ocr.get("ocr_text_raw") or "").strip()
    first_ocr = str(ocr.get("ocr_first_result") or "").strip()
    llm_meaning = str(ocr.get("ocr_llm_meaning") or "").strip()
    llm_source = str(ocr.get("ocr_llm_source") or "").strip()
    llm_warning = str(ocr.get("ocr_llm_warning") or "").strip()
    llm_confidence = ocr.get("ocr_llm_confidence", "")

    coverage_raw = ocr.get("ocr_frame_coverage", ocr.get("cobertura", 0))
    coverage = float(coverage_raw or 0) * 100
    word_count = ocr.get("ocr_word_count", ocr.get("cantidad_palabras", 0))
    frames_total = ocr.get("ocr_frame_count", ocr.get("frames_totales", 0))
    frames_text = ocr.get("ocr_frames_with_text", ocr.get("frames_con_texto", 0))
    density = ocr.get("visual_text_density", ocr.get("densidad", "—"))
    intent = ocr.get("tipo_contenido_estimado") or ocr.get("content_intent") or "general/branding"
    role = ocr.get("rol_texto_en_pantalla") or ocr.get("ocr_role") or "sin función clara"

    relevance_raw = ocr.get("ocr_relevance_score", ocr.get("relevancia_texto_en_pantalla", 0))
    relevance = float(relevance_raw or 0) * 100
    overlap = float(ocr.get("overlap_contextual", ocr.get("ocr_context_overlap", 0)) or 0) * 100

    if relevance >= 70:
        relevance_label = "alta"
    elif relevance >= 40:
        relevance_label = "media"
    elif int(float(word_count or 0)) > 0:
        relevance_label = "baja"
    else:
        relevance_label = "nula"

    base_interpretation = ocr.get("interpretacion_ocr") or "El texto detectado se interpreta según el objetivo estimado del video."
    if str(intent).startswith("musical"):
        practical_reading = (
            "En un video musical, el texto en pantalla puede ser letra, título, artista, subtítulo o rótulo de ambiente. "
            "No debe evaluarse como CTA ni como venta si no hay intención comercial."
        )
        improve = (
            "Si la relevancia es baja, mejora la legibilidad de la letra o rótulos clave y evita exceso de texto que compita con la música o el performance."
        )
    elif str(intent).startswith("humor"):
        practical_reading = (
            "En un video de humor, el texto en pantalla no tiene que vender ni tener CTA. "
            "Debe ayudar a entender el contexto, preparar el chiste, reforzar el remate o hacer legible la situación sin audio."
        )
        improve = (
            "Si la relevancia es baja, usa subtítulos/captions más conectados con el remate o una frase corta que explique la situación graciosa. "
            "Evita recomendaciones de producto si el contenido no está vendiendo nada."
        )
    elif str(intent).startswith("comercial"):
        practical_reading = (
            "En una pieza comercial, el OCR debe reforzar beneficio, oferta, prueba o acción. "
            "Aquí sí tiene sentido revisar CTA, promoción y claridad de propuesta."
        )
        improve = "Si la relevancia es baja, reemplaza texto decorativo por beneficio concreto, prueba o CTA visible."
    elif str(intent).startswith("educativo"):
        practical_reading = (
            "En un video educativo, el texto en pantalla debe resumir pasos, conceptos o aprendizajes. "
            "La prioridad es claridad, no venta."
        )
        improve = "Si la relevancia es baja, usa rótulos que sinteticen el aprendizaje o el paso que se está explicando."
    else:
        practical_reading = (
            "El OCR se evalúa como soporte del mensaje general: contexto, énfasis visual o recordación. "
            "No necesariamente debe funcionar como CTA comercial."
        )
        improve = "Si la relevancia es baja, define qué función cumple el texto: contexto, énfasis, subtítulo, dato o cierre."

    cta = "sí" if ocr.get("tiene_cta") or ocr.get("ocr_cta_flag") else "no"
    promo = "sí" if ocr.get("tiene_promocion") or ocr.get("ocr_promo_flag") else "no"
    urgencia = "sí" if ocr.get("tiene_urgencia") or ocr.get("ocr_urgency_flag") else "no"
    confianza = "sí" if ocr.get("tiene_confianza") or ocr.get("ocr_trust_flag") else "no"

    return f"""
### Lectura OCR del video

El OCR mide el texto visible en pantalla y evalúa si ese texto ayuda al objetivo real del video. La lectura cambia según el tipo de contenido: humor, educación, información, branding o venta.

### Resultado detectado

| Métrica OCR | Valor |
|---|---:|
| Tipo de contenido estimado | {intent} |
| Rol probable del texto | {role} |
| Fuente interpretación OCR | {llm_source or "reglas"} |
| Confianza interpretación | {llm_confidence if llm_confidence != "" else "—"} |
| Palabras detectadas | {word_count} |
| Frames analizados | {frames_total} |
| Frames con texto | {frames_text} |
| Cobertura de texto | {_fmt_pct(coverage)} |
| Densidad visual de texto | {density} |
| Conexión con título/transcripción | {_fmt_pct(overlap)} |
| Relevancia del texto en pantalla | {_fmt_pct(relevance)} |
| Nivel de relevancia | {relevance_label} |

### Señales detectadas solo si aplican

| Señal | Detectada |
|---|---|
| CTA visible | {cta} |
| Promoción/oferta | {promo} |
| Urgencia | {urgencia} |
| Confianza/prueba | {confianza} |

### Texto detectado e interpretado

**Primer resultado OCR:** {first_ocr or "—"}

**Texto normalizado usado para el análisis:**

{preview or "No se detectó texto relevante en pantalla."}

**OCR original crudo:**

{raw_text if raw_text and raw_text != texto else "Sin diferencias relevantes frente al texto normalizado."}

### Interpretación

{llm_meaning or base_interpretation}

{base_interpretation}

{practical_reading}

{f"**Advertencia OCR/LLM:** {llm_warning}" if llm_warning else ""}

### Qué se debe mejorar

{improve}

### Conclusión

El texto en pantalla debe cumplir una función concreta. En videos musicales puede ser letra o rótulo; en humor debe reforzar contexto o remate; en educativos debe explicar; en comerciales debe reforzar beneficio o acción. Si el sistema detecta baja relevancia, el problema no es necesariamente ausencia de CTA, sino falta de conexión entre el texto visible y el objetivo real del contenido.
""".strip()

def _policy_markdown(policy: Dict[str, Any]) -> str:
    categorias = policy.get("policy_risk_categories") or []
    return f"""
### Evaluación preliminar de políticas de YouTube Ads

Este screening estima si el contenido podría requerir revisión o ajustes antes de pautar. No reemplaza la revisión oficial de la plataforma.

### Resultado

| Criterio | Resultado |
|---|---|
| Nivel de riesgo | {policy.get("policy_risk_level", "—")} |
| Estado estimado | {policy.get("youtube_ad_status_estimate", "—")} |
| Categorías sensibles | {", ".join(categorias) if categorias else "No se detectaron categorías sensibles."} |

### Explicación

{policy.get("policy_explanation", "No se generó explicación de políticas.")}

### Recomendación

Si aparece riesgo medio, alto o revisión humana, revisa claims, promesas, lenguaje sensible, texto en pantalla y contexto del guion antes de invertir pauta.
""".strip()


def _visual_markdown(visual: Dict[str, Any]) -> str:
    if not visual.get("visual_ok"):
        return f"""
### Análisis visual

No se pudo completar el análisis visual.

**Detalle:** {visual.get("warning", "Sin video o frames disponibles.")}
""".strip()

    score = float(visual.get("composition_score", 0) or 0) * 100
    thirds = float(visual.get("rule_of_thirds_score", 0) or 0) * 100
    golden = float(visual.get("golden_ratio_score", 0) or 0) * 100
    balance = float(visual.get("geometric_balance_score", 0) or 0) * 100
    focus = float(visual.get("focal_clarity_score", 0) or 0) * 100
    contrast = float(visual.get("contrast_score", 0) or 0) * 100
    complexity = float(visual.get("visual_complexity_score", 0) or 0) * 100
    motion = float(visual.get("motion_score", 0) or 0) * 100
    theory = visual.get("golden_ratio_theory", {}) or {}

    return f"""
### Análisis visual y composición

Se analizaron **{visual.get("frames_analyzed", "—")} frames** para medir cómo se organiza la atención visual del video. Los frames de abajo muestran tercios, retícula áurea y centroide de saliencia.

### Qué mide este apartado

- **Regla de tercios:** si el sujeto, producto o CTA cae cerca de zonas naturalmente fuertes del encuadre.
- **Composición áurea:** si el peso visual se aproxima a líneas 0.382 / 0.618 de la proporción 1.618.
- **Balance geométrico:** si el peso visual está repartido sin vacíos o sobrecarga extrema.
- **Claridad focal:** si la mirada tiene un punto principal o se dispersa entre muchos elementos.
- **Contraste, brillo y complejidad:** si la pieza se entiende rápido en pantalla móvil.

### Métricas visuales

| Dimensión | Score |
|---|---:|
| Score global de composición | {score:.0f}/100 |
| Regla de tercios | {thirds:.0f}/100 |
| Composición áurea | {golden:.0f}/100 |
| Balance geométrico | {balance:.0f}/100 |
| Claridad del foco | {focus:.0f}/100 |
| Contraste | {contrast:.0f}/100 |
| Complejidad visual | {complexity:.0f}/100 |
| Movimiento entre frames | {motion:.0f}/100 |

### Teoría de composición áurea

La composición áurea usa la proporción **1.618** para ubicar el peso visual en líneas **0.382** y **0.618** del encuadre. En anuncios ayuda a que rostro, producto, beneficio o CTA se sientan ordenados sin depender de un centro rígido.

{theory.get("description", "")}

### Hallazgos principales

**Resumen:** {visual.get("visual_summary", "No se generó resumen visual.")}

**Recomendaciones:**
{_list_md(visual.get("visual_recommendations"))}

### Conclusión visual

{visual.get("visual_conclusion", "No se generó conclusión visual.")}
""".strip()


def _script_markdown(script: Dict[str, Any]) -> str:
    intent = script.get("content_intent", "—")
    transcript_words = script.get("transcript_word_count", "—")
    transcript_available = bool(script.get("transcript_available")) and safe_float(transcript_words, 0) > 0
    if not transcript_available:
        return f"""
### Análisis de guion y transcripción

No hay transcripción suficiente para analizar el guion.

Este apartado no muestra score, tono, hook ni recomendaciones específicas porque esos datos dependen del audio o de una transcripción real. En videos musicales, piezas con solo música, audio bajo o clips sin voz, el sistema no debe inventar un análisis de guion.

### Qué hacer

- Si el video sí tiene voz, revisa que el audio sea claro o carga una transcripción manual.
- Si el video es musical o no tiene diálogo, evalúalo principalmente desde composición visual, OCR, sentimiento, métricas y contexto del canal.
- Para una lectura narrativa más completa, agrega letra, subtítulos o descripción manual.

### Conclusión

Sin transcripción no se puede evaluar estructura verbal, hook hablado, ritmo narrativo ni claridad del mensaje oral.
""".strip()
    hook_type = script.get("hook_type", "—")
    cta_clarity = script.get("cta_clarity", "—")
    intent_fit = script.get("intent_fit_score", script.get("value_proposition_score", "—"))
    hook_score = script.get("hook_score", "—")
    clarity_score = script.get("clarity_score", "—")
    strengths = script.get("strengths") or script.get("script_strengths") or script.get("main_strengths")
    weaknesses = script.get("weaknesses") or script.get("main_weaknesses")
    recs = script.get("recommended_script_improvements") or script.get("recommendations") or script.get("what_to_improve")

    if str(intent).startswith("humor"):
        criteria = (
            "Este guion se evalúa como humor/entretenimiento. Por eso no se penaliza por no tener producto o CTA comercial. "
            "Se revisa si la transcripción ayuda a preparar contexto, setup, tensión y remate."
        )
    elif str(intent).startswith("comercial"):
        criteria = (
            "Este guion se evalúa como pieza comercial/promocional. Se revisa si comunica beneficio, prueba, oferta y acción."
        )
    elif str(intent).startswith("educativo"):
        criteria = (
            "Este guion se evalúa como contenido educativo. Se revisa claridad, secuencia y aprendizaje prometido."
        )
    else:
        criteria = (
            "Este guion se evalúa según claridad narrativa y coherencia con el objetivo detectado, no con una plantilla fija de venta."
        )

    return f"""
### Análisis de guion y transcripción

Este apartado analiza la transcripción real del MP4 o del texto manual. El sistema primero estima la intención del contenido y luego aplica criterios acordes; no todos los videos necesitan CTA o producto.

### Criterio aplicado

{criteria}

### Resultado general

| Variable | Resultado |
|---|---:|
| Tipo de contenido estimado | {intent} |
| Palabras transcritas analizadas | {transcript_words} |
| Score de calidad | {script.get("script_quality_score", "—")}/100 |
| Score de hook | {hook_score}/100 |
| Score de claridad | {clarity_score}/100 |
| Ajuste a intención | {intent_fit}/100 |
| Tono | {script.get("tone", "—")} |
| Tipo de hook | {hook_type} |
| Claridad de CTA/cierre | {cta_clarity} |

### Interpretación de la transcripción

{script.get("transcript_interpretation") or script.get("transcript_summary") or "No se generó interpretación de la transcripción."}

### Fragmento usado para el análisis

{script.get("transcript_preview") or "No hay transcripción disponible."}

### Fortalezas

{_list_md(strengths)}

### Debilidades detectadas

{_list_md(weaknesses)}

### Qué se debe mejorar

{_list_md(recs)}

### Conclusión

El análisis ya no usa una plantilla única de pauta. Si el video es humorístico, se prioriza claridad del chiste, setup, remate y ritmo; si es comercial, se prioriza propuesta de valor y acción; si es educativo, claridad y secuencia.
""".strip()


def _transcript_markdown(transc: Dict[str, Any], script: Dict[str, Any], redac: Dict[str, Any], transcript_text: str) -> str:
    """Muestra transcripción + resultados del análisis y LLM en la pestaña Transcripción."""
    text = (transcript_text or transc.get("preview") or "").strip()
    preview = text[:1800] + ("..." if len(text) > 1800 else "")
    if not text:
        return f"""
### Transcripción automática

No se obtuvo transcripción útil del video.

| Campo | Resultado |
|---|---:|
| Fuente | {transc.get("fuente", "—")} |
| Palabras transcritas | 0 |
| Estado | Sin texto analizable |

### Interpretación

Si el video es musical, contiene solo ambiente, tiene audio muy bajo o no incluye voz, este resultado es normal. En ese caso, el sistema no debe inventar análisis de guion. Usa los apartados de composición visual, OCR, sentimiento, métricas y canal para evaluar la pieza.

### Qué hacer

- Carga una transcripción manual si el video sí tiene voz y quieres evaluar guion.
- Si es un video musical, evalúa el rendimiento desde visuales, engagement, comentarios y texto en pantalla.
- Revisa que el MP4 tenga audio claro si esperabas transcripción automática.
""".strip()
    intent = script.get("content_intent", transc.get("tipo_contenido_estimado", "—"))
    recs = script.get("recommended_script_improvements") or script.get("recommendations") or script.get("what_to_improve") or []
    strengths = script.get("strengths") or script.get("main_strengths") or []
    redac_txt = (redac.get("recomendacion") or "").strip()
    if len(redac_txt) > 1000:
        redac_txt = redac_txt[:1000] + "..."

    return f"""
### Transcripción automática y análisis LLM

| Campo | Resultado |
|---|---:|
| Fuente | {transc.get("fuente", "—")} |
| Palabras transcritas | {transc.get("cantidad_palabras", 0)} |
| Tipo de contenido estimado | {intent} |
| Score de guion | {script.get("script_quality_score", "—")}/100 |
| Motor de recomendación | {redac.get("source", "—")} |

### Qué entendió el sistema de la transcripción

{script.get("transcript_summary") or script.get("transcript_interpretation") or "No se generó lectura de la transcripción."}

### Fortalezas de la transcripción

{_list_md(strengths)}

### Qué se debe mejorar

{_list_md(recs)}

### Resultado del LLM aplicado a esta pieza

{redac_txt or "El LLM no generó texto. Revisa la pestaña Recomendación LLM o usa modo rules/gemini."}

### Texto transcrito

{preview or "No se obtuvo transcripción. Si el audio es bajo, tiene música encima o está cortado, carga una transcripción manual."}
""".strip()


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def analyze_video(
    demo_case: str,
    youtube_url: str,
    channel_url: str,
    video_file: Any,
    title: str, description: str, category: str, topic: str,
    views: float, likes: float, comments: float,
    shares: float, retention_rate: float, average_watch_time: float,
    hours_since_publication: float, followers_count: float, avg_channel_reach: float,
    duration_seconds: float,
    manual_transcript: str,
    manual_ocr_text: str,
    manual_comments: str,
    video_type: str,
    cpm_estimado: float,
    presupuesto: float,
    llm_mode: str,
) -> Tuple[
    str,            # 0  resumen ejecutivo md
    Dict[str,Any],  # 1  result_json
    str,            # 2  transcripción
    str,            # 3  OCR texto
    str,            # 4  metricas narrativas
    str,            # 5  analisis_llm narrativo
    str,            # 6  analisis_ocr narrativo
    str,            # 7  analisis_politicas narrativo
    str,            # 8  analisis_visual narrativo
    str,            # 9  analisis_guion narrativo
    str,            # 10 analisis_sentimiento narrativo
    List[str],      # 11 frames galería
    str,            # 12 score_chart
    str,            # 13 projection_chart
    str,            # 14 policy_chart
    str,            # 15 advertencias md
    str,            # 16 recomendación redactada md
    str,            # 17 redacción meta narrativa
    str,            # 18 sentiment_bar_chart
    str,            # 19 wordcloud_positive
    str,            # 20 wordcloud_neutral
    str,            # 21 wordcloud_negative
]:
    """Pipeline completo. Devuelve 22 outputs."""

    # ── Demo precalculado ────────────────────────────────────────────────────
    demo = _load_demo(demo_case)
    if demo:
        result  = _result_from_demo(demo)
        charts  = create_analysis_charts(result)
        exec_s  = build_executive_summary(
            final_recommendation=result,
            prediction=result.get("modelo", {}),
            features=result.get("metricas", {}),
            operational_metrics=result.get("metricas", {}).get("operational", {}),
            policy_block=result.get("analisis_politicas", {}),
            visual_analysis=result.get("analisis_visual"),
            script_analysis=result.get("analisis_guion"),
            metadata=result.get("metadata_youtube", {}),
            cpm=safe_float(result.get("cpm_estimado", DEFAULT_CPM)),
            budget=safe_float(presupuesto, 0),
        )
        result["resumen_ejecutivo"] = exec_s
        diag = _build_diagnostic(result, {}, {})
        redac = generate_recommendation_with_llm(diag, mode=(llm_mode or "auto"))
        result["recomendacion_redactada"] = redac
        redac_md = _redac_md(redac)
        sentiment_demo = result.get("analisis_sentimiento", {}) or {}
        sentiment_demo_charts = create_sentiment_visuals(sentiment_demo)
        demo_transc_blk = result.get("analisis_transcripcion", {}) or {
            "fuente": "demo/manual",
            "cantidad_palabras": len((demo.get("transcript_text", "") or "").split()),
            "preview": demo.get("transcript_text", ""),
        }
        transcript_demo_md = _transcript_markdown(
            demo_transc_blk,
            result.get("analisis_guion", {}) or {},
            redac,
            demo.get("transcript_text", ""),
        )
        return (
            exec_s["markdown"], result,
            transcript_demo_md, demo.get("ocr_text", ""),
            _metrics_markdown(result.get("metricas", {})),
            _llm_markdown(result.get("analisis_llm", {})),
            _ocr_markdown(result.get("analisis_ocr", {})),
            _policy_markdown(result.get("analisis_politicas", {})),
            _visual_markdown(result.get("analisis_visual", {})),
            _script_markdown(result.get("analisis_guion", {})),
            build_sentiment_markdown(sentiment_demo),
            [],
            charts.get("score_chart", ""), charts.get("projection_chart", ""), charts.get("policy_chart", ""),
            "\n".join(f"- {w}" for w in result.get("warnings", [])),
            redac_md,
            _redac_meta_md(redac),
            sentiment_demo_charts.get("sentiment_bar_chart", ""),
            sentiment_demo_charts.get("wordcloud_positive", ""),
            sentiment_demo_charts.get("wordcloud_neutral", ""),
            sentiment_demo_charts.get("wordcloud_negative", ""),
        )

    warnings: List[str] = []

    # ── Metadata YouTube ─────────────────────────────────────────────────────
    api_payload = (
        fetch_youtube_metadata(youtube_url)
        if youtube_url else {"ok": False, "warning": "Sin URL.", "video_id": None}
    )
    if youtube_url and not api_payload.get("ok"):
        warnings.append(f"YouTube API: {api_payload.get('warning', '')}")

    # ── Descarga automática ──────────────────────────────────────────────────
    download_info: Dict[str, Any] = {"ok": False, "attempts": [], "warning": ""}
    video_path = _coerce_video(video_file)
    if youtube_url and not video_path:
        download_info = download_youtube_video_360p(youtube_url)
        if download_info.get("ok"):
            video_path = download_info["video_path"]
            for k, v in (download_info.get("metadata") or {}).items():
                if v and not api_payload.get(k):
                    api_payload[k] = v
            warnings.append(f"Video descargado vía {download_info['backend_used']} (360p, ≤60s).")
            if download_info.get("trimmed_to_max_duration"):
                warnings.append("Video recortado a 60s.")
        else:
            warnings.append(f"Descarga: {download_info.get('warning', '')}")

    metadata = {
        "title":            (title or "").strip() or api_payload.get("title", ""),
        "description":      (description or "").strip() or api_payload.get("description", ""),
        "category_id":      (category or "").strip() or str(api_payload.get("category_id", "unknown") or "unknown"),
        "topic":            (topic or "").strip(),
        "views":            safe_float(views) if views else safe_float(api_payload.get("views", 0)),
        "likes":            safe_float(likes) if likes else safe_float(api_payload.get("likes", 0)),
        "comments":         safe_float(comments) if comments else safe_float(api_payload.get("comments", 0)),
        "duration_seconds": safe_float(duration_seconds) if duration_seconds else safe_float(api_payload.get("duration_seconds", 0)),
        "published_at":     api_payload.get("published_at"),
        "channel_title":    api_payload.get("channel_title", ""),
    }
    auto_hours_since_publication = _hours_since_published(metadata.get("published_at"))
    if auto_hours_since_publication is not None:
        metadata["hours_since_publication_auto"] = auto_hours_since_publication

    # ── Métricas públicas del canal ──────────────────────────────────────────
    channel_metrics: Dict[str, Any] = {}
    channel_hint = (channel_url or "").strip()
    api_channel_id = api_payload.get("channel_id") or ""
    if channel_hint or api_channel_id:
        channel_metrics = get_channel_reach_estimate(
            channel_hint,
            channel_id=api_channel_id,
            max_videos=20,
        )
        if channel_metrics.get("ok"):
            metadata["channel_metrics"] = channel_metrics
            if not followers_count:
                followers_count = safe_float(channel_metrics.get("subscriber_count", 0))
            if not avg_channel_reach:
                avg_channel_reach = safe_float(channel_metrics.get("avg_views_recent", 0))
        elif channel_metrics.get("warning"):
            warnings.append(f"Canal: {channel_metrics.get('warning')}")

    video_summary: Dict[str, Any] = {}
    if video_path:
        video_summary = summarize_video(video_path, video_type=video_type)
        if video_summary.get("warning"):
            warnings.append(f"Video: {video_summary['warning']}")
        if video_summary.get("duration_seconds") and not metadata.get("duration_seconds"):
            metadata["duration_seconds"] = video_summary["duration_seconds"]

    # ── OCR + frames SIEMPRE juntos ──────────────────────────────────────────
    # Los 10 frames se extraen una sola vez en extract_ocr_from_video.
    # analyze_visual_composition reutiliza los mismos timestamps.
    manual_ocr = (manual_ocr_text or "").strip()
    if video_path:
        ocr_result = extract_ocr_from_video(video_path, video_type=video_type)
        if ocr_result.get("ocr_warning"):
            warnings.append(f"OCR: {ocr_result['ocr_warning']}")
        # El texto manual tiene precedencia si se pegó; si no, usamos el automático
        ocr_features = {k: v for k, v in ocr_result.items()
                        if k.startswith("ocr_") or k == "visual_text_density"}
        if manual_ocr:
            ocr_features["ocr_text"] = manual_ocr
        ocr_status = {"engine": ocr_result.get("ocr_engine", "none"), "warning": ocr_result.get("ocr_warning", "")}
    else:
        ocr_features = {
            "ocr_text": manual_ocr, "ocr_word_count": len(manual_ocr.split()) if manual_ocr else 0,
            "ocr_frame_count": 0, "ocr_frames_with_text": 0, "ocr_frame_coverage": 0.0,
            "visual_text_density": "sin_texto" if not manual_ocr else "media",
            "ocr_cta_flag": 0, "ocr_promo_flag": 0, "ocr_trust_flag": 0,
            "ocr_urgency_flag": 0, "ocr_excess_text_flag": 0,
        }
        ocr_status = {"engine": "manual_o_sin_video", "warning": ""}

    # ── Análisis visual (reutiliza los mismos 10 frames) ─────────────────────
    visual_analysis: Dict[str, Any] = {"visual_ok": False, "warning": "Sin video."}
    if video_path:
        visual_dir = RUNTIME_CACHE_DIR / "visual"
        visual_analysis = analyze_visual_composition(
            video_path,
            ocr_text=ocr_features.get("ocr_text", ""),
            output_dir=visual_dir,
            n_frames=10,
        )
        if visual_analysis.get("warning"):
            warnings.append(f"Visual: {visual_analysis['warning']}")

    # ── Transcripción automática MP4 (video→WAV→faster-whisper/Google Speech) ───────────────────────────────
    transcript_result = resolve_transcript(manual_transcript, video_path)
    if transcript_result.get("warning"):
        warnings.append(f"Transcripción: {transcript_result.get('warning')}")

    if transcript_result.get("requires_human_review"):
        warnings.append(
        "Bloqueo parcial: sin transcripción válida, la evaluación completa de políticas y guion queda limitada."
    )
    transcript_text   = transcript_result.get("transcript_text", "")
    trans_status      = {"source": transcript_result.get("source"), "warning": transcript_result.get("warning")}
    if transcript_result.get("warning"):
        warnings.append(f"Transcripción: {transcript_result['warning']}")

    # ── Interpretación/corrección OCR con LLM ────────────────────────────────
    # El OCR automático puede leer mal letras o partir frases. Usamos el primer
    # resultado legible como ancla y lo normalizamos antes de calcular relevancia.
    ocr_features = _apply_llm_ocr_interpretation(
        ocr_features,
        title=metadata.get("title", ""),
        description=metadata.get("description", ""),
        transcript_text=transcript_text,
        mode=llm_mode or "auto",
    )
    if ocr_features.get("ocr_llm_warning"):
        warnings.append(f"OCR/LLM: {ocr_features.get('ocr_llm_warning')}")

    # ── Sentimiento de comentarios ────────────────────────────────────────────
    video_id = api_payload.get("video_id") or extract_video_id(youtube_url or "")
    if video_id and YOUTUBE_API_KEY:
        sentiment = fetch_and_analyze_comments(video_id, api_key=YOUTUBE_API_KEY, max_results=50)
    elif manual_comments and manual_comments.strip():
        comment_list = [c.strip() for c in manual_comments.strip().splitlines() if c.strip()]
        sentiment = analyze_comments_sentiment(comment_list)
        sentiment["source"] = "manual"
    else:
        sentiment = analyze_comments_sentiment([])

    # ── Features ─────────────────────────────────────────────────────────────
    features = build_feature_row(
        title=metadata.get("title", ""), description=metadata.get("description", ""),
        transcript_text=transcript_text, ocr_text=ocr_features.get("ocr_text", ""),
        category_id=metadata.get("category_id", "unknown"),
        duration_seconds=metadata.get("duration_seconds", 0),
        views=metadata.get("views", 0), likes=metadata.get("likes", 0),
        comments=metadata.get("comments", 0), published_at=metadata.get("published_at"),
        video_type=video_type, extra=ocr_features,
    )

    # ── Modelo predictivo ────────────────────────────────────────────────────
    prediction = predict_from_features(features)
    if prediction.get("model_warning"):
        warnings.append(f"Modelo: {prediction['model_warning']}")

    # ── Análisis de guion ────────────────────────────────────────────────────
    script_analysis = analyze_video_script(
        title=metadata.get("title", ""), description=metadata.get("description", ""),
        transcript=transcript_text, category=metadata.get("category_id", ""),
        topic=metadata.get("topic", ""), duration_seconds=metadata.get("duration_seconds", 0),
    ) if (transcript_text or metadata.get("description")) else {
        "script_quality_score": 0, "recommendations": [],
        "warning": "Sin transcripción ni descripción.",
    }

    # ── Políticas ────────────────────────────────────────────────────────────
    combined_text = " ".join([
        metadata.get("title", ""), metadata.get("description", ""), transcript_text,
    ]).strip()
    has_transcript = bool(transcript_text.strip())
    policy = evaluate_youtube_ad_policy_risk(combined_text, has_transcript=has_transcript)

    # ── Métricas operativas ──────────────────────────────────────────────────
    effective_hours = auto_hours_since_publication if auto_hours_since_publication is not None else safe_float(hours_since_publication, 24)
    ops = _derive_ops(metadata.get("views", 0), shares, retention_rate, effective_hours)
    ops.update({
        "average_watch_time": safe_float(average_watch_time),
        "followers_count":    safe_float(followers_count),
        "avg_channel_reach":  safe_float(avg_channel_reach),
        "topic":              (topic or "").strip(),
        "published_at":       metadata.get("published_at"),
        "hours_source":       "youtube_published_at" if auto_hours_since_publication is not None else "manual",
    })

    # ── Segundo modelo: XGBoost para pauta pagada ────────────────────────────
    paid_xgb = predict_paid_ad_boost(
        features=features,
        metadata=metadata,
        operational_metrics=ops,
        model_probability=safe_float(prediction.get("probability", 0)),
        budget=safe_float(presupuesto, 0),
        manual_cpm=safe_float(cpm_estimado, DEFAULT_CPM),
    )
    if paid_xgb.get("warning"):
        warnings.append(f"XGBoost pauta: {paid_xgb['warning']}")
    cpm_para_pauta = safe_float(
        paid_xgb.get("predicted_cpm"), safe_float(cpm_estimado, DEFAULT_CPM)
    ) if paid_xgb.get("eligible_for_paid_xgboost") else safe_float(cpm_estimado, DEFAULT_CPM)

    # ── Recomendación ────────────────────────────────────────────────────────
    has_text = bool((metadata.get("title") or metadata.get("description") or transcript_text).strip())
    final_rec = build_final_recommendation(
        prediction, features,
        cpm=cpm_para_pauta,
        budget=safe_float(presupuesto, 0),
        operational_metrics=ops,
        policy_block=policy,
        requires_human_review_due_to_missing_transcript=not transcript_result.get("ok"),
        has_evaluable_text=has_text,
    )

    # ── Proyección ───────────────────────────────────────────────────────────
    proyeccion = build_boost_projection(
        features=features,
        performance_probability=safe_float(prediction.get("probability", 0)),
        budget=safe_float(presupuesto, 0),
        cpm=cpm_para_pauta,
        operational_metrics=ops,
        policy_risk_level=policy.get("policy_risk_level", "bajo"),
    )

    # ── LLM redactor técnico (rules, no trae torch) ──────────────────────────
    llm = analyze_with_llm(
        text_total=features.get("text_total", ""),
        features=features, final_recommendation=final_rec,
        prefer="rules",
        title=metadata.get("title", ""), description=metadata.get("description", ""),
        transcript_text=transcript_text, ocr_text=ocr_features.get("ocr_text", ""),
        policy_block=policy,
    )

    # ── LLM provider: Gemini / open source local / rules ───────────────────────────────
    diag = _build_diagnostic_full(
        final_rec=final_rec, metadata=metadata, features=features,
        ops=ops, policy=policy, script=script_analysis,
        visual=visual_analysis, proyeccion=proyeccion,
        ocr_text=ocr_features.get("ocr_text", ""),
        sentiment=sentiment, channel_metrics=channel_metrics, cpm=safe_float(cpm_estimado, DEFAULT_CPM),
        presupuesto=safe_float(presupuesto, 0),
    )
    redac = generate_recommendation_with_llm(diag, mode=(llm_mode or "auto"))

    # ── Construir result ─────────────────────────────────────────────────────
    metricas = metricas_block(features)
    metricas["operational"] = ops
    metricas["xgboost_pauta"] = paid_xgb
    if channel_metrics:
        metricas["channel_metrics"] = channel_metrics

    transc_blk = transcripcion_block(features, extra_status=trans_status)
    transc_blk["requires_human_review"] = not transcript_result.get("ok")
    ocr_blk = ocr_block(features, extra_status=ocr_status)

    result: Dict[str, Any] = {
        "prediccion_rendimiento":       final_rec["prediccion_rendimiento"],
        "probabilidad_rendimiento":     final_rec["probabilidad_rendimiento"],
        "recomendacion_impulso":        final_rec["recomendacion_impulso"],
        "requiere_ajustes":             final_rec["requiere_ajustes"],
        "nivel_prioridad":              final_rec["nivel_prioridad"],
        "alcance_estimado_por_dolar":   final_rec["alcance_estimado_por_dolar"],
        "alcance_estimado_total":       final_rec["alcance_estimado_total"],
        "metricas":                     metricas,
        "analisis_transcripcion":       transc_blk,
        "analisis_ocr":                 ocr_blk,
        "analisis_llm":                 llm,
        "analisis_politicas":           policy,
        "analisis_visual":              visual_analysis,
        "analisis_guion":               script_analysis,
        "analisis_sentimiento":         sentiment,
        "ajustes_sugeridos":            final_rec["ajustes_sugeridos"],
        "justificacion":                final_rec["justificacion"],
        "accion_final":                 final_rec["accion_final"],
        "score_hibrido":                final_rec["score_hibrido"],
        "score_hibrido_detalle":        final_rec["score_hibrido_detalle"],
        "policy_risk_level":            final_rec["policy_risk_level"],
        "youtube_ad_status_estimate":   final_rec["youtube_ad_status_estimate"],
        "modelo":                       prediction,
        "metadata_youtube":             {**metadata, "api_metadata": api_payload,
                                         "video_summary": video_summary,
                                         "download_info": download_info},
        "metricas_canal":               channel_metrics,
        "proyeccion_pauta":             proyeccion,
        "xgboost_pauta":               paid_xgb,
        "explicabilidad":               simple_explanation(features, prediction),
        "cpm_estimado":                 final_rec["cpm_estimado"],
        "cpm_modelado_xgboost":        cpm_para_pauta,
        "multiplicador_potencial":      final_rec["multiplicador_potencial"],
        "alcance_base_por_dolar":       final_rec["alcance_base_por_dolar"],
        "nota_metodologica":            final_rec["nota_metodologica"],
        "recomendacion_redactada":      redac,
        "warnings":                     warnings,
    }

    exec_s = build_executive_summary(
        final_recommendation=final_rec, prediction=prediction, features=features,
        operational_metrics=ops, policy_block=policy,
        visual_analysis=visual_analysis, script_analysis=script_analysis,
        metadata=metadata, cpm=cpm_para_pauta,
        budget=safe_float(presupuesto, 0),
    )
    result["resumen_ejecutivo"] = exec_s

    charts = create_analysis_charts(result)
    sentiment_charts = create_sentiment_visuals(sentiment)
    sentiment_md = build_sentiment_markdown(sentiment)
    frames_gallery = visual_analysis.get("annotated_frame_paths", []) or []
    redac_md = _redac_md(redac)
    warnings_md = "\n".join(f"- {w}" for w in warnings) if warnings else "_Sin advertencias._"

    return (
        exec_s["markdown"], result,
        transcript_text, ocr_features.get("ocr_text", ""),
        _metrics_markdown(metricas),
        _llm_markdown(llm),
        _ocr_markdown(ocr_blk),
        _policy_markdown(policy),
        _visual_markdown(visual_analysis),
        _script_markdown(script_analysis),
        sentiment_md,
        frames_gallery,
        charts.get("score_chart", ""), charts.get("projection_chart", ""), charts.get("policy_chart", ""),
        warnings_md, redac_md,
        _redac_meta_md(redac),
        sentiment_charts.get("sentiment_bar_chart", ""),
        sentiment_charts.get("wordcloud_positive", ""),
        sentiment_charts.get("wordcloud_neutral", ""),
        sentiment_charts.get("wordcloud_negative", ""),
    )


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _redac_md(redac: Dict[str, Any]) -> str:
    src  = redac.get("source", "?")
    mode = redac.get("mode_requested", "?")
    secs = redac.get("elapsed_s", 0)
    txt  = redac.get("recomendacion") or "_Sin texto._"
    md   = f"_Fuente: **{src}** · modo: `{mode}` · {secs}s_\n\n{txt}"
    if redac.get("warning"):
        md += f"\n\n> ⚠️ {redac['warning']}"
    return md




def _redac_meta_md(redac: Dict[str, Any]) -> str:
    warning = redac.get("warning") or "Sin advertencias del redactor."
    return f"""
**Fuente:** {redac.get("source", "—")}  
**Modo solicitado:** {redac.get("mode_requested", "—")}  
**Tiempo de generación:** {redac.get("elapsed_s", 0)}s  
**Estado:** {warning}
""".strip()


def _build_diagnostic(result: Dict[str, Any], ops: Dict[str, Any], sentiment: Dict[str, Any]) -> Dict[str, Any]:
    """Construye diagnóstico mínimo para camino demo."""
    met = result.get("metricas", {}) or {}
    pol = result.get("analisis_politicas", {}) or {}
    vis = result.get("analisis_visual", {}) or {}
    gui = result.get("analisis_guion", {}) or {}
    proy = result.get("proyeccion_pauta", {}) or {}
    return {
        "titulo":          (result.get("metadata_youtube") or {}).get("title", ""),
        "accion":          result.get("accion_final"),
        "probabilidad_pct": int(round(safe_float(result.get("probabilidad_rendimiento", 0)) * 100)),
        "score_0_100":     int(round(safe_float(result.get("score_hibrido", 0)) * 100)),
        "metricas_publicas": {
            "views": int(safe_float(met.get("views", 0))),
            "likes": int(safe_float(met.get("likes", 0))),
            "comments": int(safe_float(met.get("comments", 0))),
        },
        "metricas_privadas": ops,
        "politica": {"nivel": pol.get("policy_risk_level"), "estado": pol.get("youtube_ad_status_estimate"),
                     "categorias": pol.get("policy_risk_categories") or []},
        "visual": {
            "score_0_100": int(round(safe_float(vis.get("composition_score", 0)) * 100)) if vis.get("visual_ok") else None,
            "resumen": vis.get("visual_summary"),
            "sugerencias": (vis.get("visual_recommendations") or [])[:4],
        },
        "guion": {"score": gui.get("script_quality_score"), "tono": gui.get("tone"), "mejoras": []},
        "proyeccion": {"cpm": proy.get("cpm"), "presupuesto": proy.get("budget"),
                       "views_esperadas": proy.get("projected_views_after_boost"),
                       "likes_esperados": proy.get("projected_likes_after_boost")},
        "sentimiento": sentiment or result.get("analisis_sentimiento", {}),
    }


def _build_diagnostic_full(
    *, final_rec, metadata, features, ops, policy, script, visual, proyeccion, ocr_text, sentiment, channel_metrics, cpm, presupuesto,
) -> Dict[str, Any]:
    content_intent = script.get("content_intent") or features.get("content_intent") or "general/branding"
    transcript_text = features.get("transcript_text", "") or ""
    ocr_relevance_pct = round(safe_float(features.get("ocr_relevance_score", 0)) * 100, 1)
    ocr_overlap_pct = round(safe_float(features.get("ocr_context_overlap", 0)) * 100, 1)
    return {
        "titulo":          metadata.get("title", ""),
        "content_intent":  content_intent,
        "accion":          final_rec.get("accion_final"),
        "probabilidad_pct": int(round(safe_float(final_rec.get("probabilidad_rendimiento", 0)) * 100)),
        "score_0_100":     int(round(safe_float(final_rec.get("score_hibrido", 0)) * 100)),
        "metricas_publicas": {
            "views": int(safe_float(metadata.get("views", 0))),
            "likes": int(safe_float(metadata.get("likes", 0))),
            "comments": int(safe_float(metadata.get("comments", 0))),
            "engagement_pct": round(safe_float(features.get("engagement_rate", 0)) * 100, 2),
        },
        "metricas_privadas": {
            "shares": int(safe_float(ops.get("shares", 0))),
            "retention_pct": round(safe_float(ops.get("retention_rate", 0)) * 100, 1),
            "average_watch_time": int(safe_float(ops.get("average_watch_time", 0))),
            "hours_since_publication": int(safe_float(ops.get("hours_since_publication", 0))),
            "published_age_label": ops.get("published_age_label") or _format_age_from_hours(ops.get("hours_since_publication", 0)),
        },
        "politica": {
            "nivel": policy.get("policy_risk_level"),
            "estado": policy.get("youtube_ad_status_estimate"),
            "categorias": policy.get("policy_risk_categories") or [],
        },
        "guion": {
            "score": script.get("script_quality_score"),
            "tono": script.get("tone"),
            "content_intent": content_intent,
            "hook_type": script.get("hook_type"),
            "interpretacion": script.get("transcript_interpretation") or script.get("transcript_summary"),
            "mejoras": script.get("recommended_script_improvements") or script.get("recommendations") or [],
        },
        "transcripcion": {
            "preview": transcript_text[:1200],
            "word_count": int(safe_float(features.get("transcript_word_count", 0))),
            "source": "automatic/manual",
        },
        "ocr": {
            "text": ocr_text[:800],
            "content_intent": content_intent,
            "role": features.get("ocr_role"),
            "relevance_pct": ocr_relevance_pct,
            "overlap_pct": ocr_overlap_pct,
            "interpretation": features.get("ocr_interpretation"),
        },
        "ocr_text": ocr_text[:800],
        "transcript_text": transcript_text[:1200],
        "visual": {
            "score_0_100": int(round(safe_float(visual.get("composition_score", 0)) * 100)) if visual.get("visual_ok") else None,
            "resumen": visual.get("visual_summary"),
            "sugerencias": (visual.get("visual_recommendations") or [])[:4],
        },
        "sentimiento": sentiment,
        "canal": {
            "ok": bool((channel_metrics or {}).get("ok")),
            "titulo": (channel_metrics or {}).get("channel_title"),
            "suscriptores": int(safe_float((channel_metrics or {}).get("subscriber_count", 0))),
            "alcance_promedio_estimado": int(safe_float((channel_metrics or {}).get("avg_views_recent", 0))),
            "mediana_views_recientes": int(safe_float((channel_metrics or {}).get("median_views_recent", 0))),
            "engagement_publico_promedio_pct": round(safe_float((channel_metrics or {}).get("avg_public_engagement_recent", 0)) * 100, 2),
            "views_por_suscriptor": round(safe_float((channel_metrics or {}).get("views_per_subscriber_recent", 0)), 4),
            "videos_muestreados": int(safe_float((channel_metrics or {}).get("videos_sampled", 0))),
        },
        "proyeccion": {
            "cpm": cpm,
            "presupuesto": presupuesto,
            "views_esperadas": int(safe_float(proyeccion.get("projected_views_after_boost", 0))),
            "likes_esperados": int(safe_float(proyeccion.get("projected_likes_after_boost", 0))),
            "comments_esperados": int(safe_float(proyeccion.get("projected_comments_after_boost", 0))),
            "shares_esperados": int(safe_float(proyeccion.get("projected_shares_after_boost", 0))),
        },
    }


# ---------------------------------------------------------------------------
# Función "Rellenar con enlace"
# ---------------------------------------------------------------------------

def fill_from_url(youtube_url: str, channel_url: str = ""):
    """Rellena datos del video y, si es posible, métricas públicas del canal.

    Devuelve 14 valores en el orden de los componentes del formulario.
    """
    empty = ("", "", "unknown", "", 0.0, 0.0, 0.0, 0.0, 24.0, 0.0, 0.0, 0.0, "", "⚠️ Sin datos o sin API key.")
    url = (youtube_url or "").strip()
    if not url and not (channel_url or "").strip():
        return empty

    api = fetch_youtube_metadata(url)
    channel_metrics: Dict[str, Any] = {}
    channel_hint = (channel_url or "").strip()

    if api.get("ok") or channel_hint:
        channel_metrics = get_channel_reach_estimate(
            channel_hint,
            channel_id=api.get("channel_id", "") if api.get("ok") else "",
            max_videos=20,
        )

    if not api.get("ok"):
        msg = api.get("warning") or "No se obtuvieron datos de YouTube."
        if channel_metrics.get("ok"):
            return (
                "", "", "unknown", "", 0.0, 0.0, 0.0, 0.0, 24.0,
                float(channel_metrics.get("subscriber_count", 0) or 0),
                float(channel_metrics.get("avg_views_recent", 0) or 0),
                0.0,
                channel_metrics.get("channel_url", ""),
                f"⚠️ Video: {msg} · ✅ Canal cargado: {channel_metrics.get('channel_title', '')}",
            )
        return ("", "", "unknown", "", 0.0, 0.0, 0.0, 0.0, 24.0, 0.0, 0.0, 0.0, "", f"⚠️ {msg}")

    auto_hours = _hours_since_published(api.get("published_at")) or 24.0
    published_msg = f" · publicado hace {_format_age_from_hours(auto_hours)} ({api.get('published_at')})" if api.get("published_at") else ""

    followers = float(channel_metrics.get("subscriber_count", 0) or 0) if channel_metrics.get("ok") else 0.0
    avg_reach = float(channel_metrics.get("avg_views_recent", 0) or 0) if channel_metrics.get("ok") else 0.0
    channel_msg = (
        f" · canal: {channel_metrics.get('channel_title', '')} · suscriptores: {followers:,.0f} · alcance estimado: {avg_reach:,.0f}"
        if channel_metrics.get("ok") else
        f" · canal no calculado: {channel_metrics.get('warning', 'sin datos')}"
    )

    channel_url_out = channel_metrics.get("channel_url") if channel_metrics.get("ok") else api.get("channel_url", "")

    return (
        api.get("title", ""),                                    # title
        api.get("description", ""),                              # description
        str(api.get("category_id", "unknown") or "unknown"),    # category
        "",                                                      # topic (no viene de API)
        float(api.get("views", 0) or 0),                         # views
        float(api.get("likes", 0) or 0),                         # likes
        float(api.get("comments", 0) or 0),                      # comments
        float(api.get("duration_seconds", 0) or 0),              # duration
        float(auto_hours),                                       # hours_since_publication
        followers,                                               # followers_count
        avg_reach,                                               # avg_channel_reach
        float(0),                                                # shares (no viene)
        channel_url_out or "",                                   # channel_url
        f"✅ Datos cargados desde YouTube API — {api.get('channel_title', '')}{published_msg}{channel_msg}",
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#0b0d12",
        body_background_fill_dark="#0b0d12",
        background_fill_primary="#11141c",
        background_fill_primary_dark="#11141c",
        background_fill_secondary="#161a22",
        background_fill_secondary_dark="#161a22",
        border_color_primary="#262b36",
        border_color_primary_dark="#262b36",
        block_background_fill="#11141c",
        block_background_fill_dark="#11141c",
        input_background_fill="#161a22",
        input_background_fill_dark="#161a22",
        button_primary_background_fill="linear-gradient(135deg,#a78bfa,#7c3aed)",
        button_primary_background_fill_dark="linear-gradient(135deg,#a78bfa,#7c3aed)",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
    )

    with gr.Blocks(title="YouTube Boost AI", theme=theme, css=CUSTOM_CSS) as demo:
        gr.HTML("""
        <div class="yba-hero">
          <span class="yba-pill">🎬 VIDEO</span>
          <span class="yba-pill">🛡️ YT ADS POLICY</span>
          <span class="yba-pill">📊 PROYECCIÓN</span>
          <span class="yba-pill">🎨 COMPOSICIÓN</span>
          <span class="yba-pill">💬 SENTIMIENTO</span>
          <h1>YouTube Boost AI</h1>
          <p>Pega una URL o sube un MP4 · Transcripción automática MP4 vía faster-whisper + Google Speech ·
          OCR + análisis visual siempre juntos · Análisis de sentimiento de comentarios ·
          Recomendación por Gemini 2.5 Flash Lite o SmolLM2-135M local.</p>
        </div>""")

        with gr.Row():
            # ══════════════════════ ENTRADA ══════════════════════
            with gr.Column(scale=5):

                with gr.Group():
                    gr.Markdown("### 1. Demo / enlace")
                    demo_case = gr.Dropdown(
                        choices=list(DEMO_CASES.keys()), value="Ninguno",
                        label="🎯 Caso demo (activo → ignora el resto)",
                    )
                    with gr.Row():
                        youtube_url = gr.Textbox(
                            label="🔗 URL YouTube (se descarga automáticamente a 360p ≤60s)",
                            placeholder="https://www.youtube.com/watch?v=...",
                            scale=4,
                        )
                        fill_btn = gr.Button("📋 Rellenar con enlace", scale=1, variant="secondary")
                    channel_url = gr.Textbox(
                        label="🔗 Canal de YouTube que subió el video",
                        placeholder="Se rellena automáticamente desde el video o pega https://www.youtube.com/@canal",
                        lines=1,
                    )
                    fill_status = gr.Markdown(value="", visible=True)
                    video_file = gr.Video(label="📁 O sube MP4 manualmente (≤60s)", sources=["upload"])
                    video_type = gr.Dropdown(
                        choices=["auto", "con narrador", "sin narrador"],
                        value="auto", label="Tipo de video",
                    )

                with gr.Group():
                    gr.Markdown("### 2. Datos del video")
                    title       = gr.Textbox(label="Título / copy", lines=1)
                    description = gr.Textbox(label="Descripción", lines=3)
                    with gr.Row():
                        category = gr.Textbox(label="Categoría (id)", value="unknown")
                        topic    = gr.Textbox(label="Tópico", value="")
                    with gr.Row():
                        views    = gr.Number(label="Visualizaciones", value=0)
                        likes    = gr.Number(label="Likes", value=0)
                        comments = gr.Number(label="Comentarios", value=0)
                    with gr.Row():
                        shares           = gr.Number(label="Shares (Analytics)", value=0)
                        retention_rate   = gr.Number(label="Retención [0-1]", value=0.0)
                        average_watch_time = gr.Number(label="Watch time (s)", value=0)
                    with gr.Row():
                        hours_since_publication = gr.Number(label="Horas desde publicación (se muestra como días/semanas/meses/años en resultados)", value=24)
                        followers_count         = gr.Number(label="Followers", value=0)
                        avg_channel_reach       = gr.Number(label="Alcance prom. canal", value=0)
                    duration_seconds = gr.Number(label="Duración (s)", value=0)

                with gr.Group():
                    gr.Markdown("### 3. Texto manual (opcional)")
                    manual_transcript = gr.Textbox(
                        label="Transcripción manual (opcional; si subes MP4 se intentará automático)",
                        lines=4,
                        placeholder="Pega aquí el texto si no subes video…",
                    )
                    manual_ocr_text = gr.Textbox(label="OCR manual (sobreescribe el automático)", lines=2)
                    manual_comments = gr.Textbox(
                        label="Comentarios para análisis de sentimiento (uno por línea)",
                        lines=4,
                        placeholder="Pega comentarios aquí si no hay YouTube API key…",
                    )

                with gr.Group():
                    gr.Markdown("### 4. Pauta y LLM")
                    with gr.Row():
                        cpm_estimado = gr.Number(label="CPM USD", value=DEFAULT_CPM)
                        presupuesto  = gr.Number(label="Presupuesto USD", value=20)
                    llm_mode = gr.Dropdown(
                        choices=["auto", "gemini", "local_open_source", "rules"],
                        value="auto",
                        label="Motor LLM",
                        info=(
                            "auto: usa Gemini si GEMINI_API_KEY existe, sino SmolLM2 si instalado, sino rules. "
                            "gemini: Gemini 2.5 Flash Lite (necesita GEMINI_API_KEY). "
                            "local_open_source: Qwen/SmolLM2 local opcional con LoRA/QLoRA (instala requirements-llm.txt). "
                            "Puede tardar más en CPU. Si falla, usará rules. "
                            "rules: redacción determinista sin LLM."
                        ),
                    )

                analyze_btn = gr.Button("🚀 ANALIZAR VIDEO", variant="primary", size="lg")

            # ══════════════════════ RESULTADOS ══════════════════════
            with gr.Column(scale=7):
                with gr.Tab("📋 Resumen ejecutivo"):
                    summary_md = gr.Markdown(value="*Carga un video o URL y analiza.*", elem_classes="yba-result-card")

                with gr.Tab("🧠 Recomendación LLM"):
                    redaccion_md_out  = gr.Markdown(value="*La recomendación del LLM aparecerá aquí.*")
                    redaccion_meta_out = gr.Markdown(
                        value="*La fuente y estado de generación aparecerán aquí.*",
                        elem_classes="yba-result-card",
                    )

                with gr.Tab("📊 Gráficos"):
                    score_img      = gr.Image(label="Diagnóstico", interactive=False, height=380)
                    projection_img = gr.Image(label="Actual vs esperado con pauta", interactive=False, height=380)
                    policy_img     = gr.Image(label="Riesgo publicitario", interactive=False, height=320)

                with gr.Tab("🎨 Composición visual + OCR"):
                    visual_panel = gr.Markdown(
                        value="*El análisis visual aparecerá aquí.*",
                        elem_classes="yba-result-card",
                    )
                    frames_gallery = gr.Gallery(
                        label="Frames anotados: tercios + centroide focal",
                        columns=5, rows=2, object_fit="contain", height=520, preview=True,
                    )
                    ocr_panel = gr.Markdown(
                        value="*El análisis OCR aparecerá aquí.*",
                        elem_classes="yba-result-card",
                    )

                with gr.Tab("💬 Sentimiento"):
                    sentiment_panel = gr.Markdown(
                        value="*El análisis de sentimiento aparecerá aquí.*",
                        elem_classes="yba-result-card",
                    )
                    sentiment_bar_img = gr.Image(
                        label="Distribución de sentimiento (%)",
                        interactive=False,
                        height=320,
                    )
                    with gr.Row():
                        wordcloud_pos_img = gr.Image(
                            label="Nube de palabras positivas",
                            interactive=False,
                            height=260,
                        )
                        wordcloud_neu_img = gr.Image(
                            label="Nube de palabras neutras",
                            interactive=False,
                            height=260,
                        )
                        wordcloud_neg_img = gr.Image(
                            label="Nube de palabras negativas",
                            interactive=False,
                            height=260,
                        )

                with gr.Tab("✍️ Guion"):
                    script_panel = gr.Markdown(
                        value="*El análisis de guion aparecerá aquí.*",
                        elem_classes="yba-result-card",
                    )

                with gr.Tab("🛡️ Políticas"):
                    policy_panel = gr.Markdown(
                        value="*La evaluación de políticas aparecerá aquí.*",
                        elem_classes="yba-result-card",
                    )

                with gr.Tab("📝 Transcripción"):
                    transcript_out = gr.Markdown(
                        value="*La transcripción automática y su análisis aparecerán aquí.*",
                        elem_classes="yba-result-card",
                    )

                with gr.Tab("👁️ OCR texto"):
                    ocr_out = gr.Textbox(label="Texto detectado en pantalla", lines=6)

                with gr.Tab("🔬 LLM técnico"):
                    llm_panel = gr.Markdown(
                        value="*El diagnóstico comunicacional aparecerá aquí.*",
                        elem_classes="yba-result-card",
                    )

                with gr.Tab("📈 Métricas"):
                    metricas_panel = gr.Markdown(
                        value="*Las métricas aparecerán aquí.*",
                        elem_classes="yba-result-card",
                    )

                with gr.Tab("⚠️ Advertencias"):
                    warnings_md = gr.Markdown(value="_Sin análisis aún._")

                with gr.Tab("🧬 JSON / API"):
                    result_json = gr.JSON(label="Salida completa")

        # ── Botón "Rellenar con enlace" ──────────────────────────────────────
        fill_btn.click(
            fill_from_url,
            inputs=[youtube_url, channel_url],
            outputs=[title, description, category, topic, views, likes, comments,
                     duration_seconds, hours_since_publication, followers_count, avg_channel_reach, shares, channel_url, fill_status],
        )

        # ── Botón principal ──────────────────────────────────────────────────
        analyze_btn.click(
            analyze_video,
            inputs=[
                demo_case, youtube_url, channel_url, video_file,
                title, description, category, topic,
                views, likes, comments,
                shares, retention_rate, average_watch_time,
                hours_since_publication, followers_count, avg_channel_reach,
                duration_seconds,
                manual_transcript, manual_ocr_text, manual_comments,
                video_type, cpm_estimado, presupuesto, llm_mode,
            ],
            outputs=[
                summary_md, result_json,
                transcript_out, ocr_out,
                metricas_panel, llm_panel, ocr_panel, policy_panel,
                visual_panel, script_panel, sentiment_panel,
                frames_gallery,
                score_img, projection_img, policy_img,
                warnings_md, redaccion_md_out, redaccion_meta_out,
                sentiment_bar_img, wordcloud_pos_img, wordcloud_neu_img, wordcloud_neg_img,
            ],
            api_name="analyze",
        )

    return demo


if __name__ == "__main__":
    build_demo().launch()

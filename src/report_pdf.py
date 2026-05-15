"""Generación de reporte ejecutivo PDF para YouTube AI Recommendations.

El PDF se genera desde el JSON de diagnóstico que ya produce la app. El LLM no
calcula métricas: solo se incorpora la redacción final ya generada/estabilizada.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen.canvas import Canvas

PAGE_W, PAGE_H = landscape(A4)
MARGIN = 28

# Paleta blanca/elegante, inspirada en tarjetas tipo dashboard web.
INK = colors.HexColor("#101828")
MUTED = colors.HexColor("#667085")
SOFT = colors.HexColor("#F8FAFC")
BORDER = colors.HexColor("#E5E7EB")
BLUE = colors.HexColor("#2563EB")
VIOLET = colors.HexColor("#7C3AED")
GREEN = colors.HexColor("#16A34A")
YELLOW = colors.HexColor("#D97706")
RED = colors.HexColor("#DC2626")
SLATE = colors.HexColor("#334155")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _fmt_num(value: Any) -> str:
    try:
        v = float(value or 0)
        if abs(v) >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if abs(v) >= 1_000:
            return f"{v/1_000:.1f}K"
        if v == int(v):
            return f"{int(v):,}".replace(",", ".")
        return f"{v:.2f}"
    except Exception:
        return "-"


def _fmt_money(value: Any) -> str:
    v = _safe_float(value, 0.0)
    return f"${v:.2f}"


def _fmt_pct(value: Any, *, already_pct: bool = False) -> str:
    v = _safe_float(value, 0.0)
    if not already_pct and abs(v) <= 1.5:
        v *= 100
    return f"{v:.1f}%"


def _first(*values: Any, default: str = "-") -> str:
    for value in values:
        if value not in (None, "", [], {}):
            return str(value)
    return default


def _clean_text(text: Any, max_chars: int = 420) -> str:
    raw = str(text or "")
    raw = re.sub(r"```.*?```", " ", raw, flags=re.DOTALL)
    raw = re.sub(r"[#*_`>|]", " ", raw)
    raw = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    if len(raw) > max_chars:
        return raw[: max_chars - 1].rstrip() + "…"
    return raw


def _decision(result: Dict[str, Any]) -> Tuple[str, colors.Color, str, str]:
    """Decisión canónica del PDF.

    Evita contradicciones entre el titular y la explicación. La prioridad es:
    política/riesgo > acción final del sistema > gate XGBoost > score.
    """
    xgb = result.get("xgboost_pauta") or (result.get("metricas", {}) or {}).get("xgboost_pauta") or {}
    policy = result.get("analisis_politicas", {}) or {}
    exec_xgb = ((result.get("resumen_ejecutivo") or {}).get("xgboost_summary") or {})

    raw = str(
        result.get("accion_final")
        or result.get("recomendacion_impulso")
        or result.get("decision")
        or exec_xgb.get("decision")
        or "MONITOREAR"
    ).strip().upper()
    exec_decision = str(exec_xgb.get("decision") or "").upper()
    policy_level = str(result.get("policy_risk_level") or policy.get("policy_risk_level") or "bajo").lower()
    score = _safe_float(result.get("score_hibrido", result.get("probabilidad_rendimiento", 0)), 0.0)
    eligible = bool(xgb.get("eligible_for_paid_xgboost"))
    gate_passed = bool(xgb.get("gate_passed", eligible))

    if policy_level in {"alto", "high", "revisión humana", "revision humana"} or "REVIS" in raw or "HUMAN" in raw:
        return "RECOMENDACIÓN: REVISIÓN HUMANA", VIOLET, "Requiere validación del equipo antes de decidir.", "review"

    if "NO INVERT" in exec_decision or ("NO" in raw and ("IMPUL" in raw or "PAUT" in raw)) or "DESCART" in raw:
        return "RECOMENDACIÓN: NO PAUTAR", RED, "No conviene invertir con las señales actuales.", "no"

    # Si el segundo modelo/gate no habilita pauta, el PDF no debe decir que sí inviertas aunque exista texto LLM positivo.
    if xgb and not eligible and not gate_passed:
        return "RECOMENDACIÓN: AJUSTAR ANTES DE PAUTAR", YELLOW, "El gate de pauta no se supera; optimiza y vuelve a evaluar.", "adjust"

    if "AJUSTAR" in raw or "MONIT" in raw:
        return "RECOMENDACIÓN: AJUSTAR ANTES DE PAUTAR", YELLOW, "Optimiza puntos críticos antes de invertir presupuesto.", "adjust"

    if ("SÍ INVERT" in exec_decision or "SI INVERT" in exec_decision or "IMPULSAR" in raw or "PAUTAR" in raw) and (eligible or score >= 0.60):
        return "RECOMENDACIÓN: PAUTAR", GREEN, "El contenido muestra señales suficientes para inversión controlada.", "invest"

    return "RECOMENDACIÓN: MONITOREAR, NO PAUTAR AÚN", SLATE, "Aún faltan señales para invertir con seguridad.", "monitor"


def _consistent_executive_text(result: Dict[str, Any], decision_key: str, decision_subtitle: str) -> str:
    """Texto determinista y alineado con la decisión del titular.

    No usa la redacción cruda del LLM cuando puede contradecir la decisión final.
    """
    metricas = result.get("metricas", {}) or {}
    ops = metricas.get("operational", {}) or {}
    xgb = result.get("xgboost_pauta") or metricas.get("xgboost_pauta") or {}
    policy = result.get("analisis_politicas", {}) or {}

    prob = _fmt_pct(result.get("probabilidad_rendimiento", 0))
    score = _fmt_pct(result.get("score_hibrido", 0))
    cpm = _fmt_money(xgb.get("predicted_cpm") or result.get("cpm_estimado") or 0)
    impressions = _fmt_num(xgb.get("unified_estimated_impressions") or xgb.get("estimated_budget_impressions") or result.get("impresiones_estimadas_promedio") or 0)
    retention = _fmt_pct(ops.get("retention_rate", 0))
    risk = str(result.get("policy_risk_level") or policy.get("policy_risk_level") or "bajo")

    if decision_key == "invest":
        return (
            f"{decision_subtitle} La probabilidad del modelo es {prob}, el score híbrido es {score} y el CPM estimado es {cpm}. "
            f"Con el presupuesto evaluado se proyectan aproximadamente {impressions} impresiones. La pauta debe iniciar como prueba controlada, "
            f"monitoreando CPM real, CTR y retención ({retention}) antes de escalar inversión."
        )
    if decision_key == "no":
        return (
            f"{decision_subtitle} Aunque exista un CPM de referencia ({cpm}), la decisión operativa es no invertir. "
            f"El score híbrido ({score}), la probabilidad ({prob}) o el riesgo ({risk}) no justifican activar pauta. "
            "La prioridad es corregir los puntos débiles y volver a evaluar antes de asignar presupuesto."
        )
    if decision_key == "review":
        return (
            f"{decision_subtitle} El contenido requiere revisión por riesgo de políticas o señales sensibles. "
            f"No debe pautarse hasta validar claims, texto en pantalla, guion y contexto. CPM referencial: {cpm}; no debe interpretarse como autorización de inversión."
        )
    if decision_key == "adjust":
        return (
            f"{decision_subtitle} La pauta no queda habilitada todavía. Probabilidad: {prob}; score híbrido: {score}; retención: {retention}; CPM referencial: {cpm}. "
            "Conviene optimizar hook, claridad del mensaje, CTA/cierre, texto en pantalla y coherencia creativa antes de invertir."
        )
    return (
        f"{decision_subtitle} El sistema recomienda observar más señales antes de invertir. Probabilidad: {prob}; score híbrido: {score}; "
        f"riesgo: {risk}; CPM referencial: {cpm}. Usa esta lectura para seguimiento, no para activar pauta inmediata."
    )


def _draw_wrapped(
    c: Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    font: str = "Helvetica",
    size: int = 8,
    leading: float = 10,
    color: colors.Color = INK,
    max_lines: int = 6,
) -> float:
    c.setFont(font, size)
    c.setFillColor(color)
    lines: List[str] = []
    for paragraph in str(text or "").split("\n"):
        lines.extend(simpleSplit(paragraph.strip(), font, size, width))
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = lines[-1].rstrip(" .,") + "…"
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def _card(c: Canvas, x: float, y: float, w: float, h: float, title: str = "") -> None:
    c.setFillColor(colors.white)
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, 12, fill=1, stroke=1)
    if title:
        c.setFont("Helvetica-Bold", 8.5)
        c.setFillColor(SLATE)
        c.drawString(x + 12, y + h - 18, title.upper())


def _metric_card(c: Canvas, x: float, y: float, w: float, h: float, label: str, value: str, color: colors.Color = BLUE) -> None:
    _card(c, x, y, w, h)
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(x + 10, y + h - 16, label.upper())
    c.setFont("Helvetica-Bold", 17)
    c.setFillColor(color)
    c.drawString(x + 10, y + 16, value)


def _draw_table_rows(c: Canvas, rows: Iterable[Tuple[str, str]], x: float, y: float, w: float, row_h: float = 16) -> None:
    c.setFont("Helvetica", 8)
    for idx, (label, value) in enumerate(rows):
        yy = y - idx * row_h
        if idx % 2 == 0:
            c.setFillColor(SOFT)
            c.roundRect(x, yy - 3, w, row_h, 4, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7.7)
        c.drawString(x + 7, yy + 1, label)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 8.2)
        c.drawRightString(x + w - 7, yy + 1, value)


def _draw_bar(c: Canvas, x: float, y: float, w: float, label: str, value: float, color: colors.Color) -> None:
    value = max(0.0, min(100.0, value))
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(x, y + 9, label)
    c.setFillColor(colors.HexColor("#EEF2F7"))
    c.roundRect(x + 68, y + 7, w - 108, 7, 3.5, fill=1, stroke=0)
    c.setFillColor(color)
    c.roundRect(x + 68, y + 7, (w - 108) * value / 100.0, 7, 3.5, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 7.5)
    c.drawRightString(x + w - 8, y + 7.8, f"{value:.1f}%")


def _valid_image(path: Any) -> Optional[str]:
    if not path:
        return None
    p = Path(str(path))
    if p.exists() and p.is_file():
        return str(p)
    return None


def _draw_image_fit(c: Canvas, path: str, x: float, y: float, w: float, h: float) -> bool:
    try:
        img = ImageReader(path)
        iw, ih = img.getSize()
        if iw <= 0 or ih <= 0:
            return False
        scale = min(w / iw, h / ih)
        dw, dh = iw * scale, ih * scale
        c.drawImage(img, x + (w - dw) / 2, y + (h - dh) / 2, dw, dh, preserveAspectRatio=True, mask="auto")
        return True
    except Exception:
        return False


def _module_status(result: Dict[str, Any]) -> List[Tuple[str, str]]:
    modules = result.get("modulos_analizados") or {}
    mapping = [
        ("Modelo predictivo", result.get("modelo") or result.get("prediccion_rendimiento")),
        ("XGBoost pauta", modules.get("xgboost_pauta") or result.get("xgboost_pauta")),
        ("OCR", modules.get("ocr") or result.get("analisis_ocr")),
        ("Transcripción", modules.get("transcripcion") or result.get("analisis_transcripcion")),
        ("Visual", modules.get("composicion_visual") or result.get("analisis_visual")),
        ("Sentimiento", modules.get("sentimiento") or result.get("analisis_sentimiento")),
        ("Políticas", modules.get("politicas") or result.get("analisis_politicas")),
        ("LLM", modules.get("llm") or result.get("recomendacion_redactada")),
    ]
    return [(name, "OK" if bool(value) else "-") for name, value in mapping]


def generate_executive_pdf(result: Dict[str, Any], output_dir: str | Path = "outputs/reports") -> str:
    """Genera un PDF ejecutivo de una página y devuelve la ruta.

    Parameters
    ----------
    result:
        JSON/dict de salida de `analyze_video`.
    output_dir:
        Carpeta de salida. Por defecto `outputs/reports`.
    """
    if not isinstance(result, dict) or not result:
        raise ValueError("No hay datos analizados para generar el PDF.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = out_dir / f"reporte_ejecutivo_youtube_{stamp}.pdf"

    metricas = result.get("metricas", {}) or {}
    ops = metricas.get("operational", {}) or {}
    xgb = result.get("xgboost_pauta") or metricas.get("xgboost_pauta") or {}
    sentiment = result.get("analisis_sentimiento", {}) or {}
    policy = result.get("analisis_politicas", {}) or {}
    visual = result.get("analisis_visual", {}) or {}
    script = result.get("analisis_guion", {}) or {}
    trans = result.get("analisis_transcripcion", {}) or {}
    ocr = result.get("analisis_ocr", {}) or {}
    metadata = result.get("metadata_youtube", {}) or {}
    redac = result.get("recomendacion_redactada", {}) or {}

    decision, decision_color, decision_subtitle, decision_key = _decision(result)
    title = _first(metadata.get("title"), result.get("title"), default="Video analizado")
    channel = _first(metadata.get("channel_title"), metadata.get("channel"), default="Canal no especificado")

    c = Canvas(str(pdf_path), pagesize=landscape(A4))
    c.setTitle("Reporte ejecutivo YouTube AI Recommendations")
    c.setAuthor("YouTube AI Recommendations")

    # Fondo
    c.setFillColor(colors.white)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#F5F7FB"))
    c.rect(0, 0, PAGE_W, 88, fill=1, stroke=0)

    # Header principal
    header_x, header_y, header_w, header_h = MARGIN, PAGE_H - 86, PAGE_W - 2 * MARGIN, 58
    c.setFillColor(decision_color)
    c.roundRect(header_x, header_y, header_w, header_h, 16, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(header_x + 20, header_y + 31, decision)
    c.setFont("Helvetica", 9)
    c.drawString(header_x + 20, header_y + 15, decision_subtitle)
    c.setFont("Helvetica-Bold", 8)
    c.drawRightString(header_x + header_w - 20, header_y + 36, "YOUTUBE AI RECOMMENDATIONS")
    c.setFont("Helvetica", 7.5)
    c.drawRightString(header_x + header_w - 20, header_y + 20, datetime.now().strftime("%d/%m/%Y %H:%M"))

    # Título/metadata
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 12)
    _draw_wrapped(c, title, MARGIN, PAGE_H - 104, PAGE_W - 2 * MARGIN - 135, font="Helvetica-Bold", size=11.5, leading=12, max_lines=1)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN, PAGE_H - 119, f"Canal: {channel}")

    # KPI cards
    kpi_y = PAGE_H - 178
    kpi_w = (PAGE_W - 2 * MARGIN - 24) / 4
    prob = _fmt_pct(result.get("probabilidad_rendimiento", 0))
    score = _fmt_pct(result.get("score_hibrido", 0))
    risk = str(result.get("policy_risk_level") or policy.get("policy_risk_level") or "bajo").upper()
    cpm = _fmt_money(xgb.get("predicted_cpm") or result.get("cpm_estimado") or 0)
    _metric_card(c, MARGIN, kpi_y, kpi_w, 46, "Probabilidad", prob, BLUE)
    _metric_card(c, MARGIN + kpi_w + 8, kpi_y, kpi_w, 46, "Score híbrido", score, VIOLET)
    _metric_card(c, MARGIN + 2 * (kpi_w + 8), kpi_y, kpi_w, 46, "Riesgo", risk, RED if risk in {"ALTO", "REVISIÓN HUMANA"} else GREEN)
    _metric_card(c, MARGIN + 3 * (kpi_w + 8), kpi_y, kpi_w, 46, "CPM estimado", cpm, YELLOW)

    # Cards intermedias
    mid_y, mid_h = 280, 130
    left_x, left_w = MARGIN, 248
    center_x, center_w = left_x + left_w + 14, 230
    right_x, right_w = center_x + center_w + 14, PAGE_W - MARGIN - (center_x + center_w + 14)

    _card(c, left_x, mid_y, left_w, mid_h, "Estadística extraída")
    engagement = _fmt_pct(metricas.get("engagement_rate", 0))
    retention = _fmt_pct(ops.get("retention_rate", 0))
    metric_rows = [
        ("Visualizaciones", _fmt_num(metricas.get("views", 0))),
        ("Likes", _fmt_num(metricas.get("likes", 0))),
        ("Comentarios", _fmt_num(metricas.get("comments", 0))),
        ("Engagement", engagement),
        ("Retención", retention),
        ("Views/hora", _fmt_num(ops.get("views_per_hour", 0))),
    ]
    _draw_table_rows(c, metric_rows, left_x + 10, mid_y + mid_h - 40, left_w - 20, row_h=15.5)

    _card(c, center_x, mid_y, center_w, mid_h, "Análisis de sentimiento")
    _draw_bar(c, center_x + 13, mid_y + 86, center_w - 26, "Positivo", _safe_float(sentiment.get("pct_positivo")), GREEN)
    _draw_bar(c, center_x + 13, mid_y + 61, center_w - 26, "Neutral", _safe_float(sentiment.get("pct_neutro")), MUTED)
    _draw_bar(c, center_x + 13, mid_y + 36, center_w - 26, "Negativo", _safe_float(sentiment.get("pct_negativo")), RED)
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(center_x + 13, mid_y + 14, f"Total comentarios analizados: {_fmt_num(sentiment.get('total', 0))}")

    _card(c, right_x, mid_y, right_w, mid_h, "Cobertura de módulos")
    status_rows = _module_status(result)
    sx, sy = right_x + 12, mid_y + mid_h - 41
    col_w = (right_w - 24) / 2
    for idx, (name, status) in enumerate(status_rows[:8]):
        col = idx % 2
        row = idx // 2
        x = sx + col * col_w
        y = sy - row * 21
        c.setFillColor(GREEN if status == "OK" else MUTED)
        c.circle(x + 4, y + 4, 3.2, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(x + 12, y + 1, name)

    # Frames + recomendación
    bot_y, bot_h = 104, 155
    frames_w = 352
    reco_x = MARGIN + frames_w + 14
    reco_w = PAGE_W - MARGIN - reco_x
    _card(c, MARGIN, bot_y, frames_w, bot_h, "Capturas de frames representativos")
    frame_paths = []
    for key in ("annotated_frame_paths", "frame_paths", "frames"):
        vals = visual.get(key) or []
        if isinstance(vals, list):
            frame_paths.extend(vals)
    frame_paths = [p for p in (_valid_image(p) for p in frame_paths) if p][:3]
    thumb_w = (frames_w - 34) / 3
    if frame_paths:
        for idx, path in enumerate(frame_paths):
            x = MARGIN + 10 + idx * (thumb_w + 7)
            y = bot_y + 33
            c.setFillColor(SOFT)
            c.roundRect(x, y, thumb_w, 78, 8, fill=1, stroke=0)
            _draw_image_fit(c, path, x + 2, y + 2, thumb_w - 4, 74)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7)
        c.drawString(MARGIN + 12, bot_y + 16, "Frames extraídos por el módulo visual/OCR para validar composición y texto en pantalla.")
    else:
        c.setFillColor(SOFT)
        c.roundRect(MARGIN + 12, bot_y + 35, frames_w - 24, 75, 8, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 8)
        c.drawCentredString(MARGIN + frames_w / 2, bot_y + 71, "Sin frames disponibles. Sube un MP4 para OCR, visual y capturas.")

    _card(c, reco_x, bot_y, reco_w, bot_h, "Explicación ejecutiva")
    reco_text = _clean_text(_consistent_executive_text(result, decision_key, decision_subtitle), 760)
    _draw_wrapped(c, reco_text, reco_x + 12, bot_y + bot_h - 40, reco_w - 24, size=8.2, leading=10.2, color=INK, max_lines=9)

    # Footer módulos específicos
    footer_y = 30
    footer_h = 58
    _card(c, MARGIN, footer_y, PAGE_W - 2 * MARGIN, footer_h, "Resumen operativo por módulo")
    visual_score = visual.get("composition_score") or visual.get("score") or 0
    if _safe_float(visual_score) <= 1.5:
        visual_label = _fmt_pct(visual_score)
    else:
        visual_label = f"{_safe_float(visual_score):.0f}/100"
    footer_items = [
        ("Visual", visual_label),
        ("Guion", f"{_safe_float(script.get('script_quality_score', 0)):.0f}/100"),
        ("OCR", f"{_fmt_num(ocr.get('ocr_word_count') or ocr.get('cantidad_palabras') or 0)} palabras"),
        ("Transcripción", f"{_fmt_num(trans.get('cantidad_palabras') or trans.get('word_count') or 0)} palabras"),
        ("Pauta", f"{_fmt_num(xgb.get('unified_estimated_impressions') or result.get('impresiones_estimadas_promedio') or 0)} imp."),
    ]
    item_w = (PAGE_W - 2 * MARGIN - 22) / 5
    for i, (label, value) in enumerate(footer_items):
        x = MARGIN + 12 + i * item_w
        c.setFont("Helvetica", 7.3)
        c.setFillColor(MUTED)
        c.drawString(x, footer_y + 25, label.upper())
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(INK)
        c.drawString(x, footer_y + 11, value)

    # Nota metodológica corta
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 6.8)
    c.drawRightString(PAGE_W - MARGIN, 14, "El LLM redacta la explicación; la decisión y métricas provienen de los módulos analíticos de la app.")

    c.showPage()
    c.save()
    return str(pdf_path)

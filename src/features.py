"""Construcción de variables para entrenamiento e inferencia."""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from .config import BENEFIT_TERMS, CTA_TERMS, PRICE_TERMS, PROMO_TERMS, TRUST_TERMS, URGENCY_TERMS


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convierte a float aceptando None, strings y NaN sin fallar."""
    try:
        if value is None or value == "":
            return default
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    return int(round(safe_float(value, default=default)))


def contains_any(text: str, terms: Iterable[str]) -> int:
    lower = (text or "").lower()
    return int(any(term in lower for term in terms))


def uppercase_ratio(text: str) -> float:
    letters = [c for c in text or "" if c.isalpha()]
    if not letters:
        return 0.0
    return round(sum(1 for c in letters if c.isupper()) / len(letters), 4)


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text or "", flags=re.UNICODE))




STOPWORDS_RELEVANCE = {
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o",
    "en", "a", "por", "para", "con", "sin", "del", "al", "que", "como",
    "más", "mas", "muy", "este", "esta", "estos", "estas", "eso", "esa",
    "es", "son", "ser", "fue", "hay", "se", "su", "sus", "tu", "tus",
    "mi", "mis", "lo", "le", "les", "video", "youtube", "canal",
    "comentario", "comentarios", "contenido", "link", "http", "https",
    "www", "com", "reel", "short", "shorts", "post", "publicacion", "publicación"
}

HUMOR_TERMS = {
    "jaja", "jajaja", "jeje", "risa", "humor", "broma", "chiste", "meme",
    "parodia", "sketch", "comedia", "gracioso", "graciosa", "sarcasmo",
    "irónico", "ironico", "pov", "cuando", "fail", "fails", "absurdo", "random"
}
EDU_TERMS = {
    "tutorial", "aprende", "aprender", "guía", "guia", "paso a paso", "cómo",
    "como hacer", "explico", "explicar", "consejos", "tips", "clase", "curso"
}
COMMERCIAL_TERMS = {
    "compra", "comprar", "precio", "descuento", "oferta", "promoción", "promocion",
    "cupón", "cupon", "envío", "envio", "agenda", "reserva", "servicio",
    "producto", "tienda", "cliente", "clientes", "venta", "ventas", "gratis",
    "whatsapp", "cotiza", "cotizar", "link en bio"
}
MUSIC_TERMS = {
    "música", "musica", "canción", "cancion", "letra", "lyrics", "cover",
    "artista", "cantante", "banda", "álbum", "album", "single", "concierto",
    "verso", "coro", "beat", "remix", "videoclip"
}


def relevance_tokens(text: str) -> set[str]:
    """Tokens útiles para medir si el texto OCR conecta con el mensaje."""
    clean = (text or "").lower()
    clean = re.sub(r"https?://\S+", " ", clean)
    clean = re.sub(r"www\.\S+", " ", clean)
    clean = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", clean, flags=re.UNICODE)
    clean = re.sub(r"\s+", " ", clean).strip()
    return {
        token for token in clean.split()
        if len(token) >= 3 and token not in STOPWORDS_RELEVANCE
    }


def _count_custom_terms(text: str, terms: Iterable[str]) -> int:
    low = (text or "").lower()
    return sum(1 for term in terms if str(term).lower() in low)


def infer_content_intent_light(title: str = "", description: str = "", transcript_text: str = "", ocr_text: str = "") -> str:
    """Detecta intención básica para no aplicar lógica comercial a todo.

    Retorna una etiqueta simple usada por OCR/guion: humor, comercial,
    educativo, informativo o general.
    """
    text = " ".join([title or "", description or "", transcript_text or "", ocr_text or ""]).lower()
    humor = _count_custom_terms(text, HUMOR_TERMS)
    edu = _count_custom_terms(text, EDU_TERMS)
    music = _count_custom_terms(text, MUSIC_TERMS)
    commercial = _count_custom_terms(text, COMMERCIAL_TERMS) + contains_any(text, CTA_TERMS) + contains_any(text, PROMO_TERMS) + contains_any(text, PRICE_TERMS)
    if music >= 1 and commercial < 3:
        return "musical"
    if humor >= 1 and commercial < 3:
        return "humor/entretenimiento"
    if commercial >= 3:
        return "comercial/promocional"
    if edu >= 2:
        return "educativo/tutorial"
    if any(x in text for x in ["noticia", "actualidad", "denuncia", "entrevista", "comunicado"]):
        return "informativo/noticioso"
    return "general/branding"


def _ocr_role_for_intent(ocr_text: str, intent: str, overlap: float, marketing_signal: float, humor_signal: float) -> str:
    if not (ocr_text or "").strip():
        return "sin_texto_detectado"
    if intent.startswith("musical"):
        return "letra_musical_o_rotulo" if overlap >= 0.15 else "texto_musical_poco_conectado"
    if intent.startswith("humor"):
        if humor_signal > 0 or overlap >= 0.25:
            return "apoya_contexto_o_remate"
        return "texto_desconectado_del_chiste"
    if intent.startswith("comercial"):
        if marketing_signal >= 0.34:
            return "refuerza_oferta_cta_o_beneficio"
        if overlap >= 0.25:
            return "acompaña_mensaje_comercial"
        return "texto_comercial_poco_claro"
    if intent.startswith("educativo"):
        return "apoyo_explicativo" if overlap >= 0.20 else "texto_explicativo_poco_conectado"
    return "contexto_visual" if overlap >= 0.20 else "texto_visible_sin_funcion_clara"


def compute_ocr_relevance_score(
    ocr_text: str,
    title: str = "",
    description: str = "",
    transcript_text: str = "",
    ocr_metrics: Dict[str, Any] | None = None,
) -> float:
    """Score 0-1 de relevancia del texto en pantalla.

    La versión anterior trataba casi todo como pieza comercial. Esta versión es
    sensible al tipo de contenido: humor/entretenimiento, educativo, comercial,
    informativo o general. Por eso no exige CTA ni producto cuando el video no
    está vendiendo nada.
    """
    ocr_metrics = ocr_metrics or {}
    ocr_text = ocr_text or ""
    if not ocr_text.strip():
        return 0.0

    intent = str(ocr_metrics.get("ocr_llm_content_intent") or infer_content_intent_light(title, description, transcript_text, ocr_text))
    ocr_tokens = relevance_tokens(ocr_text)
    context_tokens = relevance_tokens(" ".join([title or "", description or "", transcript_text or ""]))
    overlap = (len(ocr_tokens & context_tokens) / max(len(ocr_tokens), 1)) if ocr_tokens and context_tokens else 0.0

    commercial_flags = [
        contains_any(ocr_text, CTA_TERMS), contains_any(ocr_text, BENEFIT_TERMS),
        contains_any(ocr_text, URGENCY_TERMS), contains_any(ocr_text, TRUST_TERMS),
        contains_any(ocr_text, PROMO_TERMS), contains_any(ocr_text, PRICE_TERMS),
        _count_custom_terms(ocr_text, COMMERCIAL_TERMS) > 0,
    ]
    marketing_signal = sum(bool(x) for x in commercial_flags) / max(len(commercial_flags), 1)
    humor_signal = min(_count_custom_terms(ocr_text, HUMOR_TERMS) / 2, 1.0)
    edu_signal = min(_count_custom_terms(ocr_text, EDU_TERMS) / 2, 1.0)

    coverage = min(safe_float(ocr_metrics.get("ocr_frame_coverage", 0)), 1.0)
    wc = count_words(ocr_text)
    if 2 <= wc <= 18:
        density_fit = 1.0
    elif 19 <= wc <= 40:
        density_fit = 0.65
    elif wc > 40:
        density_fit = 0.35
    else:
        density_fit = 0.45

    if intent.startswith("musical"):
        # En música el texto puede ser letra, título de canción, artista o rótulo;
        # no se exige CTA ni venta.
        score = 0.42 * overlap + 0.25 * coverage + 0.23 * density_fit + 0.10 * max(humor_signal, edu_signal)
    elif intent.startswith("humor"):
        # En humor el texto puede ser caption, setup, meme/subtítulo o remate;
        # no tiene que tener CTA ni beneficio comercial.
        score = 0.45 * overlap + 0.20 * humor_signal + 0.20 * coverage + 0.15 * density_fit
    elif intent.startswith("educativo"):
        score = 0.45 * overlap + 0.20 * edu_signal + 0.20 * coverage + 0.15 * density_fit
    elif intent.startswith("comercial"):
        score = 0.38 * overlap + 0.34 * marketing_signal + 0.18 * coverage + 0.10 * density_fit
    else:
        score = 0.50 * overlap + 0.20 * coverage + 0.20 * density_fit + 0.10 * max(marketing_signal, humor_signal, edu_signal)

    return round(max(0.0, min(score, 1.0)), 4)


def describe_ocr_relevance(
    ocr_text: str,
    title: str = "",
    description: str = "",
    transcript_text: str = "",
    ocr_metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Devuelve explicación del OCR sensible al objetivo del video."""
    ocr_metrics = ocr_metrics or {}
    intent = str(ocr_metrics.get("ocr_llm_content_intent") or infer_content_intent_light(title, description, transcript_text, ocr_text))
    ocr_tokens = relevance_tokens(ocr_text)
    context_tokens = relevance_tokens(" ".join([title or "", description or "", transcript_text or ""]))
    overlap = (len(ocr_tokens & context_tokens) / max(len(ocr_tokens), 1)) if ocr_tokens and context_tokens else 0.0
    commercial_flags = [
        contains_any(ocr_text, CTA_TERMS), contains_any(ocr_text, BENEFIT_TERMS),
        contains_any(ocr_text, URGENCY_TERMS), contains_any(ocr_text, TRUST_TERMS),
        contains_any(ocr_text, PROMO_TERMS), contains_any(ocr_text, PRICE_TERMS),
        _count_custom_terms(ocr_text, COMMERCIAL_TERMS) > 0,
    ]
    marketing_signal = sum(bool(x) for x in commercial_flags) / max(len(commercial_flags), 1)
    humor_signal = min(_count_custom_terms(ocr_text, HUMOR_TERMS) / 2, 1.0)
    score = compute_ocr_relevance_score(ocr_text, title, description, transcript_text, ocr_metrics)
    role = str(ocr_metrics.get("ocr_llm_role") or _ocr_role_for_intent(ocr_text, intent, overlap, marketing_signal, humor_signal))
    if not (ocr_text or "").strip():
        interpretation = "No se detectó texto en pantalla; la lectura visual depende del audio, imagen y ritmo."
    elif intent.startswith("musical"):
        interpretation = "El texto en pantalla se evalúa como letra, rótulo, título, artista o apoyo visual del videoclip; no como CTA comercial."
    elif intent.startswith("humor"):
        interpretation = "El texto en pantalla se evalúa como soporte del chiste, contexto o remate; no como CTA de venta."
    elif intent.startswith("comercial"):
        interpretation = "El texto en pantalla se evalúa por su capacidad de reforzar oferta, beneficio, prueba o acción."
    elif intent.startswith("educativo"):
        interpretation = "El texto en pantalla se evalúa como apoyo explicativo: debe aclarar conceptos, pasos o ideas clave."
    else:
        interpretation = "El texto en pantalla se evalúa por su conexión con el tema central y por si ayuda a entender el video."
    return {
        "content_intent": intent,
        "ocr_role": role,
        "ocr_context_overlap": round(overlap, 4),
        "ocr_marketing_signal": round(marketing_signal, 4),
        "ocr_humor_signal": round(humor_signal, 4),
        "ocr_relevance_score": score,
        "ocr_interpretation": interpretation,
    }

def parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    except Exception:
        return None


def days_since_publication(published_at: Any) -> float:
    dt = parse_date(published_at)
    if not dt:
        return 1.0
    now = datetime.now(timezone.utc)
    days = (now - dt).total_seconds() / 86400
    return max(days, 1.0)


def duration_fit_score(duration_seconds: float) -> float:
    """Score heurístico de ajuste de duración para anuncios y contenido corto."""
    d = safe_float(duration_seconds)
    if d <= 0:
        return 0.5
    if d <= 7:
        return 0.95
    if d <= 15:
        return 0.90
    if d <= 30:
        return 0.80
    if d <= 60:
        return 0.70
    if d <= 180:
        return 0.55
    return 0.40


def compute_text_power_score(text: str, ocr_metrics: Dict[str, Any] | None = None) -> float:
    """Score 0-1 de potencia comunicacional por señales básicas."""
    ocr_metrics = ocr_metrics or {}
    score = 0.0
    score += 0.20 * contains_any(text, CTA_TERMS)
    score += 0.18 * contains_any(text, BENEFIT_TERMS)
    score += 0.15 * contains_any(text, URGENCY_TERMS)
    score += 0.12 * contains_any(text, TRUST_TERMS)
    score += 0.10 * contains_any(text, PROMO_TERMS)
    score += 0.08 * contains_any(text, PRICE_TERMS)
    wc = count_words(text)
    if 8 <= wc <= 120:
        score += 0.10
    elif wc > 120:
        score += 0.04
    if safe_float(ocr_metrics.get("ocr_frame_coverage", 0)) > 0:
        score += 0.07
    return round(min(score, 1.0), 4)


def compute_engagement_metrics(views: Any, likes: Any, comments: Any, published_at: Any = None) -> Dict[str, float]:
    views_f = max(safe_float(views), 0.0)
    likes_f = max(safe_float(likes), 0.0)
    comments_f = max(safe_float(comments), 0.0)
    denom = max(views_f, 1.0)
    days = days_since_publication(published_at)
    return {
        "views": views_f,
        "likes": likes_f,
        "comments": comments_f,
        "engagement_rate": round((likes_f + comments_f) / denom, 6),
        "like_rate": round(likes_f / denom, 6),
        "comment_rate": round(comments_f / denom, 6),
        "views_per_day": round(views_f / days, 6),
        "log_views": round(float(np.log1p(views_f)), 6),
        "log_likes": round(float(np.log1p(likes_f)), 6),
        "log_comments": round(float(np.log1p(comments_f)), 6),
    }


def build_text_features(title: str, description: str, transcript_text: str, ocr_text: str) -> Dict[str, Any]:
    title = title or ""
    description = description or ""
    transcript_text = transcript_text or ""
    ocr_text = ocr_text or ""
    text_total = " ".join([title, description, transcript_text, ocr_text]).strip()
    return {
        "title": title,
        "description": description,
        "transcript_text": transcript_text,
        "ocr_text": ocr_text,
        "text_total": text_total,
        "title_len": len(title),
        "description_len": len(description),
        "transcript_len": len(transcript_text),
        "ocr_text_len": len(ocr_text),
        "text_total_len": len(text_total),
        "title_word_count": count_words(title),
        "description_word_count": count_words(description),
        "transcript_word_count": count_words(transcript_text),
        "ocr_word_count": count_words(ocr_text),
        "text_total_word_count": count_words(text_total),
        "exclamation_count": text_total.count("!"),
        "question_count": text_total.count("?"),
        "uppercase_ratio": uppercase_ratio(text_total),
        "digit_count": sum(c.isdigit() for c in text_total),
        "cta_flag": contains_any(text_total, CTA_TERMS),
        "urgency_flag": contains_any(text_total, URGENCY_TERMS),
        "trust_flag": contains_any(text_total, TRUST_TERMS),
        "promo_flag": contains_any(text_total, PROMO_TERMS),
        "benefit_flag": contains_any(text_total, BENEFIT_TERMS),
        "price_flag": contains_any(text_total, PRICE_TERMS),
    }


def build_feature_row(
    title: str = "",
    description: str = "",
    transcript_text: str = "",
    ocr_text: str = "",
    category_id: Any = "",
    duration_seconds: Any = 0,
    views: Any = 0,
    likes: Any = 0,
    comments: Any = 0,
    published_at: Any = None,
    video_type: str = "auto",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Construye una fila de features para inferencia."""
    extra = extra or {}
    text_features = build_text_features(title, description, transcript_text, ocr_text)
    engagement = compute_engagement_metrics(views, likes, comments, published_at)
    d = safe_float(duration_seconds)
    ocr_relevance_info = describe_ocr_relevance(
        ocr_text=ocr_text,
        title=title,
        description=description,
        transcript_text=transcript_text,
        ocr_metrics=extra,
    )
    ocr_relevance = safe_float(ocr_relevance_info.get("ocr_relevance_score", 0))
    content_intent = ocr_relevance_info.get("content_intent") or infer_content_intent_light(title, description, transcript_text, ocr_text)
    text_power = compute_text_power_score(text_features["text_total"], extra)
    vt = video_type or "auto"
    if vt == "auto":
        vt = "sin narrador" if d and d <= 8 else "con narrador/desconocido"
    return {
        **text_features,
        **engagement,
        "category_id": str(category_id or "unknown"),
        "duration_seconds": d,
        "duration_fit_score": duration_fit_score(d),
        "duration_bucket": "short" if d <= 15 else "medium" if d <= 60 else "long",
        "video_type": vt,
        "content_intent": content_intent,
        "text_power_score": text_power,
        "ocr_relevance_score": ocr_relevance,
        **ocr_relevance_info,
        **extra,
    }


def metricas_block(features: Dict[str, Any]) -> Dict[str, Any]:
    """Subdiccionario `metricas` para el contrato de salida final."""
    return {
        "views": int(safe_float(features.get("views", 0))),
        "likes": int(safe_float(features.get("likes", 0))),
        "comments": int(safe_float(features.get("comments", 0))),
        "engagement_rate": round(safe_float(features.get("engagement_rate", 0)), 6),
        "like_rate": round(safe_float(features.get("like_rate", 0)), 6),
        "comment_rate": round(safe_float(features.get("comment_rate", 0)), 6),
        "views_per_day": round(safe_float(features.get("views_per_day", 0)), 4),
        "duration_seconds": round(safe_float(features.get("duration_seconds", 0)), 2),
        "duration_fit_score": round(safe_float(features.get("duration_fit_score", 0)), 4),
        "text_power_score": round(safe_float(features.get("text_power_score", 0)), 4),
        "ocr_relevance_score": round(safe_float(features.get("ocr_relevance_score", 0)), 4),
        "title_word_count": int(safe_float(features.get("title_word_count", 0))),
        "description_word_count": int(safe_float(features.get("description_word_count", 0))),
        "transcript_word_count": int(safe_float(features.get("transcript_word_count", 0))),
        "ocr_word_count": int(safe_float(features.get("ocr_word_count", 0))),
        "ocr_frame_coverage": round(safe_float(features.get("ocr_frame_coverage", 0)), 4),
        "visual_text_density": features.get("visual_text_density", "no_disponible"),
        "video_type": features.get("video_type", "auto"),
        "content_intent": features.get("content_intent", "general/branding"),
    }


def transcripcion_block(features: Dict[str, Any], extra_status: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Subdiccionario `analisis_transcripcion` para el contrato."""
    extra_status = extra_status or {}
    text = features.get("transcript_text", "") or ""
    return {
        "disponible": bool(text.strip()),
        "fuente": extra_status.get("source", "manual_o_no_ejecutado"),
        "advertencia": extra_status.get("warning", ""),
        "longitud_caracteres": len(text),
        "cantidad_palabras": count_words(text),
        "tiene_cta": bool(features.get("cta_flag")),
        "tiene_beneficio": bool(features.get("benefit_flag")),
        "tiene_urgencia": bool(features.get("urgency_flag")),
        "tiene_confianza": bool(features.get("trust_flag")),
        "tipo_contenido_estimado": features.get("content_intent", "general/branding"),
        "preview": text[:900] + ("..." if len(text) > 900 else ""),
    }


def ocr_block(features: Dict[str, Any], extra_status: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Subdiccionario `analisis_ocr` para el contrato."""
    extra_status = extra_status or {}
    text = features.get("ocr_text", "") or ""
    return {
        "disponible": bool(text.strip()),
        "motor": extra_status.get("engine", "no_disponible"),
        "advertencia": extra_status.get("warning", ""),
        "frames_con_texto": int(safe_float(features.get("ocr_frames_with_text", 0))),
        "frames_totales": int(safe_float(features.get("ocr_frame_count", 0))),
        "cobertura": round(safe_float(features.get("ocr_frame_coverage", 0)), 4),
        "cantidad_palabras": int(safe_float(features.get("ocr_word_count", 0))),
        "densidad": features.get("visual_text_density", "sin_texto"),
        "relevancia_texto_en_pantalla": round(safe_float(features.get("ocr_relevance_score", 0)), 4),
        "relevancia_pct": round(safe_float(features.get("ocr_relevance_score", 0)) * 100, 1),
        "tiene_cta": bool(features.get("ocr_cta_flag")),
        "tiene_promocion": bool(features.get("ocr_promo_flag")),
        "tiene_urgencia": bool(features.get("ocr_urgency_flag")),
        "tiene_confianza": bool(features.get("ocr_trust_flag")),
        "tiene_exceso_texto": features.get("visual_text_density") == "alta",
        "tipo_contenido_estimado": features.get("ocr_llm_content_intent") or features.get("content_intent", "general/branding"),
        "rol_texto_en_pantalla": features.get("ocr_llm_role") or features.get("ocr_role", "sin_funcion_clara"),
        "ocr_text_raw": features.get("ocr_text_raw", ""),
        "ocr_first_result": features.get("ocr_first_result", ""),
        "ocr_llm_source": features.get("ocr_llm_source", ""),
        "ocr_llm_warning": features.get("ocr_llm_warning", ""),
        "ocr_llm_meaning": features.get("ocr_llm_meaning", ""),
        "ocr_llm_confidence": features.get("ocr_llm_confidence", ""),
        "ocr_llm_content_intent": features.get("ocr_llm_content_intent", ""),
        "ocr_llm_role": features.get("ocr_llm_role", ""),
        "overlap_contextual": round(safe_float(features.get("ocr_context_overlap", 0)), 4),
        "senal_comercial": round(safe_float(features.get("ocr_marketing_signal", 0)), 4),
        "senal_humor": round(safe_float(features.get("ocr_humor_signal", 0)), 4),
        "interpretacion_ocr": features.get("ocr_interpretation", ""),
        "preview": text[:900] + ("..." if len(text) > 900 else ""),
    }


def normalize_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas frecuentes de datasets públicos de YouTube.

    Implementación **vectorizada** para que escale a cientos de miles de filas:
    se evita ``df.apply(axis=1)`` y se usan operaciones nativas de NumPy/Pandas.
    """
    rename_candidates = {
        "view_count": "views",
        "views_count": "views",
        "likes_count": "likes",
        "like_count": "likes",
        "comment_count": "comments",
        "comments_count": "comments",
        "publishedAt": "published_at",
        "publish_time": "published_at",
        "categoryId": "category_id",
        "channelTitle": "channel_title",
        "channel": "channel_title",  # archive/daily_trending_videos.csv usa "channel"
    }
    normalized = df.copy()
    for src_col, dst_col in rename_candidates.items():
        if src_col in normalized.columns and dst_col not in normalized.columns:
            normalized = normalized.rename(columns={src_col: dst_col})
    for col in ["title", "description", "tags"]:
        if col not in normalized.columns:
            normalized[col] = ""
        normalized[col] = normalized[col].fillna("").astype(str)
    for col in ["views", "likes", "comments"]:
        if col not in normalized.columns:
            normalized[col] = 0
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce").fillna(0)
    if "published_at" not in normalized.columns:
        normalized["published_at"] = None
    if "category_id" not in normalized.columns:
        normalized["category_id"] = "unknown"
    normalized["category_id"] = normalized["category_id"].fillna("unknown").astype(str)
    if "duration_seconds" not in normalized.columns:
        normalized["duration_seconds"] = 0
    normalized["duration_seconds"] = pd.to_numeric(normalized["duration_seconds"], errors="coerce").fillna(0)

    # Texto combinado para TF-IDF.
    normalized["text_total"] = (
        normalized["title"] + " "
        + normalized["description"] + " "
        + normalized["tags"]
    )

    # Métricas de longitud y palabras (vectorizado).
    normalized["title_len"] = normalized["title"].str.len().fillna(0).astype(int)
    normalized["description_len"] = normalized["description"].str.len().fillna(0).astype(int)
    normalized["text_total_word_count"] = (
        normalized["text_total"].str.findall(r"\w+", flags=re.UNICODE).str.len().fillna(0).astype(int)
    )

    # Engagement vectorizado.
    views = normalized["views"].astype(float).clip(lower=0)
    likes = normalized["likes"].astype(float).clip(lower=0)
    comments = normalized["comments"].astype(float).clip(lower=0)
    denom = views.replace(0, np.nan).fillna(1.0)
    normalized["engagement_rate"] = ((likes + comments) / denom).round(6)
    normalized["like_rate"] = (likes / denom).round(6)
    normalized["comment_rate"] = (comments / denom).round(6)
    normalized["log_views"] = np.log1p(views).round(6)
    normalized["log_likes"] = np.log1p(likes).round(6)
    normalized["log_comments"] = np.log1p(comments).round(6)

    # Días desde publicación, vectorizado con clip a 1 día mínimo.
    pub = pd.to_datetime(normalized["published_at"], errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    days = (now - pub).dt.total_seconds() / 86400.0
    days = days.fillna(1.0).clip(lower=1.0)
    normalized["views_per_day"] = (views / days).round(6)

    # Asegura que las columnas escalares existan en el orden conocido.
    normalized["views"] = views
    normalized["likes"] = likes
    normalized["comments"] = comments

    # Flags textuales vectorizados con regex (más rápido que apply en datasets grandes).
    text_lower = normalized["text_total"].str.lower()

    def _flag(terms):
        pattern = "|".join(re.escape(t) for t in terms)
        return text_lower.str.contains(pattern, regex=True, na=False).astype(int)

    normalized["cta_flag"] = _flag(CTA_TERMS)
    normalized["urgency_flag"] = _flag(URGENCY_TERMS)
    normalized["trust_flag"] = _flag(TRUST_TERMS)
    normalized["promo_flag"] = _flag(PROMO_TERMS)
    normalized["benefit_flag"] = _flag(BENEFIT_TERMS)
    normalized["price_flag"] = _flag(PRICE_TERMS)

    # text_power_score vectorizado replicando los pesos de compute_text_power_score.
    score = (
        0.20 * normalized["cta_flag"]
        + 0.18 * normalized["benefit_flag"]
        + 0.15 * normalized["urgency_flag"]
        + 0.12 * normalized["trust_flag"]
        + 0.10 * normalized["promo_flag"]
        + 0.08 * normalized["price_flag"]
    )
    wc = normalized["text_total_word_count"]
    score = score + np.where((wc >= 8) & (wc <= 120), 0.10, np.where(wc > 120, 0.04, 0.0))
    normalized["text_power_score"] = score.clip(upper=1.0).round(4)

    # duration_fit_score vectorizado.
    d = normalized["duration_seconds"].astype(float)
    fit = pd.Series(0.40, index=normalized.index)
    fit = np.where(d <= 7, 0.95, fit)
    fit = np.where((d > 7) & (d <= 15), 0.90, fit)
    fit = np.where((d > 15) & (d <= 30), 0.80, fit)
    fit = np.where((d > 30) & (d <= 60), 0.70, fit)
    fit = np.where((d > 60) & (d <= 180), 0.55, fit)
    fit = np.where(d <= 0, 0.50, fit)  # duración desconocida
    normalized["duration_fit_score"] = fit
    return normalized


def create_boost_candidate_target(df: pd.DataFrame, percentile: float = 0.70) -> pd.DataFrame:
    """Crea etiqueta boost_candidate a partir de score compuesto."""
    work = df.copy()

    def robust_minmax(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").fillna(0)
        lo, hi = s.quantile(0.02), s.quantile(0.98)
        clipped = s.clip(lo, hi)
        if hi - lo == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (clipped - lo) / (hi - lo)

    work["organic_performance_score"] = (
        0.35 * robust_minmax(work["views_per_day"])
        + 0.25 * robust_minmax(work["engagement_rate"])
        + 0.15 * robust_minmax(work["like_rate"])
        + 0.10 * robust_minmax(work["comment_rate"])
        + 0.10 * robust_minmax(work["text_power_score"])
        + 0.05 * robust_minmax(work["duration_fit_score"])
    )
    threshold = work["organic_performance_score"].quantile(percentile)
    work["boost_candidate"] = (work["organic_performance_score"] >= threshold).astype(int)
    work["target_threshold"] = threshold
    return work

"""OCR + análisis de frames: siempre juntos, nunca excluyentes.

Motor de OCR (cascada):
1. EasyOCR con GPU=False (español + inglés) — si está instalado.
2. pytesseract — fallback.
3. OpenCV EAST / threshold puro — para detectar presencia de texto aunque
   no se pueda leer (nunca hay "sin OCR disponible": siempre devolvemos
   al menos la detección de texto en pantalla).

Los frames que se extraen para OCR son EXACTAMENTE los mismos que se usan en
``visual_composition.py`` (10 frames uniformes). El análisis de composición
reutiliza los frames ya extraídos: no hay doble extracción.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .config import CTA_TERMS, PROMO_TERMS, RUNTIME_CACHE_DIR, TRUST_TERMS, URGENCY_TERMS
from .video_processing import (
    get_video_duration_seconds,
    hash_file,
    validate_video_path,
)

N_OCR_FRAMES = 10   # sincronizado con visual_composition.FRAMES_FOR_VISUAL_ANALYSIS


# ---------------------------------------------------------------------------
# Helpers de texto
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _deduplicate_lines(lines: Iterable[str]) -> List[str]:
    seen: set = set()
    output: List[str] = []
    for line in lines:
        clean = _normalize_text(line)
        if not clean:
            continue
        key = clean.lower()
        if key not in seen:
            seen.add(key)
            output.append(clean)
    return output


def _contains_any(text: str, terms: Iterable[str]) -> int:
    lower = (text or "").lower()
    return int(any(term in lower for term in terms))


# ---------------------------------------------------------------------------
# Extracción de frames uniforme (10 frames)
# ---------------------------------------------------------------------------

def _evenly_spaced_timestamps(duration: float, n: int) -> List[float]:
    if duration <= 0:
        return []
    if n <= 1:
        return [duration / 2.0]
    start = min(0.5, duration * 0.05)
    end   = max(duration - 0.5, duration * 0.95)
    if end <= start:
        return [duration / 2.0] * n
    step = (end - start) / (n - 1)
    return [round(start + i * step, 2) for i in range(n)]


def extract_frames_uniform(video_path: str, n: int = N_OCR_FRAMES) -> List[Tuple[float, np.ndarray]]:
    """Extrae ``n`` frames uniformes. Devuelve lista de (timestamp, imagen BGR)."""
    cap = cv2.VideoCapture(video_path)
    frames: List[Tuple[float, np.ndarray]] = []
    if not cap.isOpened():
        return frames
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total_frames / fps if fps > 0 and total_frames > 0 else 0.0
    timestamps = _evenly_spaced_timestamps(duration, n)
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append((ts, frame))
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Motor OCR con cascada EasyOCR → pytesseract → OpenCV threshold
# ---------------------------------------------------------------------------

def _ocr_frame_easyocr(reader: Any, frame_bgr: np.ndarray) -> List[str]:
    """Lee texto en un frame con EasyOCR."""
    try:
        results = reader.readtext(frame_bgr, detail=0, paragraph=True)
        return [str(r).strip() for r in results if str(r).strip()]
    except Exception:
        return []


def _ocr_frame_tesseract(frame_bgr: np.ndarray) -> List[str]:
    """Lee texto en un frame con pytesseract."""
    try:
        import pytesseract  # type: ignore
        config = "--oem 3 --psm 11 -l spa+eng"
        text = pytesseract.image_to_string(frame_bgr, config=config)
        return [l.strip() for l in text.splitlines() if l.strip()]
    except Exception:
        return []


def _ocr_frame_opencv(frame_bgr: np.ndarray) -> List[str]:
    """Detección de texto con OpenCV puro (umbralización + contornos).

    No reconoce el texto verbalmente, pero detecta PRESENCIA de texto
    en pantalla. Útil cuando ni EasyOCR ni tesseract están disponibles.
    Devuelve "TEXTO_DETECTADO" si encuentra bloques que parecen texto.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # MSER para encontrar regiones estables (típico en texto)
    mser = cv2.MSER_create()
    mser.setMinArea(50)
    mser.setMaxArea(2000)
    try:
        regions, _ = mser.detectRegions(gray)
        if regions and len(regions) > 5:
            return ["[texto detectado por OpenCV]"]
    except Exception:
        pass
    # Fallback: threshold + contornos horizontales
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_like = [c for c in contours if cv2.contourArea(c) > 200]
    if len(text_like) >= 3:
        return ["[texto detectado por OpenCV]"]
    return []


def _load_easyocr_reader():
    """Carga EasyOCR lazy. Devuelve reader o None."""
    try:
        import easyocr  # type: ignore
        return easyocr.Reader(["es", "en"], gpu=False, verbose=False)
    except Exception:
        return None


def _has_tesseract() -> bool:
    try:
        import pytesseract  # type: ignore
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_frames(frames: List[Tuple[float, np.ndarray]]) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Aplica OCR a cada frame y devuelve (texto_total, engine_usado, per_frame).

    Siempre devuelve algo — aunque sea la detección de texto con OpenCV.
    """
    reader = _load_easyocr_reader()
    use_tess = _has_tesseract() if reader is None else False

    all_lines: List[str] = []
    per_frame: List[Dict[str, Any]] = []
    engine = "easyocr" if reader else ("tesseract" if use_tess else "opencv")

    for ts, frame in frames:
        if reader:
            lines = _ocr_frame_easyocr(reader, frame)
        elif use_tess:
            lines = _ocr_frame_tesseract(frame)
        else:
            lines = _ocr_frame_opencv(frame)
        all_lines.extend(lines)
        per_frame.append({"timestamp": round(ts, 2), "lines": lines, "has_text": bool(lines)})

    unique = _deduplicate_lines(all_lines)
    text = " ".join(unique)
    return text, engine, per_frame


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def extract_ocr_from_video(
    video_path: Optional[str],
    video_type: str = "auto",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """OCR + métricas textuales sobre 10 frames uniformes del video.

    Siempre se ejecuta si hay video — el análisis de composición visual
    (visual_composition.py) reutilizará los mismos timestamps.

    Returns:
        dict con: ``ocr_text``, ``ocr_engine``, ``ocr_word_count``,
        ``ocr_frame_count``, ``ocr_frames_with_text``, ``ocr_frame_coverage``,
        ``visual_text_density``, ``ocr_per_frame``, ``ocr_ok``, ``ocr_warning``,
        flags léxicos (``ocr_cta_flag``, ``ocr_promo_flag``, etc.)
    """
    empty = {
        "ocr_text": "", "ocr_engine": "none", "ocr_word_count": 0,
        "ocr_frame_count": 0, "ocr_frames_with_text": 0, "ocr_frame_coverage": 0.0,
        "visual_text_density": "sin_texto", "ocr_per_frame": [],
        "ocr_ok": False, "ocr_warning": "Sin video.",
        "ocr_cta_flag": 0, "ocr_promo_flag": 0, "ocr_trust_flag": 0,
        "ocr_urgency_flag": 0, "ocr_excess_text_flag": 0,
    }
    if not video_path:
        return empty

    validation = validate_video_path(video_path)
    if not validation.get("ok"):
        return {**empty, "ocr_warning": validation.get("warning", "Archivo no válido.")}

    # Caché
    cache_file: Optional[Path] = None
    if use_cache:
        digest = hash_file(video_path)
        if digest:
            cache_dir = RUNTIME_CACHE_DIR / "ocr"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{digest}_ocr.json"
            if cache_file.exists():
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

    frames = extract_frames_uniform(video_path, N_OCR_FRAMES)
    if not frames:
        return {**empty, "ocr_warning": "No se pudieron extraer frames del video."}

    ocr_text, engine, per_frame = _ocr_frames(frames)

    frames_with_text = sum(1 for f in per_frame if f["has_text"])
    frame_count = len(per_frame)
    coverage = frames_with_text / frame_count if frame_count > 0 else 0.0
    word_count = len(ocr_text.split()) if ocr_text else 0

    if word_count == 0:
        density = "sin_texto"
    elif word_count < 10:
        density = "baja"
    elif word_count < 35:
        density = "media"
    else:
        density = "alta"

    result = {
        "ocr_text": ocr_text,
        "ocr_engine": engine,
        "ocr_word_count": word_count,
        "ocr_frame_count": frame_count,
        "ocr_frames_with_text": frames_with_text,
        "ocr_frame_coverage": round(coverage, 3),
        "visual_text_density": density,
        "ocr_per_frame": per_frame,
        "ocr_ok": bool(ocr_text),
        "ocr_warning": "" if ocr_text else "No se detectó texto legible en los frames.",
        "ocr_cta_flag":      _contains_any(ocr_text, CTA_TERMS),
        "ocr_promo_flag":    _contains_any(ocr_text, PROMO_TERMS),
        "ocr_trust_flag":    _contains_any(ocr_text, TRUST_TERMS),
        "ocr_urgency_flag":  _contains_any(ocr_text, URGENCY_TERMS),
        "ocr_excess_text_flag": int(word_count > 35),
    }

    if cache_file:
        try:
            cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return result

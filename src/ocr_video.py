"""OCR robusto para videos cortos.

Objetivo de esta versión:
- Mejorar legibilidad sin inventar texto.
- Evitar que el fallback de OpenCV contamine el análisis con frases como
  "[texto detectado]".
- Usar Tesseract como motor por defecto por ser más liviano y estable en
  Hugging Face CPU.
- Dejar EasyOCR como opción explícita mediante ENABLE_EASYOCR=1.

Salida compatible con app.py y features.py.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .config import CTA_TERMS, PROMO_TERMS, RUNTIME_CACHE_DIR, TRUST_TERMS, URGENCY_TERMS
from .video_processing import hash_file, validate_video_path

N_OCR_FRAMES = int(os.getenv("FAST_OCR_FRAMES", os.getenv("N_OCR_FRAMES", "8")))
MIN_TEXT_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "38"))


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = (text or "").replace("\x0c", " ")
    text = re.sub(r"[|_~`^]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _line_quality(line: str) -> float:
    """Score simple para descartar ruido de OCR."""
    clean = _normalize_text(line)
    if not clean:
        return 0.0
    if len(clean) < 3:
        return 0.0
    alnum = sum(ch.isalnum() for ch in clean)
    letters = sum(ch.isalpha() for ch in clean)
    weird = sum((not ch.isalnum() and not ch.isspace() and ch not in "áéíóúÁÉÍÓÚñÑ.,:;!?¿¡%$#@+-/()") for ch in clean)
    ratio_alnum = alnum / max(len(clean), 1)
    ratio_letters = letters / max(len(clean), 1)
    ratio_weird = weird / max(len(clean), 1)
    repeated = bool(re.search(r"(.)\1{4,}", clean.lower()))
    if repeated:
        return 0.0
    if ratio_alnum < 0.35 and ratio_letters < 0.25:
        return 0.0
    return max(0.0, min(1.0, ratio_alnum + ratio_letters * 0.4 - ratio_weird * 1.4))


def _clean_lines(lines: Iterable[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        clean = _normalize_text(str(line or ""))
        if _line_quality(clean) >= 0.38:
            out.append(clean)
    return out


def _deduplicate_lines(lines: Iterable[str]) -> List[str]:
    result: List[str] = []
    for line in _clean_lines(lines):
        low = line.lower()
        duplicate = False
        for existing in result:
            ex = existing.lower()
            if low == ex or low in ex or ex in low:
                duplicate = True
                break
            if SequenceMatcher(None, low, ex).ratio() >= 0.86:
                duplicate = True
                break
        if not duplicate:
            result.append(line)
    return result


def _contains_any(text: str, terms: Iterable[str]) -> int:
    lower = (text or "").lower()
    return int(any(term in lower for term in terms))


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def _evenly_spaced_timestamps(duration: float, n: int) -> List[float]:
    if duration <= 0:
        return []
    n = max(1, int(n))
    if n == 1:
        return [duration / 2.0]
    start = min(0.55, duration * 0.08)
    end = max(duration - 0.55, duration * 0.92)
    if end <= start:
        return [duration / 2.0] * n
    step = (end - start) / (n - 1)
    return [round(start + i * step, 2) for i in range(n)]


def extract_frames_uniform(video_path: str, n: int = N_OCR_FRAMES) -> List[Tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    frames: List[Tuple[float, np.ndarray]] = []
    if not cap.isOpened():
        return frames
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total / fps if fps > 0 and total > 0 else 0.0
    for ts in _evenly_spaced_timestamps(duration, n):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append((ts, frame))
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Preprocesamiento OCR
# ---------------------------------------------------------------------------

def _resize_for_ocr(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    target_w = 1280
    if w < target_w:
        scale = target_w / max(w, 1)
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return frame_bgr


def _preprocess_variants(frame_bgr: np.ndarray) -> List[np.ndarray]:
    """Genera variantes conservadoras para mejorar Tesseract."""
    frame = _resize_for_ocr(frame_bgr)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Nitidez suave
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.6, blur, -0.6, 0)

    # Umbrales alternativos
    th_otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th_inv = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    th_adapt = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)

    return [sharp, th_otsu, th_inv, th_adapt]


def _has_tesseract_binary() -> bool:
    try:
        subprocess.run(["tesseract", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=3)
        return True
    except Exception:
        return False


def _ocr_tesseract(frame_bgr: np.ndarray) -> Tuple[List[str], float]:
    try:
        import pytesseract  # type: ignore
    except Exception:
        return [], 0.0
    if not _has_tesseract_binary():
        return [], 0.0

    best_lines: List[str] = []
    best_conf = 0.0
    best_score = 0.0
    configs = [
        "--oem 3 --psm 6 -l spa+eng",
        "--oem 3 --psm 11 -l spa+eng",
        "--oem 3 --psm 7 -l spa+eng",
    ]
    for img in _preprocess_variants(frame_bgr):
        for cfg in configs:
            try:
                data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
                words: List[str] = []
                confs: List[float] = []
                n = len(data.get("text", []))
                for i in range(n):
                    txt = _normalize_text(data["text"][i])
                    try:
                        conf = float(data["conf"][i])
                    except Exception:
                        conf = -1.0
                    if txt and conf >= MIN_TEXT_CONFIDENCE and _line_quality(txt) > 0:
                        words.append(txt)
                        confs.append(conf)
                text = " ".join(words)
                lines = _clean_lines([text]) if text else []
                conf_avg = float(np.mean(confs)) if confs else 0.0
                score = len(" ".join(lines)) * 0.7 + conf_avg * 0.3
                if score > best_score:
                    best_lines = lines
                    best_conf = conf_avg
                    best_score = score
            except Exception:
                continue
    return best_lines, best_conf


def _ocr_easyocr(frame_bgr: np.ndarray) -> Tuple[List[str], float]:
    if os.getenv("ENABLE_EASYOCR", "0").strip().lower() not in {"1", "true", "yes"}:
        return [], 0.0
    try:
        import easyocr  # type: ignore
        reader = _get_easyocr_reader()
        if reader is None:
            return [], 0.0
        results = reader.readtext(_resize_for_ocr(frame_bgr), detail=1, paragraph=False)
        lines: List[str] = []
        confs: List[float] = []
        for item in results:
            if len(item) >= 3:
                text = _normalize_text(item[1])
                conf = float(item[2]) * 100
                if conf >= MIN_TEXT_CONFIDENCE and _line_quality(text) >= 0.38:
                    lines.append(text)
                    confs.append(conf)
        return _clean_lines(lines), float(np.mean(confs)) if confs else 0.0
    except Exception:
        return [], 0.0


_EASYOCR_READER = None

def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    try:
        import easyocr  # type: ignore
        _EASYOCR_READER = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
    except Exception:
        _EASYOCR_READER = None
    return _EASYOCR_READER


def _detect_text_presence_opencv(frame_bgr: np.ndarray) -> bool:
    try:
        gray = cv2.cvtColor(_resize_for_ocr(frame_bgr), cv2.COLOR_BGR2GRAY)
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 4))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_like = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < 350:
                continue
            ratio = w / max(h, 1)
            if 1.5 <= ratio <= 18 and 8 <= h <= 120:
                text_like += 1
        return text_like >= 1
    except Exception:
        return False


def _ocr_frame(frame_bgr: np.ndarray) -> Tuple[List[str], str, float, bool]:
    # Tesseract primero: en despliegues CPU suele ser más predecible que EasyOCR.
    tess_lines, tess_conf = _ocr_tesseract(frame_bgr)
    easy_lines, easy_conf = _ocr_easyocr(frame_bgr)

    if easy_lines and (not tess_lines or easy_conf > tess_conf + 8):
        return easy_lines, "easyocr", easy_conf, True
    if tess_lines:
        return tess_lines, "tesseract", tess_conf, True

    presence = _detect_text_presence_opencv(frame_bgr)
    return [], "opencv_presence", 0.0, presence


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def extract_ocr_from_video(video_path: Optional[str], video_type: str = "auto", use_cache: bool = True) -> Dict[str, Any]:
    empty = {
        "ocr_text": "",
        "ocr_text_raw": "",
        "ocr_engine": "none",
        "ocr_word_count": 0,
        "ocr_frame_count": 0,
        "ocr_frames_with_text": 0,
        "ocr_frame_coverage": 0.0,
        "ocr_confidence_avg": 0.0,
        "ocr_low_confidence": True,
        "visual_text_density": "sin_texto",
        "ocr_per_frame": [],
        "ocr_ok": False,
        "ocr_warning": "Sin video.",
        "ocr_cta_flag": 0,
        "ocr_promo_flag": 0,
        "ocr_trust_flag": 0,
        "ocr_urgency_flag": 0,
        "ocr_excess_text_flag": 0,
        "ocr_text_presence_detected": False,
    }
    if not video_path:
        return empty

    validation = validate_video_path(video_path)
    if not validation.get("ok"):
        return {**empty, "ocr_warning": validation.get("warning", "Archivo no válido.")}

    cache_file: Optional[Path] = None
    if use_cache:
        digest = hash_file(video_path)
        if digest:
            cache_dir = RUNTIME_CACHE_DIR / "ocr"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{digest}_ocr_v2.json"
            if cache_file.exists():
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

    frames = extract_frames_uniform(video_path, N_OCR_FRAMES)
    if not frames:
        return {**empty, "ocr_warning": "No se pudieron extraer frames del video."}

    all_lines: List[str] = []
    per_frame: List[Dict[str, Any]] = []
    engines: List[str] = []
    confs: List[float] = []
    presence_count = 0

    for ts, frame in frames:
        lines, engine, conf, presence = _ocr_frame(frame)
        engines.append(engine)
        if conf > 0:
            confs.append(conf)
        if presence:
            presence_count += 1
        clean = _deduplicate_lines(lines)
        all_lines.extend(clean)
        per_frame.append({
            "timestamp": round(float(ts), 2),
            "lines": clean,
            "engine": engine,
            "confidence": round(float(conf), 2),
            "has_text": bool(clean),
            "text_presence_detected": bool(presence),
        })

    unique = _deduplicate_lines(all_lines)
    ocr_text = " ".join(unique)
    frame_count = len(per_frame)
    frames_with_text = sum(1 for f in per_frame if f["has_text"])
    coverage = frames_with_text / frame_count if frame_count else 0.0
    word_count = len(ocr_text.split()) if ocr_text else 0
    conf_avg = float(np.mean(confs)) if confs else 0.0
    low_conf = bool(word_count > 0 and conf_avg < 52)

    if word_count == 0:
        density = "sin_texto"
    elif word_count < 10:
        density = "baja"
    elif word_count < 35:
        density = "media"
    else:
        density = "alta"

    engine_counts = {e: engines.count(e) for e in sorted(set(engines))}
    main_engine = max(engine_counts.items(), key=lambda kv: kv[1])[0] if engine_counts else "none"

    if ocr_text:
        warning = ""
        if low_conf:
            warning = "OCR con confianza media/baja; revisar visualmente antes de usar como texto definitivo."
    elif presence_count > 0:
        warning = "Se detectó posible texto en pantalla, pero no fue posible leerlo con suficiente confianza."
    else:
        warning = "No se detectó texto legible en los frames."

    result = {
        "ocr_text": ocr_text,
        "ocr_text_raw": ocr_text,
        "ocr_engine": main_engine,
        "ocr_engine_counts": engine_counts,
        "ocr_word_count": word_count,
        "ocr_frame_count": frame_count,
        "ocr_frames_with_text": frames_with_text,
        "ocr_frame_coverage": round(coverage, 3),
        "ocr_confidence_avg": round(conf_avg, 2),
        "ocr_low_confidence": low_conf,
        "visual_text_density": density,
        "ocr_per_frame": per_frame,
        "ocr_ok": bool(ocr_text),
        "ocr_warning": warning,
        "ocr_cta_flag": _contains_any(ocr_text, CTA_TERMS),
        "ocr_promo_flag": _contains_any(ocr_text, PROMO_TERMS),
        "ocr_trust_flag": _contains_any(ocr_text, TRUST_TERMS),
        "ocr_urgency_flag": _contains_any(ocr_text, URGENCY_TERMS),
        "ocr_excess_text_flag": int(word_count > 35),
        "ocr_text_presence_detected": bool(presence_count > 0),
    }

    if cache_file:
        try:
            cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return result

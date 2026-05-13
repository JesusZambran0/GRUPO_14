"""Utilidades de procesamiento de video con OpenCV y dependencias opcionales."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import MAX_OCR_FRAMES, MAX_VIDEO_DURATION_SECONDS

VALID_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash MD5 del archivo para usar como clave de caché."""
    h = hashlib.md5()
    p = Path(path)
    if not p.exists():
        return ""
    try:
        with p.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def validate_video_path(video_path: str | Path | None) -> Dict[str, Any]:
    """Valida que el archivo de video exista y tenga extensión usable."""
    if not video_path:
        return {"ok": False, "warning": "No se cargó video.", "path": None}
    p = Path(str(video_path))
    if not p.exists():
        return {"ok": False, "warning": f"El archivo no existe: {p}", "path": str(p)}
    if p.suffix.lower() not in VALID_VIDEO_EXTENSIONS:
        return {
            "ok": False,
            "warning": f"Extensión no soportada: {p.suffix}. Se recomienda MP4.",
            "path": str(p),
        }
    return {"ok": True, "warning": "", "path": str(p)}


def get_video_duration_seconds(video_path: str) -> Optional[float]:
    """Obtiene duración aproximada del video mediante OpenCV."""
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps <= 0 or frame_count <= 0:
            return None
        return round(float(frame_count / fps), 2)
    except Exception:
        return None


def select_frame_timestamps(duration_seconds: float, video_type: str = "auto") -> List[float]:
    """Selecciona timestamps para OCR según tipo de video.

    - sin narrador o videos cortos (<=8s): paso 0.5-1.0s.
    - con narrador o videos más largos: paso 3-5s (usa 4s).
    """
    if not duration_seconds or duration_seconds <= 0:
        return []
    vt = (video_type or "auto").lower()
    if vt == "sin narrador" or duration_seconds <= 8:
        step = 0.75
    else:
        step = 4.0
    timestamps: List[float] = []
    t = 0.5
    while t < max(duration_seconds - 0.2, 0.6):
        timestamps.append(round(t, 2))
        t += step
    # Asegura puntos importantes (inicio, cuartiles y final).
    for t in [1.0, duration_seconds * 0.25, duration_seconds * 0.5, duration_seconds * 0.75, max(duration_seconds - 0.5, 0)]:
        if 0 <= t <= duration_seconds:
            timestamps.append(round(float(t), 2))
    unique_sorted = sorted(set(timestamps))
    # Aplica el límite global de frames OCR.
    if len(unique_sorted) > MAX_OCR_FRAMES:
        # Submuestreo uniforme.
        idxs = [round(i * (len(unique_sorted) - 1) / (MAX_OCR_FRAMES - 1)) for i in range(MAX_OCR_FRAMES)]
        unique_sorted = [unique_sorted[i] for i in idxs]
    return unique_sorted


def extract_frames(video_path: str, timestamps: List[float], output_dir: str | Path) -> List[str]:
    """Extrae frames en timestamps específicos y retorna rutas de imagen."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frames: List[str] = []
    try:
        import cv2  # type: ignore
    except Exception:
        return frames
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return frames
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps <= 0:
            cap.release()
            return frames
        for idx, ts in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            file_path = output_path / f"frame_{idx:03d}_{ts:.2f}.jpg"
            try:
                cv2.imwrite(str(file_path), frame)
                frames.append(str(file_path))
            except Exception:
                continue
        cap.release()
    except Exception:
        return frames
    return frames


def extract_audio_to_wav(video_path: str, output_path: str | Path) -> Optional[str]:
    """Extrae audio a WAV usando moviepy si está disponible.

    Soporta moviepy 1.x (``moviepy.editor``) y 2.x (``moviepy``). Devuelve None si
    el video no tiene audio, si moviepy no está instalado o si el audio falla.
    """
    try:
        try:
            from moviepy.editor import VideoFileClip  # type: ignore
        except Exception:
            from moviepy import VideoFileClip  # type: ignore

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        clip = VideoFileClip(str(video_path))
        try:
            if clip.audio is None:
                return None
            try:
                clip.audio.write_audiofile(str(out), verbose=False, logger=None)
            except TypeError:
                clip.audio.write_audiofile(str(out), logger=None)
            return str(out)
        finally:
            try:
                clip.close()
            except Exception:
                pass
    except Exception:
        return None


def summarize_video(video_path: str, video_type: str = "auto") -> Dict[str, Any]:
    """Resumen liviano del video para la app, con validación y aviso de duración."""
    validation = validate_video_path(video_path)
    summary: Dict[str, Any] = {
        "duration_seconds": None,
        "selected_timestamps": [],
        "video_type_inferred": video_type,
        "warning": "",
        "valid": validation.get("ok", False),
    }
    if not validation.get("ok"):
        summary["warning"] = validation.get("warning", "")
        return summary

    duration = get_video_duration_seconds(video_path) or 0
    summary["duration_seconds"] = duration
    if duration > MAX_VIDEO_DURATION_SECONDS:
        summary["warning"] = (
            f"Duración detectada {duration:.1f}s supera el máximo de demo ({MAX_VIDEO_DURATION_SECONDS:.0f}s). "
            "Se procesará igual, pero el OCR puede tardar más."
        )
    summary["video_type_inferred"] = "sin narrador" if (duration and duration <= 8) else "con narrador/desconocido"
    summary["selected_timestamps"] = select_frame_timestamps(duration, video_type=video_type)
    return summary

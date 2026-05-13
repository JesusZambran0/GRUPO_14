"""Descarga robusta de videos de YouTube para análisis académico.

Implementa una estrategia con **múltiples fallbacks** porque yt-dlp falla con
frecuencia en entornos como Hugging Face Spaces por restricciones de IP y
ausencia de cookies de sesión.

Orden de intentos:
1. ``yt-dlp`` con headers y formatos amplios.
2. ``pytubefix`` (fork mantenido de pytube, sin dependencias nativas).
3. ``pytube`` clásico (último recurso).

Si los tres fallan, devuelve metadata vía oEmbed (público, no requiere clave) y
un mensaje claro indicando que el usuario debe subir el MP4 manualmente.

Restricciones:
- Resolución máxima 360p (limitar uso de disco en HF Spaces).
- Duración máxima parametrizable (default 60 s).
- Recorte automático con ffmpeg si el archivo supera el límite.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from .config import MAX_VIDEO_DURATION_SECONDS

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_video_id(url: str) -> Optional[str]:
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.lower()
        if "youtu.be" in host:
            return parsed.path.strip("/").split("/")[0] or None
        if "youtube.com" in host:
            if parsed.path.startswith("/watch"):
                qs = urllib.parse.parse_qs(parsed.query)
                if "v" in qs:
                    return qs["v"][0]
            if parsed.path.startswith("/shorts/"):
                return parsed.path.split("/shorts/")[1].split("/")[0]
            if parsed.path.startswith("/embed/"):
                return parsed.path.split("/embed/")[1].split("/")[0]
        return None
    except Exception:
        return None


def _oembed_metadata(url: str) -> Dict[str, Any]:
    """Metadata vía oEmbed público. Sirve cuando todo descargador falla."""
    try:
        endpoint = "https://www.youtube.com/oembed?" + urllib.parse.urlencode({"url": url, "format": "json"})
        req = urllib.request.Request(endpoint, headers={"User-Agent": DEFAULT_USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {
            "title": data.get("title") or "",
            "channel_title": data.get("author_name") or "",
            "webpage_url": url,
            "thumbnail_url": data.get("thumbnail_url") or "",
            "duration_seconds": None,
            "views": None,
            "likes": None,
            "comments": None,
            "description": "",
            "published_at": None,
        }
    except Exception:
        return {}


def _probe_duration_ffprobe(path: str | Path) -> Optional[float]:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return float(proc.stdout.strip())
    except Exception:
        pass
    return None


def _trim_to_max_duration(path: str | Path, max_seconds: float) -> Optional[Path]:
    src = Path(str(path))
    if not src.exists():
        return None
    duration = _probe_duration_ffprobe(src)
    if duration is None or duration <= max_seconds:
        return src
    out = src.with_name(src.stem + "_trim.mp4")
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-t", f"{max_seconds:.0f}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
             "-c:a", "aac", "-b:a", "96k", str(out)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90,
        )
        if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out
    except Exception:
        pass
    return src


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _meta_from_ytdlp(info: Optional[Dict[str, Any]], fallback_url: str) -> Dict[str, Any]:
    info = info or {}
    return {
        "title": info.get("title") or "",
        "description": info.get("description") or "",
        "duration_seconds": info.get("duration"),
        "views": info.get("view_count"),
        "likes": info.get("like_count"),
        "comments": info.get("comment_count"),
        "channel_title": info.get("channel") or info.get("uploader") or "",
        "published_at": info.get("upload_date") or info.get("timestamp"),
        "webpage_url": info.get("webpage_url") or fallback_url,
        "thumbnail_url": info.get("thumbnail") or "",
    }


def _meta_from_pytube(yt: Any, fallback_url: str) -> Dict[str, Any]:
    try:
        return {
            "title": getattr(yt, "title", "") or "",
            "description": getattr(yt, "description", "") or "",
            "duration_seconds": getattr(yt, "length", None),
            "views": getattr(yt, "views", None),
            "likes": None,
            "comments": None,
            "channel_title": getattr(yt, "author", "") or "",
            "published_at": str(getattr(yt, "publish_date", "") or "") or None,
            "webpage_url": getattr(yt, "watch_url", fallback_url) or fallback_url,
            "thumbnail_url": getattr(yt, "thumbnail_url", "") or "",
        }
    except Exception:
        return {"webpage_url": fallback_url}


def _download_with_ytdlp(url: str, out_dir: Path, max_duration: float) -> Dict[str, Any]:
    try:
        import yt_dlp  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": f"yt-dlp no disponible: {exc}", "backend": "yt-dlp"}
    outtmpl = str(out_dir / "video.%(ext)s")
    opts = {
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 2,
        "fragment_retries": 2,
        "merge_output_format": "mp4",
        "http_headers": {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
        },
        "format": (
            "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]"
            "/best[height<=360][ext=mp4]"
            "/18"
            "/best[height<=360]/best"
        ),
    }
    try:
        with yt_dlp.YoutubeDL({**opts, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        duration = float(info.get("duration") or 0)
        if duration and duration > max_duration:
            return {
                "ok": False,
                "error": f"El video dura {duration:.0f}s y supera el límite de {max_duration:.0f}s. Sube un clip más corto.",
                "backend": "yt-dlp",
                "metadata": _meta_from_ytdlp(info, url),
                "duration_exceeded": True,
            }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        mp4s = sorted(out_dir.glob("*.mp4")) or sorted(out_dir.glob("*"))
        if not mp4s:
            return {"ok": False, "error": "yt-dlp no produjo archivo.", "backend": "yt-dlp"}
        return {"ok": True, "video_path": str(mp4s[0]), "metadata": _meta_from_ytdlp(info, url), "backend": "yt-dlp"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "backend": "yt-dlp"}


def _download_with_pytubefix(url: str, out_dir: Path, max_duration: float) -> Dict[str, Any]:
    try:
        from pytubefix import YouTube  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": f"pytubefix no disponible: {exc}", "backend": "pytubefix"}
    try:
        yt = YouTube(url, use_oauth=False, allow_oauth_cache=False)
        duration = float(yt.length or 0)
        if duration and duration > max_duration:
            return {
                "ok": False,
                "error": f"El video dura {duration:.0f}s y supera el límite de {max_duration:.0f}s.",
                "backend": "pytubefix",
                "metadata": _meta_from_pytube(yt, url),
                "duration_exceeded": True,
            }
        stream = (
            yt.streams.filter(progressive=True, file_extension="mp4", res="360p").first()
            or yt.streams.filter(progressive=True, file_extension="mp4", res="240p").first()
            or yt.streams.filter(progressive=True, file_extension="mp4").get_lowest_resolution()
            or yt.streams.filter(file_extension="mp4").first()
        )
        if not stream:
            return {"ok": False, "error": "pytubefix sin stream MP4 progresivo.", "backend": "pytubefix"}
        stream.download(output_path=str(out_dir), filename="video.mp4")
        target = out_dir / "video.mp4"
        if not target.exists() or target.stat().st_size == 0:
            return {"ok": False, "error": "pytubefix no escribió archivo.", "backend": "pytubefix"}
        return {"ok": True, "video_path": str(target), "metadata": _meta_from_pytube(yt, url), "backend": "pytubefix"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "backend": "pytubefix"}


def _download_with_pytube(url: str, out_dir: Path, max_duration: float) -> Dict[str, Any]:
    try:
        from pytube import YouTube  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": f"pytube no disponible: {exc}", "backend": "pytube"}
    try:
        yt = YouTube(url)
        duration = float(yt.length or 0)
        if duration and duration > max_duration:
            return {
                "ok": False,
                "error": f"El video dura {duration:.0f}s y supera el límite de {max_duration:.0f}s.",
                "backend": "pytube",
                "metadata": _meta_from_pytube(yt, url),
                "duration_exceeded": True,
            }
        stream = (
            yt.streams.filter(progressive=True, file_extension="mp4", res="360p").first()
            or yt.streams.filter(progressive=True, file_extension="mp4").get_lowest_resolution()
        )
        if not stream:
            return {"ok": False, "error": "pytube sin stream MP4 progresivo.", "backend": "pytube"}
        stream.download(output_path=str(out_dir), filename="video.mp4")
        target = out_dir / "video.mp4"
        if not target.exists() or target.stat().st_size == 0:
            return {"ok": False, "error": "pytube no escribió archivo.", "backend": "pytube"}
        return {"ok": True, "video_path": str(target), "metadata": _meta_from_pytube(yt, url), "backend": "pytube"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "backend": "pytube"}


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def download_youtube_video_360p(
    url: str,
    max_duration_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Descarga un video de YouTube a <=360p y devuelve metadata.

    Estrategia: yt-dlp → pytubefix → pytube → fallback oembed.

    Returns:
        dict con ``ok``, ``video_path``, ``metadata``, ``warning``, ``backend_used``,
        ``attempts``, ``duration_exceeded``, ``trimmed_to_max_duration``.
    """
    url = (url or "").strip()
    max_duration = float(max_duration_seconds or MAX_VIDEO_DURATION_SECONDS)
    if not url:
        return {
            "ok": False, "video_path": None, "metadata": {}, "warning": "No se ingresó URL.",
            "backend_used": None, "attempts": [], "duration_exceeded": False,
        }

    out_dir = Path(tempfile.mkdtemp(prefix="ytboost_url_"))
    attempts: list = []

    for fn in (_download_with_ytdlp, _download_with_pytubefix, _download_with_pytube):
        result = fn(url, out_dir, max_duration)
        attempts.append({"backend": result.get("backend"), "ok": result.get("ok"), "error": result.get("error", "")})
        if result.get("ok"):
            trimmed = _trim_to_max_duration(result["video_path"], max_duration)
            trimmed_flag = False
            if trimmed and str(trimmed) != result["video_path"]:
                result["video_path"] = str(trimmed)
                trimmed_flag = True
            return {
                "ok": True,
                "video_path": result["video_path"],
                "metadata": result.get("metadata", {}),
                "warning": "",
                "backend_used": result.get("backend"),
                "attempts": attempts,
                "duration_exceeded": False,
                "trimmed_to_max_duration": trimmed_flag,
            }
        if result.get("duration_exceeded"):
            return {
                "ok": False, "video_path": None,
                "metadata": result.get("metadata", {}),
                "warning": result.get("error", "Duración excedida."),
                "backend_used": result.get("backend"),
                "attempts": attempts, "duration_exceeded": True,
            }

    meta = _oembed_metadata(url)
    msg = (
        "No se pudo descargar el video automáticamente (todos los backends fallaron). "
        "Sube el MP4 manualmente (≤60 s) para continuar el análisis."
    )
    return {
        "ok": False, "video_path": None,
        "metadata": meta or {"webpage_url": url},
        "warning": msg, "backend_used": None,
        "attempts": attempts, "duration_exceeded": False,
    }


def is_ytdlp_available() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except Exception:
        return False

"""Extracción de metadatos públicos desde YouTube Data API v3.

El módulo es opcional: si no existe ``YOUTUBE_API_KEY``, el sistema continúa en
modo manual sin caer.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import requests

from .config import YOUTUBE_API_KEY

YOUTUBE_VIDEO_ENDPOINT = "https://www.googleapis.com/youtube/v3/videos"


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Extrae el ID de video desde una URL de YouTube (estándar, shorts, youtu.be)
    o retorna el ID si ya fue provisto.
    """
    if not url_or_id:
        return None
    value = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    parsed = urlparse(value)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        candidate = parsed.path.strip("/").split("/")[0]
        return candidate if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate) else None
    if "youtube.com" in host or "m.youtube.com" in host or "music.youtube.com" in host:
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            candidate = qs["v"][0]
            return candidate if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate) else None
        match = re.search(r"/(shorts|embed|live)/([A-Za-z0-9_-]{11})", parsed.path)
        if match:
            return match.group(2)
    return None


def iso8601_duration_to_seconds(duration: str) -> Optional[int]:
    """Convierte duración ISO 8601 de YouTube (PT1M05S) a segundos."""
    if not duration:
        return None
    try:
        import isodate  # type: ignore

        parsed = isodate.parse_duration(duration)
        return int(parsed.total_seconds())
    except Exception:
        # Parser regex simple de respaldo.
        hours = minutes = seconds = 0
        match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
        if not match:
            return None
        if match.group(1):
            hours = int(match.group(1))
        if match.group(2):
            minutes = int(match.group(2))
        if match.group(3):
            seconds = int(match.group(3))
        return hours * 3600 + minutes * 60 + seconds


def fetch_youtube_metadata(url_or_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Obtiene metadatos públicos del video usando YouTube Data API.

    Devuelve siempre un diccionario con clave ``ok`` y ``warning`` para que la app
    pueda continuar en modo manual cuando no hay clave o la API falla.
    """
    video_id = extract_video_id(url_or_id) if url_or_id else None
    if url_or_id and not video_id:
        return {"ok": False, "warning": "No se pudo extraer video_id desde la URL.", "video_id": None}

    key = (api_key or YOUTUBE_API_KEY or "").strip()
    if not key:
        return {
            "ok": False,
            "warning": "No se configuró YOUTUBE_API_KEY. Continuar en modo manual.",
            "video_id": video_id,
        }
    if not video_id:
        return {"ok": False, "warning": "No se proporcionó URL ni ID de video.", "video_id": None}

    params = {
        "id": video_id,
        "part": "snippet,contentDetails,statistics",
        "key": key,
    }
    try:
        response = requests.get(YOUTUBE_VIDEO_ENDPOINT, params=params, timeout=20)
        if response.status_code == 403:
            return {
                "ok": False,
                "warning": "YouTube API rechazó la solicitud (403). Posible cuota agotada o clave inválida.",
                "video_id": video_id,
            }
        response.raise_for_status()
        payload = response.json()
        items = payload.get("items", [])
        if not items:
            return {"ok": False, "warning": "La API no devolvió datos para el video.", "video_id": video_id}
        item = items[0]
        snippet = item.get("snippet", {})
        content = item.get("contentDetails", {})
        stats = item.get("statistics", {})
        duration_seconds = iso8601_duration_to_seconds(content.get("duration", ""))
        return {
            "ok": True,
            "warning": "",
            "video_id": video_id,
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "channel_id": snippet.get("channelId", ""),
            "channel_url": f"https://www.youtube.com/channel/{snippet.get('channelId', '')}",
            "category_id": snippet.get("categoryId", ""),
            "published_at": snippet.get("publishedAt", ""),
            "tags": snippet.get("tags", []),
            "duration_iso": content.get("duration", ""),
            "duration_seconds": duration_seconds,
            "views": int(stats.get("viewCount", 0) or 0),
            "likes": int(stats.get("likeCount", 0) or 0),
            "comments": int(stats.get("commentCount", 0) or 0),
        }
    except requests.RequestException as exc:
        return {
            "ok": False,
            "warning": f"Falló consulta a YouTube API: {exc}",
            "video_id": video_id,
        }
    except Exception as exc:
        return {
            "ok": False,
            "warning": f"Error inesperado consultando YouTube API: {exc}",
            "video_id": video_id,
        }

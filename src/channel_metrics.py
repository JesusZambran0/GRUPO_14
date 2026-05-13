"""Métricas públicas de canal de YouTube.

Este módulo usa YouTube Data API v3 con ``requests`` para evitar dependencias
adicionales. No intenta leer métricas privadas de YouTube Studio: el alcance real
(impressions/reach) requiere YouTube Analytics API + OAuth del dueño del canal.

La app calcula un alcance público estimado como promedio/mediana de views de los
últimos N videos públicos del canal.
"""
from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from .config import YOUTUBE_API_KEY

YOUTUBE_CHANNELS_ENDPOINT = "https://www.googleapis.com/youtube/v3/channels"
YOUTUBE_SEARCH_ENDPOINT = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_PLAYLIST_ITEMS_ENDPOINT = "https://www.googleapis.com/youtube/v3/playlistItems"
YOUTUBE_VIDEOS_ENDPOINT = "https://www.googleapis.com/youtube/v3/videos"


def _key(api_key: Optional[str] = None) -> str:
    return (api_key or YOUTUBE_API_KEY or "").strip()


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def extract_channel_hint(channel_url_or_id: str) -> Dict[str, str]:
    """Extrae una pista usable desde URL/id/handle del canal.

    Soporta:
    - https://www.youtube.com/@handle
    - https://www.youtube.com/channel/UCxxxx
    - https://www.youtube.com/c/nombre
    - https://www.youtube.com/user/nombre
    - UCxxxx
    - @handle
    """
    raw = (channel_url_or_id or "").strip()
    if not raw:
        return {"type": "", "value": ""}

    if raw.startswith("UC") and len(raw) >= 20:
        return {"type": "id", "value": raw}

    if raw.startswith("@"):
        return {"type": "handle", "value": raw}

    parsed = urlparse(raw)
    path = parsed.path.strip("/")

    if path.startswith("@"):
        return {"type": "handle", "value": path}

    parts = path.split("/") if path else []
    if len(parts) >= 2 and parts[0] == "channel":
        return {"type": "id", "value": parts[1]}
    if len(parts) >= 2 and parts[0] in {"c", "user"}:
        return {"type": parts[0], "value": parts[1]}

    return {"type": "query", "value": raw}


def resolve_channel_id(channel_url_or_id: str, api_key: Optional[str] = None) -> Optional[str]:
    key = _key(api_key)
    if not key:
        return None

    hint = extract_channel_hint(channel_url_or_id)
    if hint["type"] == "id":
        return hint["value"]

    # Handles oficiales: @canal
    if hint["type"] == "handle":
        handle = hint["value"]
        if not handle.startswith("@"):
            handle = "@" + handle
        try:
            resp = requests.get(
                YOUTUBE_CHANNELS_ENDPOINT,
                params={"part": "id", "forHandle": handle, "maxResults": 1, "key": key},
                timeout=20,
            )
            if resp.ok:
                items = resp.json().get("items", [])
                if items:
                    return items[0].get("id")
        except Exception:
            pass

    # Fallback por búsqueda pública.
    query = re.sub(r"^@", "", hint.get("value", "")).strip()
    if not query:
        return None
    try:
        resp = requests.get(
            YOUTUBE_SEARCH_ENDPOINT,
            params={"part": "snippet", "q": query, "type": "channel", "maxResults": 1, "key": key},
            timeout=20,
        )
        if not resp.ok:
            return None
        items = resp.json().get("items", [])
        if not items:
            return None
        return items[0].get("snippet", {}).get("channelId")
    except Exception:
        return None


def get_channel_public_stats(channel_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    key = _key(api_key)
    if not key:
        return {"ok": False, "warning": "Falta YOUTUBE_API_KEY para consultar métricas del canal."}
    try:
        resp = requests.get(
            YOUTUBE_CHANNELS_ENDPOINT,
            params={
                "part": "snippet,statistics,contentDetails",
                "id": channel_id,
                "maxResults": 1,
                "key": key,
            },
            timeout=20,
        )
        if resp.status_code == 403:
            return {"ok": False, "warning": "YouTube API rechazó métricas de canal (403): cuota agotada o key inválida."}
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            return {"ok": False, "warning": "La API no devolvió datos para el canal."}

        ch = items[0]
        snippet = ch.get("snippet", {}) or {}
        stats = ch.get("statistics", {}) or {}
        content = ch.get("contentDetails", {}) or {}
        uploads_playlist = (content.get("relatedPlaylists", {}) or {}).get("uploads")

        return {
            "ok": True,
            "warning": "",
            "channel_id": channel_id,
            "channel_title": snippet.get("title", ""),
            "channel_description": snippet.get("description", ""),
            "channel_custom_url": snippet.get("customUrl", ""),
            "channel_published_at": snippet.get("publishedAt", ""),
            "channel_thumbnail": ((snippet.get("thumbnails", {}) or {}).get("high", {}) or {}).get("url", ""),
            "subscriber_count": _to_int(stats.get("subscriberCount")),
            "hidden_subscriber_count": bool(stats.get("hiddenSubscriberCount", False)),
            "channel_view_count": _to_int(stats.get("viewCount")),
            "channel_video_count": _to_int(stats.get("videoCount")),
            "uploads_playlist_id": uploads_playlist,
            "channel_url": f"https://www.youtube.com/channel/{channel_id}",
        }
    except requests.RequestException as exc:
        return {"ok": False, "warning": f"Falló consulta de canal: {exc}"}
    except Exception as exc:
        return {"ok": False, "warning": f"Error inesperado consultando canal: {exc}"}


def get_recent_upload_video_ids(uploads_playlist_id: str, max_videos: int = 20, api_key: Optional[str] = None) -> List[str]:
    key = _key(api_key)
    if not key or not uploads_playlist_id:
        return []
    try:
        resp = requests.get(
            YOUTUBE_PLAYLIST_ITEMS_ENDPOINT,
            params={
                "part": "contentDetails",
                "playlistId": uploads_playlist_id,
                "maxResults": min(max_videos, 50),
                "key": key,
            },
            timeout=20,
        )
        if not resp.ok:
            return []
        return [
            item.get("contentDetails", {}).get("videoId")
            for item in resp.json().get("items", [])
            if item.get("contentDetails", {}).get("videoId")
        ]
    except Exception:
        return []


def get_video_stats(video_ids: List[str], api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    key = _key(api_key)
    if not key or not video_ids:
        return []
    try:
        resp = requests.get(
            YOUTUBE_VIDEOS_ENDPOINT,
            params={
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(video_ids[:50]),
                "maxResults": 50,
                "key": key,
            },
            timeout=20,
        )
        if not resp.ok:
            return []
        videos: List[Dict[str, Any]] = []
        for item in resp.json().get("items", []):
            stats = item.get("statistics", {}) or {}
            snippet = item.get("snippet", {}) or {}
            views = _to_int(stats.get("viewCount"))
            likes = _to_int(stats.get("likeCount"))
            comments = _to_int(stats.get("commentCount"))
            videos.append({
                "video_id": item.get("id"),
                "title": snippet.get("title", ""),
                "published_at": snippet.get("publishedAt", ""),
                "views": views,
                "likes": likes,
                "comments": comments,
                "engagement_public": round((likes + comments) / views, 4) if views else 0,
            })
        return videos
    except Exception:
        return []


def get_channel_reach_estimate(
    channel_url_or_id: str = "",
    *,
    channel_id: str = "",
    api_key: Optional[str] = None,
    max_videos: int = 20,
) -> Dict[str, Any]:
    """Devuelve métricas públicas del canal y alcance estimado por views recientes."""
    key = _key(api_key)
    if not key:
        return {"ok": False, "warning": "Falta YOUTUBE_API_KEY para métricas del canal."}

    resolved_id = (channel_id or "").strip() or resolve_channel_id(channel_url_or_id, api_key=key)
    if not resolved_id:
        return {"ok": False, "warning": "No se pudo resolver el canal de YouTube."}

    channel = get_channel_public_stats(resolved_id, api_key=key)
    if not channel.get("ok"):
        return channel

    uploads_playlist_id = channel.get("uploads_playlist_id")
    if not uploads_playlist_id:
        return {
            **channel,
            "ok": False,
            "warning": "No se pudo encontrar la playlist de videos subidos del canal.",
        }

    video_ids = get_recent_upload_video_ids(uploads_playlist_id, max_videos=max_videos, api_key=key)
    videos = get_video_stats(video_ids, api_key=key)

    views = [v["views"] for v in videos if v.get("views") is not None]
    likes = [v["likes"] for v in videos if v.get("likes") is not None]
    comments = [v["comments"] for v in videos if v.get("comments") is not None]

    avg_views = round(sum(views) / len(views), 1) if views else 0
    median_views = round(statistics.median(views), 1) if views else 0
    avg_likes = round(sum(likes) / len(likes), 1) if likes else 0
    avg_comments = round(sum(comments) / len(comments), 1) if comments else 0
    subscribers = _to_int(channel.get("subscriber_count"))

    return {
        **channel,
        "ok": True,
        "videos_sampled": len(videos),
        "avg_views_recent": avg_views,
        "median_views_recent": median_views,
        "avg_likes_recent": avg_likes,
        "avg_comments_recent": avg_comments,
        "avg_public_engagement_recent": round((avg_likes + avg_comments) / avg_views, 4) if avg_views else 0,
        "views_per_subscriber_recent": round(avg_views / subscribers, 4) if subscribers else 0,
        "reach_estimate_label": "Alcance público promedio estimado por views recientes",
        "recent_videos": videos,
        "warning": (
            "Este no es alcance real de YouTube Studio. Es una estimación pública basada en visualizaciones de videos recientes."
        ),
    }

"""Genera un dataset YouTube-like para pruebas offline del repositorio.

Este archivo NO reemplaza los datasets públicos de Kaggle. Su propósito es que
la demo, el entrenamiento y las pruebas funcionen aunque el usuario todavía no
haya descargado los CSV reales en data/raw/.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "demo_youtube_public_training.csv"

random.seed(42)
np.random.seed(42)

CATEGORIES = ["22", "24", "26", "27", "28", "10", "20", "23", "17", "1"]
CHANNELS = ["Marca Creativa", "Tech Lab", "Fitness Hoy", "Cocina Plus", "Educa Fácil", "Viajes Ya", "Negocios Pro", "Entretenimiento Max"]
TOPICS = [
    "nuevo lanzamiento", "tutorial rápido", "oferta exclusiva", "caso real", "reseña completa",
    "antes y después", "guía práctica", "beneficio comprobado", "historia de cliente", "comparativa"
]
CTA = ["compra ahora", "suscríbete", "descubre más", "agenda hoy", "aprovecha", "visita el enlace", "regístrate gratis"]
BENEFITS = ["ahorra tiempo", "mejora tus resultados", "aprende fácil", "crece rápido", "solución práctica", "resultados reales"]
TRUST = ["oficial", "garantía", "clientes reales", "verificado", "expertos", "calidad comprobada"]
PROMO = ["oferta", "descuento", "gratis", "solo hoy", "promoción limitada"]
FILLER = ["video", "contenido", "experiencia", "producto", "servicio", "estrategia", "herramienta", "recomendación"]


def make_text(latent: float) -> tuple[str, str, str]:
    topic = random.choice(TOPICS)
    title_parts = [topic.title()]
    desc_parts = [f"Este video presenta {topic} para YouTube."]
    tags = [topic.replace(" ", "_")]
    if latent > 0.25:
        b = random.choice(BENEFITS)
        title_parts.append(b)
        desc_parts.append(f"Beneficio principal: {b}.")
        tags.append("beneficio")
    if latent > 0.45:
        c = random.choice(CTA)
        title_parts.append(c.title())
        desc_parts.append(f"Llamado a la acción: {c}.")
        tags.append("cta")
    if latent > 0.60:
        t = random.choice(TRUST)
        desc_parts.append(f"Señal de confianza: {t}.")
        tags.append("confianza")
    if latent > 0.75:
        p = random.choice(PROMO)
        title_parts.append(p.title())
        desc_parts.append(f"Incluye {p} para aumentar urgencia.")
        tags.append("promo")
    if random.random() < 0.18:
        desc_parts.append(" ".join(random.choices(FILLER, k=random.randint(8, 30))))
    return " | ".join(title_parts), " ".join(desc_parts), "|".join(tags)


def main(n: int = 1500) -> None:
    rows = []
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    for i in range(n):
        latent = np.random.beta(2.2, 2.8)
        category = random.choice(CATEGORIES)
        title, desc, tags = make_text(latent)
        days_old = int(np.random.gamma(4, 35)) + 1
        published = now - timedelta(days=days_old, hours=random.randint(0, 23))
        duration = random.choice([6, 7, 10, 15, 25, 30, 45, 60, 90, 120])
        duration_fit = 1.0 if duration <= 15 else 0.8 if duration <= 60 else 0.55
        # Genera métricas correlacionadas con calidad latente, duración y ruido.
        base_daily = np.exp(4.4 + 3.2 * latent + 0.25 * duration_fit + np.random.normal(0, 0.7))
        views = int(max(80, base_daily * days_old))
        er = max(0.002, min(0.22, 0.015 + 0.10 * latent + np.random.normal(0, 0.018)))
        like_share = max(0.55, min(0.95, 0.78 + np.random.normal(0, 0.08)))
        likes = int(views * er * like_share)
        comments = int(views * er * (1 - like_share) * np.random.uniform(0.6, 1.4))
        rows.append({
            "video_id": f"demo{i:07d}",
            "title": title,
            "description": desc,
            "tags": tags,
            "channel_title": random.choice(CHANNELS),
            "category_id": category,
            "published_at": published.isoformat().replace("+00:00", "Z"),
            "views": views,
            "likes": max(likes, 0),
            "comments": max(comments, 0),
            "duration_seconds": duration,
            "country": random.choice(["US", "MX", "EC", "CO", "ES", "AR"]),
            "source_file": "demo_generated",
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Dataset demo generado: {OUT} ({n} filas)")


if __name__ == "__main__":
    main()

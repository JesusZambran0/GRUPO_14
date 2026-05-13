"""Análisis visual con principios clásicos de composición sobre 10 frames del video.

Cubre teorías de composición visual útiles para marketing y pauta:

1. **Regla de tercios**: dividir el frame en 3×3 y colocar el punto de interés
   sobre intersecciones o líneas. Puntaje basado en distancia al cruce de tercios
   más cercano.

2. **Composición áurea**: usa la proporción 1.618 para evaluar si el centro de
   masa visual cae cerca de líneas/intersecciones 0.382 y 0.618. Es una guía más
   fina que tercios para ubicar sujeto, producto, rostro o CTA con mayor armonía.

3. **Composición geométrica / balance**: distribución del peso visual entre las
   mitades izquierda-derecha y superior-inferior. Puntaje basado en simetría
   aceptable (≤30% de desbalance se considera bueno).

4. **Centro de masa visual / foco**: ubicación del punto focal estimado por el
   centroide de los bordes detectados.

Además calcula contraste, brillo, densidad de bordes, complejidad visual y
movimiento entre frames, y genera **una imagen anotada** por frame con tercios,
retícula áurea y centroide focal.

El análisis es liviano (no depende de modelos ML, solo OpenCV) para correr en
Hugging Face Spaces CPU.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .video_processing import extract_frames, get_video_duration_seconds, select_frame_timestamps

# Número de frames extraídos para el análisis composicional.
FRAMES_FOR_VISUAL_ANALYSIS = 10


def analyze_visual_composition(
    video_path: Optional[str],
    ocr_text: str = "",
    output_dir: Optional[Path] = None,
    n_frames: int = FRAMES_FOR_VISUAL_ANALYSIS,
) -> Dict[str, Any]:
    """Análisis visual completo a partir de hasta ``n_frames`` (default 10) frames.

    Args:
        video_path: ruta al MP4. Si no hay video, retorna estructura vacía.
        ocr_text: texto OCR detectado, usado para complejidad visual.
        output_dir: directorio donde guardar los frames anotados (PNG). Si es
            ``None`` se crea un temporal y se mantiene mientras el proceso vive.
        n_frames: cuántos frames extraer (10 por defecto).

    Returns:
        dict con todas las métricas, recomendaciones por teoría y rutas a los
        frames anotados.
    """
    if not video_path:
        return _empty("No se recibió video para análisis visual.")
    p = Path(str(video_path))
    if not p.exists():
        return _empty(f"El archivo de video no existe: {p}")
    try:
        import cv2  # type: ignore
    except Exception as exc:
        return _empty(f"OpenCV no está disponible: {type(exc).__name__}: {exc}")

    duration = get_video_duration_seconds(str(p)) or 0
    if duration <= 0:
        return _empty("No se pudo determinar la duración del video.")

    timestamps = select_frame_timestamps(duration, video_type="auto")
    # Forzar hasta n_frames timestamps espaciados uniformemente.
    timestamps = _evenly_spaced_timestamps(duration, n_frames)

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="ytboost_visual_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        raw_frame_paths = extract_frames(str(p), timestamps, tmp)
        analyses: List[Dict[str, Any]] = []
        annotated_paths: List[str] = []
        prev_gray = None
        motion_values: List[float] = []

        for idx, fp in enumerate(raw_frame_paths):
            frame = cv2.imread(fp)
            if frame is None:
                continue
            ts = timestamps[idx] if idx < len(timestamps) else timestamps[-1]
            a = _analyze_frame(frame, idx=idx, timestamp=ts)
            analyses.append(a)
            # Frame anotado con grid de tercios + centroide
            annotated = _draw_overlay(frame, a)
            out_path = Path(output_dir) / f"frame_{idx:02d}_t{int(ts):03d}s.png"
            cv2.imwrite(str(out_path), annotated)
            annotated_paths.append(str(out_path))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                h = min(gray.shape[0], prev_gray.shape[0])
                w = min(gray.shape[1], prev_gray.shape[1])
                diff = cv2.absdiff(gray[:h, :w], prev_gray[:h, :w])
                motion_values.append(float(np.mean(diff)) / 255.0)
            prev_gray = gray

    if not analyses:
        return _empty("No se pudieron extraer frames útiles para análisis visual.")

    avg = lambda k: float(np.mean([a[k] for a in analyses]))
    rule_thirds = avg("rule_thirds_score")
    golden_ratio = avg("golden_ratio_score")
    geometric_balance = avg("geometric_balance_score")
    visual_focus = avg("focal_clarity_score")
    contrast = avg("contrast_score")
    brightness = avg("brightness_score")
    edge_density = avg("edge_density")
    motion = float(np.mean(motion_values)) if motion_values else 0.0
    complexity = _complexity_score(edge_density=edge_density, ocr_text=ocr_text)

    # Puntaje global compuesto.
    composition_score = round(
        0.22 * rule_thirds
        + 0.18 * golden_ratio
        + 0.22 * geometric_balance
        + 0.18 * visual_focus
        + 0.12 * contrast
        + 0.08 * (1 - complexity),
        3,
    )

    recommendations_by_theory = _theory_based_recommendations(
        rule_thirds=rule_thirds,
        golden_ratio=golden_ratio,
        geometric_balance=geometric_balance,
        visual_focus=visual_focus,
        contrast=contrast,
        brightness=brightness,
        edge_density=edge_density,
        complexity=complexity,
        motion=motion,
        ocr_text=ocr_text,
    )
    flat_recommendations = []
    for items in recommendations_by_theory.values():
        flat_recommendations.extend(items)

    return {
        "visual_ok": True,
        "frames_analyzed": len(analyses),
        "duration_seconds": round(duration, 2),
        "composition_score": composition_score,
        "rule_of_thirds_score": round(rule_thirds, 3),
        "golden_ratio_score": round(golden_ratio, 3),
        "geometric_balance_score": round(geometric_balance, 3),
        "focal_clarity_score": round(visual_focus, 3),
        "contrast_score": round(contrast, 3),
        "brightness_score": round(brightness, 3),
        "edge_density": round(edge_density, 3),
        "motion_score": round(motion, 3),
        "visual_complexity_score": round(complexity, 3),
        "composition_principles": [
            "regla de tercios",
            "composición áurea / proporción 1.618",
            "composición geométrica / balance",
            "centro de masa visual / foco",
            "contraste",
            "carga cognitiva visual",
        ],
        "frame_diagnostics": analyses,
        "annotated_frame_paths": annotated_paths,
        "visual_recommendations": flat_recommendations[:8],
        "recommendations_by_theory": recommendations_by_theory,
        "golden_ratio_theory": {
            "name": "Composición áurea",
            "phi": 1.618,
            "guide_lines": [0.382, 0.618],
            "description": (
                "Evalúa si el centro de masa visual se aproxima a las líneas e intersecciones "
                "de la proporción áurea. En piezas de pauta ayuda a ubicar rostro, producto, "
                "beneficio o CTA en zonas armónicas sin saturar el centro."
            ),
        },
        "visual_summary": _visual_summary(rule_thirds, golden_ratio, geometric_balance, visual_focus, complexity),
        "visual_conclusion": _visual_conclusion(composition_score, rule_thirds, golden_ratio, geometric_balance, visual_focus, complexity),
        "warning": "",
    }


def _evenly_spaced_timestamps(duration: float, n: int) -> List[float]:
    """``n`` timestamps uniformemente espaciados, evitando los primeros y últimos 0.5s."""
    if duration <= 0:
        return []
    if n <= 1:
        return [duration / 2.0]
    start = min(0.5, duration * 0.05)
    end = max(duration - 0.5, duration * 0.95)
    if end <= start:
        return [duration / 2.0] * n
    step = (end - start) / (n - 1)
    return [round(start + i * step, 2) for i in range(n)]


def _empty(warning: str) -> Dict[str, Any]:
    return {
        "visual_ok": False,
        "frames_analyzed": 0,
        "composition_score": 0.0,
        "rule_of_thirds_score": 0.0,
        "golden_ratio_score": 0.0,
        "geometric_balance_score": 0.0,
        "focal_clarity_score": 0.0,
        "contrast_score": 0.0,
        "brightness_score": 0.0,
        "edge_density": 0.0,
        "motion_score": 0.0,
        "visual_complexity_score": 0.0,
        "composition_principles": [],
        "frame_diagnostics": [],
        "annotated_frame_paths": [],
        "visual_recommendations": [
            "No se pudo ejecutar el análisis visual. La recomendación se apoyará en transcripción, OCR y métricas."
        ],
        "recommendations_by_theory": {},
        "golden_ratio_theory": {},
        "visual_summary": "Análisis visual no disponible.",
        "visual_conclusion": "No se pudo generar conclusión visual por falta de frames analizables.",
        "warning": warning,
    }


def _analyze_frame(frame: np.ndarray, idx: int, timestamp: float) -> Dict[str, Any]:
    import cv2  # type: ignore

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    ys, xs = np.nonzero(edges)
    edge_density = float(len(xs)) / float(max(w * h, 1))
    brightness = float(np.mean(gray)) / 255.0
    contrast = max(0.0, min(1.0, float(np.std(gray)) / 80.0))
    brightness_score = 1.0 - min(abs(brightness - 0.52) / 0.52, 1.0)

    # Centroide de bordes (proxy del punto focal).
    if len(xs) > 0:
        cx = float(np.mean(xs)) / w
        cy = float(np.mean(ys)) / h
    else:
        cx, cy = 0.5, 0.5

    # Regla de tercios: cercanía al cruce más próximo.
    intersections = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
    dist = min(math.dist((cx, cy), pt) for pt in intersections)
    rule_score = 1.0 - min(dist / 0.55, 1.0)

    # Composición áurea: cercanía a líneas/intersecciones 0.382 / 0.618.
    # La proporción áurea se usa aquí como retícula compositiva práctica:
    # permite ubicar el peso visual en zonas armónicas sin obligarlo al centro.
    golden_lines = (0.382, 0.618)
    golden_intersections = [(gx, gy) for gx in golden_lines for gy in golden_lines]
    golden_dist = min(math.dist((cx, cy), pt) for pt in golden_intersections)
    golden_score = 1.0 - min(golden_dist / 0.52, 1.0)

    # Balance geométrico (izq vs der, sup vs inf en cantidad de píxeles de borde).
    if len(xs) > 0:
        left = float((xs < w / 2).sum())
        right = float((xs >= w / 2).sum())
        up = float((ys < h / 2).sum())
        down = float((ys >= h / 2).sum())
        lr = abs(left - right) / max(left + right, 1.0)
        ud = abs(up - down) / max(up + down, 1.0)
        # Buen balance ≈ desbalance < 0.30. Convertimos a puntaje 0-1.
        balance_score = 1.0 - min((lr + ud) / 2 / 0.60, 1.0)
    else:
        balance_score = 0.3

    # Claridad de foco: cuán concentrado está el centroide (varianza inversa).
    if len(xs) > 100:
        var_x = float(np.var(xs)) / (w * w)
        var_y = float(np.var(ys)) / (h * h)
        focal_clarity = 1.0 - min((var_x + var_y) / 0.18, 1.0)
    else:
        focal_clarity = 0.2

    return {
        "frame_index": idx,
        "timestamp": round(float(timestamp), 2),
        "saliency_center_x": round(cx, 3),
        "saliency_center_y": round(cy, 3),
        "rule_thirds_score": round(rule_score, 3),
        "golden_ratio_score": round(golden_score, 3),
        "geometric_balance_score": round(balance_score, 3),
        "focal_clarity_score": round(focal_clarity, 3),
        "contrast_score": round(contrast, 3),
        "brightness_score": round(brightness_score, 3),
        "edge_density": round(edge_density, 3),
    }


def _draw_overlay(frame: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
    """Dibuja tercios, retícula áurea, intersecciones y centroide focal."""
    import cv2  # type: ignore

    out = frame.copy()
    h, w = out.shape[:2]
    color_grid = (200, 200, 60)  # tercios
    color_golden = (90, 180, 255)  # retícula áurea
    color_inter = (60, 200, 255)
    color_centroid = (50, 80, 255)  # rojo cálido

    # Líneas tercios
    for i in (1, 2):
        cv2.line(out, (int(w * i / 3), 0), (int(w * i / 3), h), color_grid, 1, cv2.LINE_AA)
        cv2.line(out, (0, int(h * i / 3)), (w, int(h * i / 3)), color_grid, 1, cv2.LINE_AA)
    # Retícula áurea 0.382 / 0.618
    for f in (0.382, 0.618):
        cv2.line(out, (int(w * f), 0), (int(w * f), h), color_golden, 1, cv2.LINE_AA)
        cv2.line(out, (0, int(h * f)), (w, int(h * f)), color_golden, 1, cv2.LINE_AA)

    # Intersecciones de tercios y puntos áureos
    for (fx, fy) in [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]:
        cv2.circle(out, (int(w * fx), int(h * fy)), 6, color_inter, 2, cv2.LINE_AA)
    for (fx, fy) in [(0.382, 0.382), (0.618, 0.382), (0.382, 0.618), (0.618, 0.618)]:
        cv2.circle(out, (int(w * fx), int(h * fy)), 4, color_golden, 1, cv2.LINE_AA)
    # Centroide
    cx = int(analysis["saliency_center_x"] * w)
    cy = int(analysis["saliency_center_y"] * h)
    cv2.drawMarker(out, (cx, cy), color_centroid, markerType=cv2.MARKER_CROSS, markerSize=22, thickness=3)
    cv2.circle(out, (cx, cy), 14, color_centroid, 2, cv2.LINE_AA)

    # Etiqueta superior
    label = f"t={analysis['timestamp']}s | tercios={analysis['rule_thirds_score']:.2f} | aurea={analysis.get('golden_ratio_score',0):.2f} | foco=({analysis['saliency_center_x']:.2f},{analysis['saliency_center_y']:.2f})"
    cv2.rectangle(out, (0, 0), (w, 26), (15, 15, 25), -1)
    cv2.putText(out, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 245), 1, cv2.LINE_AA)
    return out


def _complexity_score(edge_density: float, ocr_text: str = "") -> float:
    text_words = len((ocr_text or "").split())
    text_factor = min(text_words / 45.0, 1.0)
    edge_factor = min(edge_density / 0.18, 1.0)
    return max(0.0, min(1.0, 0.65 * edge_factor + 0.35 * text_factor))


def _visual_summary(rule_thirds: float, golden_ratio: float, balance: float, focal: float, complexity: float) -> str:
    parts = []
    parts.append("composición alineada con tercios" if rule_thirds >= 0.55 else "composición centrada o con foco fuera de tercios")
    parts.append("buena aproximación a proporción áurea" if golden_ratio >= 0.55 else "aprovechamiento áureo mejorable")
    parts.append("buen balance geométrico" if balance >= 0.55 else "balance geométrico mejorable")
    parts.append("foco visual claro" if focal >= 0.40 else "foco visual difuso")
    parts.append("carga visual controlada" if complexity < 0.55 else "alta carga visual")
    return "; ".join(parts) + "."


def _visual_conclusion(score: float, rule_thirds: float, golden_ratio: float, balance: float, focal: float, complexity: float) -> str:
    score100 = int(round(score * 100))
    if score >= 0.70:
        verdict = "La pieza tiene una base visual sólida para sostener atención."
    elif score >= 0.50:
        verdict = "La pieza es usable, pero requiere ajustes de encuadre o claridad antes de escalar pauta."
    else:
        verdict = "La pieza necesita correcciones visuales antes de invertir presupuesto de forma agresiva."
    weakest = sorted([
        (rule_thirds, "encuadre por tercios"),
        (golden_ratio, "composición áurea"),
        (balance, "balance geométrico"),
        (focal, "claridad del foco"),
        (1 - complexity, "carga visual"),
    ], key=lambda x: x[0])[0][1]
    return f"Score visual {score100}/100. {verdict} El área que más conviene revisar es {weakest}."


def _theory_based_recommendations(
    *, rule_thirds: float, golden_ratio: float, geometric_balance: float, visual_focus: float,
    contrast: float, brightness: float, edge_density: float,
    complexity: float, motion: float, ocr_text: str,
) -> Dict[str, List[str]]:
    """Recomendaciones agrupadas por teoría de composición."""
    out: Dict[str, List[str]] = {
        "regla_de_tercios": [],
        "composicion_aurea": [],
        "composicion_geometrica": [],
        "foco_visual": [],
        "iluminacion_y_contraste": [],
        "carga_cognitiva": [],
    }

    if rule_thirds < 0.45:
        out["regla_de_tercios"].append(
            "Reposicionar el sujeto principal sobre una intersección de tercios (no en el centro). "
            "Mover la cámara o reencuadrar para que rostros, productos o texto clave caigan en los cruces."
        )
    elif rule_thirds < 0.65:
        out["regla_de_tercios"].append(
            "El foco está cerca de los tercios pero podría afinarse: ajustar levemente el encuadre."
        )
    else:
        out["regla_de_tercios"].append(
            "Buena aplicación de la regla de tercios; mantener este encuadre como referencia."
        )

    if golden_ratio < 0.45:
        out["composicion_aurea"].append(
            "La proporción áurea no se aprovecha bien. Ubicar rostro, producto o CTA cerca de las líneas 0.382/0.618 para crear una lectura más armónica y menos rígida que un centro exacto."
        )
    elif golden_ratio < 0.65:
        out["composicion_aurea"].append(
            "La composición se aproxima a zonas áureas, pero puede afinarse moviendo el punto focal ligeramente hacia una intersección 0.382/0.618."
        )
    else:
        out["composicion_aurea"].append(
            "Buena alineación con proporción áurea; conservar esta distribución para escenas donde producto, rostro o CTA son protagonistas."
        )

    if geometric_balance < 0.45:
        out["composicion_geometrica"].append(
            "El peso visual está desbalanceado. Redistribuir elementos para equilibrar mitades "
            "izquierda/derecha o superior/inferior, o añadir un elemento de contrapeso."
        )
    elif geometric_balance >= 0.65:
        out["composicion_geometrica"].append(
            "La composición presenta buen balance geométrico; reforzarlo con líneas guía consistentes."
        )

    if visual_focus < 0.35:
        out["foco_visual"].append(
            "El foco visual está disperso. Concentrar el interés en un único elemento por escena: "
            "rostro, producto, demostración o titular en pantalla."
        )
    if edge_density < 0.025:
        out["foco_visual"].append(
            "Las escenas tienen pocos bordes/elementos visibles. Agregar un punto focal claro en los primeros segundos."
        )

    if contrast < 0.40:
        out["iluminacion_y_contraste"].append(
            "Contraste bajo entre sujeto y fondo. Aumentar iluminación clave, separar el sujeto del fondo "
            "con color o profundidad de campo."
        )
    if brightness < 0.35:
        out["iluminacion_y_contraste"].append(
            "Frames demasiado oscuros/sobreexpuestos. Calibrar exposición y balance de blancos."
        )

    if complexity > 0.65:
        out["carga_cognitiva"].append(
            "Carga visual alta: demasiados elementos simultáneos. Reducir gráficos, texto y elementos "
            "secundarios para que el mensaje principal se lea en menos de 2 segundos."
        )
    if motion > 0.28:
        out["carga_cognitiva"].append(
            "Cortes y movimiento muy rápidos. Mantener planos clave 1.5–3 s para mejorar comprensión."
        )
    if not (ocr_text or "").strip() and complexity < 0.35:
        out["carga_cognitiva"].append(
            "No se detectó texto en pantalla. Si el video será anuncio, agregar una frase breve de beneficio o un CTA visible."
        )

    # Eliminar listas vacías.
    return {k: v for k, v in out.items() if v}

"""Pruebas rápidas del repositorio YouTube Boost AI.

Ejecutar:
    python tests/smoke_test.py

Si falta una dependencia opcional (gradio, faster-whisper, easyocr, torch+transformers)
la prueba relacionada se marca SKIP en lugar de fallar.
"""
from __future__ import annotations

import json
import os
import sys
import cv2
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS: List[Tuple[str, str, str]] = []


def ok(name: str, detail: str = "") -> None:
    RESULTS.append(("PASS", name, detail))
    print(f"  ✅ PASS  {name}  {detail}")


def skip(name: str, detail: str) -> None:
    RESULTS.append(("SKIP", name, detail))
    print(f"  ⚠️  SKIP  {name}  {detail}")


def fail(name: str, detail: str) -> None:
    RESULTS.append(("FAIL", name, detail))
    print(f"  ❌ FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_imports() -> None:
    name = "test_imports"
    try:
        from src import (  # noqa: F401
            config, features, predict, recommender, llm_analyzer,
            ocr_video, transcription, video_processing, youtube_api, explain,
            policy_evaluator,
        )
    except Exception as exc:
        fail(name, f"Falló import de src.*: {exc}")
        return
    try:
        import app  # noqa: F401
        ok(name, "src.* y app importan")
    except ModuleNotFoundError as exc:
        if "gradio" in str(exc).lower():
            skip(name + "_app", "gradio no está instalado en el entorno")
            ok(name + "_src_only", "src.* importan correctamente")
        else:
            fail(name, f"Falló import de app: {exc}")


def test_manual_prediction() -> None:
    name = "test_manual_prediction"
    try:
        from app import analyze_video  # type: ignore
    except Exception as exc:
        skip(name, f"No se pudo importar app: {exc}")
        return
    try:
        out = analyze_video(
            "Ninguno", "", None,
            "Tutorial: aprende a programar en Python",
            "Curso introductorio con ejemplos prácticos paso a paso",
            "28", "programación",
            150000, 8500, 420,
            120, 0.5, 240,
            48, 25000, 3000,
            900,
            "Hola, en este video te enseño los primeros pasos en Python con ejemplos prácticos.",
            "APRENDE PYTHON",
            "",   # manual_comments
            "con narrador", 5.0, 50.0, "rules",
        )
        result = out[1]
        assert len(out) == 22, f"Esperaba 22 outputs, recibí {len(out)}"
        assert result["accion_final"] in {"IMPULSAR", "AJUSTAR ANTES DE IMPULSAR", "MONITOREAR", "NO IMPULSAR", "REVISIÓN HUMANA"}
        assert result["recomendacion_impulso"] in {"impulsar", "no impulsar", "ajustar antes de impulsar", "monitorear"}
        assert "score_hibrido" in result
        assert "analisis_politicas" in result
        assert "analisis_visual" in result
        assert "analisis_guion" in result
        assert "proyeccion_pauta" in result
        assert "resumen_ejecutivo" in result
        assert "analisis_sentimiento" in result
        ok(name, f"accion={result['accion_final']}, score={result['score_hibrido']}, 22 outputs")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_demo_case() -> None:
    name = "test_demo_case"
    try:
        from app import analyze_video  # type: ignore
    except Exception as exc:
        skip(name, f"No se pudo importar app: {exc}")
        return
    try:
        out = analyze_video(
            "Demo: video con narrador", "", None,
            "", "", "unknown", "",
            0, 0, 0,
            0, 0, 0,
            24, 0, 0,
            0,
            "", "", "",  # manual_comments vacío
            "auto", 5, 20, "rules",
        )
        result = out[1]
        assert result.get("accion_final")
        assert result.get("metricas")
        ok(name, "demo precalculado ok")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_json_contract() -> None:
    name = "test_json_contract"
    try:
        from app import analyze_video  # type: ignore
    except Exception as exc:
        skip(name, f"No se pudo importar app: {exc}")
        return
    required = [
        "prediccion_rendimiento", "probabilidad_rendimiento", "recomendacion_impulso",
        "requiere_ajustes", "nivel_prioridad", "alcance_estimado_por_dolar",
        "alcance_estimado_total", "metricas", "analisis_transcripcion",
        "analisis_ocr", "analisis_llm", "ajustes_sugeridos", "justificacion",
        "accion_final", "score_hibrido", "analisis_politicas",
        "policy_risk_level", "youtube_ad_status_estimate",
        "analisis_visual", "analisis_guion", "proyeccion_pauta", "resumen_ejecutivo",
        "recomendacion_redactada", "analisis_sentimiento",
    ]
    try:
        out = analyze_video(
            "Ninguno", "", None,
            "Mi video", "Descripción", "27", "tema",
            1000, 100, 10,
            5, 0.4, 60,
            24, 1000, 200,
            30,
            "Texto manual de prueba", "OCR de prueba", "",
            "con narrador", 5, 20, "rules",
        )
        result = out[1]
        missing = [k for k in required if k not in result]
        assert not missing, f"Claves faltantes: {missing}"
        json.dumps(result, ensure_ascii=False)
        ok(name, "todas las claves del contrato presentes y JSON serializable")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_transcription_obligatory() -> None:
    """Política nueva: sin transcripción pero CON título y descripción legibles,
    la app **no** escala a REVISIÓN HUMANA. Solo escala si hay 3+ categorías de
    alta severidad O si NO hay nada de texto evaluable.
    """
    name = "test_transcription_obligatory"
    try:
        from app import analyze_video  # type: ignore
    except Exception as exc:
        skip(name, f"No se pudo importar app: {exc}")
        return
    try:
        # Caso A: hay título + descripción evaluables (texto neutro) y NO hay transcripción.
        # Esperado: NO ser REVISIÓN HUMANA. policy debe ser bajo + partial_evaluation=True.
        out_partial = analyze_video(
            "Ninguno", "", None,
            "Título cualquiera", "Descripción cualquiera",
            "22", "tema",
            5000, 200, 30,
            0, 0, 0,
            24, 1000, 100,
            60,
            "",  # sin transcript manual
            "", "",  # sin OCR, sin comments
            "auto", 5, 20, "rules",
        )
        r_partial = out_partial[1]
        assert r_partial["accion_final"] != "REVISIÓN HUMANA", (
            f"Con título+descripción no debería ser REVISIÓN HUMANA. Obtuvo {r_partial['accion_final']}"
        )
        assert r_partial["analisis_politicas"].get("partial_evaluation") is True

        # Caso B: TODO vacío (sin título, sin descripción, sin transcript) → sí REVISIÓN HUMANA.
        out_empty = analyze_video(
            "Ninguno", "", None,
            "", "", "", "",
            0, 0, 0, 0, 0, 0, 24, 0, 0, 0,
            "", "", "", "auto", 5, 20, "rules",
        )
        r_empty = out_empty[1]
        assert r_empty["accion_final"] == "REVISIÓN HUMANA", (
            f"Sin texto evaluable debería escalar a REVISIÓN HUMANA. Obtuvo {r_empty['accion_final']}"
        )
        ok(name, f"partial={r_partial['accion_final']} | empty={r_empty['accion_final']}")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_policy_evaluator() -> None:
    """policy_evaluator detecta categorías sensibles."""
    name = "test_policy_evaluator"
    try:
        from src.policy_evaluator import evaluate_youtube_ad_policy_risk
        # Caso inocuo
        r1 = evaluate_youtube_ad_policy_risk("Tutorial de cocina italiana paso a paso")
        assert r1["policy_risk_level"] == "bajo"
        assert r1["youtube_ad_status_estimate"] == "apto"
        # Caso alto riesgo (≥1 categoría alta = alto; 3+ = revisión humana)
        r2 = evaluate_youtube_ad_policy_risk("Pierde 10 kilos en una semana, dieta milagrosa garantizada")
        assert r2["policy_risk_level"] in {"alto", "revisión humana"}
        assert "salud o pérdida de peso" in r2["policy_risk_categories"]
        # Caso vacío → bajo + partial_evaluation True (nueva política permisiva)
        r3 = evaluate_youtube_ad_policy_risk("")
        assert r3["policy_risk_level"] == "bajo"
        assert r3.get("partial_evaluation") is True
        assert r3.get("evaluation_coverage") == "ninguna"
        # Sin transcripción pero con texto: parcial
        r4 = evaluate_youtube_ad_policy_risk("Hola mundo tutorial", has_transcript=False)
        assert r4.get("partial_evaluation") is True
        ok(name, f"bajo/{r1['youtube_ad_status_estimate']} | alto/{r2['youtube_ad_status_estimate']} | vacío→bajo+partial | sin_trans→partial")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_hybrid_score_action() -> None:
    """compute_hybrid_score y determine_final_action."""
    name = "test_hybrid_score_action"
    try:
        from src.recommender import compute_hybrid_score, determine_final_action, FINAL_ACTIONS
        # Score alto, sin riesgo → IMPULSAR
        s = compute_hybrid_score(0.85, 0.08, 0.02, 0.6, 500, "bajo")
        a = determine_final_action(s["hybrid_score"], "bajo", requires_adjustments=False)
        assert a == "IMPULSAR", f"esperaba IMPULSAR, got {a} ({s['hybrid_score']})"
        # Riesgo alto → NO IMPULSAR
        a2 = determine_final_action(0.8, "alto")
        assert a2 == "NO IMPULSAR"
        # Policy=revisión humana (3+ categorías) → REVISIÓN HUMANA
        a3 = determine_final_action(0.8, "revisión humana")
        assert a3 == "REVISIÓN HUMANA"
        # Sin transcripción PERO con texto evaluable → NO debe escalar
        a4 = determine_final_action(
            0.8, "bajo", requires_human_review_due_to_missing_transcript=True, has_evaluable_text=True,
        )
        assert a4 != "REVISIÓN HUMANA", f"con texto evaluable no debe ser REVISIÓN HUMANA, got {a4}"
        # Sin transcripción Y sin nada de texto → REVISIÓN HUMANA
        a4b = determine_final_action(
            0.8, "bajo", requires_human_review_due_to_missing_transcript=True, has_evaluable_text=False,
        )
        assert a4b == "REVISIÓN HUMANA"
        # Score bajo → NO IMPULSAR
        a5 = determine_final_action(0.2, "bajo")
        assert a5 == "NO IMPULSAR"
        for action in [a, a2, a3, a4, a4b, a5]:
            assert action in FINAL_ACTIONS
        ok(name, "6 ramas de acción cubiertas (incluye nueva semántica has_evaluable_text)")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_predict_fallback() -> None:
    name = "test_predict_fallback"
    try:
        from src.predict import heuristic_probability, predict_from_features
        from src.features import build_feature_row
        feat = build_feature_row(
            title="Aprende rápido", description="Suscríbete hoy",
            transcript_text="", ocr_text="OFERTA",
            category_id="22", duration_seconds=7,
            views=1000, likes=80, comments=10, video_type="sin narrador",
        )
        prob = heuristic_probability(feat)
        assert 0 <= prob <= 1
        out = predict_from_features(feat)
        assert "probability" in out and "level" in out
        ok(name, f"heurística={prob:.4f}; predict_level={out['level']}")
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_ocr_fallback() -> None:
    name = "test_ocr_fallback"
    try:
        from src.ocr_video import extract_ocr_from_video
        result = extract_ocr_from_video("ruta/que/no/existe.mp4", "auto")
        assert "ocr_text" in result and "ocr_warning" in result
        assert result.get("ocr_ok") is False
        ok(name, "OCR fallback OK ante video ausente")
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_transcription_fallback() -> None:
    """resolve_transcript maneja sin video/sin manual sin romper."""
    name = "test_transcription_fallback"
    try:
        from src.transcription import resolve_transcript
        # Sin nada → ok=False pero NO lanza excepción
        r = resolve_transcript("", None)
        assert r["ok"] is False
        assert r["transcript_text"] == ""
        assert isinstance(r.get("warning"), str)
        # Con manual → ok=True, source=manual
        r2 = resolve_transcript("Texto manual de prueba", None)
        assert r2["ok"] is True
        assert r2["source"] == "manual"
        assert r2["transcript_text"] == "Texto manual de prueba"
        # Con video inexistente → ok=False pero no rompe
        r3 = resolve_transcript("", "ruta/inexistente.mp4")
        assert r3["ok"] is False
        ok(name, "3 ramas de resolve_transcript OK (nueva semántica sin bloqueo)")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_youtube_id_extraction() -> None:
    name = "test_youtube_id_extraction"
    try:
        from src.youtube_api import extract_video_id
        cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("not a url", None),
        ]
        for url, expected in cases:
            got = extract_video_id(url)
            assert got == expected, f"Para {url!r} esperaba {expected!r}, obtuvo {got!r}"
        ok(name, f"{len(cases)} casos OK")
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_rule_based_recommendation() -> None:
    """rule_based_recommendation siempre devuelve contrato LLM completo."""
    name = "test_rule_based_recommendation"
    try:
        from src.llm_analyzer import rule_based_recommendation, LLM_OUTPUT_KEYS
        from src.features import build_feature_row
        feat = build_feature_row(title="Aprende", description="Hoy")
        rec = {"prediccion_rendimiento": "alto", "recomendacion_impulso": "impulsar", "ajustes_sugeridos": []}
        out = rule_based_recommendation(feat, rec)
        missing = [k for k in LLM_OUTPUT_KEYS if k not in out]
        assert not missing, f"faltan en LLM contract: {missing}"
        assert out["llm_engine"] == "rules"
        ok(name, f"engine={out['llm_engine']}, claves OK")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_gradio_builds() -> None:
    name = "test_gradio_builds"
    try:
        import gradio  # noqa: F401
    except Exception as exc:
        skip(name, f"gradio no instalado: {exc}")
        return
    # Si es el stub de testing (no tiene .themes), saltar el test.
    if not hasattr(gradio, "themes"):
        skip(name, "gradio stub sin .themes (entorno de testing); en runtime real funciona")
        return
    try:
        from app import build_demo  # type: ignore
        demo = build_demo()
        assert demo is not None
        try:
            demo.close()
        except Exception:
            pass
        ok(name, "build_demo() ok")
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def _force_clean_exit(code: int) -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


def main() -> int:
    print("== Smoke tests YouTube Boost AI ==\n")
    test_imports()
    test_manual_prediction()
    test_demo_case()
    test_json_contract()
    test_transcription_obligatory()
    test_policy_evaluator()
    test_hybrid_score_action()
    test_predict_fallback()
    test_ocr_fallback()
    test_transcription_fallback()
    test_youtube_id_extraction()
    test_rule_based_recommendation()
    test_gradio_builds()
    test_script_analyzer_basic()
    test_script_analyzer_detects_exaggerated_claim()
    test_new_modules_lightweight()
    test_exec_summary_marketing_voice()
    test_analytics_charts_dark_theme()
    test_downloader_graceful_no_backends()
    # llm_provider (4 modos + lazy loading)
    test_llm_rules_always_returns_text()
    test_llm_auto_fallback_without_key()
    test_llm_local_smol_fallback_if_missing_dependencies()
    test_llm_output_has_source()
    test_llm_provider_does_not_import_torch_eagerly()
    # Cambios nuevos: sentimiento, transcripción WAV, OCR OpenCV, fill_from_url
    test_comment_sentiment_basic()
    test_comment_sentiment_empty()
    test_transcription_wav_no_crash()
    test_ocr_opencv_no_crash()
    test_fill_from_url_no_key()

    n_pass = sum(1 for r in RESULTS if r[0] == "PASS")
    n_skip = sum(1 for r in RESULTS if r[0] == "SKIP")
    n_fail = sum(1 for r in RESULTS if r[0] == "FAIL")
    print(f"\n== Resumen: {n_pass} PASS · {n_skip} SKIP · {n_fail} FAIL ==")
    if n_fail:
        print("Pruebas fallidas:")
        for status, name, detail in RESULTS:
            if status == "FAIL":
                print(f"  - {name}: {detail}")
        return 1
    print("OK: smoke tests completados")
    return 0


def test_new_modules_lightweight() -> None:
    name = "test_new_modules_lightweight"
    try:
        from src.visual_composition import analyze_visual_composition
        from src.metric_forecaster import build_boost_projection
        from src.youtube_downloader import download_youtube_video_360p
        from src.features import build_feature_row
        v = analyze_visual_composition(None)
        assert v["visual_ok"] is False
        d = download_youtube_video_360p("")
        assert d["ok"] is False
        feat = build_feature_row(
            title="Aprende rápido", description="Guarda este video", transcript_text="Contenido de prueba con CTA",
            ocr_text="", category_id="22", duration_seconds=30, views=1000, likes=80, comments=5
        )
        proj = build_boost_projection(feat, performance_probability=0.7, budget=100, cpm=5, operational_metrics={"retention_rate": 0.45, "share_rate": 0.01}, policy_risk_level="bajo")
        assert proj["projected_views_after_boost"] >= 0
        ok(name, "visual fallback + downloader empty + metric projection OK")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_script_analyzer_basic() -> None:
    name = "test_script_analyzer_basic"
    try:
        from src.script_analyzer import analyze_video_script
        r = analyze_video_script(
            "Cómo mejorar tu productividad en 3 pasos",
            "Tutorial breve con consejos prácticos",
            "En este video aprenderás tres pasos para mejorar tu productividad. Guarda el video y prueba estos consejos hoy.",
            category="educación",
            topic="productividad",
            duration_seconds=45,
        )
        assert 0 <= r["script_quality_score"] <= 100
        assert r["transcript_available"] is True
        assert r["suggested_hook"]
        ok(name, f"score={r['script_quality_score']}, tono={r['tone']}")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_script_analyzer_detects_exaggerated_claim() -> None:
    name = "test_script_analyzer_detects_exaggerated_claim"
    try:
        from src.script_analyzer import analyze_video_script
        r = analyze_video_script(
            "Pierde peso garantizado",
            "Método 100% seguro",
            "Pierde peso rápido sin esfuerzo con este método milagroso y garantizado.",
        )
        assert r["policy_claim_risk"] == "alto"
        assert r["recommended_script_improvements"]
        ok(name, f"risk={r['policy_claim_risk']}")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_exec_summary_marketing_voice() -> None:
    """El resumen ejecutivo debe ser legible para marketing: sin jerga técnica,
    con porcentajes, dólares formateados y bullets cortos."""
    name = "test_exec_summary_marketing_voice"
    try:
        from src.exec_summary import build_executive_summary, ACTION_PHRASES
        # Caso fuerte sin riesgo
        rec = {
            "accion_final": "IMPULSAR", "prediccion_rendimiento": "alto",
            "probabilidad_rendimiento": 0.85, "score_hibrido": 0.72,
            "ajustes_sugeridos": [], "alcance_estimado_por_dolar": 300,
            "alcance_estimado_total": 6000, "cpm_estimado": 5.0,
            "multiplicador_potencial": 1.3,
        }
        out = build_executive_summary(
            final_recommendation=rec,
            prediction={"level": "alto", "probability": 0.85},
            features={"engagement_rate": 0.06, "views": 10000, "likes": 500, "comments": 30},
            operational_metrics={"shares": 30, "retention_rate": 0.55},
            policy_block={"policy_risk_level": "bajo", "policy_risk_categories": []},
            metadata={"title": "Tutorial Python"},
            cpm=5.0, budget=20.0,
        )
        # Verificar contrato del bloque ejecutivo
        for k in ("headline", "score_global_0_100", "probabilidad_pct", "policy_status_human",
                  "por_que", "que_hacer_ahora", "forecast", "markdown"):
            assert k in out, f"falta {k}"
        # No debe haber jerga técnica
        forbidden = ["score_hibrido", "feature engineering", "logistic regression", "f1-score"]
        for term in forbidden:
            assert term.lower() not in out["markdown"].lower(), f"jerga técnica filtrada: {term}"
        assert out["score_global_0_100"] == 72
        assert out["probabilidad_pct"] == 85
        assert out["action"] == "IMPULSAR"
        # Headline para IMPULSAR debe ser positivo y claro
        assert "listo para pauta" in out["headline"].lower() or "adelante" in out["headline"].lower()
        ok(name, f"headline='{out['headline'][:40]}...' | score 0-100={out['score_global_0_100']}")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_analytics_charts_dark_theme() -> None:
    """create_analysis_charts debe generar 3 PNG con tema oscuro."""
    name = "test_analytics_charts_dark_theme"
    try:
        from src.analytics_viz import create_analysis_charts, PALETTE
        from pathlib import Path as _P
        # Verificar palette dark
        assert PALETTE["bg"] == "#0b0d12", "background debe ser dark"
        # Generar charts a partir de un resultado mínimo
        result = {
            "probabilidad_rendimiento": 0.7,
            "policy_risk_level": "bajo",
            "metricas": {
                "engagement_rate": 0.04,
                "views": 10000, "likes": 500, "comments": 30,
                "operational": {"retention_rate": 0.5, "shares": 25},
            },
            "analisis_guion": {"script_quality_score": 65},
            "analisis_politicas": {"policy_risk_categories": []},
            "proyeccion_pauta": {
                "projected_views_after_boost": 25000,
                "projected_likes_after_boost": 1200,
                "projected_comments_after_boost": 80,
                "projected_shares_after_boost": 60,
            },
            "multiplicador_potencial": 1.3,
        }
        charts = create_analysis_charts(result)
        for key in ("score_chart", "projection_chart", "policy_chart"):
            assert charts.get(key), f"falta {key}"
            assert _P(charts[key]).exists(), f"chart no existe: {key}"
            assert _P(charts[key]).stat().st_size > 1000, f"chart vacío: {key}"
        ok(name, "3 charts dark generados (score, projection, policy)")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_downloader_graceful_no_backends() -> None:
    """download_youtube_video_360p NO debe romper aunque no haya backends instalados."""
    name = "test_downloader_graceful_no_backends"
    try:
        from src.youtube_downloader import download_youtube_video_360p
        # URL vacía
        r0 = download_youtube_video_360p("")
        assert r0["ok"] is False
        assert "warning" in r0
        # URL real: si ninguna lib está instalada, debe reportar attempts y un warning claro
        r = download_youtube_video_360p("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert "attempts" in r, "debe reportar attempts por backend"
        assert isinstance(r["attempts"], list)
        # En este sandbox no hay yt-dlp ni pytubefix ni pytube → debe ser graceful
        if not r["ok"]:
            assert r["warning"], "debe haber mensaje de advertencia"
            assert any("no disponible" in (a.get("error") or "") or a.get("ok") for a in r["attempts"])
        ok(name, f"graceful fail: attempts={len(r['attempts'])}, backend_used={r.get('backend_used')}")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


# ---------------------------------------------------------------------------
# Tests del llm_provider: rules, auto sin key, local_smol sin deps, source
# ---------------------------------------------------------------------------

def _sample_diagnostic() -> dict:
    return {
        "titulo": "Tutorial de Python para principiantes",
        "accion": "AJUSTAR ANTES DE IMPULSAR",
        "probabilidad_pct": 78,
        "score_0_100": 64,
        "metricas_publicas": {"views": 15000, "likes": 900, "comments": 45, "engagement_pct": 6.3},
        "metricas_privadas": {"shares": 50, "retention_pct": 55.0, "average_watch_time": 180, "hours_since_publication": 24},
        "politica": {"nivel": "bajo", "estado": "apto", "categorias": []},
        "guion": {"score": 70, "tono": "educativo", "mejoras": ["Reforzar CTA"]},
        "ocr_text": "APRENDE PYTHON",
        "visual": {"score_0_100": 65, "resumen": "Composición correcta, balance medio.", "sugerencias": ["Mantener encuadre"]},
        "proyeccion": {"cpm": 5.0, "presupuesto": 50.0, "views_esperadas": 5000, "likes_esperados": 300,
                       "comments_esperados": 25, "shares_esperados": 15},
    }


def test_llm_rules_always_returns_text() -> None:
    """rules siempre debe producir texto válido, sin red, sin LLM, sin dependencias."""
    name = "test_llm_rules_always_returns_text"
    try:
        from src.llm_provider import generate_recommendation_with_llm
        d = _sample_diagnostic()
        out = generate_recommendation_with_llm(d, mode="rules")
        assert isinstance(out, dict), "debe devolver dict"
        assert out.get("source") == "rules", f"source debe ser rules, got {out.get('source')}"
        assert isinstance(out.get("recomendacion"), str)
        assert len(out["recomendacion"]) > 40, "el texto debe ser sustancial"
        # No debe prometer cosas prohibidas
        forbidden = ["viralidad garantizada", "roi garantizado", "youtube va a aprobar"]
        low = out["recomendacion"].lower()
        for term in forbidden:
            assert term not in low, f"texto contiene frase prohibida: '{term}'"
        ok(name, f"rules → {len(out['recomendacion'])} chars, sin promesas prohibidas")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_llm_auto_fallback_without_key() -> None:
    """auto sin GEMINI_API_KEY (y sin torch) debe caer en rules."""
    name = "test_llm_auto_fallback_without_key"
    try:
        from src.llm_provider import generate_recommendation_with_llm
        prev_gemini = os.environ.pop("GEMINI_API_KEY", None)
        try:
            d = _sample_diagnostic()
            out = generate_recommendation_with_llm(d, mode="auto")
            # Sin GEMINI_API_KEY y sin torch → rules
            assert out["source"] in {"rules", "local_smol"}, f"sin key auto debe ser rules o smol, got {out['source']}"
            assert out["mode_requested"] == "auto"
            assert out["recomendacion"]
        finally:
            if prev_gemini is not None:
                os.environ["GEMINI_API_KEY"] = prev_gemini
        ok(name, f"auto sin key → {out['source']}")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_llm_local_smol_fallback_if_missing_dependencies() -> None:
    """local_smol sin transformers/torch instalados debe caer en rules sin romper."""
    name = "test_llm_local_smol_fallback_if_missing_dependencies"
    try:
        # Si transformers/torch NO están en el entorno, debe caer a rules.
        # Si SÍ están, lo saltamos (no podemos forzar el ImportError sin trucos invasivos).
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            skip(name, "transformers/torch instalados en este entorno; el fallback se prueba en CI sin esas libs")
            return
        except ImportError:
            pass
        from src.llm_provider import generate_recommendation_with_llm
        d = _sample_diagnostic()
        out = generate_recommendation_with_llm(d, mode="local_smol")
        assert out["source"] == "rules", f"sin transformers/torch local_smol debe caer en rules, got {out['source']}"
        assert out["mode_requested"] == "local_smol"
        assert "rules" in (out.get("warning") or "").lower() or "smollm" in (out.get("warning") or "").lower()
        ok(name, f"local_smol sin deps → rules ({out.get('warning', '')[:60]}...)")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_llm_output_has_source() -> None:
    """Cada modo debe devolver source explícito y campos completos."""
    name = "test_llm_output_has_source"
    try:
        from src.llm_provider import generate_recommendation_with_llm
        d = _sample_diagnostic()
        for mode in ("auto", "gemini", "local_smol", "rules"):
            out = generate_recommendation_with_llm(d, mode=mode)
            for k in ("recomendacion", "source", "warning", "mode_requested", "elapsed_s"):
                assert k in out, f"falta {k} en modo {mode}"
            assert out["source"] in {"gemini", "local_smol", "rules"}, f"source inválido: {out['source']}"
            assert out["mode_requested"] == mode
            assert isinstance(out["recomendacion"], str) and out["recomendacion"]
        ok(name, "los 4 modos devuelven dict con source explícito y texto")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_llm_provider_does_not_import_torch_eagerly() -> None:
    """El módulo llm_provider NO debe importar torch/transformers en su carga."""
    name = "test_llm_provider_does_not_import_torch_eagerly"
    try:
        # Forzar reimport del módulo en aislamiento
        import importlib
        # Limpiar si ya estaba
        for k in list(sys.modules.keys()):
            if k == "src.llm_provider" or k.startswith("src.llm_provider."):
                del sys.modules[k]
        # Estado de torch/transformers antes
        had_torch_before = any(k == "torch" or k.startswith("torch.") for k in sys.modules)
        had_tx_before = any(k == "transformers" or k.startswith("transformers.") for k in sys.modules)
        import src.llm_provider  # noqa: F401
        importlib.reload(src.llm_provider)
        had_torch_after = any(k == "torch" or k.startswith("torch.") for k in sys.modules)
        had_tx_after = any(k == "transformers" or k.startswith("transformers.") for k in sys.modules)
        if had_torch_before or had_tx_before:
            skip(name, "torch/transformers ya estaban en sys.modules antes del import")
            return
        assert not had_torch_after, "llm_provider importó torch al cargarse"
        assert not had_tx_after, "llm_provider importó transformers al cargarse"
        ok(name, "lazy loading correcto (no trae torch ni transformers al importar)")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


# ---------------------------------------------------------------------------
# Tests de los cambios nuevos de esta entrega
# ---------------------------------------------------------------------------

def test_comment_sentiment_basic() -> None:
    name = "test_comment_sentiment_basic"
    try:
        from src.comment_sentiment import analyze_comments_sentiment
        comments = [
            "Excelente video, me encantó muchísimo",
            "Horrible, no sirve para nada",
            "Está bien, gracias",
            "Increíble explicación, muy útil",
            "Malo y aburrido",
        ]
        r = analyze_comments_sentiment(comments)
        assert r["total"] == 5
        assert r["positivos"] >= 1
        assert r["sentimiento_dominante"] in {"positivo", "neutro", "negativo"}
        for k in ("total", "positivos", "neutros", "negativos", "pct_positivo",
                  "sentimiento_dominante", "ejemplos"):
            assert k in r, f"falta {k}"
        ok(name, f"pos={r['positivos']} neg={r['negativos']} dom={r['sentimiento_dominante']}")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_comment_sentiment_empty() -> None:
    name = "test_comment_sentiment_empty"
    try:
        from src.comment_sentiment import analyze_comments_sentiment
        r = analyze_comments_sentiment([])
        assert r["total"] == 0 and r["sentimiento_dominante"] == "neutro"
        ok(name, "lista vacía → estructura válida")
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_transcription_wav_no_crash() -> None:
    name = "test_transcription_wav_no_crash"
    try:
        from src.transcription import resolve_transcript
        r = resolve_transcript("", None)
        assert not r["ok"] and isinstance(r.get("warning"), str)
        r2 = resolve_transcript("Texto manual de prueba.", None)
        assert r2["ok"] and r2["source"] == "manual"
        ok(name, "resolve_transcript funciona sin video ni SpeechRecognition")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_ocr_opencv_no_crash() -> None:
    name = "test_ocr_opencv_no_crash"
    try:
        from src.ocr_video import extract_ocr_from_video, _ocr_frame_opencv
        import numpy as np
        r = extract_ocr_from_video(None)
        assert r["ocr_ok"] is False and "ocr_text" in r
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(frame, "TEST TEXT", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        lines = _ocr_frame_opencv(frame)
        assert isinstance(lines, list)
        ok(name, f"sin video→vacío OK | frame sintético→{len(lines)} líneas detectadas")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


def test_fill_from_url_no_key() -> None:
    name = "test_fill_from_url_no_key"
    try:
        from app import fill_from_url  # type: ignore
        r0 = fill_from_url("")
        assert r0[-1].startswith("⚠️")
        r1 = fill_from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert isinstance(r1, tuple) and len(r1) in {10, 11}
        ok(name, f"vacía={r0[-1][:35]}... | con url→tuple válido")
    except AssertionError as exc:
        fail(name, str(exc))
    except Exception as exc:
        fail(name, f"Excepción: {exc}")


if __name__ == "__main__":
    _force_clean_exit(main())

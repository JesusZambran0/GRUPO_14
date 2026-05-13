# Demo Script — YouTube Boost AI

Guion sugerido para la defensa académica. Duración estimada: 6-8 minutos.

## 0. Pre-flight (antes de empezar)

```bash
# Verificar entorno
pip install -r requirements.txt
python tests/smoke_test.py   # debe terminar con 18 PASS · 1 SKIP · 0 FAIL, exit code 0
```

Abrir la app:
```bash
python app.py
```

La interfaz aparece en `http://127.0.0.1:7860` con tema oscuro.

### Pestañas disponibles en resultados

- **📋 Resumen ejecutivo** — markdown legible por marketing (headline, métricas, "por qué", "qué hacer ahora").
- **📊 Gráficos** — 3 imágenes: diagnóstico, actual vs esperado con $, riesgo política.
- **🎨 Composición visual** — JSON + galería con 10 frames anotados (regla de tercios + centroide).
- **✍️ Guion** — análisis de guion basado en patrones del dataset.
- **🛡️ Políticas** — detalle de evaluación YouTube Ads (12 categorías).
- **📝 Transcripción** — texto producido por faster-whisper o pegado manualmente.
- **👁️ OCR** — texto detectado sobre los frames.
- **🧠 LLM** — recomendación redactada (NO decide).
- **📈 Métricas** — features + métricas operativas.
- **⚠️ Advertencias** — cualquier paso del pipeline que falló.
- **🧬 JSON / API** — salida completa para integración.

---

## 1. Apertura (45 s)

> "YouTube Boost AI es un sistema explicable que analiza un video y entrega:
> 1) una **predicción** de potencial,
> 2) una **evaluación de riesgo** según políticas de YouTube Ads,
> 3) una **acción recomendada** de cinco valores: IMPULSAR, AJUSTAR ANTES DE IMPULSAR, MONITOREAR, NO IMPULSAR o REVISIÓN HUMANA.
>
> El modelo predictivo fue entrenado con **datos reales** de Kaggle (711.493 filas → 68.526 únicas tras dedup). No usa Gemini ni APIs pagas: corre con un LLM local pequeño opcional, o con reglas locales por defecto."

---

## 2. Caso 1 — Video con narrador, mensaje claro (90 s)

En la app, en **Modo demo seguro**, elegir **"Demo: video con narrador"** y click en **Analizar**.

Mostrar resultados:

- **Pestaña Resultado ejecutivo**:
  - Acción: `AJUSTAR ANTES DE IMPULSAR` (score híbrido 0.75)
  - Riesgo de políticas: `bajo` → estado `apto`
- **Pestaña Métricas**: engagement, retention, views/h, score híbrido detalle.
- **Pestaña Transcripción**: muestra el texto fuente.
- **Pestaña Políticas YouTube Ads**: 0 categorías detectadas.

> "Aquí el modelo predice potencial alto y las políticas dan luz verde. Sin embargo, el recomendador detecta que falta optimizar el CTA, por eso recomienda **ajustar antes de impulsar** en lugar de impulsar directamente."

---

## 3. Caso 2 — Saturación de texto visual (90 s)

Elegir **"Demo: video sin narrador"**.

- Acción: `AJUSTAR ANTES DE IMPULSAR`, riesgo `medio`.
- Pestaña OCR: muchas palabras detectadas, densidad `alta`.
- En las pestañas, ver el listado de ajustes: "reducir cantidad de texto visible".

> "Mismo nivel de acción pero por motivos distintos: hay saturación visual. El sistema separa el origen del problema (OCR vs políticas vs métricas)."

---

## 4. Caso 3 — Riesgo de políticas (90 s)

Elegir **"Demo: requiere ajustes"**.

- Acción: **`REVISIÓN HUMANA`**.
- Pestaña Políticas: categorías detectadas → `claims financieros / afirmaciones exageradas / salud o pérdida de peso`.
- Estado YouTube Ads: `revisión humana`.

> "Aquí el modelo solo detecta texto promocional fuerte, pero el evaluador de políticas detecta múltiples categorías de alta severidad. El sistema escala automáticamente a **revisión humana** independientemente de qué dijo el modelo. **El LLM no decide; solo redacta.**"

---

## 5. Caso 4 — Sin transcripción (60 s)

Desactivar el demo (elegir "Ninguno"). Llenar manualmente:
- Título: "Mi video"
- Descripción: "Una descripción cualquiera"
- Views: 5000, Likes: 100, Comments: 10
- Dejar **vacío** el campo Transcript manual y NO subir video.

Click **Analizar**.

- Acción: **`REVISIÓN HUMANA`**.
- Pestaña Advertencias: "Políticas: revisión humana obligatoria (falta transcripción)."

> "La **transcripción es obligatoria** para evaluar políticas. Si el usuario no sube video ni pega transcript manual, el sistema rehúsa evaluar y escala a revisión humana. No improvisa."

---

## 6. Caso 5 — Caso manual completo (90 s)

Volver a llenar el formulario con un video real (o ficticio):

- Título: "Curso introductorio a Python para principiantes"
- Descripción: "Aprende desde cero los fundamentos de Python..."
- Categoría: 28, Tópico: programación
- Views: 50000, Likes: 3000, Comments: 200
- Shares: 80, Retention: 0.55, Average watch time: 240
- Hours since publication: 48
- Followers: 25000, Channel reach: 5000
- Duration: 900
- Transcript manual: "Hola, en este video te enseño los conceptos básicos de Python..."

Click **Analizar**.

Mostrar:
- **Resultado ejecutivo**: acción concreta + score híbrido.
- **JSON técnico**: contrato completo con `accion_final`, `policy_risk_level`, `score_hibrido_detalle`, `analisis_llm`.

> "Las variables `shares` y `retention_rate` no son predictores del modelo (no estaban en el dataset de entrenamiento). Entran solo en el score híbrido. Eso lo documentamos abiertamente en `docs/DATASET_CARD.md`."

---

## 7. Cierre — Limitaciones y honestidad metodológica (45 s)

> "Tres cosas honestas para cerrar:
>
> 1. **Sesgo de selección**: el modelo aprende de videos en trending. Discrimina 'muy bueno' de 'bueno', no 'bueno' de 'malo'. Las probabilidades tienden a ser altas; las reglas de ajuste compensan.
>
> 2. **Anti-leakage**: las variables que componen la etiqueta NO se usan como predictores. F1 = 0.52, AUC = 0.73. Métricas honestas.
>
> 3. **No es ROI causal**: el alcance por dólar es estimación parametrizada por CPM, no retorno real.
>
> La demo corre sin internet, sin Gemini, sin API keys. Open source end-to-end."

---

## Apéndice — Comandos rápidos

```bash
# Reentrenar con datos reales
python -m src.train

# Regenerar demos
python scripts/build_demo_cache.py

# Pre-cargar Whisper para transcripción real
python scripts/download_whisper_model.py tiny

# Pre-cargar LLM local (opcional)
python scripts/download_local_llm.py
```

## Apéndice — API Gradio

```python
from gradio_client import Client
client = Client("http://127.0.0.1:7860")
out = client.predict(
    "Ninguno", "", None,
    "Tutorial Python", "Curso desde cero", "28", "programación",
    50000, 3000, 200,
    80, 0.55, 240,
    48, 25000, 5000,
    900,
    "Hola, en este video te enseño Python.",
    "", "con narrador",
    5.0, 50.0, "rules", True,
    api_name="/analyze",
)
summary, result_json, transcript, ocr, metricas, llm, ocr_block, policy_block, warnings = out
```

## Secuencia recomendada para demostrar el análisis del guion

1. Abrir el enlace público de Hugging Face Spaces.
2. Seleccionar modo `rules` en la sección LLM para evitar dependencia de modelos externos.
3. Ingresar un título, descripción y transcripción manual.
4. Completar views, likes, comments, shares, retention rate, duración y horas desde publicación.
5. Presionar **Analizar**.
6. Mostrar la pestaña **Resultado ejecutivo**.
7. Mostrar la pestaña **Políticas YouTube Ads**.
8. Mostrar la pestaña **Análisis del guion**.
9. Explicar que el sistema no solo predice potencial, sino que sugiere mejoras concretas de gancho, claridad, CTA y promesas del copy.

Frase sugerida:

> En esta sección se observa el análisis semántico del guion. El sistema evalúa gancho inicial, claridad, propuesta de valor, llamado a la acción y riesgo de promesas exageradas. Si el video tiene buen potencial cuantitativo, pero el guion no está listo para pauta, la recomendación cambia a ajustar antes de impulsar.

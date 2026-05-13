# Análisis exhaustivo y estado definitivo del repositorio

## Decisión de alcance
El proyecto queda delimitado exclusivamente a YouTube. La solución evalúa videos orgánicos o anuncios publicados, estima su potencial de rendimiento y recomienda si conviene impulsarlos, ajustarlos o monitorearlos. El sistema usa metadatos públicos, métricas de engagement, transcripción, OCR de texto visible y análisis semántico-sintáctico con LLM o reglas locales.

## Arquitectura validada
1. Entrada por URL de YouTube, video MP4 o formulario manual.
2. Extracción opcional de metadatos con YouTube Data API.
3. Procesamiento de video con OpenCV.
4. OCR con EasyOCR y fallback probado con pytesseract.
5. Transcripción con Whisper si está instalado y fallback manual/reglas.
6. Construcción de variables textuales, sintácticas, semánticas y de engagement.
7. Modelo predictivo supervisado para `boost_candidate`.
8. Recomendación final, métricas y alcance por dólar vía CPM parametrizado.
9. Demo Gradio con endpoint `/analyze`.

## Modelos entrenados y comparados
Se entrenaron tres modelos compatibles con CPU y entornos gratuitos. El mejor modelo guardado fue **linear_svc**.

|   accuracy |   precision |   recall |       f1 |   roc_auc |   tn |   fp |   fn |   tp | model               |
|-----------:|------------:|---------:|---------:|----------:|-----:|-----:|-----:|-----:|:--------------------|
|   0.983333 |    0.947368 | 1        | 0.972973 |  0.999683 |  205 |    5 |    0 |   90 | linear_svc          |
|   0.97     |    0.917526 | 0.988889 | 0.951872 |  0.999048 |  202 |    8 |    1 |   89 | logistic_regression |
|   0.923333 |    0.831683 | 0.933333 | 0.879581 |  0.977249 |  193 |   17 |    6 |   84 | random_forest       |

## Archivos generados
- `models/best_model.joblib`: modelo final entrenado.
- `models/model_metadata.json`: metadatos del entrenamiento.
- `outputs/model_comparison.csv`: comparación de modelos.
- `outputs/eda_summary.md`: resumen EDA.
- `data/processed/youtube_training_dataset.csv`: dataset procesado con etiqueta.

## Estado de pruebas
El repositorio fue probado localmente con:
- carga de casos demo precargados;
- inferencia manual con modelo entrenado;
- construcción de interfaz Gradio;
- OCR sobre video MP4 de ejemplo mediante fallback pytesseract;
- ejecución de entrenamiento completo con tres modelos;
- generación de salidas EDA y métricas.

## Limitaciones honestas
El dataset incluido es una muestra de validación funcional generada localmente para que el repositorio ejecute sin credenciales ni descargas externas. Para la entrega académica final se recomienda agregar datasets públicos reales de Kaggle en `data/raw/`, ejecutar nuevamente `python -m src.train` y reemplazar los resultados del modelo. La estimación de alcance por dólar depende del CPM ingresado y no representa ROI causal real.

## Datasets públicos recomendados para reemplazo/complemento
1. YouTube Trending Videos 2025 updated daily.
2. YouTube Trending Video Dataset updated daily.
3. Global YouTube Trending Dataset 2022–2025.
4. Trending YouTube Video Statistics.
5. YouTube top channel videos o YouTube analytics datasets recientes.

## Recomendación final de implementación
Para defensa, usar Hugging Face Spaces + Gradio. La app funciona con modo seguro, permite API mediante `gradio_client` y no requiere AWS.

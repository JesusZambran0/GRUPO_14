# XGBoost para rendimiento pagado y CPM

## Flujo implementado

1. La regresión logística estima `probability` de candidatura orgánica/publicitaria.
2. Si `probability >= 0.51`, se activa `src.paid_ads_xgboost.predict_paid_ad_boost`.
3. XGBoost estima:
   - `predicted_paid_performance_score` en escala 0–100.
   - `predicted_cpm`.
   - `estimated_impressions_per_dollar = 1000 / CPM`.
   - `ad_niche` por reglas transparentes de NLP.

## Métricas del modelo entrenado

| target                 |     mae |     rmse |       r2 |   mape_pct |   test_mean |
|:-----------------------|--------:|---------:|---------:|-----------:|------------:|
| paid_performance_score | 8.81276 | 10.9026  | 0.377508 |    19.4576 |     50.607  |
| cpm                    | 4.50506 |  5.91092 | 0.508643 |   149.559  |     10.0111 |

## Datasets usados

| file                                           |   rows | description                                                                                          | source_url                                                                                                     |
|:-----------------------------------------------|-------:|:-----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|
| facebook_kag_conversion.csv                    |   1143 | Fuente raw Facebook/KAG con impressions, clicks, spent y conversions                                 | https://raw.githubusercontent.com/mGalarnyk/Python_Tutorials/master/Kaggle/Facebook/KAG_conversion_data.csv    |
| onyx_marketing_campaign.csv                    |   9900 | Fuente raw Onyx con campaign, channel, device, impressions, CTR, clicks, CPC, spend y conversions    | https://raw.githubusercontent.com/Github-sanket07sett/Market-Campaign-Analysis/refs/heads/main/onyx%20data.csv |
| paid_ads_xgboost_training_processed.csv        |  10763 | Dataset procesado con features, targets y métricas auxiliares para auditoría                         | Derivado de los dos CSV raw anteriores                                                                         |
| paid_ads_xgboost_training_ready_no_leakage.csv |  10763 | Dataset listo para entrenamiento XGBoost sin columnas de fuga; targets: paid_performance_score y cpm | Derivado de paid_ads_xgboost_training_processed.csv                                                            |
| paid_ads_xgboost_train_ready_no_leakage.csv    |   8610 | Partición train 80% sin columnas de fuga                                                             | Derivado de paid_ads_xgboost_training_ready_no_leakage.csv                                                     |
| paid_ads_xgboost_test_ready_no_leakage.csv     |   2153 | Partición test 20% sin columnas de fuga                                                              | Derivado de paid_ads_xgboost_training_ready_no_leakage.csv                                                     |

## Advertencia

El modelo usa datasets públicos de campañas pagadas y debe usarse como estimación inicial. Para una tesis sólida, la calibración final debe hacerse con campañas reales del anunciante o con experimentos controlados de bajo presupuesto.

# Comparación predictiva sobre precios originales — Train

Baseline recuperada de MLflow: 43e383e9b91d43efbdeda5ec1ec28f71. Mismas filas y predictoras.

| Métrica | Baseline_OLS_01 | OLS_02_Log_Target | Diferencia |
|---|---|---|---|
| R² (escala original) | 0.893313416689 | 0.927524646297 | 0.0342112296074 |
| MAE (RUB) | 1442336.50357 | 1093408.72367 | -348927.779896 |
| MSE (RUB²) | 3.65528082297e+12 | 2.48314045037e+12 | -1.17214037261e+12 |
| RMSE (RUB) | 1911878.87246 | 1575798.35333 | -336080.519131 |

Diferencia = OLS_02_Log_Target − Baseline_OLS_01. Predicción nueva = exp(predicción log), sin corrección.

## Estadísticas del modelo en ln(RUB), sin diferencias entre escalas

| Métrica | OLS_02_Log_Target | Unidad / Observación |
|---|---|---|
| R² (ln(RUB)) | 0.955694187914 | Train; ajuste log |
| R² ajustado (ln(RUB)) | 0.955486757116 | Train; ajuste log |
| MAE (ln(RUB)) | 0.0933372242985 | ln(RUB) |
| RMSE (ln(RUB)) | 0.117406144956 | ln(RUB) |
| RSS (ln(RUB)) | 523.579161946 | ln(RUB)² |
| MSE (ln(RUB)) | 0.0137842028735 | ln(RUB)² |
| AIC (ln(RUB)) | -54582.3481458 | Train; likelihood ln(RUB); k=rank |
| BIC (ln(RUB)) | -53061.3523329 | Train; likelihood ln(RUB); k=rank |
| F-statistic (ln(RUB)) | 4607.29167676 | Train; ajuste log |
| Probabilidad F-statistic (ln(RUB)) | 0 | Train; ajuste log |
| n_obs | 37984 | Train |
| n_predictors | 177 | Train |
| n_parameters_effective | 178 | Train |
| n_coefficients | 178 | Train |
| df_model | 177 | Train |
| df_resid | 37806 | Train |
| log_likelihood (ln(RUB)) | 27469.1740729 | Train; ajuste log |
| Durbin-Watson (ln(RUB)) | 1.79418181535 | Train; ajuste log |
| n_coefficients_p_gt_005 (ln(RUB)) | 26 | Train; ajuste log |
| n_coefficients_p_undefined (ln(RUB)) | 0 | Train; ajuste log |
| condition_number | 19159.1660483 | Train |
| rank_deficiency | 0 | Train |
| jarque_bera (ln(RUB)) | 183.542393045 | Train; ajuste log |
| jarque_bera_pvalue (ln(RUB)) | 1.39404166021e-40 | Train; ajuste log |
| skewness (ln(RUB)) | -0.166356738106 | Train; ajuste log |
| kurtosis (ln(RUB)) | 3.07260911624 | Train; ajuste log |
| omnibus (ln(RUB)) | 180.778930354 | Train; ajuste log |
| omnibus_pvalue (ln(RUB)) | 5.5507804677e-40 | Train; ajuste log |
| VIF máximo | 102.584323807 | Sin intercepto |
| vif_infinite_count | 0 | Train |
| vif_undefined_count | 0 | Train |
| bp_lm (ln(RUB)) | 1195.72279526 | Train; ajuste log |
| bp_lm_pvalue (ln(RUB)) | 1.19611077672e-150 | Train; ajuste log |
| bp_f (ln(RUB)) | 6.94238224449 | Train; ajuste log |
| bp_f_pvalue (ln(RUB)) | 1.11625003601e-153 | Train; ajuste log |

AIC/BIC, R² y RSS nativos log no son directamente comparables con las métricas nativas en RUB de la baseline.

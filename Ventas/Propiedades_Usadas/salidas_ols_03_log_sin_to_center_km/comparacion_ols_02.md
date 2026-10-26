# Comparación exclusiva con OLS_02_Log_Target — Train

Referencia MLflow: 2dfd0a0d8d7a41818305f1137704b26e. Mismas filas, mismo objetivo ln(RUB), misma inversa exp sin corrección.

Δ = Experimento 03 − Experimento 02. VIF sin intercepto; media y conteos de OLS_02 derivados de su artefacto vif.csv.

| Métrica | OLS_02_Log_Target | OLS_03_Log_Sin_to_center_km | Diferencia |
|---|---|---|---|
| R² (ln(RUB)) | 0.955694187914 | 0.955388277995 | -0.000305909919117 |
| R² ajustado (ln(RUB)) | 0.955486757116 | 0.955180600499 | -0.000306156616895 |
| MAE (ln(RUB)) | 0.0933372242985 | 0.0936698768163 | 0.000332652517833 |
| RMSE (ln(RUB)) | 0.117406144956 | 0.11781076368 | 0.000404618724259 |
| RSS (ln(RUB)) | 523.579161946 | 527.194219464 | 3.61505751823 |
| MSE (ln(RUB)) | 0.0137842028735 | 0.013879376039 | 9.51731654967e-05 |
| AIC (ln(RUB)) | -54582.3481458 | -54322.9884864 | 259.3596594 |
| BIC (ln(RUB)) | -53061.3523329 | -52810.5375938 | 250.814739103 |
| F-statistic (ln(RUB)) | 4607.29167676 | 4600.34572177 | -6.94595499313 |
| Probabilidad F-statistic (ln(RUB)) | 0 | 0 | 0 |
| n_obs | 37984 | 37984 | 0 |
| n_predictors | 177 | 176 | -1 |
| n_parameters_effective | 178 | 177 | -1 |
| n_coefficients | 178 | 177 | -1 |
| df_model | 177 | 176 | -1 |
| df_resid | 37806 | 37807 | 1 |
| log_likelihood (ln(RUB)) | 27469.1740729 | 27338.4942432 | -130.6798297 |
| Durbin-Watson (ln(RUB)) | 1.79418181535 | 1.79682392767 | 0.00264211232534 |
| n_coefficients_p_gt_005 (ln(RUB)) | 26 | 22 | -4 |
| n_coefficients_p_undefined (ln(RUB)) | 0 | 0 | 0 |
| condition_number | 19159.1660483 | 14533.7195965 | -4625.44645177 |
| rank_deficiency | 0 | 0 | 0 |
| jarque_bera (ln(RUB)) | 183.542393045 | 179.17967428 | -4.36271876462 |
| jarque_bera_pvalue (ln(RUB)) | 1.39404166021e-40 | 1.23488949282e-39 | 1.0954853268e-39 |
| skewness (ln(RUB)) | -0.166356738106 | -0.163908965601 | 0.00244777250443 |
| kurtosis (ln(RUB)) | 3.07260911624 | 3.0758233672 | 0.00321425095787 |
| omnibus (ln(RUB)) | 180.778930354 | 176.476612361 | -4.30231799284 |
| omnibus_pvalue (ln(RUB)) | 5.5507804677e-40 | 4.7707925594e-39 | 4.21571451263e-39 |
| VIF máximo | 102.584323807 | 17.4312021082 | -85.1531216992 |
| VIF promedio | 3.66833820845 | 2.24384256672 | -1.42449564173 |
| Cantidad VIF > 5 | 18 | 4 | -14 |
| Cantidad VIF > 10 | 11 | 3 | -8 |
| vif_infinite_count | 0 | 0 | 0 |
| vif_undefined_count | 0 | 0 | 0 |
| bp_lm (ln(RUB)) | 1195.72279526 | 1180.14332552 | -15.5794697338 |
| bp_lm_pvalue (ln(RUB)) | 1.19611077672e-150 | 3.54121812942e-148 | 3.52925702166e-148 |
| bp_f (ln(RUB)) | 6.94238224449 | 6.88812426253 | -0.0542579819676 |
| bp_f_pvalue (ln(RUB)) | 1.11625003601e-153 | 4.03799297691e-151 | 4.02683047655e-151 |
| R² (precio original) | 0.927524646297 | 0.927055578331 | -0.000469067965507 |
| R² ajustado (precio original) | 0.927185331436 | 0.926716005812 | -0.000469325623967 |
| MAE (precio original) | 1093408.72367 | 1097249.03335 | 3840.30967657 |
| RMSE (precio original) | 1575798.35333 | 1580889.49377 | 5091.14043785 |
| MAPE (precio original) | 9.40748518507 | 9.44103877685 | 0.0335535917768 |
| RSS (precio original) | 9.43196068667e+16 | 9.49300530921e+16 | 6.10446225367e+14 |
| MSE (precio original) | 2.48314045037e+12 | 2.49921159151e+12 | 16071141148 |

## Aporte de to_center_km y contraste complementario

| Diagnóstico | Valor |
|---|---|
| incremental_r2_removed_predictor_log | 0.000305909919118 |
| partial_r2_removed_predictor_log | 0.00685716456053 |
| partial_f_descriptive | 261.031902084 |
| partial_f_df_num | 1 |
| partial_f_df_den | 37806 |
| partial_f_classical_inference_valid | 0 |
| removed_predictor_coefficient_log | -0.00774987058828 |
| removed_predictor_t_nonrobust | -16.1564817359 |
| removed_predictor_pvalue_nonrobust | 1.60644192582e-58 |
| removed_predictor_se_hc3 | 0.000480343071437 |
| removed_predictor_wald_hc3 | 260.306995478 |
| removed_predictor_pvalue_hc3 | 1.47095236043e-58 |
| n_significance_changes | 6 |
| n_significant_to_nonsignificant | 1 |
| n_nonsignificant_to_significant | 5 |
| removed_predictor_vif_reference | 102.584323807 |

F parcial: solo descriptivo; p-value clásico omitido por heterocedasticidad. HC3: Wald asintótico bajo independencia de filas. Los cambios de significancia son nominales (nonrobust).

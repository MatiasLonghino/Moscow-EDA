# OLS_04_Log_Sin_living_area frente a OLS_03_Log_Sin_to_center_km — Train

Referencia MLflow: 7dfbe14054724f099428fcda7fbe92b7. Mismas filas, ln(RUB), exp sin corrección. Δ = nuevo − referencia.

| Métrica | OLS_03_Log_Sin_to_center_km | OLS_04_Log_Sin_living_area | Diferencia |
|---|---|---|---|
| R² (ln(RUB)) | 0.955388277995 | 0.955387057668 | -1.22032695749e-06 |
| R² ajustado (ln(RUB)) | 0.955180600499 | 0.955180559971 | -4.05279128035e-08 |
| MAE (ln(RUB)) | 0.0936698768163 | 0.0936729403337 | 3.06351732762e-06 |
| RMSE (ln(RUB)) | 0.11781076368 | 0.117812374991 | 1.61131044908e-06 |
| RSS (ln(RUB)) | 527.194219464 | 527.208640547 | 0.0144210823722 |
| MSE (ln(RUB)) | 0.013879376039 | 0.013879755701 | 3.79662025384e-07 |
| AIC (ln(RUB)) | -54322.9884864 | -54323.949471 | -0.960984622448 |
| BIC (ln(RUB)) | -52810.5375938 | -52820.0434987 | -9.50590491986 |
| F-statistic (ln(RUB)) | 4600.34572177 | 4626.62331834 | 26.2775965726 |
| Probabilidad F-statistic (ln(RUB)) | 0 | 0 | 0 |
| n_obs | 37984 | 37984 | 0 |
| n_predictors | 176 | 175 | -1 |
| n_parameters_effective | 177 | 176 | -1 |
| n_coefficients | 177 | 176 | -1 |
| df_model | 176 | 175 | -1 |
| df_resid | 37807 | 37808 | 1 |
| log_likelihood (ln(RUB)) | 27338.4942432 | 27337.9747355 | -0.519507688776 |
| Durbin-Watson (ln(RUB)) | 1.79682392767 | 1.79666881869 | -0.000155108985642 |
| n_coefficients_p_gt_005 (ln(RUB)) | 22 | 21 | -1 |
| n_coefficients_p_undefined (ln(RUB)) | 0 | 0 | 0 |
| condition_number | 14533.7195965 | 13585.6160774 | -948.103519156 |
| rank_deficiency | 0 | 0 | 0 |
| jarque_bera (ln(RUB)) | 179.17967428 | 178.832331322 | -0.347342957973 |
| jarque_bera_pvalue (ln(RUB)) | 1.23488949282e-39 | 1.46910440315e-39 | 2.34214910331e-40 |
| skewness (ln(RUB)) | -0.163908965601 | -0.163748101704 | 0.000160863896994 |
| kurtosis (ln(RUB)) | 3.0758233672 | 3.07576640995 | -5.69572444484e-05 |
| omnibus (ln(RUB)) | 176.476612361 | 176.139158049 | -0.337454312014 |
| omnibus_pvalue (ln(RUB)) | 4.7707925594e-39 | 5.64765041049e-39 | 8.7685785109e-40 |
| VIF máximo | 17.4312021082 | 10.6358240947 | -6.79537801352 |
| VIF promedio | 2.24384256672 | 2.10446526252 | -0.139377304197 |
| Cantidad VIF > 5 | 4 | 2 | -2 |
| Cantidad VIF > 10 | 3 | 1 | -2 |
| vif_infinite_count | 0 | 0 | 0 |
| vif_undefined_count | 0 | 0 | 0 |
| bp_lm (ln(RUB)) | 1180.14332552 | 1178.31105751 | -1.83226801458 |
| bp_lm_pvalue (ln(RUB)) | 3.54121812942e-148 | 2.97351914358e-148 | -5.67698985845e-149 |
| bp_f (ln(RUB)) | 6.88812426253 | 6.91656810087 | 0.0284438383458 |
| bp_f_pvalue (ln(RUB)) | 4.03799297691e-151 | 3.42998495637e-151 | -6.08008020543e-152 |
| R² (precio original) | 0.927055578331 | 0.927055846322 | 2.67990846847e-07 |
| R² ajustado (precio original) | 0.926716005812 | 0.926718213363 | 2.20755106128e-06 |
| MAE (precio original) | 1097249.03335 | 1097287.47669 | 38.4433370959 |
| RMSE (precio original) | 1580889.49377 | 1580886.58975 | -2.90402126522 |
| MAPE (precio original) | 9.44103877685 | 9.44133613346 | 0.000297356609341 |
| RSS (precio original) | 9.49300530921e+16 | 9.49297043281e+16 | -348763959488 |
| MSE (precio original) | 2.49921159151e+12 | 2.49920240965e+12 | -9181864.98242 |

## Aporte de living_area

| Diagnóstico | Valor |
|---|---|
| incremental_r2_removed_predictor_log | 1.22032695741e-06 |
| partial_r2_removed_predictor_log | 2.73536533037e-05 |
| partial_f_descriptive | 1.03418785927 |
| partial_f_df_num | 1 |
| partial_f_df_den | 37807 |
| partial_f_classical_inference_valid | 0 |
| removed_predictor_coefficient_log | 0.000209097738886 |
| removed_predictor_t_nonrobust | 1.01695027374 |
| removed_predictor_pvalue_nonrobust | 0.309183593206 |
| removed_predictor_se_hc3 | 0.000208760123331 |
| removed_predictor_wald_hc3 | 1.00323709866 |
| removed_predictor_pvalue_hc3 | 0.316528490488 |
| n_significance_changes | 0 |
| n_significant_to_nonsignificant | 0 |
| n_nonsignificant_to_significant | 0 |
| removed_predictor_vif_reference | 13.8574008587 |
| related_total_area_correlation_train | 0.963059708981 |
| related_total_area_vif_reference | 17.4312021082 |
| related_total_area_vif_new | 4.66299349574 |
| related_total_area_coefficient_reference_log | 0.0156505766983 |
| related_total_area_coefficient_new_log | 0.0157807225727 |
| related_total_area_pvalue_reference_nonrobust | 0 |
| related_total_area_pvalue_new_nonrobust | 0 |

F parcial solo descriptivo: p-value clásico omitido por supuestos no verificados; HC3 supone independencia de filas. Cambios de significancia nominales (nonrobust). Correlaciones, cuando se incluyen, corresponden a las columnas transformadas de Train.

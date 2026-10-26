# OLS_04_Log_Sin_living_area frente a OLS_03_Log_Sin_to_center_km

living_area y total_area presentan correlación de Pearson 0.9631 en las columnas transformadas de Train. Al excluir living_area, el VIF de total_area baja de 17.4312 a 4.6630; el VIF máximo pasa de 17.4312 a 10.6358 y el promedio de 2.2438 a 2.1045. Los conteos VIF > 5 / > 10 pasan de 4 / 3 a 2 / 1. El Condition Number disminuye de 14533.72 a 13585.62; la colinealidad se reduce, aunque no desaparece completamente.

El aporte incremental de living_area al R² log del modelo anterior es 1.22032696e-06; el R² ajustado cambia -4.053e-08. AIC baja 0.961 y BIC 9.506: ambos favorecen al modelo reducido, con una diferencia pequeña en AIC. En precios originales, MAE aumenta 38.44 RUB (+0.00350 %) y RMSE cambia -2.90 RUB (-0.00018 %). Son variaciones empíricas mínimas en Train, no una prueba de equivalencia.

El coeficiente recuperado de living_area es 0.000209098, t nominal=1.0170, p nominal=0.3092. Wald HC3=1.0032, p=0.3165: bajo independencia de filas, no se detecta aporte adicional al 5 %; no se demuestra un efecto exactamente nulo. El F parcial (1.0342) es solo descriptivo; se omite el p-value clásico por heterocedasticidad. HC3 no corrige dependencia temporal o por grupos.

No cambia la significancia nominal de los coeficientes restantes (0 cambios). El total no significativo pasa de 22 a 21 únicamente porque se retiró living_area. El coeficiente condicional de total_area cambia de 0.015650577 a 0.015780723; los demás cambios están documentados en cambios_significancia.csv, sin nuevas exclusiones.

Los tres gráficos de residuos son prácticamente iguales a los del Experimento 03. Persisten desviaciones en las colas y heterocedasticidad (Breusch-Pagan p=2.974e-148). En este diseño, los resultados son compatibles con que gran parte de la información de living_area esté compartida con otras superficies, especialmente total_area: quitarla mejora la colinealidad sin un deterioro apreciable observado en Train. No se afirma redundancia universal ni superioridad fuera de muestra. Se mantienen to_center_km excluida, las mismas 37.984 filas y ln(price_rub); no se utilizó Test ni se entrenaron otros modelos.

# Interpretación de OLS_02_Log_Target — Train

El QQ-Plot del ajuste logarítmico queda más próximo a la recta normal, aunque conserva desviaciones en las colas. La asimetría cambia de 0.4532 a -0.1664 y la curtosis de 4.2729 a 3.0726. Jarque-Bera pasa de 3864.73 a 183.54, pero su p-value log (1.394e-40) sigue rechazando normalidad al 5 %. La mejora visual y de los momentos no demuestra normalidad exacta.

Scale-Location presenta una curva LOWESS más horizontal y una dispersión más uniforme que la baseline. Persiste un aumento de la dispersión hacia los mayores ajustados. Residuos vs ajustados también conserva estructura en el extremo superior; los patrones no desaparecen completamente.

Breusch-Pagan robusto (Koenker) devuelve LM=1195.72, p=1.196e-150; en la baseline LM=5743.81, con p devuelto como 0 (posible subdesbordamiento numérico). Ambos rechazan homocedasticidad. El menor estadístico del nuevo ajuste no permite afirmar que la heterocedasticidad esté resuelta.

En precios originales, sin corrección de sesgo, MAE disminuye 24.19 % y RMSE 17.58 %. El R² comparable en RUB pasa de 0.893313 a 0.927525; la tabla anterior conserva también MSE y sus diferencias. AIC/BIC, RSS y R² nativos log se informan por separado y no se comparan directamente con sus equivalentes nativos en RUB.

No todos los diagnósticos mejoran: Durbin-Watson cambia de 1.9220 a 1.7942, alejándose de 2; depende del orden de filas del CSV y no constituye por sí solo una prueba temporal de este panel. La inversa exponencial mantiene el posible sesgo al estimar el precio esperado.

La menor desviación de normalidad, la dispersión más uniforme y los menores errores de Train justifican considerar ln(price_rub) en modelos posteriores. Este resultado no establece superioridad fuera de muestra ni una decisión definitiva de modelo óptimo: no se utilizó Test. Se mantienen todas las filas y las 177 predictoras, incluida to_center_km.

# OLS_03_Log_Sin_to_center_km frente a OLS_02_Log_Target

La exclusión reduce el VIF máximo de 102.5843 a 17.4312 (−83.01 %), y el VIF promedio de 3.6683 a 2.2438. Los conteos VIF > 5 pasan de 18 a 4, y VIF > 10 de 11 a 3. El Condition Number baja de 19159.17 a 14533.72; el condicionamiento mejora, pero no desaparecen todos los indicios de multicolinealidad.

El R² ajustado log cambia de 0.955486757 a 0.955180600, Δ=-0.000306157. AIC empeora 259.36 y BIC 250.81 unidades: ambos favorecen al modelo completo, pese al parámetro adicional. El aporte incremental de to_center_km al R² log es 0.000305910 (0.03059 puntos porcentuales); su R² parcial es 0.006857.

En precios originales, manteniendo exp sin corrección, MAE aumenta 3840.31 RUB (+0.351 %) y RMSE 5091.14 RUB (+0.323 %). En ln(RUB), MAE aumenta 0.00033265 y RMSE 0.00040462. Son cambios pequeños en Train, no una demostración de equivalencia ni de rendimiento fuera de muestra.

El coeficiente recuperado de to_center_km es -0.007749871, referido a la unidad de la predictora ya transformada; t nominal=-16.1565 y p nominal=1.606e-58. El F parcial algebraico es 261.0319, solo descriptivo: no se informa un p-value clásico válido porque hay heterocedasticidad. Como sensibilidad, Wald HC3=260.3070, p=1.471e-58. Bajo independencia de observaciones, HC3 respalda una contribución distinta de cero; no resuelve posible dependencia temporal o por grupos. Un VIF alto no convierte a esta variable en estadísticamente irrelevante.

Entre los coeficientes restantes cambian de significancia nominal 6: 5 pasan a p ≤ 0,05 y 1 deja de cumplir ese umbral. El total no significativo pasa de 26 a 22; incluye el intercepto. Estos cambios nonrobust se documentan en el CSV, no son evidencia suficiente para nuevas exclusiones.

Los QQ-Plot, Scale-Location y Residuos vs ajustados son visualmente muy parecidos entre ambos modelos, con desviaciones en las colas y estructura en los mayores ajustados. Breusch-Pagan sigue rechazando homocedasticidad (p=3.541e-148) y Jarque-Bera normalidad (p=1.235e-39). Quitar la variable no resuelve estos supuestos.

La reducción de multicolinealidad es grande y el deterioro empírico del error de Train es menor al 0,4 %. Eso presenta un intercambio atendible si se prioriza menor colinealidad, pero no una mejora en todos los criterios: la variable tiene aporte estadístico y AIC/BIC prefieren conservarla. Sin un umbral de equivalencia predefinido ni evaluación fuera de muestra, no se afirma que la reducción compense definitivamente la pérdida ni se determina el modelo final. No se utilizó Test ni se eliminó ninguna otra variable u observación.

# Plan 04 OLS + MLflow — Alquiler

> **Estado: Final** — contrato Q9 aceptado explícitamente por el usuario
> (2026-10-18). Terminar la entrevista no autoriza a implementar: la
> implementación requiere un pedido explícito separado.
> Documento anterior que rige 01–03: `plan_reescritura.md` (Final, 2026-10-17).
> Este plan lo supersede solo en lo que toca al notebook 04, si así se acepta.

## Objetivo (del pedido del usuario)

Rehacer el notebook 04 de alquiler basándose principalmente en
`Ventas/Propiedades_Usadas/Modelo_OLS.ipynb` de `origin/devLonghino`:
baseline OLS con `statsmodels`, Forward Stepwise por grupos con AIC, BIC
(el usuario escribió «BAIC») y R² ajustado, registro en MLflow en un
experimento nuevo, batería completa de análisis del baseline, y evaluar
Best Subset Selection posterior al stepwise y el agrupamiento de dummies
como una sola característica. Meta: el mejor modelo posible para predecir
el precio de alquiler, lo más cerca posible del punto medio del
*bias–variance tradeoff*.

## Hechos del repo (verificados, no se preguntan)

- Las 38 del baseline **no incluyen `id`** (assert en 03:
  `"id" not in FEATURES_BASE`). Composición: 3 continuas z
  (`log_area_z`, `log_center_z`, `log_metro_z`) + 3 numéricas/binarias
  (`deposit_months`, `furnished`, `pets_allowed`) + 32 dummies
  (`rooms` 3, `building_type` 5, `renovation` 3, `okrug` 10,
  `metro_line` 11). Target `log_rent`; train 16.000 / test 4.000
  (`data/processed/rentals_*_fe.csv`).
- Referencia `origin/devLonghino`: baseline con 184 predictores
  (40.000/10.000, target `price_rub`), Forward Stepwise **por grupos**
  (cada categórica original, p. ej. `district` con 100+ dummies, entra o
  sale entera), tres criterios independientes (AIC/BIC mejora ≥ 2,0;
  Adj R² ≥ 0,0001; tope 15 características), MLflow remoto en DagsHub
  (repo `matiasgastonlonghino/MOSCOW_IDA`, experimento
  `Ventas_Propiedades_Usadas`) con verificación de cada run.
- En alquiler, el agrupamiento análogo son **11 grupos** que cubren las
  38 columnas: 6 características individuales + `rooms`,
  `building_type`, `renovation`, `okrug`, `metro_line`.
- Best Subset exhaustivo: sobre 38 columnas sueltas es inviable
  (2³⁸ ≈ 2,7×10¹¹); sobre 11 grupos es trivial (2¹¹ = 2.048); sobre el
  pool de 29 columnas / 9 grupos de `04_Reduccion_Variables.ipynb`,
  512 por grupos. La unidad (grupo vs. columna) define la viabilidad.
- `04_Modelo_OLS_deprecated.ipynb` y `04_Reduccion_Variables.ipynb`
  existen solo como archivos sin versionar (`git status`: untracked).
  El segundo reduce 38 → pool de 29 y declara que la selección
  (stepwise / Best Subset) va en otro notebook posterior.
- `plan_reescritura.md` (Final) excluyó MLflow y fijó Best Subset por
  BIC como selección final con backward como benchmark. Este pedido
  revierte ambas decisiones para el 04.
- Entorno: `mlflow` 3.16.1 y `python-dotenv` están en `.venv`;
  `dagshub` **no** está instalado ni figura en `requirements.txt`.
  Existe `.env` con `DAGSHUB_USER_TOKEN` (ignorado por Git).

## Principio transversal (establecido por el usuario)

Toda acción del notebook se toma **con comparación, tradeoff y criterio
explícito declarado antes de ver el resultado**: ninguna transformación,
agrupamiento ni selección se adopta por convención, por copiar a la
referencia o por autoridad del `summary()`. Cada decisión muestra las
alternativas comparadas, qué se gana y qué se pierde con cada una, y la
regla con la que se eligió.

## Decisiones cerradas

- **Q1 — Documento (aceptada):** este archivo, `alquiler/plan_04_ols_mlflow.md`,
  nace en Draft. `plan_reescritura.md` queda intacto como historia de
  01–03.
- **Q2 — Alcance y artefactos (aceptada):** un solo notebook nuevo,
  `alquiler/04_Modelo_OLS.ipynb`, construido desde las instrucciones
  iniciales del usuario y tomando como base el `Modelo_OLS.ipynb` de
  `origin/devLonghino`. **No** se basa ni toma como referencia a
  `04_Modelo_OLS_deprecated.ipynb` ni a `04_Reduccion_Variables.ipynb`:
  ni su código, ni su estructura, ni sus resultados (incluido el pool
  de 29) se reutilizan. La disposición física final de esos dos
  borradores sin versionar se fija en el contrato de alcance (Q9).
- **Q3 — MLflow (aceptada):** remoto en DagsHub, mismo repositorio que la
  referencia (`matiasgastonlonghino/MOSCOW_IDA`), experimento nuevo
  **`Alquiler`**. Un run por modelo entrenado (el detalle de runs se
  completa con Q5–Q7), con parámetros, métricas, `summary()` y
  coeficientes, y verificación del registro remoto como en la
  referencia. Implica agregar `dagshub` (y declarar `mlflow` /
  `python-dotenv`) a `requirements.txt` e instalar en `.venv`.
- **Q4 — Batería de análisis (aceptada):** orden coherente en 7 pasos:
  (1) ajuste global del `summary()` (R², Adj R², F y Prob(F),
  Log-Likelihood, AIC/BIC, n, Df); (2) colinealidad (VIF y
  Tol = 1/VIF) antes de interpretar coeficientes; (3) coeficientes
  (coef, std err, t, P>|t|, IC 95%; corte p ≤ 0,05); (4) residuos vs.
  ajustados (sesgo/homocedasticidad) + Scale-Location; (5) normalidad:
  histograma + QQ-plot (manda lo visual) con Omnibus, Skew, Kurtosis y
  Jarque-Bera; (6) Durbin-Watson (reportado con la cautela de que en
  corte transversal depende del orden de filas) + Residuals vs.
  Leverage con distancia de Cook; (7) costo (RSS, MSE, MAE; escala en
  Q8). **Batería completa solo al baseline y al modelo final elegido**;
  a los modelos intermedios, tabla comparativa de métricas (R²,
  Adj R², AIC, BIC, RSS, MSE, MAE, VIF máximo).
- **Q5a — ¿Agrupar las dummies? (aceptada):** no se asume. Competencia
  directa en train entre Forward Stepwise **agrupado** (11
  características: 6 individuales + `rooms`, `building_type`,
  `renovation`, `okrug`, `metro_line`) y **libre** (38 columnas),
  mismos criterios y umbrales. Comparación por CV 5-fold en train
  (mismos folds, RMSE en log), BIC y n.º de columnas; evidencia de
  apoyo: cuántas dummies de cada bloque elige la variante libre.
  **Regla declarada antes de correr:** se adopta la agrupada salvo
  que la libre la supere en CV por más de 1 SE **y** en BIC con
  ΔBIC > 10; ante empate práctico gana la agrupada (menos decisiones
  de selección, categorías interpretables). Test no interviene.
- **Q5b — Forward Stepwise (aceptada):** forward desde el modelo
  solo-intercepto, tres corridas independientes (AIC, BIC, R²
  ajustado) por variante. Parada: AIC/BIC cuando ningún candidato
  mejora ≥ 2,0; Adj R² cuando ninguno mejora ≥ 0,0001 (umbrales de la
  referencia). **Sin tope artificial**: el límite es lo disponible
  (11 grupos / 38 columnas), para que la única diferencia entre
  variantes sea la unidad de selección y la comparación de Q5a no
  quede confundida por un tope asimétrico.
- **Q6a — Best Subset, universo (aceptada):** corre después del
  stepwise, en la unidad ganadora de Q5a, sobre el universo más
  amplio factible. Si gana la agrupada: exhaustivo sobre los 11
  grupos (2¹¹ = 2.048; el stepwise queda como benchmark y Best
  Subset puede recuperar grupos descartados por el greedy). Si gana
  la libre: exhaustivo sobre la unión de las columnas elegidas por
  cualquiera de las tres corridas stepwise; si la unión supera 20
  columnas, se declara inviable y el universo se reduce al conjunto
  de la corrida BIC, dejando escrito el recorte. Tradeoff asentado:
  dentro de su universo Best Subset corrige combinaciones del
  greedy, pero en variante libre no resucita columnas que las tres
  corridas descartaron. Su salida se compara contra el stepwise
  tamaño por tamaño (RSS/BIC/CV).
- **Q6b — Best Subset, campeón (aceptada):** para cada tamaño k se
  conserva el subconjunto de menor RSS, con tabla/gráficos por k
  (variables, RSS, R², Adj R², AIC, BIC, CV-RMSE en train). El
  campeón de la etapa es el de **mínimo BIC**. Si el k de mínimo
  BIC difiere del de mínimo CV, la discrepancia se documenta sin
  mezclar criterios: el arbitraje predictivo entre candidatos es
  una sola vez, en Q7.
- **Q7 — Modelo final y bias–variance (aceptada):** el «punto medio»
  se define como el punto donde agregar complejidad ya no baja el
  error de predicción de forma apreciable (no el promedio entre
  extremos). Candidatos: baseline (38), los tres stepwise de la
  variante ganadora de Q5a (AIC, BIC, Adj R²) y el campeón Best
  Subset (Q6b); los stepwise de la variante perdedora no compiten.
  **Regla:** con CV 5-fold en train (mismos folds, RMSE en log),
  gana el candidato con **menos columnas** entre los que estén
  dentro de 1 SE del mejor CV-RMSE; desempate por menor BIC. El
  baseline puede resultar final si es el único dentro del margen.
  Al elegido se le corre la batería completa (Q4) y se lo evalúa
  **una sola vez en test**, sin volver atrás a elegir otro.
- **Q8 — Métricas y escala (aceptada):** doble escala. **Log**
  (ajuste/selección): RSS, MSE = RSS/n y MAE en train para comparar
  candidatos; de acá salen el CV de Q5a/Q7 y el paso 7 de la batería
  de Q4. **RUB** (comunicación): predicciones `expm1(...)` contra
  `monthly_rent_rub`; métrica principal del modelo final, MAE en RUB
  en test (RMSE en RUB secundario), evaluado una sola vez (Q7). En
  MLflow, nombres separados por escala (`mae_log_...` / `mae_rub_...`).

## Preguntas abiertas (en orden de entrevista)

- [x] Q2. Alcance y artefactos: qué notebook(s) se crean/reemplazan y qué
  pasa con `04_Modelo_OLS_deprecated.ipynb` y `04_Reduccion_Variables.ipynb`.
  → Notebook único nuevo desde `devLonghino` + instrucciones iniciales;
  los dos borradores no son fuente (ver Decisiones cerradas).
- [x] Q3. MLflow: destino (DagsHub remoto vs. local), nombre del experimento
  nuevo, runs a registrar y dependencias a agregar. → DagsHub remoto,
  experimento `Alquiler` (ver Decisiones cerradas).
- [x] Q4. Baseline y batería de análisis: orden coherente de la batería
  (summary en 3 partes, p/t, R²/Adj R²/F, Durbin-Watson, residuos vs.
  ajustados, histograma, QQ-plot, Scale-Location, Residuals vs. Leverage,
  VIF/Tolerancia, Omnibus, Skew, Kurtosis, Jarque-Bera, MSE/RSS/MAE) y a
  qué modelos se aplica (¿solo baseline o también finales?). → Batería
  en 7 pasos; completa solo a baseline y final (ver Decisiones cerradas).
- [x] Q5. Stepwise: definición de los 11 grupos, criterios y umbrales
  de parada (AIC, BIC, Adj R²), tope de características. **Reformulada
  por el usuario:** agrupar las dummies no se asume; primero se decide
  con qué comparación, tradeoff y criterio se prueba si agrupar tiene
  sentido (Q5a), y recién después umbrales y topes (Q5b).
  - [x] Q5a. ¿Agrupar o no? → Competencia directa agrupado (11) vs.
    libre (38) en train, regla de 1 SE a favor de agrupar (ver
    Decisiones cerradas).
  - [x] Q5b. Umbrales de parada y tope del Forward Stepwise. → Umbrales
    de la referencia, sin tope artificial (ver Decisiones cerradas).
- [x] Q6. Best Subset posterior: unidad (grupos vs. columnas), conjunto
  sobre el que corre y criterio de elección dentro de Best Subset.
  - [x] Q6a. Universo → el más amplio factible en la unidad ganadora
    de Q5a (ver Decisiones cerradas).
  - [x] Q6b. Criterio para elegir el campeón dentro de Best Subset. →
    Mínimo BIC (ver Decisiones cerradas).
- [x] Q7. Elección del modelo final y «punto medio» del bias–variance:
  regla operativa para elegir entre baseline, los 3 stepwise y el
  Best Subset; rol de train/test/CV (sin seleccionar con test). →
  Regla de 1 SE sobre CV en train; test una sola vez (ver Decisiones
  cerradas).
- [x] Q8. Métricas y escala de reporte: log vs. RUB, train/test. →
  Doble escala (ver Decisiones cerradas).
- [x] Q9. Contrato de alcance y definición de terminado (scope contract).
  → Aceptado explícitamente por el usuario (2026-10-18); este plan pasa
  a Final.

## Contrato Q9 (aceptado explícitamente, 2026-10-18)

- **En alcance:** un solo notebook nuevo `alquiler/04_Modelo_OLS.ipynb`
  desde el `Modelo_OLS.ipynb` de `origin/devLonghino` y las
  instrucciones iniciales, sin usar como fuente los dos borradores
  sin versionar; baseline OLS `statsmodels` con las 38 (sin `id`,
  target `log_rent`, train 16.000); batería Q4 completa a baseline y
  modelo final y tabla comparativa a intermedios; competencia de
  agrupamiento Q5a y Forward Stepwise Q5b (2 variantes × AIC/BIC/
  Adj R²); Best Subset Q6 con campeón por BIC; elección final Q7
  por regla de 1 SE en CV y test evaluado una sola vez; métricas en
  doble escala Q8; experimento MLflow nuevo `Alquiler` en DagsHub
  remoto con un run por modelo candidato (baseline, 6 stepwise y
  campeón Best Subset; el final se marca sobre su run ya existente)
  y verificación remota de cada run; dependencias de tracking
  declaradas en `requirements.txt` e instaladas en `.venv`; este
  plan como registro. La disposición física de los dos borradores
  sin versionar forma parte de este contrato: se eliminan al
  implementar, por ser material sin versionar ya reemplazado.
- **Fuera de alcance:** cambios en 01–03 y en venta; uso de
  `origin/devLonghino_pipeline`; reincorporar dummies de `district` /
  `metro_station` ni ninguna variable fuera de las 38; re-dividir el
  split; modelos no-OLS; seleccionar o re-elegir usando test;
  corrección automática a partir del diagnóstico.
- **Terminado significa:** el notebook corre de arriba abajo con el
  kernel `.venv` (semilla 42, mismos folds CV en todas las
  competencias); están la batería del baseline en el orden Q4, la
  competencia de agrupamiento con su regla aplicada y documentada,
  los historiales stepwise, la tabla Best Subset por k con su
  campeón, la comparación de los 5 candidatos con la regla de 1 SE
  aplicada, la batería completa del final y su única evaluación en
  test en doble escala; los 8 runs figuran `FINISHED` en el
  experimento `Alquiler` y verificados contra el servidor.
- **Meta:** modelo OLS final de alquiler en log, defendible
  coeficiente por coeficiente, en el punto donde más complejidad
  ya no baja el error de predicción de forma apreciable, con error
  principal comunicable en RUB/mes.
- **Restricciones:** selección solo con train; agrupar solo si la
  competencia Q5a lo sostiene; cada acción con comparación,
  tradeoff y criterio previo (principio transversal).
- **Riesgos:** con n = 16.000, Omnibus/Jarque-Bera rechazarán por
  desvíos mínimos (manda lo visual); Durbin-Watson en corte
  transversal depende del orden de filas (se reporta con cautela);
  si gana la variante libre y la unión stepwise supera 20 columnas,
  Best Subset se acota al conjunto BIC (Q6a); DagsHub exige red y
  token válidos al ejecutar.
- **Validación:** la lista de «Terminado significa» de este contrato.

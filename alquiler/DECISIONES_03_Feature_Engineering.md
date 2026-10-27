# Decisiones para rehacer `03_Feature_Engineering.ipynb` — Draft

Estado: **Final** (contrato aceptado; implementación solicitada explícitamente el 2026-10-27, con evaluación posterior por subagente).

## Contexto: qué recibe `03` desde el `02` actual

- Artefactos: `rentals_train.csv` 14.820 × 35 (2021-2024, target winsorizada en fences de train, tope 108.550 RUB, skew 0,80) y `rentals_test.csv` 4.841 × 35 (2025-2026, target cruda, skew 1,76). Split temporal, sin solape.
- Target **cruda** `monthly_rent_rub`; ya no existe `log_rent` (el `03` actual la espera como `Y` y falla al cargar).
- Ya no existen: `okrug` (el `03` actual la usa en el one-hot), `secondary_price_per_sqm`, `newbuild_price_per_sqm`, `avg_mortgage_rate_pct`, `station_lat`, `station_lon`, `year_opened`.
- Nuevas: bloque mensual rezagado 1 mes (`rental_price_per_sqm_monthly_lag1` r = 0,773 con el target en train, variaciones, conteos y tasa clave, todo `_lag1`), `estacion_antiguedad`, `station_to_center_km`, `target_winsorizado` y 3 `flag_outlier_*` (estas últimas nunca predictoras: derivan del target o solo marcan).
- Geografía fina disponible sin codificar: `district` 129 (apoyo mínimo 88 en train), `metro_station` 104 (mínimo 76), `lat`/`lon` del aviso.
- Pares parciales diferidos desde la limpieza: `metro_distance_min`↔`to_center_km` r = 0,90 y `total_area`↔`rooms` r = 0,88.

## Decisiones pendientes (en orden)

1. [x] Target de modelado: cruda winsorizada vs. `log1p` creada en `03` vs. comparar ambas con validación temporal interna — **Decidido: A. Target cruda** `monthly_rent_rub` tal como la entrega `02` (winsorizada en train, cruda en test). Sin `log_rent` en `03`. Nota del usuario: el experimento se está reescribiendo, así que ninguna evidencia del `03` anterior (CV, formas, p = 38) ata las decisiones siguientes; todo se vuelve a medir.
2. [x] Precio distrital rezagado: predictor, referencia o fuera de los artefactos FE — **Decidido: fuera, por principio general.** El objetivo del análisis es explicar y predecir el alquiler desde las características de la vivienda y su ubicación; ninguna variable derivada de la propia target ni agregados de alquileres (del estilo de `rent_per_sqm` o `rental_price_per_sqm_monthly`, incluso rezagados) entra a la matriz ni a los artefactos FE. Explicar el alquiler con el alquiler promedio del distrito sería circular. El rezago resolvía la fuga temporal, pero no esta circularidad.
3. [x] Geografía fina: cómo entran (o no) `district` y `metro_station` a la matriz — **Decidido: B. One-hot k−1 de `district` (128 columnas); `metro_station` fuera como identidad.** Evidencia: bloque geográfico grueso R² = 0,350; `district` solo R² = 0,620; agregar `metro_station` (103 columnas) no suma (0,626 sin cambios). Target encoding descartado por el principio de la decisión 2 (es un agregado de la target).
4. [x] Esquema de validación dentro de train para comparar transformaciones (temporal vs. KFold mezclado) — **Decidido: C. Walk-forward validation con ventana expandible** (validaciones 2022, 2023 y 2024; cada forma se ajusta solo con el pasado y se evalúa en el año siguiente). Queda descartado el KFold mezclado del `03` anterior por violar el orden temporal.
5. [x] Pares parciales: ambas distancias y superficie/ambientes en la matriz base, o selección en `03` — **Decidido: B. `03` resuelve los pares por aporte incremental** (a confirmar con walk-forward al implementar). Con `district` en la base: `to_center_km` suma 0,000 y `metro_distance_min` +0,006; `total_area` lleva R² de 0,703 a 0,874 y `rooms` sobre superficie suma +0,0004. Matriz base con `metro_distance_min` y `total_area`; fuera `to_center_km` y `rooms` como columnas separadas.
6. [x] Alcance y artefactos de `03`: tamaño de la matriz base, columnas que salen y contrato para `04` — **Decidido: C. Matriz base cerrada de p ≈ 152** (128 district + 19 building_type/renovation/metro_line + total_area, metro_distance_min, deposit_months, furnished, pets_allowed), con las candidatas débiles viajando en los artefactos FE fuera de la lista de features. Formas funcionales por walk-forward y continuas estandarizadas con parámetros de train. `rental_price_per_sqm_monthly_lag1` sale también de los artefactos.

## Alcance (borrador)

- En alcance: rehacer solo `alquiler/03_Feature_Engineering.ipynb` sobre los artefactos vigentes de `02`.
- Fuera de alcance hasta nueva instrucción: modificar `01`, `02`, `04` y los CSV.
- La implementación empieza recién con pedido explícito posterior a esta entrevista.

## Contrato asentado (aceptado)

**Objetivo.** Rehacer `03_Feature_Engineering.ipynb` como experimento nuevo sobre los artefactos vigentes de `02` (train temporal 14.820, test temporal 4.841, target cruda): extraer candidatas, elegir formas y codificaciones con validación temporal, y entregar la matriz base para el OLS de `04`.

**Principio rector.** Ninguna variable derivada de la target ni agregados de alquileres —del estilo de `rent_per_sqm` o `rental_price_per_sqm_monthly`, incluso rezagados— entra a la matriz ni viaja en los artefactos FE. El alquiler se explica desde características de la vivienda, su ubicación y contexto exógeno (venta, macro), no desde alquileres promediados.

**Decisiones (en orden de aplicación).**

1. Target de modelado **cruda** `monthly_rent_rub` (winsorizada en train por `02`, cruda en test). Sin `log_rent`. La evidencia del `03` anterior no ata nada: el experimento se reescribe y todo se vuelve a medir.
2. `rental_price_per_sqm_monthly_lag1` fuera de la matriz **y de los artefactos FE** por el principio rector.
3. Geografía fina: **one-hot k−1 de `district`** (128 columnas, R² geográfico 0,620 contra 0,350 del bloque grueso); `metro_station` fuera como identidad (+0,000 sobre district); target encoding descartado por el principio rector.
4. Comparación de formas y codificaciones con **walk-forward validation**, ventana expandible, validaciones 2022, 2023 y 2024. Nada de KFold mezclado.
5. Pares parciales resueltos en `03` por aporte incremental a confirmar con walk-forward: entran `metro_distance_min` y `total_area`; fuera `to_center_km` (suma 0,000 con district) y `rooms` como columnas separadas (+0,0004 sobre superficie).
6. **Matriz base cerrada p ≈ 152**: 128 district + 5 building_type + 3 renovation + 11 metro_line + total_area, metro_distance_min, deposit_months, furnished, pets_allowed. Candidatas débiles (pisos, años, estación, contexto de venta/macro rezagado, lat/lon) viajan en los artefactos fuera de la lista de features. Continuas estandarizadas con parámetros de train. Nunca predictoras: `id`, fechas, target, `target_winsorizado`, `flag_outlier_*`.

**Fuera de alcance.** Modificar `01`, `02`, `04`, los CSV crudos y cualquier decisión de selección/modelo final.

**Riesgos aceptados.** Matriz ancha (p ≈ 152) sostenida por 14.820 filas; distritos como efectos fijos sin suavizar; candidatas débiles disponibles pero no evaluadas en la base.

**Validación esperada al implementar.** Walk-forward impreso por forma/codificación, VIF de la matriz final, asserts de forma y de ausencia de columnas derivadas de la target en los artefactos FE, y re-chequeo de que ningún parámetro (escalado, bases del one-hot) se ajustó fuera de train.

**Sin resolver.** Nada dentro de `03`. Quedan para `04`: selección final entre las 152, posible apertura del pool de candidatas y diagnóstico de residuos del OLS.

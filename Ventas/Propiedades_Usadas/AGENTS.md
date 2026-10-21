# Experimentos de propiedades usadas

- `PropUsadas_OLS.csv` es la única fuente principal del notebook. Cargarla una vez y conservar el DataFrame base; generar variantes en memoria sin sobrescribir columnas originales.
- No exportar nuevas versiones completas del dataset salvo necesidad técnica explícita. Guardar configuraciones, mapas, listas de características, transformadores ajustados y modelos como artefactos en MLflow.
- Usar exclusivamente el experimento existente `Ventas_Propiedades_Usadas`; no crear experimentos independientes ni sobrescribir notas generales de otros algoritmos.
- Ajustar toda transformación aprendida únicamente con el entrenamiento correspondiente. Los mapas supervisados de Train completo no sirven dentro de folds. Verificar huellas, fechas, características del OLS preliminar y preprocesamiento antes de reutilizar.
- Mantener procedimientos independientes salvo combinación explícita. Registrar Runs de origen, pasos, características originales/transformadas, columnas efectivas y notas interpretativas automáticas.
- Conservar las ocho métricas acordadas según aplicabilidad y calcular métricas predictivas en RUB. Test permanece reservado hasta cerrar la elección.
- Para agrupamientos, usar `transformaciones_experimentales.py` y JSON con todos los distritos observados, contexto de Train/fold y fallback aprendido solo de entrenamiento. Ante incompatibilidad, recalcular expresamente; nunca adaptar mapas con Validation o Test.
- Feature Selection conserva listas, criterios y contexto; Feature Extraction conserva el transformador ajustado y sus salidas; regularización conserva hiperparámetros, modelo y procedencia. Cambiar las variables del OLS preliminar invalida el mapa anterior.
- Configuración provisional vigente: OLS log con Duan, K-Means K=8; excluir `mortgage_rate_at_listing`, `cbr_key_rate_pct_lag1` y `n_listings_secondary_lag1`. La combinación todavía no fue evaluada; no atribuirle métricas de las ablaciones independientes ni del K-Means original.
- El notebook consolidado prepara `df_trabajo` sin nuevos ajustes ni Runs. Los procedimientos reproducibles se centralizan en `experimentos_ols.py`; invocarlos solo cuando corresponda al experimento solicitado.
- Para CV usar `datos_base` y `preparar_fold_trabajo(fold)`, nunca el DataFrame agrupado con Train completo. Mantener explícita la definición original de los efectos que generaron el mapa K=8; las tres exclusiones se aplican después. La preparación estructural de filas reservadas no autoriza entrenar ni evaluar con Test.

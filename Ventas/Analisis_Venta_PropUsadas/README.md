# Pipeline OLS de propiedades usadas

Todo el desarrollo está contenido en esta carpeta. Los notebooks existentes
continúan disponibles para EDA. No se generan CSV integrados ni DataFrames en
disco. `requirements.txt` es independiente del archivo de dependencias del repositorio.

Desde la raíz del repositorio, en PowerShell:

```powershell
python -m pip install -r Ventas/Analisis_Venta_PropUsadas/requirements.txt
python Ventas/Analisis_Venta_PropUsadas/scripts/run_experiment.py
python Ventas/Analisis_Venta_PropUsadas/scripts/run_experiment.py --include total_area rooms renovation building_type --selection-method forward --criterion bic
python Ventas/Analisis_Venta_PropUsadas/scripts/run_experiment.py --include total_area rooms renovation --selection-method best_subset --criterion bic --selection-params '{"max_features":3,"max_models":32}'
python -m pytest Ventas/Analisis_Venta_PropUsadas/tests -q
```

Se verificó el desarrollo por etapas: carga/preprocesamiento; OLS de referencia;
pipeline/CLI/selectores; tracking externo. Las pruebas usan datos sintéticos.

## API y uso en notebooks

```python
from Ventas.Analisis_Venta_PropUsadas.pipeline import run_pipeline, predict_pipeline

result = run_pipeline(
    # secondary_path="...", monthly_path="...", metro_path="...",
    train_end="2024-01-01",
    validation_end="2025-01-01",
    # test_end="2026-05-01",  # opcional, límite exclusivo
    include=["total_area", "rooms", "renovation", "building_type"],
    exclude=[],
    selection_method="stepwise",
    criterion="bic",
    selection_params={"max_steps": 30},
    preprocessing={"monthly_lag_months": 1, "numeric_strategy": "median"},
    tracking=False,
    return_frames=True,
)
result["model"].summary()
result["selected_groups"]
result["selected_features"]
result["metrics"]
result["diagnostics"]
result["selection"]  # historial, cantidad de modelos y convergencia
result["frames"]["original"]  # integrado, antes de preparación
result["frames"]["features"]  # derivadas/exclusiones, antes de imputar
result["frames"]["prepared"]["train"]  # imputado, sin encoding
result["frames"]["encoded"]["train"]   # numérico, conserva nombres
result["frames"]["targets"]["validation"]
```

Los imports del ejemplo requieren la raíz del repositorio en `sys.path` (o como
directorio de trabajo). El CLI resuelve sus imports y rutas predeterminadas desde
la ubicación del script y funciona desde cualquier directorio. Para notebooks
ubicados en otra carpeta se puede añadir la raíz a `sys.path` explícitamente.

El resultado también contiene `preprocessor`, `predictions` por partición,
`candidate_features`, `candidate_groups`, `join_audit`, `config` y `run_id`.
`predict_pipeline(result, integrated_frame)` reutiliza el estado aprendido; el
frame debe integrarse con `data.load_data` usando el mismo rezago. Una propiedad
sin precio puede predecirse si su columna `price_rub` está presente como nula;
la validación del objetivo ocurre únicamente en `run_pipeline`.

## Integridad y disponibilidad temporal

`data.load_data` recibe las tres rutas de CSV. Los defaults apuntan a `Data/` del
repositorio. Rechaza claves nulas y duplicadas en tablas auxiliares e IDs de
propiedades repetidos. Los joins son izquierdos, muchos-a-uno, con comprobación
de conservación de filas. Los atributos añadidos llevan prefijos `monthly_` y
`metro_`; las coordenadas originales de la propiedad se conservan.

Cada propiedad consulta **el mes anterior**, por distrito y okrug; el rezago debe
ser un entero >= 1. No hay relleno desde meses posteriores. Por defecto se supone
que el agregado del mes anterior está publicado al comenzar el siguiente mes.
El CSV no incluye fechas efectivas de publicación ni revisiones históricas:
hay que validar esa hipótesis con el proveedor y aumentar `monthly_lag_months`
si corresponde. El rezago no garantiza disponibilidad de versiones históricas
si el proveedor revisa los valores. Para un experimento sin esa hipótesis,
excluir las columnas `monthly_*` o usar `include` con variables de la propiedad.

Las estaciones aún no inauguradas en el año del anuncio no aportan atributos
del catálogo. El catálogo no contiene fechas históricas de cambios de línea ni
mes de inauguración; su uso supone estabilidad histórica de la identificación
y de los atributos geográficos. Puede omitirse mediante `include`.

`join_audit` informa faltantes y coincidencias fallidas. En los CSV actuales,
el rezago de un mes deja 678 propiedades sin agregado mensual anterior; se
conservan y se imputan. `require_matches=True` convierte faltas de coincidencia
en error. `price_per_sqm`, `price_rub`, identificador y fechas crudas nunca
pueden incorporarse como features, incluso con inclusión explícita.

## Particiones y preprocesamiento

Las fechas son límites **exclusivos**: train antes de `train_end`; validación
desde ese límite hasta `validation_end`; test desde `validation_end` hasta
`test_end` (o el final del dataset). Las tres particiones deben tener filas.
Las filas posteriores a `test_end` no participan del modelado.

Sólo train determina imputaciones, niveles categóricos, constantes descartadas y
selección. Las derivaciones son deterministas: año, seno/coseno del mes, edad
del edificio y piso relativo. La imputación numérica admite `median` o `mean`;
columnas completamente vacías en train usan cero. Las categóricas nulas usan
`__MISSING__`. Infinitos numéricos se tratan como faltantes; objetivos inválidos
generan error. No se eliminan outliers automáticamente.

One-hot omite la primera categoría ordenada de cada feature (referencia).
Categorías nuevas en validación/test se codifican como la referencia por defecto;
`handle_unknown="error"` permite rechazarlas. No se aprenden categorías futuras.
Los grupos por variable original mantienen juntas todas sus dummies.

Opciones de `preprocessing` (argumento dict, o JSON en `--preprocessing`):

| Opción | Default |
| --- | --- |
| `monthly_lag_months` | `1` |
| `require_matches` | `False` |
| `numeric_strategy` | `"median"` |
| `categorical_fill` | `"__MISSING__"` |
| `handle_unknown` | `"ignore"` |
| `derive` | `True` |

## Selección, métricas y diagnósticos

`none` es la referencia con todas las candidatas no constantes. `forward`,
`backward` y `stepwise` admiten `aic`, `bic` o `pvalue`; este último usa una
prueba F parcial del bloque completo, con `entry_threshold=0.05` y
`exit_threshold=0.10`. Otros parámetros son `max_steps=100` y
`min_improvement=1e-6` para AIC/BIC. El historial contiene cambios aceptados;
`converged=False` señala un límite de pasos o ciclo.

`best_subset` admite AIC/BIC. `max_features=5` limita **grupos**, no dummies;
`max_groups=15` y `max_models=4096` restringen la búsqueda. Siempre incluye el
modelo de intercepto. Antes de ajustar comprueba el presupuesto completo y
rechaza búsquedas que lo exceden. Conviene limitar candidatas con `include`:
buscar sobre todas las variables y dummies del dataset puede ser costoso.
Los modelos intermedios no se retienen en memoria; se cachean sus estadísticas.
Para añadir selectores basta una función con la misma firma y registrarla en
`selection.SELECTORS`, sin cambiar la coordinación principal.

El modelo permanece ajustado sólo en train, también después de evaluarlo.
Las métricas llevan prefijos `train_`, `validation_` y `test_`: R², R² ajustado,
MAE y RMSE en RUB. El R² ajustado fuera de train usa el rango efectivo del modelo
de train y se informa como NaN cuando no hay grados de libertad suficientes.
AIC/BIC se registran **sólo en train**, pues corresponden al ajuste estadístico.
Validación permite comparar configuraciones; reservar test para la evaluación
final, sin elegir variables o hiperparámetros en función de sus resultados.

Se informan media/desvío de residuos, Jarque–Bera, Durbin–Watson (filas ordenadas
por fecha), Breusch–Pagan, número de condición, rango y VIF. La colinealidad
exacta se marca y los VIF correspondientes se informan como infinito.
Statsmodels usa pseudoinversa en diseños de rango deficiente: las predicciones
siguen disponibles, pero los coeficientes individuales no son identificables.
Los p-valores/pruebas F son clásicos, sujetos a los supuestos de OLS y a sesgo
posterior a selección; los diagnósticos son exploratorios.

## MLflow externo

```powershell
$env:MLFLOW_TRACKING_URI = "https://tu-servidor-mlflow"
# Autenticación según el servidor: variables MLFLOW_TRACKING_TOKEN o
# MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD, fuera del código.
python Ventas/Analisis_Venta_PropUsadas/scripts/run_experiment.py --tracking --experiment-name Moscow_OLS --run-name baseline
python Ventas/Analisis_Venta_PropUsadas/scripts/compare_experiments.py --experiment-name Moscow_OLS
```

Tracking desactivado no importa ni conecta MLflow. Activado exige HTTP(S),
comprueba conexión y nunca recurre al almacén `mlruns` local. El servidor debe
tener un almacenamiento de artefactos accesible y configurado por su administrador.
No incluir credenciales en la URI.

Registra parámetros, candidatas y seleccionadas, decisiones de selección,
métricas de las tres particiones, resumen OLS, coeficientes, diagnósticos y
gráficos, estado del preprocesamiento/modelo (`model.pkl`), ID de ejecución,
hashes SHA-256 de los datasets, versiones del entorno y copia exacta del código.
El modelo puede restaurarse con `pickle` e inspeccionarse o usarse con
`predict_pipeline`. Cargar pickle sólo de ejecuciones propias y confiables.
Los artefactos se preparan en una carpeta temporal que se elimina al terminar;
los DataFrames intermedios no se exportan. Las rutas y versiones registradas
permiten reproducir el experimento con los mismos CSV y entorno.

Un error de conexión, autenticación o subida genera `RuntimeError`; si ya existe
la ejecución, se intenta marcarla `FAILED`. No se silencian errores ni se expone
texto de terceros que pueda contener credenciales. En Python puede recuperarse
el resultado ya entrenado mediante `error.result`.

La comparación retorna un DataFrame y ordena por RMSE de validación. `--output`
permite exportar una comparación explícitamente; por defecto sólo la imprime.
Comparar ejecuciones con los mismos hashes de datasets, fechas, candidatas y
preprocesamiento, variando únicamente la selección. Esos campos están registrados
para comprobarlo; la herramienta no afirma que ejecuciones arbitrarias sean comparables.

No se configuró un servidor en este entorno durante el desarrollo: las pruebas
verifican el cliente, registro, artefactos, fallos y comparación con dobles de
prueba. La conexión real queda pendiente de definir `MLFLOW_TRACKING_URI`.

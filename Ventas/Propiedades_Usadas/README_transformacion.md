# Datos y resultados de propiedades usadas

Los únicos resultados permanentes del procesamiento local son
`data/train_transformado.csv` y `data/test_transformado.csv`. Ambos conservan los
bytes de la entrega anterior. `scripts/contrato_datasets.py` contiene el esquema y las
huellas del dataset congelado para cargar Train sin red ni ejecutar el EDA.

No redefinir el corte cronológico ni recombinar los CSV. Tukey reduce solo Train;
Test conserva todas sus observaciones. `price_rub` es la respuesta; `id`,
`date_posted_dt` y `dataset_split` son auxiliares, ajenos a X. La selección y
comparación de modelos utilizan exclusivamente Train y su validación interna.

Los modelos, métricas, diagnósticos, tablas y configuraciones permanecen en
[DagsHub/MLflow](https://dagshub.com/matiasgastonlonghino/MOSCOW_IDA.mlflow).
Los tres artefactos históricos de transformación están en `transformacion/`
del Run Baseline_OLS_01 (`43e383e9b91d43efbdeda5ec1ec28f71`):
`transformadores_prop_usadas.joblib`, `manifiesto_exportacion.json` y
`resumen_final.json`. La reorganización conserva el contrato original y sus
huellas; que el módulo actual tenga otro hash no invalida los CSV congelados.

```python
from pathlib import Path
import sys
base = Path("Ventas/Propiedades_Usadas").resolve()
sys.path.insert(0, str(base))
from scripts import baseline_ols as b
X_train, y_train, contexto = b.cargar_train(base)
```

Para recuperar transformadores sin conservar copias locales:

```python
import joblib
from scripts.almacenamiento_mlflow import SesionTemporal, RUN_TRANSFORMACION
cliente = b.conectar(base.parents[1])
with SesionTemporal() as sesion:
    ruta = cliente.download_artifacts(
        RUN_TRANSFORMACION, "transformacion/transformadores_prop_usadas.joblib",
        str(sesion.carpeta("lectura")))
    reglas = joblib.load(ruta)
# reglas sigue en memoria; el archivo temporal fue eliminado.
```

`pt.aplicar_flujo(nuevos, reglas)[0]` aplica solamente `transform`, sin Tukey ni
ajustes nuevos. Para observaciones futuras se requieren `id`, `date_posted_dt`
y `dataset_split="test"`; la respuesta puede omitirse. District conserva su
codificador independiente y las predictoras siguen la convención snake_case ASCII.

Los notebooks 2, 3 y 4 usan una `SesionTemporal` fuera del repositorio.
Cada celda se ejecuta dentro de `sesion.paso()`: un error o una interrupción
limpia los temporales y obliga a reiniciar la ejecución desde la primera celda.
La última celda libera la sesión al finalizar correctamente. Los scripts deben
usar `with SesionTemporal() as sesion:` y pasar sus carpetas temporales a las
funciones de exportación, selección y diagnóstico. Estas funciones rechazan
rutas de artefactos dentro del repositorio. No se descargan referencias permanentes.

El notebook 2 conserva transformadores en memoria durante el resumen y publica
su reproducibilidad en el Run existente bajo
`transformacion/<sha256_train>/<sha256_modulo>/`, con verificación de descarga.
Los CSV se escriben directamente en `data`. Si se cambia intencionalmente el
procesamiento, actualizar conjuntamente los dos CSV y su contrato fuente; la
limpieza estructural no modifica los datasets congelados ni sus parámetros.

El resumen de transformación permanece al final del notebook. El script
`scripts/actualizar_indice_transformacion.py --check` verifica su índice y ubicación.
`scripts/verificar_normalizacion_nombres.py` valida las huellas y el esquema contra los
artefactos remotos sin ajustar modelos ni escribir auditorías permanentes.

Los algoritmos, hiperparámetros, criterios de parada y conclusiones no cambian.
Para revisar resultados históricos se descargan los artefactos del Run original;
no se necesita repetir el entrenamiento. No ejecutar los notebooks de ajuste
completos para verificar una reorganización de archivos.

# Entrega de transformación de propiedades usadas

Decisión permanente: Train y Test se almacenan y utilizan por separado, salvo
instrucción expresa de cambio. Ejecutar de principio a fin
`2_Transformacion_PropUsadas.ipynb` regenera en `salidas_transformacion/`:

- `train_transformado.csv`: entrenamiento, comparación y selección de modelos,
  incluidos OLS, Best Subset Selection y Stepwise Selection.
- `test_transformado.csv`: evaluación final de los modelos seleccionados.
- `transformadores_prop_usadas.joblib`: imputación, codificadores, extracción,
  selección final, escalado, orden de predictoras y configuración aprendida con Train.
- `manifiesto_exportacion.json`: esquema, recuentos, fechas y huellas SHA-256 de
  los tres archivos, publicado al final para detectar entregas incompletas o mezcladas.

No redefinir el corte cronológico ni recombinar los CSV. Tukey reduce solo Train;
Test conserva sus observaciones y precios extremos. `price_rub` es la respuesta;
`id`, `date_posted_dt` y `dataset_split` son auxiliares de trazabilidad, ajenos a X.
Los dos archivos tienen exactamente las mismas predictoras en el mismo orden.
Los artefactos de `Data/Propiedades_Usadas/transformacion_80_20/` son de auditoría;
las siguientes etapas deben consumir esta entrega oficial.

Comparar y seleccionar variables/modelos dentro de Train. En validación temporal,
reajustar todo el flujo con el subtrain crudo de cada fold, no con datos de su
validación. Los transformadores finales ajustados con todo Train no sustituyen
ese ajuste interno. Nunca calcular estadísticas ni ajustar decisiones con Test.

Ejemplo desde la raíz del repositorio, en un proceso nuevo con las dependencias
registradas en `configuracion["versiones"]`:

```python
import json
import sys
from pathlib import Path
import joblib
import pandas as pd

carpeta = Path("Ventas/Propiedades_Usadas").resolve()
sys.path.insert(0, str(carpeta))
import preparacion_train_test as pt

salida = carpeta / "salidas_transformacion"
manifiesto = json.loads((salida / "manifiesto_exportacion.json").read_text(encoding="utf-8"))
assert pt.huella_archivo(carpeta / "preparacion_train_test.py") == manifiesto["modulo_sha256"]
for nombre, esperado in manifiesto["archivos_sha256"].items():
    assert pt.huella_archivo(salida / nombre) == esperado
reglas = joblib.load(salida / "transformadores_prop_usadas.joblib")
train = pd.read_csv(salida / "train_transformado.csv", dtype={"id": "string"},
                    parse_dates=["date_posted_dt"], float_precision="round_trip")
test = pd.read_csv(salida / "test_transformado.csv", dtype={"id": "string"},
                   parse_dates=["date_posted_dt"], float_precision="round_trip")
columnas = reglas["columnas_X"]
assert columnas == manifiesto["columnas_X"]
X_train, y_train = train[columnas], train["price_rub"]
# Cargar Test no autoriza su uso para seleccionar modelos.
```

Para nuevas observaciones con el esquema crudo, incluir `id`, `date_posted_dt`
y `dataset_split="test"`; `price_rub` puede omitirse si aún no se conoce.
`pt.aplicar_flujo(nuevos, reglas)[0]` aplica únicamente `transform`, sin filtrar
la respuesta ni ajustar estimadores. El objeto `reglas["district"]` es independiente
del codificador base; actualmente aplica One-Hot Encoding, sin Target Encoding.

Si cambia cualquier transformación, regenerar conjuntamente ambos CSV, el paquete
y el manifiesto. No mezclar versiones. Conservar el módulo y las dependencias
registradas con los objetos; no se requiere el estado del kernel del notebook.

El notebook cierra obligatoriamente con **Resumen final de transformaciones y
características del dataset**. Se genera leyendo la entrega oficial y los objetos
ajustados; verifica huellas, columnas, procedencia, categorías, parámetros y totales.
No modifica datos ni ajusta estimadores. `resumen_final.json` conserva el registro
completo; las tablas del notebook agrupan visualmente cada One-Hot sin alterar X.

Tras incorporar nuevas operaciones, ejecutar desde la raíz:
`python Ventas/Propiedades_Usadas/actualizar_indice_transformacion.py`.
El script desplaza el bloque marcado de resumen al final, actualiza su numeración
y sincroniza el índice. Su opción `--check` detecta cambios pendientes.
Las nuevas salidas derivadas deben declarar su procedencia junto al transformador;
la auditoría detiene la ejecución si aparece alguna columna sin documentación.

Convención permanente: todas las predictoras finales usan `snake_case` ASCII en
minúsculas, sin prefijos técnicos ni separadores dobles. La función central
`nombre_predictor` define la regla; `NormalizadorNombres`, primer paso del filtro
final, la aplica tanto en Train como en Test/futuras observaciones. Solo renombra,
sin modificar valores. Conserva los nombres internos de los codificadores como
origen y rechaza colisiones o nombres reservados. Las categorías aprendidas y
referencias One-Hot mantienen su texto original en los metadatos.

El contrato v3 registra `convencion_nombres`, `correspondencia_nombres` y los roles
de las columnas. El paquete conserva ese normalizador; `resumen_final.json` guarda
la correspondencia de las predictoras finales y su linaje actualizado. Las claves
que documentan nombres anteriores pueden contener los prefijos técnicos originales.
No mezclar objetos anteriores a esta convención con los nuevos CSV.

La migración de nombres se audita con
`python Ventas/Propiedades_Usadas/verificar_normalizacion_nombres.py`.
Compara contra `referencia_normalizacion_nombres.json`, la instantánea histórica
previa a esta migración: cabeceras mediante correspondencia explícita y cuerpos
CSV idénticos byte a byte. También compara imputadores, categorías/referencias,
origen temporal, Tukey y estadísticas del escalador, y prueba inferencia sin `fit`.
Genera `auditoria_normalizacion_nombres.json` en los artefactos de auditoría.
La referencia es evidencia de esta migración, no una restricción sobre futuros
cambios metodológicos expresamente autorizados.

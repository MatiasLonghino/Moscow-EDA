# Primer Parcial: Moscow Real Estate Market - Sales and Rentals

## Datos institucionales

- **Universidad:** Universidad Tecnológica Nacional - Facultad Regional Tucumán (FRT).
- **Carrera:** Ingeniería en Sistemas de Información.
- **Asignatura:** Introducción al Análisis de Datos.
- **Integrantes**:
  - Longhino, Matías
   - Montilla, Tomas

## Contenido por directorio

- `data/`: conjuntos de datos originales (`rentals.csv`, `secondary_market.csv`, `new_builds.csv`, `district_prices_monthly.csv` y `metro_stations.csv`) y conjuntos procesados de alquiler en `data/processed/`.
- `alquiler/`: flujo de análisis de alquiler en 4 notebooks ordenados (EDA, limpieza, ingeniería de características y modelo OLS).
- `Ventas/Propiedades_Nuevas/`: análisis de venta de propiedades nuevas (EDA e integración, transformación OLS y modelo OLS) y sus datos derivados en `data/`.
- `Ventas/Propiedades_Usadas/`: análisis de venta de propiedades usadas (EDA e integración, y modelo OLS).

## ¿Qué pregunta responde cada análisis?

| Análisis | Pregunta que responde |
| --- | --- |
| Alquiler (`alquiler/`) | ¿Qué características de la vivienda y su ubicación explican el alquiler mensual (`monthly_rent_rub`) en Moscú? |
| Venta de propiedades nuevas (`Ventas/Propiedades_Nuevas/`) | ¿Qué características de la unidad, del complejo y de su ubicación explican el precio de venta (`price_rub`) de las propiedades nuevas en Moscú? |
| Venta de propiedades usadas (`Ventas/Propiedades_Usadas/`) | ¿Qué características de la vivienda, del edificio y de su ubicación explican el precio de venta (`price_rub`) de las propiedades usadas en Moscú? |

## Experimentos

Los experimentos y las ejecuciones de los modelos están en [MLflow en DagsHub](https://dagshub.com/matiasgastonlonghino/MOSCOW_IDA.mlflow/#/).

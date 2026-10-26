"""Verifica la migración de nombres frente a la referencia anterior, sin ajustar objetos."""
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd

import preparacion_train_test as pt


def parametros_aprendidos(objetos):
    preparacion = objetos["preparacion"]
    cat = objetos["base"].named_transformers_["cat"]
    distrito = objetos["district"].named_steps["onehot"]
    temporal = objetos["extraccion"]
    escala = objetos["escalado"].estandarizador_
    return {
        "imputadores": {k: getattr(preparacion, k).statistics_.tolist()
                        for k in ["numerico_", "mensual_", "binario_", "categorico_"]},
        "categorias": {c: v.tolist() for c, v in zip(cat.feature_names_in_, cat.categories_)},
        "district": distrito.categories_[0].tolist(), "drop_district": distrito.drop_idx_.tolist(),
        "drop_cat": cat.drop_idx_.tolist(), "mes": temporal.mes_.categories_[0].tolist(),
        "drop_mes": temporal.mes_.drop_idx_.tolist(), "origen_mes": temporal.origen_mes_,
        "media_escalado": escala.mean_.tolist(), "varianza_escalado": escala.var_.tolist(),
        "escala": escala.scale_.tolist(),
        "tukey": objetos["configuracion"]["tratamiento_objetivo"],
    }


def verificar(raiz):
    raiz = Path(raiz)
    carpeta = raiz / "Ventas/Propiedades_Usadas"
    salida = carpeta / "salidas_transformacion"
    artefactos = raiz / "Data/Propiedades_Usadas/transformacion_80_20"
    referencia = json.loads((artefactos / "referencia_normalizacion_nombres.json").read_text(encoding="utf-8"))
    manifiesto = json.loads((salida / "manifiesto_exportacion.json").read_text(encoding="utf-8"))
    for nombre, huella in manifiesto["archivos_sha256"].items():
        if pt.huella_archivo(salida / nombre) != huella:
            raise ValueError(f"Huella incorrecta: {nombre}.")
    objetos = joblib.load(salida / "transformadores_prop_usadas.joblib")
    antes = joblib.hash(objetos)
    if parametros_aprendidos(objetos) != referencia["parametros"]:
        raise ValueError("Se modificaron parámetros estadísticos durante el renombrado.")
    antiguos = referencia["columnas_X_anteriores"]
    nuevos = [pt.nombre_predictor(c) for c in antiguos]
    if nuevos != objetos["columnas_X"] or len(set(nuevos)) != len(antiguos):
        raise ValueError("La correspondencia no preserva la selección u orden de las predictoras.")
    csv = {}
    for archivo, info in referencia["csv"].items():
        ruta = salida / Path(archivo).name
        cabecera, cuerpo = ruta.read_bytes().split(b"\n", 1)
        if hashlib.sha256(cuerpo).hexdigest() != info["contenido_sha256"]:
            raise ValueError(f"Los valores o registros cambiaron: {ruta.name}.")
        esperadas = [pt.nombre_predictor(c) if c in antiguos else c for c in info["columnas_anteriores"]]
        if cabecera.decode("utf-8").strip().split(",") != esperadas:
            raise ValueError(f"Cabecera incorrecta: {ruta.name}.")
        frame = pd.read_csv(ruta, dtype={"id": "string"}, parse_dates=["date_posted_dt"], float_precision="round_trip")
        pt.validar_matriz(pt.obtener_X(frame), frame, columnas=nuevos)
        csv[ruta.name] = {"filas": len(frame), "predictoras": len(nuevos),
                          "cuerpo_csv_identico": True, "sha256": pt.huella_archivo(ruta)}
    resumen = json.loads((salida / "resumen_final.json").read_text(encoding="utf-8"))
    if resumen["columnas_X"] != nuevos or list(resumen["linaje"]) != nuevos:
        raise ValueError("El resumen JSON no corresponde a los CSV.")
    for grupo in resumen["categorias"]:
        if any(c != pt.nombre_predictor(c) for c in grupo["columnas"]):
            raise ValueError("Quedan salidas One-Hot sin normalizar en el resumen.")
    raw = pd.read_csv(artefactos / "test_crudo.csv", dtype={"id": "string"},
                      parse_dates=["date_posted_dt"], float_precision="round_trip")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reproducido = pt.aplicar_flujo(raw, objetos)[0].reset_index(drop=True)
        esperado = pd.read_csv(salida / "test_transformado.csv", dtype={"id": "string"},
                               parse_dates=["date_posted_dt"], float_precision="round_trip")
        pd.testing.assert_frame_equal(reproducido, esperado, check_dtype=False)
        inferencia = pt.aplicar_flujo(raw.iloc[:3].drop(columns=pt.OBJETIVO), objetos)[0]
    if pt.obtener_X(inferencia).columns.tolist() != nuevos or joblib.hash(objetos) != antes:
        raise ValueError("La inferencia futura alteró el esquema o los estimadores.")
    reporte = {
        "version_flujo": objetos["configuracion"]["version_flujo"],
        "convencion_nombres": "snake_case_ascii_v1", "csv": csv,
        "correspondencia_nombres_finales": dict(zip(antiguos, nuevos)),
        "parametros_aprendidos_identicos": True, "categorias_y_referencias_identicas": True,
        "columnas_seleccionadas_y_orden_preservados": True, "sin_colisiones": True,
        "objetos_test_sin_reajuste": True, "joblib_reproduce_test_exportado": True,
        "inferencia_sin_objetivo_validada": True, "resumen_y_csv_sincronizados": True,
        "modulo_sha256": pt.huella_archivo(Path(pt.__file__)),
        "archivos_sha256": manifiesto["archivos_sha256"],
        "escalado": pt.registro_escalado(objetos["escalado"]),
    }
    pt.guardar_json(artefactos / "auditoria_normalizacion_nombres.json", reporte)
    return reporte


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parents[2]
    reporte = verificar(raiz)
    print("Equivalencia exacta de valores, parámetros, categorías y orden confirmada; Joblib reutilizado sin fit.")

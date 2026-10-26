"""Baseline OLS completa: solo Train transformado; sin selección ni nuevo preprocesamiento."""
import copy
import shutil
from tempfile import TemporaryDirectory
import hashlib
import json
import os
import platform
import warnings
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import pandas as pd
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
import statsmodels.api as sm
from scipy import linalg
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera, omni_normtest
from threadpoolctl import threadpool_limits
from experimentos_ols import evaluar_regresion, tabla_markdown
from diagnosticos_residuales import ARCHIVOS, generar_diagnosticos_residuales, registrar_diagnosticos_residuales

EXPERIMENTO = "Modelos Propiedades_Usadas"
RUN = "Baseline_OLS_01"
TRACKING_URI = "https://dagshub.com/matiasgastonlonghino/MOSCOW_IDA.mlflow"

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def guardar_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def cargar_train(base):
    base = Path(base)
    archivo = base / "salidas_transformacion/train_transformado.csv"
    manifiesto = base / "salidas_transformacion/manifiesto_exportacion.json"
    contrato = json.loads(manifiesto.read_text(encoding="utf-8"))
    huella = sha256(archivo)
    assert huella == contrato["train"]["sha256"], "El CSV difiere del manifiesto."
    datos = pd.read_csv(archivo)
    predictoras = contrato["columnas_X"]
    auxiliares = ["id", "date_posted_dt", "dataset_split"]
    assert datos.columns.is_unique and len(predictoras) == len(set(predictoras))
    assert set(datos) == set(predictoras + auxiliares + ["price_rub"])
    assert len(datos) == contrato["train"]["filas"]
    assert len(predictoras) == contrato["train"]["predictoras"]
    assert datos["dataset_split"].eq("train").all()
    X = datos[predictoras].astype(float)
    y = datos["price_rub"].astype(float)
    assert np.isfinite(X).all().all() and np.isfinite(y).all() and (y > 0).all()
    contexto = {"csv": "salidas_transformacion/train_transformado.csv", "dataset_sha256": huella,
        "manifiesto_sha256": sha256(manifiesto), "objetivo": "price_rub", "unidad": "RUB",
        "transformacion_objetivo": "ninguna", "predictoras": predictoras, "auxiliares_excluidas": auxiliares,
        "n_train": len(y), "n_predictoras": len(predictoras), "test_utilizado": False,
        "preprocesamiento": "CSV transformado existente; no se ajustan transformadores"}
    return X, y, contexto

def ajustar(X, y):
    diseno = sm.add_constant(X, has_constant="add")
    with threadpool_limits(limits=1):
        modelo = sm.OLS(y, diseno, missing="raise").fit()
    assert int(modelo.nobs) == len(y) and modelo.model.exog_names == list(diseno.columns)
    return modelo, diseno

def diagnosticar(modelo, diseno, y, *, y_original=None):
    """Diagnósticos en la escala ajustada; si y=ln(precio), predicción RUB sin corrección."""
    rango = int(modelo.model.rank)
    # Reutiliza las métricas predictivas de referencia, sin su entrenamiento temporal/logarítmico.
    pred = modelo.predict(diseno)
    if y_original is not None:
        assert y.index.equals(y_original.index) and (y_original > 0).all()
        np.testing.assert_allclose(y, np.log(y_original), rtol=1e-12)
    heredadas = evaluar_regresion(y if y_original is None else y_original,
        pred if y_original is None else np.exp(pred),
        es_train=True, n_predictoras_efectivas=rango-1)
    m = {"train_r2": heredadas["r2"], "train_r2_adj": heredadas["r2_ajustado"],
        "train_mae": heredadas["mae_rub"], "train_rmse": heredadas["rmse_rub"],
        "train_mape": heredadas["mape_pct"], "train_rss": heredadas["rss_rub2"],
        "train_mse": heredadas["rss_rub2"]/len(y), "train_aic": float(modelo.aic),
        "train_bic": float(modelo.bic), "train_f_statistic": float(modelo.fvalue),
        "train_f_pvalue": float(modelo.f_pvalue), "train_n_obs": len(y),
        "train_n_predictors": diseno.shape[1]-1, "train_n_parameters_effective": rango,
        "train_n_coefficients": len(modelo.params), "train_df_model": float(modelo.df_model),
        "train_df_resid": float(modelo.df_resid), "train_log_likelihood": float(modelo.llf),
        "train_durbin_watson": float(durbin_watson(modelo.resid)),
        "train_n_coefficients_p_gt_005": int((modelo.pvalues > .05).sum()),
        "train_n_coefficients_p_undefined": int(modelo.pvalues.isna().sum()),
        "train_condition_number": float(modelo.condition_number),
        "train_rank_deficiency": diseno.shape[1]-rango}
    jb, jbp, skew, kurt = jarque_bera(modelo.resid)
    omni, omnip = omni_normtest(modelo.resid)
    m.update(dict(zip(["train_jarque_bera", "train_jarque_bera_pvalue", "train_skewness", "train_kurtosis",
                      "train_omnibus", "train_omnibus_pvalue"], map(float,[jb,jbp,skew,kurt,omni,omnip]))))
    notas = {}
    # QR conserva las normas de los residuos de cada regresión auxiliar.
    # VIF_j = TSS centrado original / RSS auxiliar; el intercepto está en todas las auxiliares.
    filas = []
    with threadpool_limits(limits=1):
        R = np.linalg.qr(diseno.to_numpy(), mode="r").copy()
        for j, nombre in enumerate(diseno.columns[1:], 1):
            tss = float(np.square(diseno.iloc[:,j]-diseno.iloc[:,j].mean()).sum())
            otros = np.delete(R,j,axis=1)
            coef = linalg.lstsq(otros,R[:,j],cond=1e-15)[0]
            rss = float(np.square(R[:,j]-otros@coef).sum())
            if tss == 0:
                valor = np.nan
            elif rss <= tss * 1e-15:
                valor = np.inf
            else:
                valor = tss/rss
            filas.append({"variable": nombre, "vif": valor})
    vif = pd.DataFrame(filas)
    m["train_vif_max"] = float(vif.vif.max())
    m["train_vif_infinite_count"] = int(np.isinf(vif.vif).sum())
    m["train_vif_undefined_count"] = int(vif.vif.isna().sum())
    notas["vif"] = "Incluye intercepto en las regresiones auxiliares y lo excluye del máximo. Constantes: indefinido; dependencia exacta: infinito. No se elimina ninguna columna."
    if rango == diseno.shape[1]:
        with threadpool_limits(limits=1):
            bp = het_breuschpagan(modelo.resid, diseno, robust=True)
        m.update(dict(zip(["train_bp_lm", "train_bp_lm_pvalue", "train_bp_f", "train_bp_f_pvalue"],map(float,bp))))
        notas["breusch_pagan"] = "Versión robusta de Koenker (robust=True)."
    else:
        notas["breusch_pagan"] = "No calculado: diseño de rango deficiente; los grados de libertad LM nominales no son válidos."
    if y_original is not None:
        # Las métricas heredadas usan precios; los diagnósticos del ajuste permanecen en ln(RUB).
        m.update(train_r2=float(modelo.rsquared), train_r2_adj=float(modelo.rsquared_adj),
            train_mae=float(np.abs(modelo.resid).mean()), train_rss=float(modelo.ssr),
            train_mse=float(modelo.ssr/len(y)), train_rmse=float(np.sqrt(modelo.ssr/len(y))))
        m.pop("train_mape")  # MAPE se registra sobre los precios originales, no sobre sus logaritmos.
        invariantes = {"train_n_obs", "train_n_predictors", "train_n_parameters_effective",
            "train_n_coefficients", "train_df_model", "train_df_resid", "train_condition_number",
            "train_rank_deficiency", "train_vif_max", "train_vif_infinite_count", "train_vif_undefined_count"}
        m = {k if k in invariantes else k+"_log":v for k,v in m.items()}
        m.update({f"train_{nombre}_original":heredadas[clave] for nombre,clave in {
            "r2":"r2", "r2_adj":"r2_ajustado", "mae":"mae_rub", "rmse":"rmse_rub",
            "mape":"mape_pct", "rss":"rss_rub2"}.items()})
        m["train_mse_original"] = heredadas["rss_rub2"]/len(y)
        notas["transformacion"] = "ln(price_rub); predicción original = exp(predicción log). Sin Duan ni otra corrección."
        notas["sesgo_inversa"] = "exp(E[ln(precio)|X]) no equivale en general a E[precio|X]; puede subestimar el precio esperado."
        notas["comparabilidad"] = "Comparar con la baseline únicamente métricas predictivas calculadas en RUB sobre las mismas filas. R², AIC, BIC y RSS nativos log no son directamente comparables con los de RUB."
    invalidas = {k:("Infinito: dependencia lineal exacta" if np.isinf(v) else "Indefinido")
                 for k,v in m.items() if not np.isfinite(v)}
    validas = {k:float(v) for k,v in m.items() if np.isfinite(v)}
    notas["metricas_no_finitas_no_registradas"] = invalidas
    notas["aic_bic"] = "Convención statsmodels OLS gaussiana en " + ("RUB" if y_original is None else "ln(RUB)") + ": -2*llf + 2*rank y -2*llf + log(n)*rank; varianza residual no contada como parámetro adicional. Comparar solo misma respuesta, escala, filas y convención."
    notas["inferencia"] = "Covarianza nonrobust; si hay rango deficiente, coeficientes no identificables individualmente. No se eliminan variables."
    notas["durbin_watson"] = "Orden de filas del CSV; no prueba autocorrelación temporal de un panel."
    notas["pvalues_cero"] = "Los p-values devueltos como 0 por statsmodels/SciPy se conservan; pueden corresponder a subdesbordamiento numérico, no a probabilidad matemáticamente nula."
    notas["pvalues"] = "Conteo incluye intercepto; p-values indefinidos se cuentan por separado."
    return validas, invalidas, vif, notas

def tabla_resultados(metricas, invalidas, notas, *, nombre_run=RUN):
    etiquetas = {"r2":"R²", "r2_adj":"R² ajustado", "mae":"MAE", "mse":"MSE", "rmse":"RMSE",
        "mape":"MAPE", "rss":"RSS", "aic":"AIC", "bic":"BIC", "vif_max":"VIF máximo",
        "durbin_watson":"Durbin-Watson", "f_statistic":"F-statistic", "f_pvalue":"Probabilidad F-statistic"}
    filas=[]
    for k,v in {**metricas, **invalidas}.items():
        clave=k.removeprefix("train_")
        escala = "log" if clave.endswith("_log") else "original" if clave.endswith("_original") else None
        if escala:
            clave=clave.removesuffix("_"+escala)
        unidad={"mae":"RUB", "rmse":"RUB", "mse":"RUB²", "rss":"RUB²", "mape":"%",
                "vif_max":"Sin intercepto", "aic":"Train; likelihood RUB; k=rank", "bic":"Train; likelihood RUB; k=rank"}.get(clave,"Train")
        if escala == "log":
            unidad = unidad.replace("RUB", "ln(RUB)")
            unidad += "; ajuste log" if unidad == "Train" else ""
        columna = "Baseline OLS" if nombre_run == RUN else nombre_run
        etiqueta=etiquetas.get(clave,clave)+(f" ({'ln(RUB)' if escala=='log' else 'precio original'})" if escala else "")
        filas.append({"Métrica":etiqueta,columna:(format(v,".12g") if isinstance(v,(float,int)) else v),"Unidad / Observación":unidad})
    if not any(k in metricas for k in ["train_bp_lm", "train_bp_lm_log"]):
        filas.append({"Métrica":"Breusch-Pagan",("Baseline OLS" if nombre_run==RUN else nombre_run):"No calculado","Unidad / Observación":notas["breusch_pagan"]})
    return tabla_markdown(pd.DataFrame(filas),decimales=10)

def exportar(base, modelo, diseno, y, contexto, metricas, invalidas, vif, notas,
             *, nombre_run=RUN, carpeta_salida=None):
    import mlflow.statsmodels
    out=Path(carpeta_salida) if carpeta_salida is not None else Path(base)/"salidas_baseline_ols"
    out.mkdir(exist_ok=True)
    temporal_pickle=out/"modelo_statsmodels.tmp"
    copy.deepcopy(modelo).save(temporal_pickle, remove_data=True)
    os.replace(temporal_pickle,out/"modelo_statsmodels.pkl")
    pd.DataFrame({"coeficiente":modelo.params,"error_estandar":modelo.bse,"t":modelo.tvalues,
        "p_value":modelo.pvalues,"ic95_inferior":modelo.conf_int()[0],"ic95_superior":modelo.conf_int()[1]}).to_csv(out/"coeficientes.csv",index_label="variable")
    vif.to_csv(out/"vif.csv",index=False)
    (out/"resumen_ols.txt").write_text(modelo.summary().as_text(),encoding="utf-8")
    (out/"metricas.md").write_text(tabla_resultados(metricas,invalidas,notas,nombre_run=nombre_run),encoding="utf-8")
    guardar_json(out/"metricas.json",metricas)
    guardar_json(out/"diagnosticos.json",notas)
    guardar_json(out/"contexto.json",contexto)
    es_log = contexto["transformacion_objetivo"] == "ln"
    guardar_json(out/"inferencia.json",{"entrada":f"DataFrame de las {len(contexto['predictoras'])} predictoras ya transformadas en el orden declarado",
        "columnas_diseno":list(diseno.columns),"intercepto":1.0,"unidad_salida":"RUB",
        "instruccion":"sm.add_constant(X[columnas_diseno[1:]].astype(float), has_constant='add'); " + ("np.exp(modelo.predict(diseno))" if es_log else "modelo.predict(diseno)"),
        "transformacion_objetivo":contexto["transformacion_objetivo"],
        **({"unidad_salida_modelo":"ln(RUB)", "transformacion_inversa":"exp", "correccion_sesgo":"ninguna"} if es_log else {})})
    paquetes=["numpy","pandas","statsmodels","scipy","scikit-learn","python-dotenv","threadpoolctl","matplotlib"]
    try:
        version("mlflow")
        paquetes.append("mlflow")
    except PackageNotFoundError:
        paquetes.append("mlflow-skinny")
    versiones={k:version(k) for k in paquetes}
    guardar_json(out/"versiones.json",{"python":platform.python_version(),**versiones})
    (out/"requirements.txt").write_text("\n".join(f"{k}=={v}" for k,v in versiones.items())+"\n",encoding="utf-8")
    from statsmodels.regression.linear_model import OLSResults
    recuperado=OLSResults.load(out/"modelo_statsmodels.pkl")
    np.testing.assert_allclose(recuperado.predict(diseno.iloc[:10]),modelo.predict(diseno.iloc[:10]),rtol=1e-12)
    carpeta_modelo=out/"modelo_mlflow"
    with TemporaryDirectory(prefix=".flavor_", dir=out) as tmp:
        temporal=Path(tmp)/"modelo"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlflow.statsmodels.save_model(copy.deepcopy(modelo),str(temporal),remove_data=True,
                pip_requirements=[f"{k}=={v}" for k,v in versiones.items()])
        shutil.copytree(temporal,carpeta_modelo,dirs_exist_ok=True)
    verificado=mlflow.statsmodels.load_model(str(carpeta_modelo))
    np.testing.assert_allclose(verificado.predict(diseno.iloc[:10]),modelo.predict(diseno.iloc[:10]),rtol=1e-12)
    generar_diagnosticos_residuales(modelo, diseno, y, nombre_run=nombre_run,
        carpeta=out/"diagnosticos_residuales", unidad_objetivo=contexto["unidad"])
    for fuente in ["baseline_ols.py", "experimentos_ols.py", "diagnosticos_residuales.py"]:
        shutil.copyfile(Path(__file__).with_name(fuente), out/fuente)
    assert sha256(Path(base)/contexto["csv"])==contexto["dataset_sha256"]
    return out

def conectar(raiz):
    from dotenv import load_dotenv
    import mlflow
    from mlflow.tracking import MlflowClient
    load_dotenv(Path(raiz)/".env",override=False)
    os.environ["MLFLOW_TRACKING_USERNAME"]=os.environ.get("DAGSHUB_USERNAME") or "matiasgastonlonghino"
    if not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"]=os.environ["DAGSHUB_USER_TOKEN"]
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"]="30"
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]="1"
    mlflow.set_tracking_uri(TRACKING_URI)
    cliente=MlflowClient(tracking_uri=TRACKING_URI)
    cliente.search_experiments(max_results=1)
    return cliente

def registrar(cliente,out,contexto,metricas, *, nombre_run=RUN):
    for nombre in ARCHIVOS:
        assert (Path(out)/"diagnosticos_residuales"/nombre).is_file(), "Generar diagnósticos de Train antes de registrar."
    exp=cliente.get_experiment_by_name(EXPERIMENTO)
    eid=exp.experiment_id if exp else cliente.create_experiment(EXPERIMENTO)
    existentes=cliente.search_runs([eid],filter_string=f"tags.mlflow.runName = '{nombre_run}'",max_results=100)
    propios=[r for r in existentes if r.data.tags.get("dataset_sha256")==contexto["dataset_sha256"] and r.info.status=="FINISHED"]
    if propios:
        rid=propios[0].info.run_id
        # Actualizar la evidencia del mismo Run sin reemplazar su modelo ni su código histórico.
        verificar(cliente,rid,eid,out,metricas,nombre_run=nombre_run,reutilizado=True)
        registrar_diagnosticos_residuales(cliente,rid,Path(out)/"diagnosticos_residuales",nombre_run=nombre_run)
    else:
        run=cliente.create_run(eid,tags={"mlflow.runName":nombre_run,"dataset_sha256":contexto["dataset_sha256"],
            "test_utilizado":"false","baseline_version":"1"})
        rid=run.info.run_id
        try:
            parametros={"modelo":"statsmodels.OLS","intercepto":True,"cov_type":"nonrobust","solver":"pinv",
                "train_fraction":1.0,"n_train":contexto["n_train"],"n_predictors":contexto["n_predictoras"],
                "n_parameters_effective":int(metricas["train_n_parameters_effective"]),
                "objetivo":"price_rub","target_unit":contexto["unidad"],
                "target_transform":"ln(Target)" if contexto["transformacion_objetivo"]=="ln" else "none",
                "feature_selection":"none","csv":contexto["csv"],"dataset_sha256":contexto["dataset_sha256"],
                "aic_bic_convention":"statsmodels_gaussian_" + ("log_RUB" if contexto["transformacion_objetivo"]=="ln" else "RUB") + "_k_rank"}
            if contexto["transformacion_objetivo"]=="ln":
                parametros.update(inverse_transform="exp", bias_correction="none", baseline_run_id=contexto["baseline_run_id"])
            for k,v in parametros.items(): cliente.log_param(rid,k,v)
            for k,v in metricas.items(): cliente.log_metric(rid,k,v)
            cliente.log_artifacts(rid,str(out))
            cliente.set_terminated(rid,"FINISHED")
        except BaseException:
            cliente.set_terminated(rid,"FAILED")
            raise
    if not propios:
        verificar(cliente,rid,eid,out,metricas,nombre_run=nombre_run)
    registro={"run_id":rid,"experiment_id":eid,"experiment_name":EXPERIMENTO,"run_name":nombre_run,
        "status":"FINISHED","url":f"{TRACKING_URI}/#/experiments/{eid}/runs/{rid}"}
    guardar_json(out/"registro_dagshub.json",registro)
    return registro

def verificar(cliente,rid,eid,out,metricas, *, nombre_run=RUN, reutilizado=False):
    from tempfile import TemporaryDirectory
    remoto=cliente.get_run(rid)
    assert remoto.info.status=="FINISHED" and remoto.info.experiment_id==eid
    assert remoto.data.tags["mlflow.runName"]==nombre_run
    for k,v in metricas.items(): np.testing.assert_allclose(remoto.data.metrics[k],v,rtol=1e-10,atol=1e-12)
    assert remoto.data.params["feature_selection"]=="none" and remoto.data.params["train_fraction"]=="1.0"
    # En un Run existente se valida el contrato y las métricas estables: serializaciones,
    # timestamps y código histórico pueden diferir de una exportación local posterior.
    if reutilizado:
        with TemporaryDirectory(prefix="verificar_contrato_") as tmp:
            for nombre in ["contexto.json", "metricas.json", "inferencia.json", "metricas.md"]:
                descargado=cliente.download_artifacts(rid,nombre,tmp)
                assert sha256(descargado)==sha256(Path(out)/nombre), nombre
        return
    # En un Run nuevo compara todo el conjunto, incluidos los componentes MLflow.
    with TemporaryDirectory(prefix="verificar_baseline_") as tmp:
        for local in Path(out).rglob("*"):
            if local.is_file() and local.name!="registro_dagshub.json":
                rel=local.relative_to(out).as_posix()
                descargado=cliente.download_artifacts(rid,rel,tmp)
                assert sha256(descargado)==sha256(local), rel

import ast
import copy
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'Fase_2/data'
PREP=ROOT/'Fase_2/Preparacion_OLS_Usadas.ipynb'
MODEL=ROOT/'Fase_2/Modelo_OLS.ipynb'
prep=json.loads(PREP.read_text(encoding='utf-8'))
model=json.loads(MODEL.read_text(encoding='utf-8'))
oldfiles=sorted(p for p in DATA.rglob('*') if p.is_file())
transformador=joblib.load(DATA/'transformadores_ols_usadas.joblib')
original=joblib.load(DATA/'modelos_ols_por_segmento.joblib')
sincentro=joblib.load(DATA/'modelos_ols_sin_to_center_km.joblib')
bss=joblib.load(DATA/'bss_features/modelos_bss_features.joblib')
orden=original['parametros_segmentacion_boxcox']['orden_segmentos']
protected=[ROOT/'Ventas/Propiedades_Usadas/data/PropUsadas_Integrado.csv',DATA/'train_ols_usadas.csv',DATA/'test_ols_usadas.csv',
           DATA/'metadatos_train_ols_usadas.csv',DATA/'metadatos_test_ols_usadas.csv',ROOT/'Fase_2/Analisis_Usadas.ipynb']
hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}

def src(c):return ''.join(c['source'])
def put(c,s):c['source']=s.splitlines(True)
def new(kind,s):
    c=dict(cell_type=kind,id=uuid.uuid4().hex[:8],metadata={},source=s.strip().splitlines(True))
    if kind=='code':c.update(execution_count=None,outputs=[])
    return c
def quitar_escrituras_json(s):
    tree=ast.parse(s);quitar=set()
    for node in tree.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='write_text':
            if '.json' in ast.get_source_segment(s,node):quitar.update(range(node.lineno,node.end_lineno+1))
    return ''.join(l for i,l in enumerate(s.splitlines(True),1) if i not in quitar)

CLAVES_PREP=['version_flujo','entrada','sha256_entrada','versiones','columnas_originales','tipos_originales','columnas_numericas_entrada',
    'correcciones','imputacion','eliminadas','alias','nominales','referencias','binarias','ordinales','desconocidas','dummies','temporales',
    'entrada_codificador','columnas_generadas','constantes_excluidas','columnas_finales','objetivo','metadatos','columnas_exportadas','tipos_predictores']
registro_minimo={k:transformador['registro'][k] for k in CLAVES_PREP}

# Preparación: CSV inmutables si ya existen, registro funcional solo en Joblib.
for c in prep['cells']:
    s=src(c)
    if c['cell_type']=='code' and 'carpeta_salida.mkdir' in s:
        old='''train_exportado.to_csv(carpeta_salida / "train_ols_usadas.csv", index=False)
test_exportado.to_csv(carpeta_salida / "test_ols_usadas.csv", index=False)
meta_train.to_csv(carpeta_salida / "metadatos_train_ols_usadas.csv", index=False, date_format="%Y-%m-%d")
meta_test.to_csv(carpeta_salida / "metadatos_test_ols_usadas.csv", index=False, date_format="%Y-%m-%d")'''
        replacement='''def guardar_dataset_necesario(datos, ruta, **opciones):
    if not ruta.exists():
        datos.to_csv(ruta, index=False, **opciones)
        return
    existente = pd.read_csv(ruta)
    assert existente.shape == datos.shape and existente.columns.tolist() == datos.columns.tolist()
    if "date_posted_dt" in datos:
        existente["date_posted_dt"] = pd.to_datetime(existente.date_posted_dt)
        pd.testing.assert_frame_equal(existente, datos.reset_index(drop=True))
    else:
        np.testing.assert_allclose(existente.to_numpy(), datos.to_numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(existente.price_rub, datos.price_rub.to_numpy())

guardar_dataset_necesario(train_exportado, carpeta_salida / "train_ols_usadas.csv")
guardar_dataset_necesario(test_exportado, carpeta_salida / "test_ols_usadas.csv")
guardar_dataset_necesario(meta_train, carpeta_salida / "metadatos_train_ols_usadas.csv", date_format="%Y-%m-%d")
guardar_dataset_necesario(meta_test, carpeta_salida / "metadatos_test_ols_usadas.csv", date_format="%Y-%m-%d")'''
        assert old in s;s=s.replace(old,replacement)
        a=s.index('ruta_registro =');b=s.index('train_releido =',a)
        s=s[:a]+f'''claves_recuperacion = {CLAVES_PREP!r}
registro_recuperable = {{k: a_json(registro_transformaciones[k]) for k in claves_recuperacion}}
ruta_transformadores = carpeta_salida / "transformadores_ols_usadas.joblib"
if not ruta_transformadores.exists():
    joblib.dump({{"registro": registro_recuperable, "codificador": codificador}}, ruta_transformadores, compress=3)

'''+s[b:]
        s=s.replace('registro_recargado = json.loads(ruta_registro.read_text(encoding="utf-8"))',
                    'registro_recargado = artefacto_recargado["registro"]\nassert registro_recargado == registro_recuperable')
        put(c,s)
    elif c['cell_type']=='markdown':
        s=s.replace('un registro JSON y un codificador reutilizable','un registro funcional y un codificador dentro de un único Joblib')
        s=s.replace('El artefacto `transformadores_ols_usadas.joblib` contiene el codificador ya ajustado y el mismo registro; el JSON permite inspeccionarlo sin cargar el artefacto.',
                    'El artefacto `transformadores_ols_usadas.joblib` conserva el codificador y únicamente el registro necesario para aplicar las transformaciones. Las métricas y auditorías permanecen en las tablas del notebook.')
        s=s.replace('Se exportan además el registro JSON y los transformadores ajustados, junto con',
                    'Se conserva un único Joblib con los transformadores y sus parámetros, junto con')
        put(c,s)

# Modelo: los parámetros Box-Cox viven solo en el artefacto de OLS original.
for c in model['cells']:
    s=src(c)
    if c['cell_type']=='code':
        if s.startswith('from pathlib import Path'):
            s=s.replace('import hashlib','import joblib\nimport hashlib',1)
            s+='''
ruta_modelos_originales = carpeta_datos / "modelos_ols_por_segmento.joblib"
ruta_modelos_sin_centro = carpeta_datos / "modelos_ols_sin_to_center_km.joblib"
cache_originales_ols = joblib.load(ruta_modelos_originales) if ruta_modelos_originales.exists() else None
cache_sin_centro_ols = joblib.load(ruta_modelos_sin_centro) if ruta_modelos_sin_centro.exists() else None
'''
        if s.startswith('cuartiles_train ='):
            old='''cuartiles_train = train.to_center_km.quantile([0.25, 0.50, 0.75])'''
            s=s.replace(old,'''cuartiles_train = (pd.Series(cache_originales_ols["parametros_segmentacion_boxcox"]["cuartiles_distancia_km"].values(), index=[0.25,0.50,0.75])
                   if cache_originales_ols is not None else train.to_center_km.quantile([0.25,0.50,0.75]))''')
        if s.startswith('train_boxcox ='):
            old='''transformados, lambda_segmento = boxcox(precios_segmento.to_numpy(dtype=float))'''
            assert old in s
            s=s.replace(old,'''if cache_originales_ols is not None:
        lambda_segmento = cache_originales_ols["parametros_segmentacion_boxcox"]["lambda_por_segmento"][segmento]
        transformados = boxcox(precios_segmento.to_numpy(dtype=float), lmbda=lambda_segmento)
    else:
        transformados, lambda_segmento = boxcox(precios_segmento.to_numpy(dtype=float))''')
        if s.startswith('condiciones_segmentos ='):
            a=s.index('ruta_parametros =');b=s.index('\ndef aplicar_parametros_boxcox',a)
            s=s[:a]+'''parametros_recargados = (cache_originales_ols["parametros_segmentacion_boxcox"]
                          if cache_originales_ols is not None else parametros_boxcox)
assert parametros_recargados == parametros_boxcox
'''+s[b:]
            s=s.replace('"Archivo": ruta_parametros.name','"Archivo": "modelos_ols_por_segmento.joblib"')
        s=s.replace('json.loads((carpeta_datos / "registro_transformaciones_ols_usadas.json").read_text(encoding="utf-8"))',
                    'joblib.load(carpeta_datos / "transformadores_ols_usadas.joblib")["registro"]')
        s=s.replace('json.loads((carpeta_datos / "contratos_features_ols.json").read_text(encoding="utf-8"))["sin_to_center_km"]',
                    '{s: a["contrato"] for s,a in analisis_features_sin_centro.items()}')
        if 'def ajustar_segmento_ols' in s:
            old='''resultado = sm.OLS(datos.boxcox_price_rub, X_ajuste, missing="raise").fit()'''
            assert old in s
            s=s.replace(old,'''cache = cache_originales_ols if "to_center_km" in columnas_predictoras_ols else cache_sin_centro_ols
    if cache is not None and segmento in cache["modelos"]:
        resultado = cache["modelos"][segmento]
        assert resultado.model.exog_names == X_ajuste.columns.tolist()
        np.testing.assert_array_equal(resultado.model.exog, X_ajuste.to_numpy())
        np.testing.assert_array_equal(resultado.model.endog, datos.boxcox_price_rub.to_numpy())
    else:
        resultado = sm.OLS(datos.boxcox_price_rub, X_ajuste, missing="raise").fit()''')
        if s.startswith('filas_comparacion_ols ='):
            a=s.index('joblib.dump(');b=s.index('comparacion_modelos_features =',a)
            s=s[:a]+'''if not ruta_modelos_originales.exists():
    joblib.dump({"modelos": modelos_ols, "registros": registros_ols,
                 "parametros_segmentacion_boxcox": parametros_recargados}, ruta_modelos_originales, compress=3)
'''+s[b:]
        if s.startswith('joblib.dump({**experimento_sin_centro'):
            b=s.index('assert sum(')
            s='''if not ruta_modelos_sin_centro.exists():
    joblib.dump({"modelos": experimento_sin_centro["modelos"], "registros": experimento_sin_centro["registros"],
                 "predictor_excluido": "to_center_km"}, ruta_modelos_sin_centro, compress=3)
'''+s[b:]
        if s.startswith('contratos_features_ols ='):
            s=quitar_escrituras_json(s)
        if 'MAX_CANDIDATAS_BSS = ' in s:
            s=s.replace('from bss_features_usadas import preparar_interno, buscar_segmento, importancia_permutacion, vif_features, inversa, errores',
                'from bss_features_usadas import preparar_interno, buscar_segmento, importancia_permutacion, vif_features, inversa, errores, entrada_codificador, recuperar_busqueda')
            s=s.replace('referencia_bss = joblib.load(carpeta_datos / "modelos_ols_sin_to_center_km.joblib")',
                'referencia_bss = {"modelos":experimento_sin_centro["modelos"], "tablas":experimento_sin_centro["tablas"], "evaluaciones":experimento_sin_centro["evaluaciones"]}')
            a=s.index('X_sub_bss, X_val_bss,');b=s.index('for f, ref in preparador_bss',a)
            s=s[:a]+'''ruta_bss = salidas_bss / "modelos_bss_features.joblib"
cache_bss = joblib.load(ruta_bss) if ruta_bss.exists() else None
if cache_bss is None:
    X_sub_bss, X_val_bss, y_sub_bss, y_val_bss, preparador_bss = preparar_interno(
        crudo_train_bss.iloc[:corte_bss], crudo_train_bss.iloc[corte_bss:], registro_preparacion_bss)
else:
    preparador_bss = {**cache_bss["preparador_validacion"], "registro":registro_preparacion_bss}
    preparador_bss["columnas"] = preparador_bss["encoder"].get_feature_names_out().tolist()
    preparador_bss["referencias"] = {columnas[0]: enc.categories_[0][0] for nombre,enc,columnas in preparador_bss["encoder"].transformers_ if nombre.startswith("cat")}
    A, y_sub_bss = entrada_codificador(crudo_train_bss.iloc[:corte_bss], preparador_bss)
    B, y_val_bss = entrada_codificador(crudo_train_bss.iloc[corte_bss:], preparador_bss)
    X_sub_bss = pd.DataFrame(preparador_bss["encoder"].transform(A), index=A.index, columns=preparador_bss["columnas"])
    X_val_bss = pd.DataFrame(preparador_bss["encoder"].transform(B), index=B.index, columns=preparador_bss["columnas"])
'''+s[b:]
            old='''busqueda = buscar_segmento(X_sub_bss.loc[i], X_val_bss.loc[j], y_sub_bss.loc[i], y_val_bss.loc[j],
        preparador_bss["mapa"], segmento, MAX_CANDIDATAS_BSS, TOLERANCIA_RMSE_BSS)'''
            assert old in s
            s=s.replace(old,'''if cache_bss is None:
        busqueda = buscar_segmento(X_sub_bss.loc[i], X_val_bss.loc[j], y_sub_bss.loc[i], y_val_bss.loc[j],
            preparador_bss["mapa"], segmento, MAX_CANDIDATAS_BSS, TOLERANCIA_RMSE_BSS)
    else:
        busqueda = recuperar_busqueda(cache_bss, X_sub_bss.loc[i], X_val_bss.loc[j], y_sub_bss.loc[i], y_val_bss.loc[j], segmento)''')
            s=''.join(line for line in s.splitlines(True) if '.to_csv(' not in line)
            s=quitar_escrituras_json(s)
            s=s.replace('len(b["resultados"])','(len(b["resultados"]) if b["resultados"] is not None else 2**len(b["candidatas_restringidas"]))')
            s=s.replace('int(b["resultados"]["rango_válido"].sum())','(int(b["resultados"]["rango_válido"].sum()) if b["resultados"] is not None else 2**len(b["candidatas_restringidas"]))')
        if s.startswith('modelos_bss = {}'):
            old='''modelo = sm.OLS(y,A,missing="raise").fit()'''
            assert old in s
            s=s.replace(old,'''if cache_bss is None:
        modelo = sm.OLS(y,A,missing="raise").fit()
    else:
        modelo = cache_bss["modelos_finales"][segmento]
        assert modelo.model.exog_names == A.columns.tolist()
        np.testing.assert_array_equal(modelo.model.exog, A.to_numpy())
        np.testing.assert_array_equal(modelo.model.endog, y.to_numpy())''')
        if s.startswith('joblib.dump({"modelos_finales":'):
            s='''if not ruta_bss.exists():
    configuracion_bss = {s:{"features":list(b["seleccionadas"]), "lambda_interno":b["lambda_interno"],
        "candidatas_totales":b["candidatas_totales"], "candidatas_restringidas":b["candidatas_restringidas"]}
        for s,b in busquedas_bss.items()}
    preparador_validacion = {k:preparador_bss[k] for k in ["imputacion","binarias","encoder","mapa"]}
    joblib.dump({"modelos_finales":modelos_bss,
        "modelos_validacion":{s:b["modelo_interno"] for s,b in busquedas_bss.items()},
        "seleccion":configuracion_bss,"preparador_validacion":preparador_validacion,
        "configuracion":{"corte":corte_bss,"max_candidatas":MAX_CANDIDATAS_BSS,"tolerancia_rmse":TOLERANCIA_RMSE_BSS,
                          "semilla":SEMILLA_BSS,"repeticiones":REPETICIONES_PERMUTACION_BSS},
        "dependencias":{"preparacion":"../transformadores_ols_usadas.joblib","segmentacion":"../modelos_ols_por_segmento.joblib"}},
        ruta_bss, compress=3)
assert sum(r["n_train"] for r in registro_final_bss.values())==40_000
assert sum(r["n_test"] for r in registro_final_bss.values())==10_000
assert all(hashlib.sha256((carpeta_datos / nombre).read_bytes()).hexdigest()==huella for nombre,huella in huellas_entrada.items())'''
        put(c,s)
    else:
        s=s.replace('Se guarda `Fase_2/data/parametros_boxcox_usadas.json`','La configuración se conserva únicamente en `data/modelos_ols_por_segmento.joblib`')
        s=s.replace('El archivo identifica que Box-Cox recibe RUB','Ese objeto identifica que Box-Cox recibe RUB')
        s=s.replace('**Parámetros guardados y verificados**','**Parámetros centralizados y verificados**')
        s=s.replace('| parametros_boxcox_usadas.json |','| modelos_ols_por_segmento.joblib |')
        s=s.replace('El contrato se exporta a `data/contratos_features_ols.json`.','El contrato se calcula desde los transformadores y modelos guardados; permanece en memoria y en las tablas, sin exportar un JSON redundante.')
        s=s.replace('Los registros completos de cada combinación, incluidas sus características, parámetros y controles de rango, quedan en `data/bss_features/subconjuntos_*.csv`.',
            'Los resultados relevantes y rankings se conservan en las tablas del notebook. Los registros exhaustivos permanecen en memoria cuando se ejecuta una búsqueda nueva, sin exportar CSV auxiliares.')
        put(c,s)

# Rankings académicos que antes solo estaban en CSV: conservarlos como Markdown.
def tabla_md(df):
    def f(v):
        if isinstance(v,str):return v.replace('|',chr(92)+'|')
        if isinstance(v,(int,np.integer)):return str(v)
        return f'{v:.6f}'
    lines=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    lines += ['| '+' | '.join(f(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(lines)
rankcells=[]
for segmento,b in bss['busquedas'].items():
    rankcells.append(new('markdown',f'**{segmento}: ranking de candidatas calculado solo con subtrain**\n\n'+tabla_md(b['ranking'])))
indice_rank=next(i for i,c in enumerate(model['cells']) if c['cell_type']=='markdown' and src(c).startswith('**Resultados de selección y validación.'))
model['cells'][indice_rank+1:indice_rank+1]=rankcells

# Auditoría exhaustiva de los 27 archivos, antes de eliminarlos.
keep={'train_ols_usadas.csv','test_ols_usadas.csv','metadatos_train_ols_usadas.csv','metadatos_test_ols_usadas.csv',
      'transformadores_ols_usadas.joblib','modelos_ols_por_segmento.joblib','modelos_ols_sin_to_center_km.joblib','bss_features/modelos_bss_features.joblib'}
rows=[]
for f in oldfiles:
    rel=f.relative_to(DATA).as_posix()
    if rel in ['train_ols_usadas.csv','test_ols_usadas.csv']:
        funcion='Dataset numérico de modelado; no se sobrescribe si ya existe.';deps='Preparacion_OLS_Usadas; Modelo_OLS';decision='Conservar'
    elif rel.startswith('metadatos_'):
        funcion='ID y fecha por fila: correspondencia y validación temporal interna.';deps='Preparacion_OLS_Usadas; Modelo_OLS/BSS';decision='Conservar'
    elif rel=='transformadores_ols_usadas.joblib':
        funcion='Codificador y parámetros de preparación; única fuente de imputación y esquema.';deps='Preparacion_OLS_Usadas; Modelo_OLS/BSS';decision='Conservar y retirar auditoría redundante interna'
    elif rel=='modelos_ols_por_segmento.joblib':
        funcion='OLS originales y única configuración externa de segmentación y Box-Cox.';deps='Modelo_OLS y etapas posteriores';decision='Conservar, comprimir sin pérdida'
    elif rel=='modelos_ols_sin_to_center_km.joblib':
        funcion='OLS del experimento y columnas necesarias para recuperar predicciones.';deps='Modelo_OLS/BSS';decision='Conservar modelos; retirar tablas y configuración duplicadas'
    elif rel=='bss_features/modelos_bss_features.joblib':
        funcion='Modelos finales y de validación, selección congelada y preparación interna aprendida.';deps='Modelo_OLS/BSS';decision='Conservar; retirar métricas, rankings y permutaciones persistidas'
    elif rel.endswith('.json'):
        funcion='Auditoría o configuración ya disponible en transformadores/modelos Joblib o derivable de ellos.';deps='Lecturas migradas a Joblib o contratos en memoria';decision='Eliminar después de resolver dependencias'
    else:
        funcion='Resultados BSS intermedios (subconjuntos, rankings o permutaciones); regenerables.';deps='Modelo_OLS; resultados relevantes conservados en Markdown y gráficos';decision='Eliminar y desactivar exportación'
    rows.append({'Archivo (data/)':rel,'Función':funcion,'Dependencias resueltas':deps,'Decisión':decision})
audit=new('markdown','''## 16. Archivos necesarios y política de persistencia

Auditoría realizada antes de retirar archivos. Se conservan cuatro CSV necesarios y cuatro Joblib. Los parámetros de preparación tienen una única fuente en el transformador; los cuartiles y λ externos viven en el OLS original. Los demás modelos no duplican esa configuración. Los modelos internos BSS se conservan porque permiten reproducir la validación y sus permutaciones sin entrenar nuevamente; los λ internos y el preparador de subtrain se mantienen separados de la preparación final.

La ejecución recupera y valida las matrices de los modelos existentes. Si no existen artefactos, el flujo permite construirlos; durante esta reorganización se reutilizan todos los entrenados, sin cambiar coeficientes. Recuperar BSS no repite su búsqueda: las selecciones permanecen congeladas, los diagnósticos se recalculan desde los modelos y los resultados históricos siguen en las celdas Markdown.

**Regla para próximas etapas:** no generar archivos de auditoría, métricas, rankings o resultados intermedios salvo que sean indispensables para una etapa posterior. Notebooks para análisis y documentación; CSV para datasets necesarios; Joblib para modelos y transformadores. JSON solo si una configuración indispensable no está ya en un Joblib.

'''+tabla_md(pd.DataFrame(rows)))
model['cells'].append(audit)

# Compilar todo y escribir notebooks con dependencias ya resueltas antes de retirar archivos.
for n in [prep,model]:
    for c in n['cells']:
        if c['cell_type']=='code':compile(src(c),'<notebook>','exec')
PREP.write_text(json.dumps(prep,ensure_ascii=False,indent=1),encoding='utf-8')
MODEL.write_text(json.dumps(model,ensure_ascii=False,indent=1),encoding='utf-8')

# Migrar solo la representación persistida; no se llama a fit ni se cambian parámetros.
joblib.dump({'registro':registro_minimo,'codificador':transformador['codificador']},DATA/'transformadores_ols_usadas.joblib',compress=3)
joblib.dump(original,DATA/'modelos_ols_por_segmento.joblib',compress=3)
joblib.dump({'modelos':sincentro['modelos'],'registros':sincentro['registros'],'predictor_excluido':'to_center_km'},DATA/'modelos_ols_sin_to_center_km.joblib',compress=3)
bss_minimo={
    'modelos_finales':bss['modelos_finales'],
    'modelos_validacion':{s:b['modelo_interno'] for s,b in bss['busquedas'].items()},
    'seleccion':{s:{'features':list(b['seleccionadas']),'lambda_interno':b['lambda_interno'],
                   'candidatas_totales':b['candidatas_totales'],'candidatas_restringidas':b['candidatas_restringidas']}
                 for s,b in bss['busquedas'].items()},
    'preparador_validacion':{k:bss['preparador_interno'][k] for k in ['imputacion','binarias','encoder','mapa']},
    'configuracion':{'corte':32000,'max_candidatas':8,'tolerancia_rmse':.01,'semilla':20261028,'repeticiones':30},
    'dependencias':{'preparacion':'../transformadores_ols_usadas.joblib','segmentacion':'../modelos_ols_por_segmento.joblib'}}
joblib.dump(bss_minimo,DATA/'bss_features/modelos_bss_features.joblib',compress=3)
for file,old,key in [('modelos_ols_por_segmento.joblib',original,'modelos'),('modelos_ols_sin_to_center_km.joblib',sincentro,'modelos'),('bss_features/modelos_bss_features.joblib',bss,'modelos_finales')]:
    loaded=joblib.load(DATA/file)
    for s,m in old[key].items():
        np.testing.assert_array_equal(loaded[key][s].params,m.params)
        np.testing.assert_array_equal(loaded[key][s].fittedvalues,m.fittedvalues)
for s,b in bss['busquedas'].items():
    assert bss_minimo['seleccion'][s]['features']==b['seleccionadas']
assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in hashes.items())

# Borrar exclusivamente archivos auditados bajo la ruta absoluta comprobada.
limite=DATA.resolve()
eliminar=[p for p in oldfiles if p.relative_to(DATA).as_posix() not in keep]
assert len(eliminar)==19
for f in eliminar:
    destino=f.resolve()
    assert destino.is_relative_to(limite) and destino.suffix in ['.csv','.json']
    f.unlink()
print('Conservados',len(keep),'archivos indispensables; eliminados',len(eliminar),'auxiliares.')
print('Coeficientes, predicciones ajustadas, selecciones y CSV protegidos: idénticos.')

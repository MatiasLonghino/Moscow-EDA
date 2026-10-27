"""Forward OLS agrupado; importar no entrena ni registra modelos."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from . import baseline_ols as b
EXCLUIDAS = ('to_center_km', 'living_area', 'rooms')
NOMBRE_RUN = 'Forward_01_Log_Grupos'
REFERENCIA = 'OLS_05_Log_Sin_rooms'


def recuperar_experimento(cliente, *, nombre=None, run_id=None,
                         experimento=b.EXPERIMENTO, parametros=None, etiquetas=None):
    """Recupera todas las métricas remotas; ante ambigüedad exige un ID explícito."""
    exp = cliente.get_experiment_by_name(experimento)
    if exp is None or exp.lifecycle_stage != 'active' or not (nombre or run_id):
        raise ValueError('Se requiere experimento existente activo y nombre o Run ID.')
    if run_id:
        candidatos = [cliente.get_run(run_id)]
    else:
        candidatos, token = [], None
        while True:
            pagina = cliente.search_runs([exp.experiment_id], max_results=1000, page_token=token)
            candidatos.extend(r for r in pagina if r.data.tags.get('mlflow.runName') == nombre)
            token = pagina.token
            if not token:
                break
    validos = [r for r in candidatos if r.info.experiment_id == exp.experiment_id
               and r.info.status == 'FINISHED'
               and (nombre is None or r.data.tags.get('mlflow.runName') == nombre)
               and all(r.data.params.get(k) == str(v) for k, v in (parametros or {}).items())
               and all(r.data.tags.get(k) == str(v) for k, v in (etiquetas or {}).items())]
    if len(validos) != 1:
        detalle = [{'run_id': r.info.run_id, 'status': r.info.status, 'params': r.data.params}
                   for r in candidatos]
        raise ValueError(f'Referencia inexistente, incompatible o ambigua; indicar Run ID: {detalle}')
    r = validos[0]
    return {'run_id': r.info.run_id, 'experiment_id': exp.experiment_id,
            'run_name': r.data.tags['mlflow.runName'], 'status': r.info.status,
            'metrics': dict(r.data.metrics), 'params': dict(r.data.params), 'tags': dict(r.data.tags)}


def grupos_caracteristicas(columnas, resumen):
    """Linaje real: bloques One-Hot e indicador de imputación junto con su origen.

    months_since_start corresponde a date_posted_dt; derivadas multientrada
    como surface_corrected conservan su identidad exportada.
    """
    columnas = list(columnas)
    if len(columnas) != len(set(columnas)) or set(EXCLUIDAS) & set(columnas):
        raise ValueError('Columnas duplicadas o exclusiones no respetadas.')
    mapa = {c: g['variable'] for g in resumen['categorias'] for c in g['columnas']}
    grupos = {}
    for c in columnas:
        info = resumen['linaje'][c]
        if info['tipo'] == 'onehot' and c not in mapa:
            raise ValueError(f'Faltan metadatos One-Hot: {c}')
        origen = info['entradas'][0] if len(info['entradas']) == 1 else info['variable']
        grupos.setdefault(mapa.get(c, origen), []).append(c)
    for g in resumen['categorias']:
        presentes = set(g['columnas']) & set(columnas)
        if presentes and (presentes != set(g['columnas']) or
                          g['referencia'] not in g['categorias'] or
                          len(g['columnas']) != len(g['categorias'])-1):
            raise ValueError(f"One-Hot parcial o sin referencia compatible: {g['variable']}")
    assert len(columnas) == sum(map(len, grupos.values()))
    assert set(columnas) == {c for cols in grupos.values() for c in cols}
    return grupos


def columnas_grupos(grupos, seleccionadas):
    if len(set(seleccionadas)) != len(seleccionadas):
        raise ValueError('Grupos repetidos.')
    return [c for g in seleccionadas for c in grupos[g]]


def forward_grupos(X, y_log, y_original, grupos, fit, val, *,
                   min_mejora_pct=0.5, max_features=12):
    """Una partición fija; OLS pinv con intercepto; exp sin corrección en validación."""
    if (not isinstance(max_features, int) or not 1 <= max_features <= 12 or
            not np.isfinite(min_mejora_pct) or min_mejora_pct < 0):
        raise ValueError('Mejora mínima no negativa y máximo absoluto de 12 características.')
    assert X.index.equals(y_log.index) and X.index.equals(y_original.index)
    np.testing.assert_allclose(y_log, np.log(y_original), rtol=1e-12)
    assert not set(fit) & set(val) and set(fit) | set(val) == set(range(len(X)))
    assert len(fit) > 1 and len(val) > 1
    assert len(X.columns) == sum(map(len, grupos.values()))
    assert set(X) == set(columnas_grupos(grupos, list(grupos)))
    posiciones = {c: i+1 for i, c in enumerate(X)}
    ids = {g: [posiciones[c] for c in cols] for g, cols in grupos.items()}
    matriz = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    ajuste, validacion = matriz[fit], matriz[val]
    objetivo, real = y_log.to_numpy()[fit], y_original.to_numpy()[val]
    seleccionadas, historial, candidatos = [], [], []
    evaluados = 0

    def evaluar(nombres):
        indices = [0] + [i for g in nombres for i in ids[g]]
        modelo = b.sm.OLS(objetivo, ajuste[:, indices], missing='raise').fit()
        with np.errstate(over='ignore'):
            pred = np.exp(modelo.predict(validacion[:, indices]))
            error = np.sqrt(np.mean(np.square(real-pred)))
        return float(error) if np.isfinite(error) else np.inf, int(modelo.model.rank)

    with threadpool_limits(limits=1):
        actual, _ = evaluar([])
        inicial = actual
        motivo = 'candidatas_agotadas'
        for iteracion in range(1, min(max_features, len(grupos))+1):
            mejores = []
            for g in grupos:
                if g in seleccionadas:
                    continue
                score, rango = evaluar(seleccionadas+[g])
                evaluados += 1
                mejores.append((score, g))
                candidatos.append({'iteracion': iteracion, 'candidata': g,
                    'rmse_validation_original': score if np.isfinite(score) else None,
                    'rango_efectivo': rango})
            score, g = min(mejores, key=lambda par: par[0])  # Empates: orden del linaje.
            mejora = 100*(actual-score)/actual if actual > 0 else -np.inf
            aceptar = bool(np.isfinite(score) and mejora >= min_mejora_pct)
            acumuladas = seleccionadas+[g] if aceptar else seleccionadas.copy()
            historial.append({'iteracion': iteracion, 'caracteristica_incorporada': g if aceptar else None,
                'mejor_candidata': g, 'aceptada': aceptar, 'caracteristicas_acumuladas': acumuladas,
                'rmse_validacion': score if aceptar else actual,
                'rmse_mejor_candidato': score if np.isfinite(score) else None,
                'mejora_pct': mejora if np.isfinite(mejora) else None,
                'n_predictores': len(columnas_grupos(grupos, acumuladas))})
            if not aceptar:
                motivo = 'min_mejora_pct'
                break
            seleccionadas, actual = acumuladas, score
            if len(seleccionadas) == max_features:
                motivo = 'max_features'
    return {'seleccionadas': seleccionadas, 'historial': historial, 'candidatos': candidatos,
            'rmse_validation_original': actual, 'rmse_intercepto_original': inicial,
            'motivo_parada': motivo, 'iteraciones': len(historial), 'candidatos_evaluados': evaluados}


def tabla_comparativa(referencia, metricas, invalidas=(), *,
                      nombre_referencia=REFERENCIA, nombre_run=NOMBRE_RUN):
    """Solo Train comparable; N/D sin derivar ni completar métricas históricas."""
    etiquetas = {'train_n_features_original': 'Cantidad de características',
                 'train_n_predictors': 'Cantidad de predictores',
                 'train_condition_number': 'Condition Number'}
    filas = []
    for k in dict.fromkeys([*metricas, *invalidas, *referencia]):
        if not k.startswith('train_'):
            continue
        etiqueta = etiquetas.get(k)
        if etiqueta is None:
            dummy = b.tabla_resultados({k: 0.0}, {}, {'breusch_pagan': 'N/D'}, nombre_run=NOMBRE_RUN)
            etiqueta = dummy.splitlines()[2].split('|')[1].strip()
        anterior, nuevo = referencia.get(k), metricas.get(k)
        a = anterior is not None and np.isfinite(anterior)
        n = nuevo is not None and np.isfinite(nuevo)
        filas.append({'Métrica': etiqueta, nombre_referencia: repr(anterior) if a else 'N/D',
                      nombre_run: repr(nuevo) if n else 'N/D',
                      'Δ': repr(nuevo-anterior) if a and n else 'N/D'})
    return b.tabla_markdown(pd.DataFrame(filas))


def preparar_experimento(base, cliente, *, reference_run_id=None, carpeta_salida):
    """Recupera y verifica la referencia ANTES del primer ajuste Forward."""
    base = Path(base)
    X, y, contexto = b.cargar_train(base)
    X = X.drop(columns=list(EXCLUIDAS))
    ref = recuperar_experimento(cliente, nombre=REFERENCIA, run_id=reference_run_id,
        parametros={'target_transform': 'ln(Target)', 'inverse_transform': 'exp',
                    'bias_correction': 'none', 'train_fraction': '1.0', 'n_train': len(y),
                    'intercepto': 'True', 'cov_type': 'nonrobust', 'solver': 'pinv',
                    'excluded_predictors': ','.join(EXCLUIDAS),
                    'aic_bic_convention': 'statsmodels_gaussian_log_RUB_k_rank'},
        etiquetas={'dataset_sha256': contexto['dataset_sha256']})
    out = b.validar_temporal(carpeta_salida)
    out.mkdir(exist_ok=True)
    ref_dir = out/'referencia_modelo'
    ref_dir.mkdir(exist_ok=True)
    ctx = json.loads(Path(cliente.download_artifacts(ref['run_id'], 'contexto.json', str(ref_dir))).read_text(encoding='utf-8'))
    assert ctx['predictoras'] == list(X) and ref['metrics']['train_n_obs'] == len(y)
    assert ctx['dataset_sha256'] == contexto['dataset_sha256'] and ctx['transformacion_objetivo'] == 'ln'
    from .almacenamiento_mlflow import leer_json, RUN_TRANSFORMACION
    resumen = leer_json(cliente, RUN_TRANSFORMACION, 'transformacion/resumen_final.json')
    assert resumen['columnas_X'] == contexto['predictoras']
    grupos = grupos_caracteristicas(X.columns, resumen)
    for g in resumen['categorias']:
        bloque = X[g['columnas']]
        assert bloque.isin([0, 1]).all().all() and bloque.sum(axis=1).le(1).all()
    contexto.update(transformacion_objetivo='ln', unidad='ln(RUB)', reference_run_id=ref['run_id'],
                    predictoras_excluidas=list(EXCLUIDAS), seleccion_caracteristicas='forward_stepwise_grouped')
    b.guardar_json(out/'referencia_mlflow.json', ref)
    b.guardar_json(out/'comparabilidad.json', {'dataset_sha256': contexto['dataset_sha256'],
        'reference_run_id': ref['run_id'], 'mismas_filas_train': True, 'target': 'ln(Target)',
        'inversa': 'exp sin corrección', 'aic_bic': 'statsmodels_gaussian_log_RUB_k_rank',
        'evaluacion': 'Train completo; validación de selección separada', 'metricas_historicas': 'MLflow exclusivamente'})
    b.guardar_json(out/'grupos_caracteristicas.json', {'grupos': grupos, 'categorias': resumen['categorias'],
        'linaje': {c: resumen['linaje'][c] for c in X}})
    return X, y, contexto, ref, grupos, out


def ejecutar_experimento(base, cliente, *, reference_run_id=None,
                        min_mejora_pct=0.5, max_features=12, random_state=42, carpeta_salida=None):
    if carpeta_salida is None:
        from .almacenamiento_mlflow import SesionTemporal
        with SesionTemporal() as sesion:
            resultado = ejecutar_experimento(base, cliente, reference_run_id=reference_run_id,
                min_mejora_pct=min_mejora_pct, max_features=max_features, random_state=random_state,
                carpeta_salida=sesion.carpeta('forward'))
            resultado.pop('salida')  # El llamador recibe resultados en memoria, nunca una ruta borrada.
            return resultado
    X, y, contexto, ref, grupos, out = preparar_experimento(base, cliente, reference_run_id=reference_run_id, carpeta_salida=carpeta_salida)
    y_log = np.log(y).rename('ln_price_rub')
    fit, val = train_test_split(np.arange(len(y)), test_size=0.2, random_state=random_state)
    np.savez_compressed(out/'particion_interna.npz', fit=fit, validation=val)
    seleccion = forward_grupos(X, y_log, y, grupos, fit, val,
                               min_mejora_pct=min_mejora_pct, max_features=max_features)
    columnas = columnas_grupos(grupos, seleccion['seleccionadas'])
    modelo, diseno = b.ajustar(X[columnas], y_log)
    assert modelo.model.rank == diseno.shape[1], 'Diseño final no identificable; no registrar.'
    assert len(y) == modelo.nobs and len(seleccion['seleccionadas']) <= 12
    metricas, invalidas, vif, notas = b.diagnosticar(modelo, diseno, y_log, y_original=y, resumir_vif=True)
    metricas.update(train_n_features_original=float(len(seleccion['seleccionadas'])),
                    selection_rmse_validation_original=seleccion['rmse_validation_original'],
                    selection_rmse_intercept_original=seleccion['rmse_intercepto_original'],
                    selection_iterations=float(seleccion['iteraciones']),
                    selection_candidates_evaluated=float(seleccion['candidatos_evaluados']))
    contexto.update(predictoras=columnas, n_predictoras=len(columnas),
                    caracteristicas_originales=seleccion['seleccionadas'])
    b.exportar(base, modelo, diseno, y_log, contexto, metricas, invalidas, vif, notas,
               nombre_run=NOMBRE_RUN, carpeta_salida=out)
    tabla = tabla_comparativa(ref['metrics'], metricas, invalidas)
    (out/'comparacion_referencia.md').write_text(tabla, encoding='utf-8')
    b.guardar_json(out/'seleccion.json', seleccion)
    b.guardar_json(out/'historial_seleccion.json', seleccion['historial'])
    b.guardar_json(out/'caracteristicas_seleccionadas.json', {g: grupos[g] for g in seleccion['seleccionadas']})
    parametros = {'algorithm': 'Forward Stepwise Selection agrupado', 'min_mejora_pct': min_mejora_pct,
        'max_features': max_features, 'random_state': random_state, 'validation_fraction': 0.2,
        'selection_metric': 'RMSE validation original RUB; exp sin corrección',
        'selection_stop_reason': seleccion['motivo_parada'], 'selection_iterations': seleccion['iteraciones'],
        'n_features_original': len(seleccion['seleccionadas']),
        'selected_features': json.dumps(seleccion['seleccionadas']),
        'feature_order': json.dumps(seleccion['seleccionadas']),
        'reference_run_name': REFERENCIA, 'reference_run_id': ref['run_id'],
        'experiment_id': ref['experiment_id'], 'notebook': '4_Seleccion_Modelos.ipynb'}
    b.guardar_json(out/'configuracion.json', parametros)
    # Evita elegir arbitrariamente el primero de varios Runs en el registrador histórico.
    existentes = cliente.search_runs([ref['experiment_id']],
        filter_string=f"tags.mlflow.runName = '{NOMBRE_RUN}'", max_results=2)
    if len(existentes) > 1 or (existentes and existentes[0].data.tags.get('dataset_sha256') != contexto['dataset_sha256']):
        raise ValueError('Runs Forward ambiguos; resolver sus IDs antes de registrar.')
    registro = b.registrar(cliente, out, contexto, metricas, nombre_run=NOMBRE_RUN, parametros_extra=parametros)
    remoto = cliente.get_run(registro['run_id'])
    assert remoto.data.params['reference_run_id'] == ref['run_id']
    assert remoto.data.params['notebook'] == '4_Seleccion_Modelos.ipynb'
    assert all(remoto.data.metrics[k] == v for k, v in metricas.items())
    despues = cliente.get_run(ref['run_id'])
    assert despues.data.metrics == ref['metrics'] and despues.data.params == ref['params']
    return {'salida': out, 'comparacion_md': tabla, 'registro': registro,
            'seleccion': seleccion, 'metricas': metricas}


def restaurar_forward(cliente, run, salida, contexto_train, referencia):
    """Recupera evidencia del Run terminado y comprueba las copias locales existentes."""
    from tempfile import TemporaryDirectory
    out = b.validar_temporal(salida)
    out.mkdir(exist_ok=True)
    assert run['params']['reference_run_id'] == referencia['run_id']
    assert run['tags']['dataset_sha256'] == contexto_train['dataset_sha256']
    assert run['params']['target_transform'] == 'ln(Target)'
    assert run['params']['inverse_transform'] == 'exp' and run['params']['bias_correction'] == 'none'
    assert run['params']['excluded_predictors'] == ','.join(EXCLUIDAS)
    nombres = ['contexto.json', 'seleccion.json', 'grupos_caracteristicas.json', 'metricas.json',
               'particion_interna.npz', 'modelo_statsmodels.pkl', 'comparacion_referencia.md',
               *['diagnosticos_residuales/'+n for n in b.ARCHIVOS]]
    with TemporaryDirectory(prefix='forward_evidencia_') as temporal:
        for nombre in nombres:
            remoto = Path(cliente.download_artifacts(run['run_id'], nombre, temporal))
            local = out/nombre
            if local.exists():
                assert b.sha256(local) == b.sha256(remoto), f'Artefacto local distinto: {nombre}'
            else:
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(remoto, local)
    leer = lambda nombre: json.loads((out/nombre).read_text(encoding='utf-8'))
    estado = {k: leer(k+'.json') for k in ['contexto', 'seleccion', 'grupos_caracteristicas', 'metricas']}
    assert estado['metricas'] == run['metrics']
    assert estado['contexto']['dataset_sha256'] == contexto_train['dataset_sha256']
    assert run['metrics']['train_n_obs'] == contexto_train['n_train']
    assert estado['seleccion']['seleccionadas'] == json.loads(run['params']['selected_features'])
    return estado


def resumen_caracteristicas(seleccionadas, grupos, modelo, *, mostrar_orden=True):
    """Una fila por característica original; cuenta predictores sin el intercepto."""
    columnas = columnas_grupos(grupos, seleccionadas)
    assert modelo.model.exog_names == ['const']+columnas
    efectivos = int(modelo.model.rank)-1
    assert efectivos == len(columnas), 'La tabla requiere un diseño final de rango completo.'
    filas = [{'Orden': i, 'Característica seleccionada': f'`{g}`',
              'Cantidad de predictores': len(grupos[g])} for i, g in enumerate(seleccionadas, 1)]
    filas.append({'Orden': 'Total', 'Característica seleccionada': f'{len(seleccionadas)} características',
                  'Cantidad de predictores': f'{efectivos} predictores'})
    if not mostrar_orden:
        filas[-1]['Característica seleccionada'] = f'Total: {len(seleccionadas)} características'
    tabla = pd.DataFrame(filas)
    return b.tabla_markdown(tabla if mostrar_orden else tabla.drop(columns='Orden'))


def importancia_permutacion_grupos(X, y_original, grupos, seleccionadas, fit, val, *,
                                   n_repeticiones=5, random_state=42, rmse_esperado=None):
    """Un ajuste interno; cada repetición permuta conjuntamente todas las columnas del grupo.

    Importancia descriptiva sobre la validación usada para seleccionar. No modifica
    el modelo final ni reajusta entre permutaciones. La correlación puede afectarla.
    """
    assert isinstance(n_repeticiones, int) and n_repeticiones > 0 and len(seleccionadas) > 0
    assert X.index.equals(y_original.index)
    assert not set(fit) & set(val) and set(fit) | set(val) == set(range(len(X)))
    assert len(fit)+len(val) == len(X)
    columnas = columnas_grupos(grupos, seleccionadas)
    modelo, _ = b.ajustar(X[columnas].iloc[fit], np.log(y_original.iloc[fit]))
    validacion = b.sm.add_constant(X[columnas].iloc[val], has_constant='add').to_numpy()
    real = y_original.iloc[val].to_numpy()
    rng, filas = np.random.default_rng(random_state), []
    posiciones = {c: i+1 for i, c in enumerate(columnas)}
    with threadpool_limits(limits=1):
        rmse = float(np.sqrt(np.mean((real-np.exp(modelo.predict(validacion)))**2)))
        assert np.isfinite(rmse) and rmse > 0
        if rmse_esperado is not None:
            np.testing.assert_allclose(rmse, rmse_esperado, rtol=1e-10, atol=1e-6)
        for g in seleccionadas:
            indices = [posiciones[c] for c in grupos[g]]
            permutada = validacion.copy()
            for repeticion in range(1, n_repeticiones+1):
                orden = rng.permutation(len(val))
                permutada[:, indices] = validacion[orden][:, indices]
                pred = np.exp(modelo.predict(permutada))
                score = float(np.sqrt(np.mean((real-pred)**2)))
                assert np.isfinite(score)
                filas.append({'caracteristica': g, 'repeticion': repeticion,
                    'rmse_original': rmse, 'rmse_permutado': score,
                    'incremento_rmse': score-rmse, 'importancia_pct': 100*(score-rmse)/rmse})
    return pd.DataFrame(filas)


def guardar_boxplot_importancia(importancias, archivo):
    b.validar_temporal(Path(archivo).parent)
    from matplotlib.figure import Figure
    orden = importancias.groupby('caracteristica')['importancia_pct'].median().sort_values(
        ascending=False, kind='stable').index.tolist()
    valores = [importancias.loc[importancias.caracteristica.eq(g), 'importancia_pct'].to_numpy() for g in orden]
    fig = Figure(figsize=(10, max(4, .55*len(orden)+1.6)), layout='constrained')
    ax = fig.subplots()
    ax.boxplot(valores, vert=False, tick_labels=orden, patch_artist=True,
               boxprops={'facecolor': '#b9dce9'}, medianprops={'color': '#b12c2c', 'linewidth': 1.7})
    ax.invert_yaxis()
    ax.axvline(0, color='#555555', linestyle='--', linewidth=1)
    ax.set(title='Permutation Importance agrupada · Validación interna de Train',
           xlabel='Incremento del RMSE (%)', ylabel='Característica original')
    ax.grid(axis='x', alpha=.2)
    fig.savefig(archivo, dpi=160)


def registrar_presentacion(cliente, run, salida, archivos):
    """Agrega y verifica artefactos al mismo Run, conservando métricas, parámetros y modelo."""
    from tempfile import TemporaryDirectory
    antes = cliente.get_run(run['run_id'])
    assert antes.info.status == 'FINISHED' and antes.data.metrics == run['metrics']
    for nombre in archivos:
        cliente.log_artifact(run['run_id'], str(Path(salida)/nombre))
    with TemporaryDirectory(prefix='forward_presentacion_') as temporal:
        for nombre in archivos:
            remoto = cliente.download_artifacts(run['run_id'], nombre, temporal)
            assert b.sha256(remoto) == b.sha256(Path(salida)/nombre), nombre
        modelo = cliente.download_artifacts(run['run_id'], 'modelo_statsmodels.pkl', temporal)
        assert b.sha256(modelo) == b.sha256(Path(salida)/'modelo_statsmodels.pkl')
    despues = cliente.get_run(run['run_id'])
    assert despues.info.status == 'FINISHED'
    assert despues.data.metrics == antes.data.metrics and despues.data.params == antes.data.params

"""Forward OLS agrupado; importar no entrena ni registra modelos."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import baseline_ols as b

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


def tabla_comparativa(referencia, metricas, invalidas=()):
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
        filas.append({'Métrica': etiqueta, REFERENCIA: repr(anterior) if a else 'N/D',
                      NOMBRE_RUN: repr(nuevo) if n else 'N/D',
                      'Δ': repr(nuevo-anterior) if a and n else 'N/D'})
    return b.tabla_markdown(pd.DataFrame(filas))


def preparar_experimento(base, cliente, *, reference_run_id=None):
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
    out = base/'salidas_forward_01_log_grupos'
    out.mkdir(exist_ok=True)
    ref_dir = out/'referencia_modelo'
    ref_dir.mkdir(exist_ok=True)
    ctx = json.loads(Path(cliente.download_artifacts(ref['run_id'], 'contexto.json', str(ref_dir))).read_text(encoding='utf-8'))
    assert ctx['predictoras'] == list(X) and ref['metrics']['train_n_obs'] == len(y)
    assert ctx['dataset_sha256'] == contexto['dataset_sha256'] and ctx['transformacion_objetivo'] == 'ln'
    resumen = json.loads((base/'salidas_transformacion/resumen_final.json').read_text(encoding='utf-8'))
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
                        min_mejora_pct=0.5, max_features=12, random_state=42):
    X, y, contexto, ref, grupos, out = preparar_experimento(base, cliente, reference_run_id=reference_run_id)
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
    shutil.copyfile(Path(__file__), out/Path(__file__).name)
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

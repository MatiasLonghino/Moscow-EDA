"""Group Lasso ponderado para seleccionar bloques; el estimador final es OLS sin penalizar.

Objetivo: ||y_c-Zb||²/(2n) + lambda * sum_g sqrt(rank(Z_g))*||b_g||₂.
Z se estandariza exclusivamente en fit; intercepto no penalizado mediante centrado.
FISTA opera sobre Gram (p x p), con reinicio adaptativo y control de estacionariedad.
Referencia: https://group-lasso.readthedocs.io/en/latest/maths.html
"""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from . import baseline_ols as b
from . import seleccion_modelos as s
NOMBRE_RUN = 'GroupLasso_02_Log_Grupos_OLS'
CONFIG = {'min_mejora_pct': .5, 'max_features': 12, 'support_tol': 1e-6,
          'solver_tol': 1e-7, 'max_iter': 20000, 'n_lambdas': 40, 'lambda_min_ratio': .001}


def resolver_group_lasso(gram, xy, ids, pesos, lam, *, inicial=None, tol=1e-7, max_iter=20000):
    """Proximal por bloques; devuelve solo soluciones convergentes y su residuo KKT."""
    L = float(np.linalg.eigvalsh(gram)[-1])
    if not np.isfinite(lam) or lam <= 0 or L <= 0:
        raise ValueError('Lambda y Lipschitz deben ser positivos.')
    beta = np.zeros(len(xy)) if inicial is None else inicial.copy()
    z, t = beta.copy(), 1.
    umbrales = lam*pesos/L
    def prox(v):
        normas = np.sqrt(np.bincount(ids, weights=v*v, minlength=len(pesos)))
        factores = np.maximum(0., 1-umbrales/np.maximum(normas, np.finfo(float).tiny))
        return v*factores[ids]
    escala = max(float(np.linalg.norm(xy, ord=np.inf)), 1e-12)
    for it in range(1, max_iter+1):
        nuevo = prox(z-(gram@z-xy)/L)
        residuo = float(L*np.linalg.norm(nuevo-prox(nuevo-(gram@nuevo-xy)/L), ord=np.inf))
        if residuo <= tol*escala:
            return nuevo, {'iteraciones_solver': it, 'residuo_kkt': residuo,
                           'residuo_kkt_relativo': residuo/escala, 'convergencia': True}
        tn = (1+np.sqrt(1+4*t*t))/2
        if np.dot(z-nuevo, nuevo-beta) > 0:
            z, t = nuevo.copy(), 1.
        else:
            z, t = nuevo+(t-1)/tn*(nuevo-beta), tn
        beta = nuevo
    raise RuntimeError(f'Group Lasso no convergió: lambda={lam}, residuo relativo={residuo/escala:g}')


def elegir_por_complejidad(candidatos, rmse_intercepto, *, min_mejora_pct=.5, max_features=12):
    """Evalúa todos los tamaños disponibles; un rechazo no detiene tamaños posteriores."""
    if not 1 <= max_features <= 12 or min_mejora_pct < 0 or not np.isfinite(min_mejora_pct):
        raise ValueError('Criterios de selección inválidos.')
    aceptado = {'seleccionadas': [], 'lambda': None, 'rmse_validation_original': rmse_intercepto}
    decisiones = []
    for k in sorted({len(c['seleccionadas']) for c in candidatos if 0 < len(c['seleccionadas']) <= max_features}):
        candidato = min((c for c in candidatos if len(c['seleccionadas']) == k),
                        key=lambda c: c['rmse_validation_original'])
        mejora = 100*(aceptado['rmse_validation_original']-candidato['rmse_validation_original'])/aceptado['rmse_validation_original']
        aceptar = bool(np.isfinite(mejora) and mejora >= min_mejora_pct)
        decisiones.append({'n_caracteristicas': k, 'lambda': candidato['lambda'],
            'seleccionadas': candidato['seleccionadas'], 'rmse_referencia': aceptado['rmse_validation_original'],
            'rmse_validation_original': candidato['rmse_validation_original'],
            'mejora_pct': mejora, 'aceptado': aceptar})
        if aceptar:
            aceptado = candidato.copy()
    return aceptado, decisiones


def buscar_group_lasso(X, y, grupos, fit, val, salida, *, lambdas=None, config=None,
                       excluidas=s.EXCLUIDAS):
    cfg = {**CONFIG, **(config or {})}
    if not 1 <= cfg['max_features'] <= 12 or cfg['min_mejora_pct'] < 0:
        raise ValueError('Máximo absoluto 12 y mejora mínima no negativa.')
    assert X.index.equals(y.index) and not set(excluidas) & set(X)
    assert not set(fit) & set(val) and len(fit)+len(val) == len(X)
    assert set(fit) | set(val) == set(range(len(X)))
    assert set(s.columnas_grupos(grupos, list(grupos))) == set(X)
    out = b.validar_temporal(salida)
    out.mkdir(exist_ok=True)
    nombres = list(grupos)
    posiciones = {c: i for i, c in enumerate(X)}
    bloques = [np.array([posiciones[c] for c in grupos[g]]) for g in nombres]
    ids = np.empty(X.shape[1], dtype=int)
    for i, bloque in enumerate(bloques): ids[bloque] = i
    scaler = StandardScaler().fit(X.iloc[fit])
    Z = scaler.transform(X.iloc[fit])
    # Se aplica el mismo escalador en validación; OLS provisional usa las escalas exportadas.
    assert np.isfinite(scaler.transform(X.iloc[val])).all()
    y_log = np.log(y)
    yc = y_log.iloc[fit].to_numpy()-y_log.iloc[fit].mean()
    historial, candidatos, cache, coeficientes = [], [], {}, []
    with threadpool_limits(limits=1):
        gram, xy = Z.T@Z/len(fit), Z.T@yc/len(fit)
        rangos = [int(np.linalg.matrix_rank(gram[np.ix_(i, i)], tol=1e-10)) for i in bloques]
        assert all(r == len(i) for r, i in zip(rangos, bloques)), 'Grupo interno no identificable.'
        assert np.linalg.matrix_rank(gram, tol=1e-10) == X.shape[1], 'Diseño interno no identificable.'
        pesos = np.sqrt(rangos)
        lambda_max = max(np.linalg.norm(xy[i])/w for i, w in zip(bloques, pesos))
        grilla = np.asarray(lambdas if lambdas is not None else
            lambda_max*np.geomspace(1., cfg['lambda_min_ratio'], cfg['n_lambdas']), dtype=float)
        if grilla.ndim != 1 or not len(grilla) or not np.isfinite(grilla).all() or (grilla <= 0).any():
            raise ValueError('Definir una grilla finita de lambdas positivos.')
        grilla = np.sort(np.unique(grilla))[::-1]
        beta = np.zeros(X.shape[1])
        interno = np.column_stack([np.ones(len(fit)), X.to_numpy()[fit]])
        validacion = np.column_stack([np.ones(len(val)), X.to_numpy()[val]])
        real = y.to_numpy()[val]
        def evaluar(seleccion):
            indices = [0]+[posiciones[c]+1 for g in seleccion for c in grupos[g]]
            modelo = b.sm.OLS(y_log.iloc[fit].to_numpy(), interno[:, indices]).fit()
            assert modelo.model.rank == len(indices), 'OLS provisional de rango deficiente.'
            score = float(np.sqrt(np.mean((real-np.exp(modelo.predict(validacion[:, indices])))**2)))
            assert np.isfinite(score)
            return score
        rmse_intercepto = evaluar([])
        for lam in grilla:
            beta, info = resolver_group_lasso(gram, xy, ids, pesos, float(lam), inicial=beta,
                tol=cfg['solver_tol'], max_iter=cfg['max_iter'])
            coeficientes.append(beta.copy())
            normas = [float(np.linalg.norm(beta[i])) for i in bloques]
            seleccionadas = [g for g, norma in zip(nombres, normas) if norma > cfg['support_tol']]
            valido = 0 < len(seleccionadas) <= cfg['max_features']
            score = None
            if valido:
                clave = tuple(seleccionadas)
                if clave not in cache: cache[clave] = evaluar(seleccionadas)
                score = cache[clave]
                candidatos.append({'lambda': float(lam), 'seleccionadas': seleccionadas,
                                   'rmse_validation_original': score})
            historial.append({'lambda': float(lam), 'seleccionadas': seleccionadas,
                'n_caracteristicas': len(seleccionadas), 'n_predictores': sum(len(grupos[g]) for g in seleccionadas),
                'valido': valido, 'rmse_validation_original': score,
                'motivo': 'evaluado' if valido else 'solo_intercepto' if not seleccionadas else 'max_features',
                'normas_grupos': dict(zip(nombres, normas)), **info})
    elegido, decisiones = elegir_por_complejidad(candidatos, rmse_intercepto,
        min_mejora_pct=cfg['min_mejora_pct'], max_features=cfg['max_features'])
    if not elegido['seleccionadas']:
        raise RuntimeError('Ningún subconjunto mejora al intercepto según el umbral; no registrar un modelo arbitrario.')
    resultado = {**elegido, 'rmse_intercepto_original': rmse_intercepto, 'historial': historial,
        'decisiones_complejidad': decisiones, 'config': cfg, 'lambda_max': float(lambda_max),
        'lambdas': grilla.tolist(), 'motivo_parada': 'grilla_completa_y_criterio_complejidad',
        'ponderacion': {g: {'p_g': r, 'peso': float(w)} for g, r, w in zip(nombres, rangos, pesos)},
        'ols_provisionales_unicos': len(cache), 'target_mean_fit': float(y_log.iloc[fit].mean())}
    joblib.dump(scaler, out/'scaler_seleccion.joblib')
    np.savez_compressed(out/'ruta_coeficientes_group_lasso.npz', lambdas=grilla,
                        coeficientes=np.asarray(coeficientes), group_ids=ids, columnas=np.asarray(list(X)))
    np.savez_compressed(out/'particion_interna.npz', fit=fit, validation=val)
    b.guardar_json(out/'seleccion.json', resultado)
    b.guardar_json(out/'historial_lambdas.json', historial)
    b.guardar_json(out/'decisiones_complejidad.json', decisiones)
    b.guardar_json(out/'grupos_caracteristicas.json', grupos)
    return resultado


def entrenar_exportar(base, X, y, grupos, seleccion, contexto_train, referencia, salida,
                     *, excluidas=s.EXCLUIDAS, nombre_run=NOMBRE_RUN):
    columnas = s.columnas_grupos(grupos, seleccion['seleccionadas'])
    modelo, diseno = b.ajustar(X[columnas], np.log(y))
    assert modelo.model.rank == len(columnas)+1 and modelo.nobs == len(y)
    metricas, invalidas, vif, notas = b.diagnosticar(modelo, diseno, np.log(y), y_original=y, resumir_vif=True)
    metricas.update(train_n_features_original=float(len(seleccion['seleccionadas'])),
        selection_rmse_validation_original=seleccion['rmse_validation_original'],
        selection_rmse_intercept_original=seleccion['rmse_intercepto_original'],
        selection_lambda=seleccion['lambda'])
    contexto = {**contexto_train, 'predictoras': columnas, 'n_predictoras': len(columnas),
        'transformacion_objetivo': 'ln', 'unidad': 'ln(RUB)', 'predictoras_excluidas': list(excluidas),
        'seleccion_caracteristicas': 'group_lasso_then_ols', 'reference_run_id': referencia['run_id'],
        'caracteristicas_originales': seleccion['seleccionadas']}
    out = b.exportar(base, modelo, diseno, np.log(y), contexto, metricas, invalidas, vif, notas,
                     nombre_run=nombre_run, carpeta_salida=salida)
    tabla = s.tabla_comparativa(referencia['metrics'], metricas, invalidas,
                                nombre_referencia=referencia['run_name'], nombre_run=nombre_run)
    resumen = s.resumen_caracteristicas(seleccion['seleccionadas'], grupos, modelo, mostrar_orden=False)
    (out/'comparacion_referencia.md').write_text(tabla, encoding='utf-8')
    (out/'caracteristicas_seleccionadas.md').write_text(resumen, encoding='utf-8')
    b.guardar_json(out/'referencia_mlflow.json', referencia)
    b.guardar_json(out/'caracteristicas_seleccionadas.json', {g: grupos[g] for g in seleccion['seleccionadas']})
    return {'modelo': modelo, 'metricas': metricas, 'contexto': contexto, 'salida': out,
            'comparacion_md': tabla, 'caracteristicas_md': resumen}


def registrar(cliente, resultado, seleccion, referencia, *, nombre_run=NOMBRE_RUN):
    cfg = seleccion['config']
    assert not cliente.search_runs([referencia['experiment_id']],
        filter_string=f"tags.mlflow.runName = '{nombre_run}'", max_results=1), 'El Run ya existe; no sobrescribir.'
    params = {'algorithm': 'Group Lasso + OLS', 'min_mejora_pct': cfg['min_mejora_pct'],
        'max_features': cfg['max_features'], 'lambda_selected': seleccion['lambda'],
        'lambda_grid': json.dumps(seleccion['lambdas']), 'group_weight': 'sqrt(effective_group_rank)',
        'scaler': 'StandardScaler fit internal_train only; OLS in original transformed columns',
        'support_tol': cfg['support_tol'], 'solver': 'pinv', 'selection_solver': 'FISTA Gram; adaptive restart',
        'selection_solver_tol': cfg['solver_tol'], 'selection_solver_max_iter': cfg['max_iter'],
        'objective': '||yc-Zb||^2/(2n) + lambda * sum(sqrt(p_g)*||b_g||_2)',
        'n_features_original': len(seleccion['seleccionadas']), 'selected_features': json.dumps(seleccion['seleccionadas']),
        'reference_run_name': referencia['run_name'], 'reference_run_id': referencia['run_id'],
        'experiment_id': referencia['experiment_id'], 'notebook': '4_Seleccion_Modelos.ipynb',
        'validation_fraction': .2, 'random_state': referencia['params']['random_state'], 'permutation_repeats': 10,
        'selection_stop_reason': seleccion['motivo_parada']}
    b.guardar_json(resultado['salida']/'configuracion.json', params)
    registro = b.registrar(cliente, resultado['salida'], resultado['contexto'], resultado['metricas'],
                           nombre_run=nombre_run, parametros_extra=params)
    actual = cliente.get_run(referencia['run_id'])
    assert actual.data.metrics == referencia['metrics'] and actual.data.params == referencia['params']
    assert actual.info.status == 'FINISHED'
    return registro

"""BSS exhaustivo en un espacio acotado de características; sin usar test."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def limpiar(datos, registro):
    d = datos.copy(deep=True)
    d['date_posted_dt'] = pd.to_datetime(d.date_posted, errors='raise')
    for c in registro['columnas_numericas_entrada']:
        d[c] = pd.to_numeric(d[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if not (np.isfinite(d.price_rub) & d.price_rub.gt(0)).all():
        raise ValueError('Objetivos inválidos: no se imputan ni se eliminan automáticamente.')
    areas = d[['total_area', 'living_area', 'kitchen_area']]
    m = (np.isfinite(areas) & areas.gt(0)).all(axis=1) & d.total_area.lt(d.living_area+d.kitchen_area)
    d.loc[m, 'total_area'] = d.loc[m, 'living_area']+d.loc[m, 'kitchen_area']
    t, f, y = d.total_floors, d.floor, d.building_year
    tv = np.isfinite(t) & t.gt(0) & t.mod(1).eq(0)
    fv = np.isfinite(f) & f.gt(0) & f.mod(1).eq(0)
    d.loc[t.notna() & ~tv, 'total_floors'] = np.nan
    d.loc[f.notna() & (~fv | (tv & f.gt(t))), 'floor'] = np.nan
    d.loc[y.notna() & (~np.isfinite(y) | y.mod(1).ne(0) | y.lt(1000) | y.gt(d.date_posted_dt.dt.year)), 'building_year'] = np.nan
    return d


def binaria(c, s):
    v = s.dropna()
    if c == 'district': return False
    if pd.api.types.is_bool_dtype(s) or c == 'has_balcony': return True
    if pd.api.types.is_numeric_dtype(s): return len(v)>0 and v.isin([0, 1]).all()
    return v.nunique()==2


def entrada_codificador(datos, parametros):
    d = limpiar(datos, parametros['registro'])
    for c, v in parametros['imputacion'].items(): d[c] = d[c].fillna(v)
    assert not d.isna().any().any()
    assert (d.floor.gt(0) & d.total_floors.gt(0) & d.floor.le(d.total_floors)).all()
    assert d[['floor', 'total_floors', 'building_year']].mod(1).eq(0).all().all()
    assert (d.building_year.ge(1000) & d.building_year.le(d.date_posted_dt.dt.year)).all()
    d = d.drop(columns=parametros['registro']['eliminadas']).rename(columns=parametros['registro']['alias'])
    X = d.drop(columns=['id', 'price_rub', 'date_posted_dt']).copy()
    X['anio_publicacion'], X['mes_publicacion'] = d.date_posted_dt.dt.year, d.date_posted_dt.dt.month
    for c, pares in parametros.get('binarias', {}).items():
        X[c] = X[c].map(dict(pares)).fillna(0).astype(float)
    return X, d.price_rub


def preparar_interno(ajuste_crudo, validacion_cruda, registro):
    """Fit solo en ajuste_crudo; validación recibe transform, nunca fit."""
    d = limpiar(ajuste_crudo, registro)
    valores = {}
    for c in d:
        if c in ['id', 'price_rub', 'date_posted', 'date_posted_dt']: continue
        s = d[c]
        if binaria(c, s):
            moda = s.mode(dropna=True)
            v = moda.iloc[0] if len(moda) else np.nan
        elif pd.api.types.is_numeric_dtype(s):
            v = s.mean() if c == 'secondary_mom_change_pct' else s.median()
        else: v = '__FALTANTE__'
        if pd.isna(v): raise ValueError(f'No se puede imputar {c} usando entrenamiento interno.')
        valores[c] = v.item() if isinstance(v, np.generic) else v
    parametros = {'registro': registro, 'imputacion': valores}
    X, y = entrada_codificador(ajuste_crudo, parametros)
    binarias = [c for c in X if binaria(c, X[c])]
    nominales = [c for c in X if c not in binarias and not pd.api.types.is_numeric_dtype(X[c])]
    numericas = [c for c in X if c not in binarias+nominales]
    mapas = {}
    for c in binarias:
        v = sorted(X[c].unique().tolist())
        mapas[c] = [(x, int(x)) for x in v] if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]) else list(zip(v, range(len(v))))
    parametros['binarias'] = mapas
    X, y = entrada_codificador(ajuste_crudo, parametros)
    V, yv = entrada_codificador(validacion_cruda, parametros)
    bloques = [('num', 'passthrough', numericas), ('bin', 'passthrough', binarias)]
    for i,c in enumerate(nominales):
        bloques.append((f'cat{i}', OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False,dtype=float),[c]))
    encoder = ColumnTransformer(bloques,verbose_feature_names_out=False)
    A = encoder.fit_transform(X)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Found unknown categories',category=UserWarning)
        B = encoder.transform(V)
    nombres = encoder.get_feature_names_out().tolist()
    mapa = {c:[c] for c in numericas+binarias if c!='to_center_km'}
    for i,c in enumerate(nominales): mapa[c] = encoder.named_transformers_[f'cat{i}'].get_feature_names_out([c]).tolist()
    parametros['encoder'], parametros['mapa'] = encoder, mapa
    parametros['columnas'] = nombres
    parametros['categorias'] = {c:encoder.named_transformers_[f'cat{i}'].categories_[0].tolist() for i,c in enumerate(nominales)}
    parametros['referencias'] = {c:v[0] for c,v in parametros['categorias'].items()}
    return pd.DataFrame(A,index=X.index,columns=nombres),pd.DataFrame(B,index=V.index,columns=nombres),y,yv,parametros


def inversa(pred, lam):
    pred=np.asarray(pred,dtype=float)
    valida=np.isfinite(pred)
    if lam: valida &= 1+lam*pred>0
    out=np.full(pred.shape,np.nan)
    with np.errstate(over='ignore',invalid='ignore'): out[valida]=inv_boxcox(pred[valida],lam)
    valida &= np.isfinite(out) & (out>0)
    return out,valida


def errores(real,pred):
    real,pred=np.asarray(real),np.asarray(pred)
    mse=np.mean((real-pred)**2)
    return {'R2':float(1-np.sum((real-pred)**2)/np.sum((real-real.mean())**2)),
            'MAE':float(np.mean(np.abs(real-pred))),'MSE':float(mse),'RMSE':float(np.sqrt(mse))}


def base_estimable(X,mapa):
    columnas=[c for cols in mapa.values() for c in cols]
    constantes=[c for c in columnas if X[c].nunique()<=1]
    candidatas=[c for c in columnas if c not in constantes]
    Z=(X[candidatas]-X[candidatas].mean())/X[candidatas].std(ddof=0)
    _,R,piv=qr(Z.to_numpy(),mode='economic',pivoting=True,check_finite=False)
    tol=np.finfo(float).eps*max(Z.shape)*np.abs(np.diag(R)).max()
    rango=int((np.abs(np.diag(R))>tol).sum())
    conservar=[candidatas[i] for i in sorted(piv[:rango].tolist())]
    bloques={f:[c for c in conservar if c in cols] for f,cols in mapa.items()}
    return bloques,{'constantes':constantes,'dependencias':[c for c in candidatas if c not in conservar],'tolerancia':float(tol)}


def buscar_segmento(X,V,y,yv,mapa,nombre,max_candidatas=8,tolerancia_rmse=.01):
    bloques,estructura=base_estimable(X,mapa)
    disponibles=[f for f,cols in bloques.items() if cols]
    columnas=[c for f in disponibles for c in bloques[f]]
    ybox,lam=boxcox(np.asarray(y,dtype=float))
    completo=sm.OLS(pd.Series(ybox,index=X.index,name='boxcox_price_rub'),sm.add_constant(X[columnas],has_constant='add')).fit()
    ranking=[]
    for f in disponibles:
        C=np.zeros((len(bloques[f]),len(completo.params)))
        C[np.arange(len(bloques[f])),[completo.params.index.get_loc(c) for c in bloques[f]]]=1
        F=float(np.asarray(completo.f_test(C).fvalue).item())
        contrib=F*len(bloques[f])/(F*len(bloques[f])+completo.df_resid)
        ranking.append({'Característica':f,'R2 parcial subtrain':contrib,'Parámetros':len(bloques[f])})
    ranking=pd.DataFrame(ranking).sort_values(['R2 parcial subtrain','Característica'],ascending=[False,True],kind='stable')
    candidatas=ranking['Característica'].head(max_candidatas).tolist()
    cols=[c for f in candidatas for c in bloques[f]]
    media=X[cols].mean();desvio=X[cols].std(ddof=0)
    Z=((X[cols]-media)/desvio).to_numpy();Zv=((V[cols]-media)/desvio).to_numpy()
    G=Z.T@Z;c=Z.T@(ybox-ybox.mean());yty=np.sum((ybox-ybox.mean())**2)
    indices={f:[cols.index(c) for c in bloques[f]] for f in candidatas}
    n=len(y);filas=[]
    for mascara in range(2**len(candidatas)):
        fs=[f for i,f in enumerate(candidatas) if mascara & (1<<i)]
        ind=[j for f in fs for j in indices[f]];k=len(ind)+1
        registro={'máscara':mascara,'características':json.dumps(fs,ensure_ascii=False),'n_features':len(fs),'n_parámetros':k,'rango_válido':True}
        try:
            if ind:
                g=G[np.ix_(ind,ind)];np.linalg.cholesky(g)
                b=np.linalg.solve(g,c[ind]);rss=yty-c[ind]@b
                pred=ybox.mean()+Zv[:,ind]@b
                vif=np.diag(np.linalg.inv(g/n));vifmax=float(vif.max())
            else: rss=yty;pred=np.full(len(yv),ybox.mean());vifmax=np.nan
            if not rss>0:raise np.linalg.LinAlgError('RSS no positivo')
            pred_rub,valida=inversa(pred,lam)
            e=errores(yv,pred_rub) if valida.all() else {'RMSE':np.inf,'R2':np.nan}
            ll=-n/2*(np.log(2*np.pi)+1+np.log(rss/n))
            registro.update({'R2 ajustado':1-(rss/(n-k))/(yty/(n-1)),'AIC':-2*ll+2*k,'BIC':-2*ll+k*np.log(n),
                'RMSE validación RUB':e['RMSE'],'R2 validación RUB':e['R2'],'VIF máximo':vifmax,'inversas inválidas':int((~valida).sum())})
        except np.linalg.LinAlgError:
            registro.update({'rango_válido':False,'RMSE validación RUB':np.inf,'VIF máximo':np.inf})
        filas.append(registro)
    resultados=pd.DataFrame(filas)
    validos=resultados.loc[resultados['rango_válido'] & np.isfinite(resultados['RMSE validación RUB'])]
    factibles=validos.loc[validos['n_features'].eq(0) | validos['VIF máximo'].lt(5)]
    if factibles.empty: raise ValueError(f'{nombre}: no existe ningún modelo válido con VIF <5; revisar sin usar test.')
    minimo=factibles['RMSE validación RUB'].min()
    equivalentes=factibles.loc[factibles['RMSE validación RUB'].le(minimo*(1+tolerancia_rmse))]
    ganador=equivalentes.sort_values(['n_features','n_parámetros','RMSE validación RUB','máscara'],kind='stable').iloc[0]
    seleccionadas=json.loads(ganador['características'])
    columnas_elegidas=[c for f in seleccionadas for c in bloques[f]]
    inner=sm.OLS(pd.Series(ybox,index=X.index,name='boxcox_price_rub'),sm.add_constant(X[columnas_elegidas],has_constant='add')).fit()
    pv=inner.predict(sm.add_constant(V[columnas_elegidas],has_constant='add'))
    rub,ok=inversa(pv,lam);assert ok.all()
    np.testing.assert_allclose(errores(yv,rub)['RMSE'],ganador['RMSE validación RUB'],rtol=1e-6)
    return {'nombre':nombre,'candidatas_totales':disponibles,'candidatas_restringidas':candidatas,'ranking':ranking,
        'bloques':bloques,'estructura':estructura,'resultados':resultados,'seleccionadas':seleccionadas,
        'lambda_interno':float(lam),'modelo_interno':inner,'ganador':ganador.to_dict(),'columnas_internas':columnas_elegidas,
        'n_subtrain':len(y),'n_validacion':len(yv),'minimo_rmse_factible':float(minimo)}


def importancia_permutacion(busqueda,V,yv,semilla=20261028,repeticiones=30):
    modelo=busqueda['modelo_interno'];cols=busqueda['columnas_internas'];lam=busqueda['lambda_interno']
    A=sm.add_constant(V[cols],has_constant='add')[modelo.model.exog_names].to_numpy()
    beta=np.asarray(modelo.params);base,ok=inversa(A@beta,lam);assert ok.all()
    rmse=errores(yv,base)['RMSE'];filas=[]
    for i,f in enumerate(busqueda['seleccionadas']):
        ind=[modelo.model.exog_names.index(c) for c in busqueda['bloques'][f]]
        for repeticion in range(repeticiones):
            rng=np.random.default_rng(np.random.SeedSequence([semilla,i,repeticion]))
            orden=rng.permutation(len(A));B=A.copy();B[:,ind]=A[orden][:,ind]
            pred,valida=inversa(B@beta,lam)
            delta=errores(yv,pred)['RMSE']-rmse if valida.all() else np.nan
            filas.append({'Característica':f,'Repetición':repeticion+1,'ΔRMSE RUB':delta,'Inversas inválidas':int((~valida).sum())})
    detalle=pd.DataFrame(filas)
    resumen=detalle.groupby('Característica',sort=False).agg(
        media=('ΔRMSE RUB','mean'),mediana=('ΔRMSE RUB','median'),desvio=('ΔRMSE RUB','std'),repeticiones_validas=('ΔRMSE RUB','count')).sort_values('mediana',ascending=False)
    resumen['Posición']=range(1,len(resumen)+1)
    return detalle,resumen.reset_index(),rmse


def vif_features(modelo,bloques):
    cols=modelo.model.exog_names[1:];X=modelo.model.exog[:,1:]
    if not cols:return pd.DataFrame(columns=['Característica','VIF máximo','Columnas VIF >5','GVIF ajustado'])
    R=np.corrcoef(X,rowvar=False);R=np.atleast_2d(R);vif=np.diag(np.linalg.inv(R));_,ld=np.linalg.slogdet(R)
    filas=[]
    for f,cs in bloques.items():
        ids=[cols.index(c) for c in cs if c in cols]
        if not ids:continue
        otros=[i for i in range(len(cols)) if i not in ids]
        sa,la=np.linalg.slogdet(R[np.ix_(ids,ids)]);sb,lb=np.linalg.slogdet(R[np.ix_(otros,otros)]) if otros else (1,0)
        gv=np.exp((la+lb-ld)/(2*len(ids))) if sa>0 and sb>0 else np.nan
        filas.append({'Característica':f,'VIF máximo':float(vif[ids].max()),'Columnas VIF >5':int((vif[ids]>5).sum()),
                      'GVIF ajustado':float(gv) if len(cs)>1 else 'No aplica'})
    return pd.DataFrame(filas).sort_values('VIF máximo',ascending=False)

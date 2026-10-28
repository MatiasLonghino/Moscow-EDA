import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

root=Path(__file__).resolve().parent
data=root/'Fase_2/data'
archivos_iniciales={p.relative_to(data).as_posix() for p in data.rglob('*') if p.is_file()}
csvs={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.glob('*.csv')}
entrada=root/'Ventas/Propiedades_Usadas/data/PropUsadas_Integrado.csv'
huella_entrada=hashlib.sha256(entrada.read_bytes()).hexdigest()
artefactos={p:joblib.load(p) for p in data.rglob('*.joblib')}
firmas={}
for p,a in artefactos.items():
    for key in ['modelos','modelos_finales','modelos_validacion']:
        if key in a:
            for s,m in a[key].items():firmas[(p,key,s)]=(m.params.copy(),m.fittedvalues.copy())

def tabla_desde_markdown(s):
    lineas=[l for l in s.splitlines() if l.startswith('|')]
    def partes(l):return [v.strip().replace('___P___','|') for v in l.replace('\\|','___P___').strip('|').split('|')]
    def convertir(v):
        try:return float(v.replace('.','').replace(',','.')) if ',' in v else float(v)
        except ValueError:return v
    return pd.DataFrame([[convertir(v) for v in partes(l)] for l in lineas[2:]],columns=partes(lineas[0]))

fit_original=sm.OLS.fit
def no_reentrenar(*args,**kwargs):raise AssertionError('La limpieza intentó reentrenar un OLS.')
sm.OLS.fit=no_reentrenar
try:
    for nombre in ['Preparacion_OLS_Usadas.ipynb','Modelo_OLS.ipynb']:
        path=root/'Fase_2'/nombre
        nb=json.loads(path.read_text(encoding='utf-8'))
        os.chdir(path.parent)
        shell=InteractiveShell.instance()
        shell.reset(new_session=False)
        numero=0
        for c in nb['cells']:
            if c['cell_type']!='code':continue
            numero+=1
            plt.show=lambda:plt.close('all')
            with capture_output() as cap:r=shell.run_cell(''.join(c['source']))
            if r.error_before_exec or r.error_in_exec:raise RuntimeError(cap.stdout+cap.stderr)
            if cap.stderr:raise RuntimeError(cap.stderr)
            print(nombre,'celda',numero,'OK',flush=True)
        if nombre=='Modelo_OLS.ipynb':
            ns=shell.user_ns
            for segmento in ns['orden_segmentos']:
                celda=next(c for c in nb['cells'] if c['cell_type']=='markdown' and ''.join(c['source']).startswith(f'**{segmento}: OLS referencia frente a BSS'))
                guardada=tabla_desde_markdown(''.join(celda['source']))
                actual=ns['comparaciones_finales_bss'][segmento]
                assert guardada['Métrica'].tolist()==actual['Métrica'].tolist()
                np.testing.assert_allclose(guardada.iloc[:,1:].to_numpy(dtype=float),actual.iloc[:,1:].to_numpy(dtype=float),rtol=1e-6,atol=2e-6)
            for p,a in artefactos.items():
                nuevo=joblib.load(p)
                for key in ['modelos','modelos_finales','modelos_validacion']:
                    if key in a:
                        for s,m in a[key].items():
                            pd.testing.assert_series_equal(nuevo[key][s].params,firmas[(p,key,s)][0])
                            pd.testing.assert_series_equal(nuevo[key][s].fittedvalues,firmas[(p,key,s)][1])
    actuales={p.relative_to(data).as_posix() for p in data.rglob('*') if p.is_file()}
    assert actuales==archivos_iniciales and len(actuales)==8
    assert not list(data.rglob('*.json'))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in csvs.items())
    assert hashlib.sha256(entrada.read_bytes()).hexdigest()==huella_entrada
    print('Dos notebooks ejecutados completos sin reentrenar. Predicciones y selecciones BSS coinciden con las tablas conservadas.')
    print('Ocho archivos necesarios; ningún auxiliar regenerado. CSV y entrada original intactos.')
finally:
    sm.OLS.fit=fit_original

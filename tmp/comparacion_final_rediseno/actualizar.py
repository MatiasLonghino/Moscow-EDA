"""Renderizar solo la comparación con evidencia ya calculada, sin red ni ajustes."""
import copy
import json
import sys
from pathlib import Path

import uuid
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import HTML, Markdown, display
from IPython.utils.capture import capture_output

base = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base / 'Ventas/Propiedades_Usadas'))
from scripts import presentacion_final as pf

cache = Path(r'C:\Users\mglon\.codex\visualizations\2026\10\27\01a17df3-a114-7142-a90c-30b6c21fa893')
leer = lambda carpeta, nombre: json.loads((cache / carpeta / nombre).read_text(encoding='utf-8'))
u = leer('bss_review', 'universo_bss.json')
historicos = {}
for clave, origen in [('forward_177', 'Forward 177'), ('backward', 'Backward 177'), ('lasso_177', 'Group Lasso 177')]:
    r = u['origenes'][origen]
    historicos[clave] = dict(seleccion={'seleccionadas': r['seleccionadas']}, metricas=r['run']['metrics'], run=r['run'])
bss = dict(universo=u, seleccion=leer('bss_review', 'seleccion.json'), metricas=leer('bss_review', 'metricas.json'),
           run=dict(run_id='9aee5b7348954fc48bf14562b6987e51', experiment_id='5', run_name='BSS_06_Log_Union_177_Grupos_OLS'))
temporal_bss = dict(resultados=leer('temporal_review', 'resultados.json'), configuracion=leer('temporal_review', 'configuracion.json'),
           run=dict(run_id='4b3e5cf9cac24ccdbda3d3568e3fb0a5', experiment_id='5', run_name='BSS_07_Log_Temporal_3F_1SE_OLS'))
path = base / 'Ventas/Propiedades_Usadas/4_Seleccion_Modelos.ipynb'
original = json.loads(path.read_text(encoding='utf-8'))
snapshot = cache / 'notebook_before_comparacion_final.ipynb'
if not snapshot.exists():
    snapshot.write_bytes(path.read_bytes())
old = json.loads(snapshot.read_text(encoding='utf-8'))
table, overlaps = [''.join(o['data']['text/markdown']) for o in old['cells'][60]['outputs']]

def md(source):
    return dict(cell_type='markdown', id=uuid.uuid4().hex[:8], metadata={}, source=source)

def code(source):
    return dict(cell_type='code', id=uuid.uuid4().hex[:8], metadata={}, source=source, execution_count=None, outputs=[])

cells = [
    md('## 10. Comparación final\n\n### ¿Qué características seleccionó cada modelo?\n\n'
       'La matriz muestra las características originales de la unión utilizada por BSS. '
       'La última columna identifica el **modelo final elegido mediante validación temporal y regla 1-SE**. '
       'Las selecciones y métricas se leen de los resultados recuperados en las secciones anteriores; no se vuelve a ajustar ningún modelo.'),
    code('from scripts import presentacion_final as pf\nfrom IPython.display import HTML\n'
         'datos_finales = pf.preparar(forward_177, backward, lasso_177, bss, temporal_bss)\n'
         'display(HTML(pf.matriz_html(datos_finales)))\ndisplay(Markdown(pf.observaciones(datos_finales)))'),
    md('### Recorrido histórico y decisión temporal\n\n'
       'Las métricas históricas describen la validación interna de selección. '
       'Los resultados temporales se presentan por separado porque utilizan otro procesamiento de Train y tres folds cronológicos.'),
    code('display(Markdown("**Selección histórica — validación interna (RUB)**"))\n'
         'display(Markdown(pf.tabla_historica(datos_finales)))\n'
         'display(Markdown("**BSS temporal — tres folds cronológicos (RUB)**"))\n'
         'display(Markdown(pf.tabla_temporal(datos_finales)))\n'
         'display(Markdown(pf.interpretar_comparaciones(datos_finales)))'),
    md('### Modelo final seleccionado: BSS + validación cruzada temporal'),
    code('caracteristicas_finales, resumen_final, justificacion_final = pf.modelo_elegido(datos_finales)\n'
         'display(Markdown(caracteristicas_finales))\ndisplay(Markdown(resumen_final))\n'
         'display(Markdown(justificacion_final))'),
    md('### Conclusión'),
    code('display(Markdown(pf.conclusion(datos_finales)))'),
    code('tabla_final, coincidencias = s.comparacion_final_presentacion(forward, lasso, backward, '
         'forward_177=forward_177, lasso_177=lasso_177, bss=bss)\n'
         'display(Markdown(pf.anexos(datos_finales, tabla_final, coincidencias)))\nsesion.cerrar()'),
]
scope = dict(pf=pf, HTML=HTML, Markdown=Markdown, display=display, bss=bss, temporal_bss=temporal_bss, **historicos)
InteractiveShell.instance()
for c in cells[:-1]:
    if c['cell_type'] != 'code':
        continue
    with capture_output() as captured:
        exec(c['source'], scope)
    assert not captured.stdout and not captured.stderr
    c['outputs'] = [dict(output_type='display_data', data=o.data, metadata=o.metadata) for o in captured.outputs]

# Conservar las tablas históricas ya visibles, sin recuperar ni ejecutar los experimentos de 174 columnas.
annex = pf.anexos(scope['datos_finales'], table, overlaps)
cells[-1]['outputs'] = [dict(output_type='display_data', data={'text/markdown': annex, 'text/plain': '<IPython.core.display.Markdown object>'}, metadata={})]
updated = copy.deepcopy(old)
updated['cells'] = old['cells'][:59] + [dict(c) for c in cells]
for c in cells:
    if c['cell_type']=='code': compile(c['source'], '<comparison>', 'exec')
assert updated['cells'][:59] == old['cells'][:59]
path.write_text(json.dumps(updated, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
datos = scope['datos_finales']
assert [len(datos['selecciones'][k]) for k in pf.NOMBRES] == [8, 12, 12, 12, 6]
assert '1.603.608' in pf.tabla_historica(datos) and '2.480.192' not in pf.tabla_historica(datos)
assert '2.480.192' in pf.tabla_temporal(datos) and '1.581.085' not in pf.tabla_temporal(datos)
assert '139 columnas transformadas más intercepto' in scope['resumen_final']
assert '128 columnas One-Hot' in scope['resumen_final']
assert 'superior en Test' in scope['justificacion_final'] and 'sigue pendiente' in pf.conclusion(datos)
matrix = pf.matriz_html(datos)
assert matrix.count('<strong>✓</strong>') == 6 and matrix.count('border-top:2px solid #93A99F;') == 11
# Comprobar que las marcas cambian con los datos y no están codificadas en el HTML.
changed = copy.deepcopy(datos)
changed['selecciones']['temporal'].remove('total_area')
assert pf.matriz_html(changed).count('<strong>✓</strong>') == 5
preview = base / 'tmp/comparacion_final_rediseno/comparacion.html'
parts = []
for c in cells:
    if c['cell_type'] == 'markdown':
        parts.append({'markdown':c['source']})
    else:
        for o in c['outputs']:
            data = o['data']
            parts.append({'html': data['text/html']} if 'text/html' in data else {'markdown':data['text/markdown']})
preview.with_suffix('.json').write_text(json.dumps(parts, ensure_ascii=False), encoding='utf-8')
print('Comparación renderizada con evidencia existente. 59 celdas anteriores idénticas; totales 8/12/12/12/6; 139 columnas + intercepto; validación de 1-SE y poblaciones separadas correcta.')
print(preview)

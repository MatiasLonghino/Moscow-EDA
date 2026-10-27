import json,re
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=Path(__file__).parent
for p in (root/'Ventas/Propiedades_Usadas').glob('[1-4]_*.ipynb'):
    nb=json.loads(p.read_text(encoding='utf-8'))
    lines=[]
    for i,c in enumerate(nb['cells']):
        s=''.join(c.get('source',[]))
        if c['cell_type']=='markdown': lines.append(f'CELL {i} MARKDOWN\n{s}')
        else:
            lines.append(f'CELL {i} CODE\n{s}')
            for o in c.get('outputs',[]):
                d=o.get('data',{})
                t=o.get('text') or d.get('text/markdown') or d.get('text/plain')
                if t:
                    if isinstance(t,list):t=''.join(t)
                    if len(t)>16000:t=t[:16000]+' [TRUNCATED IN AUDIT]'
                    lines.append(t)
    (out/(p.stem+'.txt')).write_text('\n\n'.join(lines),encoding='utf-8')
print('Audit files created')

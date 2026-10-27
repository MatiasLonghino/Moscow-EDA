exec(open(__file__.replace('fetch_comparison.py','fetch.py'),encoding='utf-8').read().split("rid='4b3e")[0])
from concurrent.futures import ThreadPoolExecutor
runs=dict(forward='9b5b90c245cb4fd8a804c1d0e4c3ff7b',backward='74f261e8575b4858ad9a88fd9373bc76',lasso='cda36a216ba1492eb1e87049344159c3',bss='9aee5b7348954fc48bf14562b6987e51')
def read(item):
    name,rid=item;r=c.get_run(rid)
    assert r.info.status=='FINISHED'
    return name,dict(run_id=rid,run_name=r.data.tags['mlflow.runName'],metrics=r.data.metrics,params=r.data.params)
d=dict(ThreadPoolExecutor(4).map(read,runs.items()))
(out/'comparison.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
for name,r in d.items():print(name,r['params'].get('selected_features'),r['metrics'].get('selection_rmse_validation_original'))

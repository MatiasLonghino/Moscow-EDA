import os,sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'Ventas/Propiedades_Usadas'))
from dotenv import load_dotenv
load_dotenv(root/'.env')
os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT']='30'
os.environ['MLFLOW_HTTP_REQUEST_MAX_RETRIES']='1'
from mlflow.tracking import MlflowClient
uri='https://dagshub.com/matiasgastonlonghino/MOSCOW_IDA.mlflow'
c=MlflowClient(tracking_uri=uri)
out=Path(__file__).parent/'mlflow'; out.mkdir(exist_ok=True)
rid='4b3e5cf9cac24ccdbda3d3568e3fb0a5'
r=c.get_run(rid)
(out/'run.json').write_text(json.dumps(dict(metrics=r.data.metrics,params=r.data.params,tags=r.data.tags),indent=2),encoding='utf-8')
print('Run',r.data.tags.get('mlflow.runName'),r.info.status,flush=True)
for a in c.list_artifacts(rid):print(a.path, a.is_dir, flush=True)
for name in ['configuracion.json','resultados.json','modelo_statsmodels.pkl','preparador.joblib']:
    c.download_artifacts(rid,name,str(out))
    print('Downloaded',name,flush=True)

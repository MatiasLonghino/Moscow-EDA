import sys,json,hashlib,os
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
os.environ['MLFLOW_DISABLE_AGENT_HINT']='1'
root=Path(__file__).resolve().parents[2]; out=Path(__file__).parent
sys.path.insert(0,str(root/'Ventas/Propiedades_Usadas'))
import numpy as np,pandas as pd,joblib,statsmodels.api as sm
from statsmodels.regression.linear_model import OLSResults
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from threadpoolctl import threadpool_limits
from scripts import bss_temporal as bt
from scripts.diagnosticos_residuales import residuos_train
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Arial','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#34444A','text.color':'#26363B','axes.edgecolor':'#B9C5C7','figure.facecolor':'white','axes.titleweight':'bold'})
teal='#176B70'; red='#AF3C36'; gray='#687C82'
cfg=json.loads((out/'mlflow/configuracion.json').read_text()); res=json.loads((out/'mlflow/resultados.json').read_text())
for f,k in [('modelo_statsmodels.pkl','model_sha256'),('preparador.joblib','preprocessor_sha256')]:
    assert hashlib.sha256((out/'mlflow'/f).read_bytes()).hexdigest()==res[k]
data,identity=bt.cargar_train_crudo(root/'Ventas/Propiedades_Usadas')
assert identity['train_sha256']==cfg['identity']['train_sha256']
prep=joblib.load(out/'mlflow/preparador.joblib'); model=OLSResults.load(out/'mlflow/modelo_statsmodels.pkl')
X=prep.transform(data)[res['columns']]; D=sm.add_constant(X,has_constant='add'); y=np.log(data.price_rub)
with threadpool_limits(limits=1):
    fitted,e,z=residuos_train(model,D,y)
    jb=jarque_bera(e); bp=het_breuschpagan(e,D)
pred=np.exp(fitted); price=data.price_rub.to_numpy(); baseline=float(np.sqrt(np.mean((price-pred)**2)))
rng=np.random.default_rng(20261027); importance=[]
for g in res['selected_features']:
    cols=cfg['groups'][g]; contribution=X[cols].to_numpy()@model.params[cols].to_numpy()
    vals=[]
    for _ in range(20):
        alt=np.exp(fitted-contribution+contribution[rng.permutation(len(data))])
        vals.append(float(np.sqrt(np.mean((price-alt)**2))-baseline))
    importance.append(dict(feature=g,mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)),values=vals))
importance.sort(key=lambda d:d['mean'],reverse=True)
stats=dict(n=len(data),train_rmse=baseline,jb_stat=float(jb[0]),jb_p=float(jb[1]),skew=float(jb[2]),kurtosis=float(jb[3]),bp_stat=float(bp[0]),bp_p=float(bp[1]),bp_f_p=float(bp[3]),importance=importance,price_median=float(data.price_rub.median()),price_skew=float(data.price_rub.skew()),surface_corr=float(data.total_area.corr(data.living_area)))
(out/'evidence.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
assets=out/'assets';assets.mkdir(exist_ok=True)
def save(fig,name):fig.savefig(assets/name,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5.5)); sm.qqplot(e,line='s',ax=ax,marker='o',markersize=2,alpha=.3,color=teal)
ax.lines[-1].set_color(red);ax.set(xlabel='Cuantiles teóricos normales',ylabel='Cuantiles de residuos en ln(RUB)');ax.grid(alpha=.15);save(fig,'qq.png')
fig,ax=plt.subplots(figsize=(9,5.5)); scale=np.sqrt(np.abs(z));ax.scatter(fitted,scale,s=4,alpha=.15,c=teal,rasterized=True)
curve=lowess(scale,fitted,frac=.3,it=3,delta=float(np.ptp(fitted))*.01,missing='raise');ax.plot(curve[:,0],curve[:,1],c=red,lw=2.5,label='LOWESS')
ax.set(xlabel='Valores ajustados en ln(RUB)',ylabel='Raíz de |residuos estandarizados|');ax.legend(frameon=False);ax.grid(alpha=.15);save(fig,'scale.png')
stats['lowess_min']=float(curve[:,1].min());stats['lowess_max']=float(curve[:,1].max());stats['lowess_endpoints']=[float(curve[0,1]),float(curve[-1,1])]
(out/'evidence.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
# Numerical data for native, editable charts. All newly computed EDA data use Train only.
charts={}
for key,col,bins in [('price','price_rub',35),('area','total_area',30)]:
    vals=data[col].to_numpy()/(1e6 if key=='price' else 1)
    counts,edges=np.histogram(vals,bins=bins)
    charts[key]=dict(labels=[f'{(a+b)/2:.1f}' for a,b in zip(edges[:-1],edges[1:])],values=counts.tolist())
vc=data.building_type.value_counts();charts['building']=dict(labels=vc.index.tolist(),values=vc.tolist())
# Scatter plots reproduce relationships explored in notebook; all Train points retained.
for key,col,xlab in [('scatter_area','total_area','Superficie total (m²)'),('scatter_center','to_center_km','Distancia al centro (km)')]:
    fig,ax=plt.subplots(figsize=(4.3,3.7));ax.scatter(data[col],price/1e6,s=3,c=teal,alpha=.12,rasterized=True)
    ax.set(xlabel=xlab,ylabel='Precio (millones de RUB)');ax.grid(alpha=.12);save(fig,key+'.png')
monthly=data.assign(month=data.date_posted_dt.dt.to_period('M')).groupby('month').price_rub.median()
charts['monthly']=dict(labels=monthly.index.astype(str).tolist(),values=(monthly/1e6).tolist())
charts['importance']=dict(labels=[x['feature'] for x in importance],values=[x['mean']/1e6 for x in importance])
charts['cv']=dict(labels=[str(r['k']) for r in res['rows']],values=[r['mean']/1e6 for r in res['rows']],threshold=[res['rule']['threshold']/1e6]*12)
(out/'charts.json').write_text(json.dumps(charts,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False,indent=2))


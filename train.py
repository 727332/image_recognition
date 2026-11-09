import argparse,json,time,platform,hashlib,gzip
from pathlib import Path
import numpy as np

def shift(x,dy,dx):
 a=x.reshape(-1,8,8);b=np.zeros_like(a)
 b[:,max(0,dy):min(8,8+dy),max(0,dx):min(8,8+dx)]=a[:,max(0,-dy):min(8,8-dy),max(0,-dx):min(8,8-dx)]
 return b.reshape(-1,64)
def split(y,seed):
 rng=np.random.default_rng(seed);parts=[[],[],[]]
 for k in range(10):
  ids=rng.permutation(np.flatnonzero(y==k)); n=len(ids); a=int(.6*n);b=int(.8*n)
  for dest,src in zip(parts,[ids[:a],ids[a:b],ids[b:]]):dest.extend(src)
 return [np.array(v) for v in parts]
def predict(z,w):return np.argmax(np.column_stack([z,np.ones(len(z))])@w,axis=1)
def main(out):
 p=Path(out);p.mkdir(parents=True,exist_ok=True)
 data=np.loadtxt(gzip.open(Path(__file__).with_name('digits.csv.gz'),'rt'),delimiter=',');x=data[:,:64]/16;y=data[:,64].astype(int);rows=[]
 dirs=[(0,1),(0,-1),(1,0),(-1,0)]
 for seed in [17,29,43]:
  tr,va,te=split(y,seed);np.savez(p/f'split_{seed}.npz',train=tr,val=va,test=te)
  for method in ['baseline','horizontal','fourway']:
   ds=[] if method=='baseline' else dirs[:2] if method=='horizontal' else dirs
   xx=np.concatenate([x[tr]]+[shift(x[tr],*s) for s in ds]);yy=np.tile(y[tr],len(ds)+1);a=np.column_stack([xx,np.ones(len(xx))]);target=np.eye(10)[yy]
   vx=np.concatenate([x[va]]+[shift(x[va],*s) for s in dirs]);vy=np.tile(y[va],5);best=None;candidates=[]
   for lam in [.0001,.001,.01,.1,1]:
    start=time.perf_counter();reg=np.eye(65)*lam;reg[-1,-1]=0;w=np.linalg.solve(a.T@a/len(a)+reg,a.T@target/len(a));elapsed=time.perf_counter()-start
    score=float(np.mean(predict(vx,w)==vy));candidates.append({'lambda':lam,'validation':score})
    if best is None or score>best[0]:best=(score,w,lam,elapsed)
   score,w,lam,elapsed=best;r={'seed':seed,'method':method,'lambda':lam,'val_score':score,'fit_seconds':elapsed,'n_train':len(xx),'candidates':candidates};preds=[]
   for name,z in [('clean',x[te])]+[(f'shift_{dy}_{dx}',shift(x[te],dy,dx)) for dy,dx in dirs]:
    pred=predict(z,w);r[name]=float(np.mean(pred==y[te]));preds.append(pred)
   r['shift_mean']=float(np.mean([r[f'shift_{dy}_{dx}'] for dy,dx in dirs]));f=[]
   for k in range(10):
    tp=np.sum((preds[0]==k)&(y[te]==k));f.append(2*tp/(np.sum(preds[0]==k)+np.sum(y[te]==k)))
   r['macro_f1']=float(np.mean(f));np.savez(p/f'{method}_{seed}.npz',weights=w,classes=np.arange(10),regularization=lam)
   np.savez(p/f'predictions_{method}_{seed}.npz',truth=y[te],predictions=np.array(preds));rows.append(r);print(seed,method,r['clean'],r['shift_mean'],flush=True)
 meta={'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'n_samples':len(y),'split_sizes':[len(tr),len(va),len(te)],'data_sha256':hashlib.sha256(data.tobytes()).hexdigest(),'results':rows}
 (p/'metrics.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',default='results');main(a.parse_args().out)

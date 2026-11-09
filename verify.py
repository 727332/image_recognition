"""Verify splits, saved predictions, weights and reported metrics."""
from pathlib import Path
import json,gzip
import numpy as np
from train import predict,shift
p=Path(__file__).parent;r=p/'results'
data=np.loadtxt(gzip.open(p/'digits.csv.gz','rt'),delimiter=',');x=data[:,:64]/16;y=data[:,64].astype(int)
meta=json.loads((r/'metrics.json').read_text())
for row in meta['results']:
 seed=row['seed'];method=row['method'];s=np.load(r/f'split_{seed}.npz');tr,va,te=(s[k] for k in ['train','val','test'])
 assert not set(tr)&set(va) and not set(tr)&set(te) and not set(va)&set(te)
 assert len(set(tr)|set(va)|set(te))==1797
 w=np.load(r/f'{method}_{seed}.npz')['weights'];assert w.shape==(65,10)
 pred=np.load(r/f'predictions_{method}_{seed}.npz');assert np.array_equal(pred['truth'],y[te])
 tests=[x[te]]+[shift(x[te],*s) for s in [(0,1),(0,-1),(1,0),(-1,0)]]
 keys=['clean','shift_0_1','shift_0_-1','shift_1_0','shift_-1_0']
 for j,(z,k) in enumerate(zip(tests,keys)):
  yp=predict(z,w);assert np.array_equal(yp,pred['predictions'][j]);assert abs(np.mean(yp==y[te])-row[k])<1e-12
 assert abs(np.mean([row[k] for k in keys[1:]])-row['shift_mean'])<1e-12
 f=[]
 for k in range(10):
  yp=pred['predictions'][0];tp=np.sum((yp==k)&(y[te]==k));f.append(2*tp/(np.sum(yp==k)+np.sum(y[te]==k)))
 assert abs(np.mean(f)-row['macro_f1'])<1e-12
 assert row['val_score']==max(c['validation'] for c in row['candidates'])
 print('PASS',seed,method)
print('All 9 models, 45 prediction groups and split isolation verified.')

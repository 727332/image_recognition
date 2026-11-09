import argparse,json,time,platform,hashlib,gzip,csv
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

METHOD_NAMES={'baseline':'原图基线','horizontal':'水平增强','fourway':'四方向增强'}
def result_table(headers, rows):
 import unicodedata
 def width(s):return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in str(s))
 rows=[[str(c) for c in r] for r in [headers]+rows]
 sizes=[max(width(r[i]) for r in rows) for i in range(len(headers))]
 line='-+-'.join('-'*n for n in sizes)
 output=[]
 for j,row in enumerate(rows):
  output.append(' | '.join(c+' '*(sizes[i]-width(c)) for i,c in enumerate(row)))
  if j==0:output.append(line)
 return '\n'.join(output)

def export_report(p,meta):
 rows=meta['results'];methods=list(METHOD_NAMES)
 summary=[]
 for method in methods:
  group=[r for r in rows if r['method']==method]
  summary.append([METHOD_NAMES[method]]+[f"{np.mean([r[k] for r in group])*100:.2f} ± {np.std([r[k] for r in group],ddof=1)*100:.2f}" for k in ['clean','shift_mean','macro_f1']])
 text='实验汇总（三次划分均值 ± 样本标准差，单位：%）\n'+result_table(['方法','原图准确率','平移平均准确率','原图 Macro-F1'],summary)
 details=[[r['seed'],METHOD_NAMES[r['method']],f"{r['lambda']:g}"]+[f"{r[k]*100:.2f}" for k in ['val_score','clean','shift_0_1','shift_0_-1','shift_1_0','shift_-1_0','shift_mean','macro_f1']] for r in rows]
 text+='\n\n逐次结果（指标单位：%）\n'+result_table(['种子','方法','λ','验证五视图','原图','右移','左移','下移','上移','平移平均','Macro-F1'],details)
 base=np.mean([r['shift_mean'] for r in rows if r['method']=='baseline']);full=np.mean([r['shift_mean'] for r in rows if r['method']=='fourway'])
 clean_delta=np.mean([r['clean'] for r in rows if r['method']=='fourway'])-np.mean([r['clean'] for r in rows if r['method']=='baseline'])
 text+=f'\n\n结果解读：四方向增强相较基线，平移平均准确率变化 {(full-base)*100:+.2f} 个百分点，原图准确率变化 {clean_delta*100:+.2f} 个百分点。\n'
 text+='平移平均：右、左、下、上四组测试准确率的等权平均，不含原图。\n验证五视图：验证集原图与四方向平移的平均准确率，仅用于选择 λ。\n标准差：三个随机划分间的样本标准差，不是置信区间。\n'
 text+=f"环境：Python {meta['python']} / NumPy {meta['numpy']}\n结果目录：{p.resolve()}\n权重：baseline_*.npz、horizontal_*.npz、fourway_*.npz\n原始记录：metrics.json；逐样本预测：predictions_*.npz；划分：split_*.npz\n"
 (p/'summary.txt').write_text(text,encoding='utf-8')
 keys=['seed','method','lambda','n_train','val_score','clean','shift_0_1','shift_0_-1','shift_1_0','shift_-1_0','shift_mean','macro_f1','fit_seconds']
 with (p/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
 print('\n'+text,flush=True)
 print('已保存：summary.txt（中文报告）、results.csv（数值为原始比例）、metrics.json（完整记录）。',flush=True)

def main(out):
 p=Path(out);p.mkdir(parents=True,exist_ok=True)
 data=np.loadtxt(gzip.open(Path(__file__).with_name('digits.csv.gz'),'rt'),delimiter=',');x=data[:,:64]/16;y=data[:,64].astype(int);rows=[]
 print('手写数字识别与平移鲁棒性实验',flush=True)
 print(f'数据：{len(y)} 张 8×8 灰度图，10 类；随机种子：17、29、43',flush=True)
 print(f'运行环境：Python {platform.python_version()} / NumPy {np.__version__}',flush=True)
 print(f'输出目录：{p.resolve()}',flush=True)
 dirs=[(0,1),(0,-1),(1,0),(-1,0)]
 for seed in [17,29,43]:
  tr,va,te=split(y,seed);np.savez(p/f'split_{seed}.npz',train=tr,val=va,test=te)
  print(f'\n随机划分 {seed}：训练 {len(tr)} / 验证 {len(va)} / 测试 {len(te)}',flush=True)
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
   np.savez(p/f'predictions_{method}_{seed}.npz',truth=y[te],predictions=np.array(preds));rows.append(r);print(f"  [{len(rows)}/9] {METHOD_NAMES[method]} | λ={lam:g} | 训练视图={len(xx)} | 原图={r['clean']:.2%} | 平移平均={r['shift_mean']:.2%} | Macro-F1={r['macro_f1']:.2%}",flush=True)
 meta={'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'n_samples':len(y),'split_sizes':[len(tr),len(va),len(te)],'data_sha256':hashlib.sha256(data.tobytes()).hexdigest(),'results':rows}
 (p/'metrics.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 export_report(p,meta)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',default='results');main(a.parse_args().out)

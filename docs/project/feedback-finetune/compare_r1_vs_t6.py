"""So sánh model task 5 round 1 với model relearn task 6 (alias_only) trên val, theo từng vật/SKU."""
import json, numpy as np, sys
from collections import defaultdict
B='full_syn_relearn_task6_20261101/data/training/full-syn-relearn'
gt=json.load(open('training/full-syn-r1/inputs/tasks/val_full.json'))
name={c['id']:c['name'] for c in gt['categories']}
rpc2lab={c['rpc_category_id']:c['id'] for c in gt['categories']}
plan=json.load(open(B+'/plan.json')); RE=set(plan['relearned'])
t6=set(); t7=set()
for t,s in ((6,t6),(7,t7)):
    for c in json.load(open(f'{B}/inputs/tasks/val_task_{t}.json'))['categories']:
        s.add(rpc2lab[int(c['name'].split('_')[0])])
assert t6|t7==RE, (sorted(t6|t7), sorted(RE))
grp=np.zeros(200,int); 
for g,(a,b) in enumerate([(0,100),(100,125),(125,150),(150,175),(175,200)]): grp[a:b]=g
runs={'r1':('training/full-syn-r1/runs/EXP-FULL-SYN/task_5/pred_val.npz',[0.1,0.15,0.18,0.18,0.2]),
      't6':(B+'/scored/task6_alias_only/task_5/pred_val.npz',[0.1,0.12,0.2,0.15,0.15])}
def iou(a,b):
    x1=np.maximum(a[:,None,0],b[None,:,0]);y1=np.maximum(a[:,None,1],b[None,:,1])
    x2=np.minimum(a[:,None,2],b[None,:,2]);y2=np.minimum(a[:,None,3],b[None,:,3])
    inter=np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None)
    ar=lambda z:(z[:,2]-z[:,0])*(z[:,3]-z[:,1])
    return inter/(ar(a)[:,None]+ar(b)[None,:]-inter+1e-9)
gimg=defaultdict(list)
for a in gt['annotations']:
    x,y,w,h=a['bbox']; gimg[a['image_id']].append((a['category_id'],[x,y,x+w,y+h]))
imgs=sorted(i['id'] for i in gt['images'])
res={}
for k,(f,th) in runs.items():
    d=np.load(f); thr=np.array(th)[grp]
    o=np.lexsort((-d['score'],d['query'],d['image_id'])); 
    iid,q=d['image_id'][o],d['query'][o]; first=np.r_[True,(iid[1:]!=iid[:-1])|(q[1:]!=q[:-1])]
    o=o[first]
    I,L,S,Bx=d['image_id'][o],d['label'][o],d['score'][o],d['boxes'][o]
    by=defaultdict(list)
    for j in range(len(I)): by[I[j]].append(j)
    per={}
    for im in imgs:
        rows=np.array(by.get(im,[]),int)
        if len(rows):
            rows=rows[np.lexsort((L[rows],-S[rows]))]
            ious=iou(Bx[rows],Bx[rows]); sup=np.zeros(len(rows),bool); keep=[]
            for i in range(len(rows)):
                if not sup[i]: keep.append(rows[i]); sup|=ious[i]>=0.45
            keep=np.array(keep,int); keep=keep[S[keep]>=thr[L[keep]]]
        else: keep=rows
        per[im]=(L[keep],Bx[keep])
    res[k]=per
def analyse(per):
    correct={}; cnt_err=np.zeros(200); conf=defaultdict(int); miss=np.zeros(200); tp=np.zeros(200); fpcls=np.zeros(200); ngt=np.zeros(200); extra=np.zeros(200)
    for im in imgs:
        g=gimg[im]; gl=np.array([c for c,_ in g]); gb=np.array([b for _,b in g],float)
        pl,pb=per[im]
        gc=np.bincount(gl,minlength=200); pc=np.bincount(pl,minlength=200) if len(pl) else np.zeros(200,int)
        correct[im]=bool((gc==pc).all()); cnt_err+=np.abs(gc-pc); ngt+=gc
        used=set()
        if len(pl):
            M=iou(gb,pb)
            for gi in np.argsort(-M.max(1)):
                cand=[j for j in np.argsort(-M[gi]) if M[gi,j]>=0.5 and j not in used]
                if not cand: miss[gl[gi]]+=1; continue
                j=cand[0]; used.add(j)
                if pl[j]==gl[gi]: tp[gl[gi]]+=1
                else: conf[(gl[gi],pl[j])]+=1
            for j in range(len(pl)):
                if j not in used: extra[pl[j]]+=1
        else:
            for c in gl: miss[c]+=1
    return dict(correct=correct,cnt_err=cnt_err,conf=conf,miss=miss,tp=tp,ngt=ngt,extra=extra)
A={k:analyse(v) for k,v in res.items()}
for k in A: print(k,'cAcc',np.mean(list(A[k]['correct'].values())))
sets={'task6 (25 SKU học lại)':t6,'task7 (25 SKU chưa học lại)':t7,'150 SKU khác':set(range(200))-RE}
def wrong_as(a,S):
    return sum(v for (t,p),v in a['conf'].items() if t in S)
def stolen_by(a,S,T): return sum(v for (t,p),v in a['conf'].items() if t in S and p in T)
print('\n== Theo vật (GT khớp box IoU>=0.5) ==')
for nm,S in sets.items():
    L=list(S)
    for k in ('r1','t6'):
        a=A[k]; n=a['ngt'][L].sum()
        print(f"{nm:28s} {k}: n={int(n)} đúng={a['tp'][L].sum()/n:.4f} nhầm={wrong_as(a,S)/n:.4f} sót={a['miss'][L].sum()/n:.4f} box_thừa={int(a['extra'][L].sum())} sai_đếm={int(a['cnt_err'][L].sum())}")
print('\n== Vật của 150 SKU khác bị gọi thành ==')
O=sets['150 SKU khác']
for k in ('r1','t6'):
    print(k,'-> task6:',stolen_by(A[k],O,t6),' -> task7:',stolen_by(A[k],O,t7),' -> SKU khác:',stolen_by(A[k],O,O))
print('\n== Ảnh ==')
c1,c2=A['r1']['correct'],A['t6']['correct']
fixed=[i for i in imgs if not c1[i] and c2[i]]; broken=[i for i in imgs if c1[i] and not c2[i]]
print('đúng->đúng',sum(c1[i] and c2[i] for i in imgs),'sai->đúng',len(fixed),'đúng->sai',len(broken))
def errclasses(per,im):
    g=np.bincount([c for c,_ in gimg[im]],minlength=200); pl=per[im][0]; p=np.bincount(pl,minlength=200) if len(pl) else np.zeros(200,int)
    return set(np.flatnonzero(g!=p))
cat=defaultdict(int)
for im in broken:
    e=errclasses(res['t6'],im)
    key=('có SKU task6' if e&t6 else '')+('|có SKU task7' if e&t7 else '')+('|có SKU khác' if e&O else '')
    cat[key.strip('|')]+=1
print('Ảnh đúng->sai, lớp bị đếm sai gồm:',dict(cat))
onlyO=sum(1 for im in broken if errclasses(res['t6'],im)<=O)
print('  ...trong đó chỉ sai ở 150 SKU khác:',onlyO)
cat=defaultdict(int)
for im in fixed:
    e=errclasses(res['r1'],im)
    key=('có SKU task6' if e&t6 else '')+('|có SKU task7' if e&t7 else '')+('|có SKU khác' if e&O else '')
    cat[key.strip('|')]+=1
print('Ảnh sai->đúng, trước đó sai ở:',dict(cat))
print('\n== 150 SKU khác: SKU tụt/tăng nhiều nhất (số vật đúng) ==')
dl=[(c,int(A['t6']['tp'][c]-A['r1']['tp'][c]),int(A['r1']['ngt'][c])) for c in O]
dl.sort(key=lambda x:x[1])
for c,dv,n in dl[:10]:
    top=sorted([(p,v) for (t,p),v in A['t6']['conf'].items() if t==c],key=lambda x:-x[1])[:3]
    top1=sorted([(p,v) for (t,p),v in A['r1']['conf'].items() if t==c],key=lambda x:-x[1])[:3]
    print(f"  {c:3d} {name[c]:22s} n={n:4d} Δđúng={dv:+d} | t6 nhầm thành {[(p,name[p],v,'T6' if p in t6 else ('T7' if p in t7 else '')) for p,v in top]} | r1 {[(p,v) for p,v in top1]}")
print('  tăng nhiều nhất:',[(c,name[c],dv) for c,dv,n in dl[-5:]])
print('\n== SKU task6: thay đổi số vật đúng ==')
for c in sorted(t6,key=lambda c:A['t6']['tp'][c]-A['r1']['tp'][c]):
    n=A['r1']['ngt'][c]; print(f"  {c:3d} {name[c]:22s} n={int(n):4d} r1={A['r1']['tp'][c]/n:.3f} t6={A['t6']['tp'][c]/n:.3f} thừa {int(A['r1']['extra'][c])}->{int(A['t6']['extra'][c])}")

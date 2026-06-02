#!/usr/bin/env python3
import glob
import json
import math
from statistics import mean, pstdev

def zscore(arr):
    if not arr: return [0]*len(arr)
    m = mean(arr)
    s = pstdev(arr)
    if s == 0: return [0]*len(arr)
    return [(x-m)/s for x in arr]

def minmax(arr):
    if not arr: return [0]*len(arr)
    mn = min(arr); mx = max(arr)
    if mx==mn: return [0.5]*len(arr)
    return [(x-mn)/(mx-mn) for x in arr]

def rankscore(arr):
    # higher -> larger score; convert ranks to (n-rank+1)
    idxs = sorted(range(len(arr)), key=lambda i: -arr[i])
    n = len(arr)
    scores = [0]*n
    for rank, i in enumerate(idxs, start=1):
        scores[i] = n - rank + 1
    # normalize to 0..1
    return [s / n for s in scores]

files = sorted(glob.glob('irsl_www/sketch_result/*_top5.json'))

methods = {
    'minmax': (lambda sc,cl: (minmax(sc), minmax(cl))),
    'zscore': (lambda sc,cl: (zscore(sc), zscore(cl))),
    'rank': (lambda sc,cl: (rankscore(sc), rankscore(cl))),
}

summary = {}
for name in methods:
    summary[name] = {'files':0,'top1_same_as_sc':0,'total_candidates':0}

for p in files:
    try:
        d = json.load(open(p,'r',encoding='utf-8'))
    except Exception:
        continue
    topk = d.get('topk', [])
    sc = []
    cl = []
    for t in topk:
        if t.get('score_sketchscape') is not None:
            sc.append(float(t.get('score_sketchcape')) if t.get('score_sketchcape') is not None else float(t.get('score_sketchscape')))
        else:
            sc.append(float(t.get('score')))
        cl.append(t.get('score_clip_text'))

    if all(v is None for v in cl):
        continue

    # filter indices where clip exists
    valid_idx = [i for i,v in enumerate(cl) if v is not None]
    sc_sub = [sc[i] for i in valid_idx]
    cl_sub = [float(cl[i]) for i in valid_idx]
    if len(sc_sub) < 1:
        continue

    sc_top1 = max(range(len(sc_sub)), key=lambda i: sc_sub[i])

    for mname, func in methods.items():
        sc_n, cl_n = func(sc_sub, cl_sub)
        # weights: use 0.5/0.5 for comparison
        fused = [0.5*s + 0.5*c for s,c in zip(sc_n, cl_n)]
        top1_fused = max(range(len(fused)), key=lambda i: fused[i])
        summary[mname]['files'] += 1
        summary[mname]['total_candidates'] += len(fused)
        if top1_fused == sc_top1:
            summary[mname]['top1_same_as_sc'] += 1

print('Fusion analysis (0.5/0.5 weights) across files with CLIP applied:')
for m, v in summary.items():
    if v['files']==0:
        print(f"  {m}: no files")
        continue
    print(f"  {m}: files={v['files']} top1_same_as_sc={v['top1_same_as_sc']} rate={v['top1_same_as_sc']/v['files']:.2f} avg_candidates={v['total_candidates']/v['files']:.1f}")

print('\nRecommendation: if minmax shows low top1 agreement and CLIP widths are small, try zscore or rank fusion.')

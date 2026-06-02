#!/usr/bin/env python3
import json
import glob
import math
from collections import defaultdict

def ranks(arr):
    # higher score -> rank 1
    sorted_idx = sorted(range(len(arr)), key=lambda i: -arr[i])
    r = [0]*len(arr)
    for rank, idx in enumerate(sorted_idx, start=1):
        r[idx] = rank
    return r

def mean(arr):
    return sum(arr)/len(arr) if arr else 0.0

def std(arr):
    if not arr: return 0.0
    m = mean(arr)
    return math.sqrt(sum((x-m)**2 for x in arr)/len(arr))

def pearson(x, y):
    if not x or not y: return 0.0
    mx = mean(x); my = mean(y)
    num = sum((a-mx)*(b-my) for a,b in zip(x,y))
    denom = math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return num/denom if denom>0 else 0.0

files = sorted(glob.glob('irsl_www/sketch_result/*_top5.json'))
results = []

for p in files:
    try:
        with open(p,'r',encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"skip {p}: {e}")
        continue
    topk = data.get('topk', [])
    sc = []
    clip = []
    for t in topk:
        if 'score_sketchcape' in t and t.get('score_sketchcape') is not None:
            # defensive: some files use 'score_sketchcape' typo
            sc.append(float(t.get('score_sketchcape')))
        elif 'score_sketchscape' in t and t.get('score_sketchcape') is not None:
            sc.append(float(t.get('score_sketchcape')))
        elif 'score_sketchscape' in t and t.get('score_sketchscape') is not None:
            sc.append(float(t.get('score_sketchscape')))
        else:
            sc.append(float(t.get('score')))
        clip.append(t.get('score_clip_text') if t.get('score_clip_text') is not None else None)
    # filter entries where clip is None
    clip_vals = [c for c in clip if c is not None]
    if not clip_vals:
        results.append((p, False, None))
        continue
    sc_min, sc_max = min(sc), max(sc)
    clip_min, clip_max = min(clip_vals), max(clip_vals)
    sc_width = sc_max - sc_min
    clip_width = clip_max - clip_min

    # Spearman via ranks
    valid_idx = [i for i,c in enumerate(clip) if c is not None]
    sc_sub = [sc[i] for i in valid_idx]
    clip_sub = [clip[i] for i in valid_idx]
    r_sc = ranks(sc_sub)
    r_clip = ranks(clip_sub)
    spearman = pearson(r_sc, r_clip)

    # top1 indices
    top1_sc = sc_sub.index(max(sc_sub)) if sc_sub else None
    top1_clip = clip_sub.index(max(clip_sub)) if clip_sub else None

    results.append((p, True, {
        'scape_width': sc_width,
        'clip_width': clip_width,
        'spearman_rank_corr': spearman,
        'top1_sc_index': top1_sc,
        'top1_clip_index': top1_clip,
        'num_candidates': len(sc_sub)
    }))

# aggregate
agg = {'files_with_clip':0,'total_files':len(files),'sum_sc_width':0.0,'sum_clip_width':0.0,'count':0,'neg_spearman':0}
for item in results:
    p, has_clip, info = item
    if not has_clip:
        print(p + ': clip not applied')
        continue
    agg['files_with_clip'] += 1
    agg['sum_sc_width'] += info['scape_width']
    agg['sum_clip_width'] += info['clip_width']
    agg['count'] += 1
    if info['spearman_rank_corr'] < -0.5:
        agg['neg_spearman'] += 1
    print(p)
    print('  scape_width=%.4f clip_width=%.4f spearman=%.3f top1_sc=%s top1_clip=%s n=%d' % (
        info['scape_width'], info['clip_width'], info['spearman_rank_corr'], info['top1_sc_index'], info['top1_clip_index'], info['num_candidates']))

if agg['count']>0:
    print('\nSummary:')
    print('  files with clip: %d / %d' % (agg['files_with_clip'], agg['total_files']))
    print('  avg scape width=%.4f avg clip width=%.4f' % (agg['sum_sc_width']/agg['count'], agg['sum_clip_width']/agg['count']))
    print('  files with spearman<-0.5: %d' % agg['neg_spearman'])
else:
    print('No files with clip applied found')

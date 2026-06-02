#!/usr/bin/env python3
import numpy as np
import os

PATH = 'CLIP_DB/cache/clip_image_embeddings.npz'

def human(x):
    if x is None: return 'None'
    return str(x)

if not os.path.exists(PATH):
    print('cache not found:', PATH)
    raise SystemExit(1)

data = np.load(PATH)
print('keys:', list(data.keys()))
ids = data.get('ids')
emb = data.get('embeddings')
print('ids shape:', None if ids is None else ids.shape)
print('embeddings shape:', None if emb is None else emb.shape)
if emb is not None:
    print('dtype:', emb.dtype)
    print('emb mean/std:', float(np.mean(emb)), float(np.std(emb)))
    norms = np.linalg.norm(emb, axis=1)
    print('per-row norm min/mean/max/std: %.6f / %.6f / %.6f / %.6f' % (float(np.min(norms)), float(np.mean(norms)), float(np.max(norms)), float(np.std(norms))))
    print('sample first-id(s):', ids[:10].tolist() if ids is not None else None)
    print('sample embedding[0][:8]:', emb[0][:8].tolist())
    print('embedding_dim:', emb.shape[1])
    print('bytes per vector (float32):', emb.shape[1]*4)

if 'meta' in data:
    print('meta:', data['meta'].tolist())

print('done')

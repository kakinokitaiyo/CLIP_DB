Tools for CLIP_DB
==================

This folder contains helper scripts used by the CLIP_DB retrieval pipeline.

1) compute_dinov2_embeddings_db.py
----------------------------------
Purpose: compute DINOv2 image embeddings for images stored in the DB and upsert
them into a separate table (default: `photo_embeddings`). Embeddings are stored
as L2-normalized `float32` bytes in a `bytea` column.

Basic usage:

```bash
python3 compute_dinov2_embeddings_db.py \
  --host localhost --port 5432 --dbname kakinoki_db \
  --user USER --password PASS \
  --schema home_robot --gallery-table photos \
  --emb-table photo_embeddings --batch-size 32 [--clear-existing]
```

Options:
- `--clear-existing`: truncate the embedding table before computing (useful when regenerating all embeddings).
- `--force`: recompute embeddings even if an entry already exists for a photo.

Notes:
- The script will create the `photo_embeddings` table if it does not exist.
- Saved embedding format: `photo_id, model, version, embedding (bytea), created_at`.
- For large collections consider exporting embeddings to Faiss for ANN search.

2) test_dinov2.py
-----------------
Quick local test to verify that DINOv2 model loads and computes a single
embedding. Run inside an environment with GPU (recommended) or CPU.

3) clip_diagnostics.py, fusion_analysis.py, check_clip_cache.py
-------------------------------------------------------------
Utilities for offline diagnostics and fusion analysis. They operate on
local result JSONs and cached CLIP embeddings.

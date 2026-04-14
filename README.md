# CLIP_DB 比較スクリプト

このディレクトリは、画像同士を OpenCLIP で比較するための作業用フォルダです。

## フォルダ構成
- `photos/` : 元画像
- `output/` : 生成画像（比較対象）
- `sketches/` : スケッチ画像（クエリ）
- `outputs/clip/` : 比較結果の保存先
- `src/run_clip_top5.py` : 比較スクリプト

## 実行例
ワークスペース上で実行する場合:

```bash
cd /home/irsl/workspace/zikken/src
python3 run_clip_top5.py \
  --gallery_dir /home/irsl/workspace/zikken/output \
  --query_dir /home/irsl/workspace/zikken/sketches \
  --output_dir /home/irsl/workspace/zikken/outputs/clip_rbte \
  --topk 5
```

Docker 風パスでも指定できます（`/CLIP_DB/...`）。

```bash
cd /home/irsl/workspace/zikken/src
python3 run_clip_top5.py \
  --gallery_dir /CLIP_DB/output \
  --query_dir /CLIP_DB/sketches \
  --output_dir /CLIP_DB/outputs/clip_rbte \
  --topk 5
```

## 出力
- 各クエリ画像ごとに `*_top5.json`
- 全体まとめの `summary.json`

## 必要パッケージ
- `torch`
- `open_clip_torch`
- `Pillow`

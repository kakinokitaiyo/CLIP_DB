# CLIP_DB

このワークスペースは、`photos` と `sketches` を比較するための作業用フォルダです。
OpenCLIP を使う比較スクリプトと、SketchScape の SBIR モデルを使う比較スクリプトを置いています。

## フォルダ構成
- `photos/` : 比較対象の写真画像
- `sketches/` : クエリとなるスケッチ画像
- `output/` : 生成済み画像などの比較対象
- `outputs/clip/` : CLIP 比較結果の保存先
- `outputs/sbir/` : SBIR 比較結果の保存先
- `src/run_clip_top5.py` : OpenCLIP による画像検索スクリプト
- `src/run_sbir_top5.py` : SketchScape の SBIR モデルによる画像検索スクリプト

## 事前準備
### 依存パッケージ
少なくとも次が必要です。
- `torch`
- `torchvision`
- `open_clip_torch`
- `Pillow`

### 学習済み SBIR モデル
SBIR 版を動かす場合は、SketchScape 側の学習済み重みが必要です。
例:
- `/home/irsl/workspace/SketchScape/models/fscoco_normal.pth`

## CLIP で比較する
`photos` を検索対象、`sketches` をクエリとして top-k を出力します。

```bash
cd /home/irsl/workspace/CLIP_DB/src
python3 run_clip_top5.py \
  --gallery_dir /home/irsl/workspace/CLIP_DB/output \
  --query_dir /home/irsl/workspace/CLIP_DB/sketches \
  --output_dir /home/irsl/workspace/CLIP_DB/outputs/clip \
  --topk 5
```

Docker 環境などで `/CLIP_DB` を使う場合は、次のようにも指定できます。

```bash
cd /CLIP_DB/src
python3 run_clip_top5.py \
  --gallery_dir /CLIP_DB/output \
  --query_dir /CLIP_DB/sketches \
  --output_dir /CLIP_DB/outputs/clip \
  --topk 5
```

### 出力
- 各クエリ画像ごとの `*_top5.json`
- 全体の `summary.json`

## SBIR で比較する
SketchScape の SBIR モデルを使って、`sketches` をクエリ、`photos` を検索対象として比較します。

```bash
cd /home/irsl/workspace/CLIP_DB/src
python3 run_sbir_top5.py \
  --photo_dir /home/irsl/workspace/CLIP_DB/photos \
  --sketch_dir /home/irsl/workspace/CLIP_DB/sketches \
  --output_dir /home/irsl/workspace/CLIP_DB/outputs/sbir \
  --model_path /home/irsl/workspace/SketchScape/models/fscoco_normal.pth \
  --topk 5 \
  --device auto
```

### `--device` について
- `auto` : CUDA が使えれば GPU、だめなら CPU に自動切り替え
- `cuda` : GPU を使う
- `cpu` : CPU を使う

RTX 5060 Ti のように、環境によっては PyTorch の CUDA ビルド更新が必要な場合があります。
その場合は `--device auto` で CPU フォールバックして実行できます。

### 出力
- 各クエリ画像ごとの `*_top5.json`
- 全体の `summary.json`

## よく使う実行例
### CLIP
```bash
python3 run_clip_top5.py --gallery_dir /home/irsl/workspace/CLIP_DB/output --query_dir /home/irsl/workspace/CLIP_DB/sketches --output_dir /home/irsl/workspace/CLIP_DB/outputs/clip --topk 5
```

### SBIR
```bash
python3 run_sbir_top5.py --photo_dir /home/irsl/workspace/CLIP_DB/photos --sketch_dir /home/irsl/workspace/CLIP_DB/sketches --output_dir /home/irsl/workspace/CLIP_DB/outputs/sbir --model_path /home/irsl/workspace/SketchScape/models/fscoco_normal.pth --topk 5 --device auto
```

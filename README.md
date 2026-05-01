# CLIP_DB

このワークスペースは、`photos` と `sketches` を比較するための作業用フォルダです。
OpenCLIP を使う比較スクリプトと、SketchScape の SBIR モデルを使う比較スクリプトを置いています。

## セットアップ

### 1. 環境変数の設定

機密情報（DB ユーザ・パスワードなど）は環境変数から読み込みます。

```bash
# .env.example をコピーして .env を作成
cp .env.example .env

# .env ファイルを編集して実際の値を設定
# PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD など
vi .env

# シェルで読み込む
export $(cat .env | xargs)
```

または、実行時に直接設定：

```bash
export PGHOST=your_db_host
export PGPORT=5432
export PGDATABASE=kakinoki_db
export PGUSER=your_db_user
export PGPASSWORD=your_db_password

python3 src/run_sbir_once_from_db.py --sketch_path sketches/writing_1.png
```

### 環境変数一覧

| 変数名 | 説明 | デフォルト |
|--------|------|----------|
| `PGHOST` | PostgreSQL ホスト | `localhost` |
| `PGPORT` | PostgreSQL ポート | `5432` |
| `PGDATABASE` | PostgreSQL データベース名 | `kakinoki_db` |
| `PGUSER` | PostgreSQL ユーザー（**必須**） | - |
| `PGPASSWORD` | PostgreSQL パスワード（**必須**） | - |
| `RBTE_DOCKER_ROOT` | RBTE Docker ルートパス | `/home/irsl/workspace/rbte_docker` |
| `CLIP_DB_ROOT` | CLIP_DB ルートパス | `/home/irsl/workspace/CLIP_DB` |
| `SKETCHSCAPE_ROOT` | SketchScape ルートパス | `/home/irsl/workspace/SketchScape` |
| `SBIR_MODEL_PATH` | SBIR 学習済み重みパス | `/home/irsl/workspace/SketchScape/models/fscoco_normal.pth` |

### 2. 依存パッケージ
少なくとも次が必要です。
- `torch`
- `torchvision`
- `open_clip_torch`
- `Pillow`

### 学習済み SBIR モデル
SBIR 版を動かす場合は、SketchScape 側の学習済み重みが必要です。
例:
- `/home/irsl/workspace/SketchScape/models/fscoco_normal.pth`

## データベース登録と SBIR 検索（新パイプライン）

### アーキテクチャ

**シンプル DB 設計：実行時 RBTE 処理**

```
DB (photos + sketches)
  │
  ├─ photos ─ [RBTE エッジ検出] ─ [SBIR 特徴抽出]
  │                               ↓
  └─ sketches ── [特徴抽出] ─────→ [比較・ランク付け]
                                  ↓
                            元写真メタデータ返却
```

- DB 登録時：`photos` と `sketches` のみ保存（シンプル）
- SBIR 実行時：RBTE BDCN でエッジ検出を**メモリ内**実行
- 表示結果：元写真のメタデータを返す

### DB 登録

```bash
cd /home/irsl/workspace/CLIP_DB/src

# photos のみ登録（スケッチは後で追加可能）
python3 register_clipdb_assets.py

# photos + sketches 両方登録
python3 register_clipdb_assets.py --sketches
```

### スケッチから SBIR 検索

```bash
export PGHOST=133.15.97.94
export PGUSER=kakinoki_taiyo
export PGPASSWORD=irsl

cd /home/irsl/workspace/CLIP_DB/src
python3 run_sbir_once_from_db.py --sketch_path ../sketches/writing_1.png
```

結果は JSON で stdout に出力されます。

### ROS 統合（自動パイプライン）

手描きスケッチから自動で SBIR 検索・結果保存を実行：

```bash
cd /home/irsl/workspace/irsl_www/script
python3 sub_writing1.py
```

- `/writing` トピックからスケッチを受信
- 自動で SBIR 実行
- 結果を `sketch_result/` に保存
- `/sbir_top5` トピックに結果を publish

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
  --gallery_dir /home/irsl/workspace/CLIP_DB/photos \
  --query_dir /home/irsl/workspace/CLIP_DB/sketches \
  --output_dir /home/irsl/workspace/CLIP_DB/outputs/sbir \
  --model_path /home/irsl/workspace/SketchScape/models/fscoco_normal.pth \
  --topk 5 \
  --device auto
```

`--photo_dir` と `--sketch_dir` も互換用に利用できます。

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
python3 run_sbir_top5.py --gallery_dir /home/irsl/workspace/CLIP_DB/photos --query_dir /home/irsl/workspace/CLIP_DB/sketches --output_dir /home/irsl/workspace/CLIP_DB/outputs/sbir --model_path /home/irsl/workspace/SketchScape/models/fscoco_normal.pth --topk 5 --device auto
```

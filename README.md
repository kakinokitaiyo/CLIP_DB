# CLIP_DB

このワークスペースは、`photos` / `output` / `sketches` を使って
OpenCLIP と SketchScape SBIR を比較するための作業用フォルダです。

現在の基本方針は次のとおりです。

- 画像ファイル本体は `photos/` `output/` `sketches/` に保存する
- PostgreSQL にはメタデータと必要な画像データだけを保存する
- SBIR 実行時に必要なら local cache を使って読み込みを高速化する

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
| `DB_IMAGE_CACHE_DIR` | DB画像のローカルキャッシュ先 | `/tmp/clip_db_image_cache` |
| `RBTE_CACHE_DIR` | RBTE edge のキャッシュ先 | `/tmp/rbte_cache` |
| `RBTE_CACHE_MAX_SIZE_GB` | RBTE キャッシュ上限サイズ | `2.0` |

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

### 推奨運用手順

1. `photos/` と `output/` を用意する
2. 必要なら `sketches/` を追加する
3. DB に登録する
4. `output` を gallery として SBIR を実行する
5. 速度が気になる場合はローカルキャッシュを有効にする

よく使うコマンド:

```bash
cd /home/irsl/workspace/CLIP_DB/src

# photos + output + sketches を登録
python3 register_clipdb_assets.py --outputs --sketches

# photos のみ登録
python3 register_clipdb_assets.py

# SBIR 実行（output を gallery として使用）
python3 run_sbir_once_from_db.py --sketch_path ../sketches/writing_1.png
```

ローカルキャッシュの保存先を変えたい場合は、`DB_IMAGE_CACHE_DIR` と `RBTE_CACHE_DIR` を設定してください。

## データベース登録と SBIR 検索

### アーキテクチャ

**DB 保存 + ローカルキャッシュのハイブリッド構成**

```
DB (photos + output + sketches)
  │
  ├─ photos ──┐
  ├─ output ───┼─ [必要に応じて local cache] ─→ [SBIR 特徴抽出] ─→ [比較]
  └─ sketches ─┘
                                   ↓
                              元写真メタデータ返却
```

- DB 登録時：`photos` `output` `sketches` を必要に応じて保存
- SBIR 実行時：`output` があればそれを直接使う
- `photo` しかない場合は RBTE BDCN でエッジ化してキャッシュする
- 表示結果：元写真のメタデータを返す

### 画像キャッシュについて

他 PC から DB を読みに行くときの遅延を減らすため、
画像本体は各 PC のローカルキャッシュにも保存できます。

- `DB_IMAGE_CACHE_DIR`: DB 画像のローカルキャッシュ
- `RBTE_CACHE_DIR`: RBTE 出力のローカルキャッシュ
- キャッシュがあれば DB 再読込や再 RBTE を省略

### source_type の使い分け

| source_type | 中身 | 使いどころ |
|-------------|------|------------|
| `photo` | 元画像 | 元写真を保持したいとき |
| `output` | RBTE 済み画像 | SBIR を速く回したいときの主力 |
| `sketch` | 手描きスケッチ | クエリや学習データとして使う |

通常運用では、`photos` と `output` を DB に入れておき、
`output` を SBIR の gallery として使うのが分かりやすいです。

### DB 登録

```bash
cd /home/irsl/workspace/CLIP_DB/src

# photos のみ登録
python3 register_clipdb_assets.py

# photos + output + sketches をまとめて登録
python3 register_clipdb_assets.py --outputs --sketches

# photos + sketches だけ登録
python3 register_clipdb_assets.py --sketches
```

### スケッチから SBIR 検索

```bash
cd /home/irsl/workspace/CLIP_DB/src
python3 run_sbir_once_from_db.py --sketch_path ../sketches/writing_1.png
```

結果は JSON で stdout に出力されます。

`output` を DB に入れている場合は、`run_sbir_once_from_db.py` は
`output` を優先して使います。`photo` だけの場合は RBTE を使います。

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

デフォルトでは `output` を gallery として参照します。

## CLIP で比較する
`output` を検索対象、`sketches` をクエリとして top-k を出力します。

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
SketchScape の SBIR モデルを使って、`sketches` をクエリ、
`output` または `photos` を検索対象として比較します。

```bash
cd /home/irsl/workspace/CLIP_DB/src
python3 run_sbir_top5.py \
  --gallery_dir /home/irsl/workspace/CLIP_DB/output \
  --query_dir /home/irsl/workspace/CLIP_DB/sketches \
  --output_dir /home/irsl/workspace/CLIP_DB/outputs/sbir \
  --model_path /home/irsl/workspace/SketchScape/models/fscoco_normal.pth \
  --topk 5 \
  --device auto
```

`--photo_dir` と `--sketch_dir` も互換用に利用できます。

`output` が DB にある場合は、SBIR 実行時に RBTE を再実行せず、そのまま使う構成が基本です。

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

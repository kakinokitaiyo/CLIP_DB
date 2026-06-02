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
| `SBIR_GALLERY_TABLE` | SBIR 検索対象テーブル | `photos_edge` |
| `SBIR_DISPLAY_TABLE` | SBIR 表示用テーブル | `photos` |
| `DB_IMAGE_CACHE_DIR` | DB画像のローカルキャッシュ先 | `/tmp/clip_db_image_cache` |
| `RBTE_CACHE_DIR` | RBTE edge のキャッシュ先 | `/tmp/rbte_cache` |
| `RBTE_CACHE_MAX_SIZE_GB` | RBTE キャッシュ上限サイズ | `2.0` |
| `SBIR_SCAPE_WEIGHT` | SketchScape スコア重み（融合使用時） | `0.7` |
| `SBIR_CLIP_WEIGHT` | CLIP スコア重み（CLIPフュージョン使用時） | `0.3` |
| `SBIR_DINO_WEIGHT` | DINOv2 スコア重み（DINOv2フュージョン使用時） | `0.3` |
| `DINO_IMAGE_EMBEDDINGS_PATH` | DINOv2 事前計算埋め込み .npz ファイルパス（未指定なら DB 読込） | - |
| `ENABLE_DINOV2_FUSION` | DINOv2 融合を有効化（true/false） | `false` |

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

## Tools

補助スクリプトは `src/tools/README.md` にまとめてあります。クローラー、埋め込み生成、診断スクリプトなどの使い方はそちらを参照してください。

See: [CLIP_DB/src/tools/README.md](src/tools/README.md)


ローカルキャッシュの保存先を変えたい場合は、`DB_IMAGE_CACHE_DIR` と `RBTE_CACHE_DIR` を設定してください。

## データベース登録と SBIR 検索

### 2テーブル構成（home_robot.photos / home_robot.photos_edge）

現在は、`photos` 登録時に RBTE を実行して、次の2テーブルを使う構成をサポートしています。

- `home_robot.photos` : 元写真
- `home_robot.photos_edge` : `photos` から生成した edge 画像

実行コマンド:

```bash
cd /home/irsl/workspace/CLIP_DB/src
python3 register_clipdb_assets.py
```

このコマンドで次を行います。

1. `photos/` を `home_robot.photos` に登録
2. 各 photo に RBTE を適用
3. edge 画像を `home_robot.photos_edge` に登録

SBIR 実行時は、デフォルトで `photos_edge` を gallery として使います。

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

### DB の削除・更新（管理用コマンド）

以下は運用・メンテナンス向けのワンライナーです。実行前に必ずバックアップを取り、適切な権限があることを確認してください。

- 環境変数を設定済みの前提（例）:

```bash
export PGHOST="***********"
export PGPORT="****"
export PGDATABASE="kakinoki_db"
export PGUSER="kakinoki_taiyo"
export PGPASSWORD="<your_password>"
```

- テーブル一覧と行数確認:

```bash
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "SELECT table_name FROM information_schema.tables WHERE table_schema='home_robot';"
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "SELECT 'photos' AS tbl, COUNT(*) FROM home_robot.photos;"
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "SELECT 'photos_edge' AS tbl, COUNT(*) FROM home_robot.photos_edge;"
```

- `photos` / `photos_edge` の中身を削除（安全順に `photos_edge` を先に削除）:

```bash
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "BEGIN; DELETE FROM home_robot.photos_edge; DELETE FROM home_robot.photos; COMMIT;"
```

- テーブルごと完全に削除したい場合（テーブル定義も消える）:

```bash
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "DROP TABLE IF EXISTS home_robot.photos_edge; DROP TABLE IF EXISTS home_robot.photos;"
```

- カラム追加（埋め込み列などを追加したいとき）:

```bash
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "ALTER TABLE home_robot.photos ADD COLUMN IF NOT EXISTS clip_model TEXT;"
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "ALTER TABLE home_robot.photos ADD COLUMN IF NOT EXISTS clip_embedding BYTEA;"
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "ALTER TABLE home_robot.photos ADD COLUMN IF NOT EXISTS clip_embedding_updated_at TIMESTAMPTZ;"
```

- 削除後に領域回収（任意）:

```bash
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "VACUUM FULL ANALYZE home_robot.photos; VACUUM FULL ANALYZE home_robot.photos_edge;"
```

注意: これらは取り消し不可の操作です。必ずバックアップを取り、実行前に確認してください。


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

## DINOv2 による再ランキング（セマンティック融合）

SketchScape（形状ベース）の SBIR 結果を、DINOv2（セマンティック）により再ランキングして **Recall@5 を改善**できます。

### 概要
- **DINOv2**: ViT ベースの自己教師あり視覚表現。スケッチと写真のセマンティック差を捉えられます
- **融合方式**: `final_score = α × norm(SketchScape) + β × norm(DINOv2)`
- **埋め込み保存**: PostgreSQL `photo_embeddings` テーブルに float32 L2-正規化済み埋め込みを bytea で保存

### セットアップ
#### 1. DINOv2 埋め込みをDB に保存

初回のみ実行してください。全ギャラリー画像（74 枚）から DINOv2 埋め込みを計算・DB に upsert します：

```bash
cd /home/irsl/workspace/CLIP_DB/src

# 初回：全画像を処理してDB に保存
python3 tools/compute_dinov2_embeddings_db.py

# オプション: 既存の埋め込みを削除してやり直す場合
python3 tools/compute_dinov2_embeddings_db.py --clear-existing

# 強制的に上書きする場合
python3 tools/compute_dinov2_embeddings_db.py --force
```

#### 2. run_sbir_once_from_db.py で DINOv2 融合を有効化

```bash
python3 run_sbir_once_from_db.py \
  --sketch_path ../sketches/writing_1.png \
  --topk 5 \
  --enable_dinov2_fusion \
  --dinov2_weight 0.3 \
  --scape_weight 0.7
```

- `--enable_dinov2_fusion`: DINOv2 再ランキングを有効化
- `--dinov2_weight`: DINOv2 スコアの重み（推奨: 0.2～0.4）
- `--scape_weight`: SketchScape スコアの重み（推奨: 0.6～0.8）
- `--dinov2_embeddings_path`: (オプション) .npz キャッシュファイル。未指定なら DB から自動読込

### 環境変数設定（全スクリプトで DINOv2 有効化）

```bash
export ENABLE_DINOV2_FUSION=true
export SBIR_DINO_WEIGHT=0.3
export SBIR_SCAPE_WEIGHT=0.7
```

その後、`sub_writing1.py` など他のスクリプトでも DINOv2 が有効になります。

### ROS 統合（DINOv2 有効）

```bash
export ENABLE_DINOV2_FUSION=true
export SBIR_DINO_WEIGHT=0.3

cd /home/irsl/workspace/irsl_www/script
python3 sub_writing1.py
```

### テスト結果例（writing_1.png）

| ランク | Baseline（SketchScape のみ） | DINOv2 融合（weight 0.3） | 改善 |
|---------|------|------|------|
| **1** | IMG_2796.JPG (0.735) | **apple.jpeg (0.879)** ✨ | rank 7→1 |
| **2** | IMG_2800.JPG (0.723) | IMG_2794.JPG (0.838) | - |
| **3** | IMG_2794.JPG (0.695) | IMG_2800.JPG (0.784) | - |
| **4** | IMG_2824.JPG (0.685) | IMG_2796.JPG (0.783) | - |
| **5** | IMG_2802.JPG (0.675) | **peach.jpeg (0.729)** ✨ | rank 9→5 |

- DINOv2 により **apple.jpeg** と **peach.jpeg** が正しく上位に浮上
- 実物写真と形状類似性が高い画像の結果品質が向上

### トラブルシューティング

#### DB に DINOv2 埋め込みがない場合
```
[WARN] DINOv2 fusion failed: No embeddings found in photo_embeddings table
```
→ `compute_dinov2_embeddings_db.py` を実行して DB に埋め込みを保存してください。

#### オンザフライ計算に切り替える場合
DB 埋め込みがなくても、`--enable_dinov2_fusion` を指定すれば候補画像だけ DINOv2 を オンザフライで計算します。初回は時間がかかりますが、2 回目以降はスキップされます。

```bash
python3 run_sbir_once_from_db.py \
  --sketch_path ../sketches/writing_1.png \
  --enable_dinov2_fusion \
  --dinov2_weight 0.3
```

### 重みのチューニング
- `--scape_weight + --dinov2_weight` の合計が 1.0 である必要はなく、比率で正規化されます
- 形状が重要なデータセット: `--scape_weight 0.8 --dinov2_weight 0.2`
- セマンティックが重要: `--scape_weight 0.6 --dinov2_weight 0.4`
- 最適値はデータセットによって異なるため、グリッド探索を推奨
## よく使う実行例
### CLIP
```bash
python3 run_clip_top5.py --gallery_dir /home/irsl/workspace/CLIP_DB/output --query_dir /home/irsl/workspace/CLIP_DB/sketches --output_dir /home/irsl/workspace/CLIP_DB/outputs/clip --topk 5
```

### SBIR（SketchScape のみ）
```bash
python3 run_sbir_top5.py --gallery_dir /home/irsl/workspace/CLIP_DB/photos --query_dir /home/irsl/workspace/CLIP_DB/sketches --output_dir /home/irsl/workspace/CLIP_DB/outputs/sbir --model_path /home/irsl/workspace/SketchScape/models/fscoco_normal.pth --topk 5 --device auto
```

### SBIR + DINOv2 融合
```bash
cd /home/irsl/workspace/CLIP_DB/src
python3 run_sbir_once_from_db.py \
  --sketch_path ../sketches/writing_1.png \
  --topk 5 \
  --enable_dinov2_fusion \
  --dinov2_weight 0.3 \
  --scape_weight 0.7
```

### ROS パイプライン（DINOv2 有効）
```bash
export ENABLE_DINOV2_FUSION=true
export SBIR_DINO_WEIGHT=0.3
cd /home/irsl/workspace/irsl_www/script
python3 sub_writing1.py
```
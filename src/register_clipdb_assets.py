from __future__ import annotations
"""
CLIP_DBアセットをPostgreSQLに登録するスクリプト

このモジュールは、CLIP_DBプロジェクトのアセット（写真、スケッチ）を
PostgreSQLデータベースに登録するための機能を提供します。

新規フロー：
    - photos: ロボット撮影写真（元写真）
    - sketches: 手描きスケッチ
    - SBIR実行時に photos をオンデマンド RBTE エッジ検出

主な機能:
    - コマンドライン引数の解析（データベース接続情報、ディレクトリパス）
    - photos と sketches の登録
    - 登録スクリプトの外部実行とプロセス管理
    - ドライラン機能によるテスト実行対応

環境変数:
    PGHOST: PostgreSQLホスト名（デフォルト: localhost）
    PGPORT: PostgreSQLポート番号（デフォルト: 5432）
    PGDATABASE: データベース名（デフォルト: kakinoki_db）
    PGUSER: PostgreSQLユーザー名
    PGPASSWORD: PostgreSQLパスワード

使用例:
    python register_clipdb_assets.py --sketches --host localhost --dbname mydb

関数:
    parse_args(): コマンドライン引数を解析して返す
    run_register(): 指定されたディレクトリのアセットを登録する
    main(): メイン処理を実行する
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTER_SCRIPT = ROOT / "src" / "register_sketches_to_db.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register CLIP_DB assets into PostgreSQL.")
    parser.add_argument("--host", type=str, default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=str, default=os.getenv("PGPORT", "5432"))
    parser.add_argument("--dbname", type=str, default=os.getenv("PGDATABASE", "kakinoki_db"))
    parser.add_argument("--user", type=str, default=os.getenv("PGUSER", ""))
    parser.add_argument("--password", type=str, default=os.getenv("PGPASSWORD", ""))
    parser.add_argument("--schema", type=str, default="home_robot")
    parser.add_argument("--table", type=str, default="sketch_images")

    parser.add_argument("--photos_dir", type=Path, default=ROOT / "photos")
    parser.add_argument("--outputs_dir", type=Path, default=ROOT / "output")
    parser.add_argument("--sketches_dir", type=Path, default=ROOT / "sketches")

    parser.add_argument("--outputs", action="store_true", help="output も登録する（RBTE 済み画像）")
    parser.add_argument("--sketches", action="store_true", help="sketches も登録する（デフォルト: photos のみ）")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def run_register(
    source_dir: Path,
    source_type: str,
    args: argparse.Namespace,
) -> None:
    cmd = [
        sys.executable,
        str(REGISTER_SCRIPT),
        "--source_dir",
        str(source_dir),
        "--recursive",
        "--source_type",
        source_type,
        "--source_app",
        "CLIP_DB",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dbname",
        args.dbname,
        "--user",
        args.user,
        "--password",
        args.password,
        "--schema",
        args.schema,
        "--table",
        args.table,
    ]
    if args.dry_run:
        cmd.append("--dry_run")

    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    # Always register photos
    run_register(args.photos_dir.expanduser().resolve(), "photo", args)

    # Optionally register RBTE outputs
    if args.outputs:
        run_register(args.outputs_dir.expanduser().resolve(), "output", args)

    # Optionally register sketches
    if args.sketches:
        run_register(args.sketches_dir.expanduser().resolve(), "sketch", args)


if __name__ == "__main__":
    main()


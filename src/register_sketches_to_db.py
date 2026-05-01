from __future__ import annotations

import argparse
import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Iterable, List

import psycopg
from PIL import Image
from psycopg import sql


DEFAULT_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register sketch image files to PostgreSQL (psycopg)."
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        required=True,
        help="登録対象フォルダ（例: irsl_www 側の保存先）",
    )
    parser.add_argument("--recursive", action="store_true", help="サブフォルダも探索する")

    parser.add_argument("--host", type=str, default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--dbname", type=str, default=os.getenv("PGDATABASE", "kakinoki_db"))
    parser.add_argument("--user", type=str, default=os.getenv("PGUSER", ""))
    parser.add_argument("--password", type=str, default=os.getenv("PGPASSWORD", ""))

    parser.add_argument("--schema", type=str, default="home_robot")
    parser.add_argument("--table", type=str, default="sketch_images")
    parser.add_argument("--source_app", type=str, default="irsl_www")
    parser.add_argument(
        "--source_type",
        type=str,
        default="sketch",
        choices=["photo", "output", "sketch", "other"],
        help="画像の種別メタデータ",
    )
    parser.add_argument("--dry_run", action="store_true", help="DB書き込みなしで件数確認のみ")

    return parser.parse_args()


def list_images(folder: Path, recursive: bool) -> List[Path]:
    if recursive:
        paths: Iterable[Path] = folder.rglob("*")
    else:
        paths = folder.glob("*")
    return sorted([p for p in paths if p.is_file() and p.suffix.lower() in DEFAULT_EXTS])


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def ensure_table(conn: psycopg.Connection, schema: str, table: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(sql.Identifier(schema)))
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.{} (
                    id BIGSERIAL PRIMARY KEY,
                    source_app TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'other',
                    source_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    mime_type TEXT,
                    width INTEGER,
                    height INTEGER,
                    image_sha256 TEXT NOT NULL UNIQUE,
                    image_data BYTEA NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            ).format(sql.Identifier(schema), sql.Identifier(table))
        )
        cur.execute(
            sql.SQL(
                "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'other';"
            ).format(sql.Identifier(schema), sql.Identifier(table))
        )



def upsert_sketch(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    source_app: str,
    source_type: str,
    image_path: Path,
) -> None:
    data = image_path.read_bytes()
    img_hash = sha256_of_bytes(data)
    width, height = get_image_size(image_path)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"

    q = sql.SQL(
        """
        INSERT INTO {}.{} (
            source_app, source_type, source_path, file_name, mime_type,
            width, height, image_sha256, image_data
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (image_sha256)
        DO UPDATE SET
            source_app = EXCLUDED.source_app,
            source_type = EXCLUDED.source_type,
            source_path = EXCLUDED.source_path,
            file_name = EXCLUDED.file_name,
            mime_type = EXCLUDED.mime_type,
            width = EXCLUDED.width,
            height = EXCLUDED.height,
            image_data = EXCLUDED.image_data,
            updated_at = NOW();
        """
    ).format(sql.Identifier(schema), sql.Identifier(table))

    with conn.cursor() as cur:
        cur.execute(
            q,
            (
                source_app,
                source_type,
                str(image_path),
                image_path.name,
                mime_type,
                width,
                height,
                img_hash,
                data,
            ),
        )



def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"source_dir not found: {source_dir}")

    images = list_images(source_dir, args.recursive)
    if not images:
        raise FileNotFoundError(f"No image files found in: {source_dir}")

    print(f"found images: {len(images)}")
    print(f"source_dir: {source_dir}")

    if args.dry_run:
        for p in images[:10]:
            print(f"[DRY_RUN] {p}")
        if len(images) > 10:
            print(f"... and {len(images) - 10} more")
        return

    if not args.user:
        raise ValueError("DB user is empty. Set --user or PGUSER.")

    with psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    ) as conn:
        ensure_table(conn, args.schema, args.table)

        for idx, path in enumerate(images, start=1):
            upsert_sketch(
                conn=conn,
                schema=args.schema,
                table=args.table,
                source_app=args.source_app,
                source_type=args.source_type,
                image_path=path,
            )
            if idx % 20 == 0 or idx == len(images):
                print(f"processed {idx}/{len(images)}")

        conn.commit()

    print("done")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import io
import mimetypes
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import psycopg
import torch
from PIL import Image
from psycopg import sql


DEFAULT_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register photos and RBTE edges into separate tables: photos / photos_edge."
    )
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")

    parser.add_argument("--host", type=str, default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--dbname", type=str, default=os.getenv("PGDATABASE", "kakinoki_db"))
    parser.add_argument("--user", type=str, default=os.getenv("PGUSER", ""))
    parser.add_argument("--password", type=str, default=os.getenv("PGPASSWORD", ""))

    parser.add_argument("--schema", type=str, default="home_robot")
    parser.add_argument("--photos_table", type=str, default="photos")
    parser.add_argument("--photos_edge_table", type=str, default="photos_edge")
    parser.add_argument("--source_app", type=str, default="CLIP_DB")

    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("RBTE_DEVICE", "auto"),
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--enable_clip", action="store_true", help="Generate and store CLIP embeddings for each photo into DB")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def list_images(folder: Path, recursive: bool) -> list[Path]:
    paths = folder.rglob("*") if recursive else folder.glob("*")
    return sorted([p for p in paths if p.is_file() and p.suffix.lower() in DEFAULT_EXTS])


def select_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda が指定されましたが CUDA が利用できません。")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def load_rgb_image_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        img = img.convert("RGBA")
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white_bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def load_bdcn_model(device: str = "cuda"):
    bdcn_root = Path(os.getenv("RBTE_DOCKER_ROOT", "/home/irsl/workspace/rbte_docker"))
    bdcn_dir = bdcn_root / "BDCN"
    bdcn_weight = bdcn_root / "bdcn_model" / "final-model" / "bdcn_pretrained_on_bsds500.pth"
    if not bdcn_weight.exists():
        raise FileNotFoundError(f"BDCN weight not found: {bdcn_weight}")

    sys.path.insert(0, str(bdcn_dir))
    import bdcn

    model = bdcn.BDCN()
    model.load_state_dict(torch.load(bdcn_weight, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def detect_edge_from_image(image_rgb: Image.Image, bdcn_model, device: str = "cuda") -> Image.Image:
    # Resize image to reduce memory usage (max 512 pixels on longest side)
    max_size = 512
    if max(image_rgb.size) > max_size:
        image_rgb.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    img_np = np.array(image_rgb)
    img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    data = img_np_bgr.astype(np.float32)
    data -= np.array([[[104.00699, 116.66877, 122.67892]]])
    data_tensor = torch.Tensor(data[np.newaxis, :, :, :].transpose(0, 3, 1, 2)).to(device)

    with torch.no_grad():
        out = bdcn_model(data_tensor)

    edge_map = torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
    edge_map = (edge_map * 255).astype(np.uint8)
    
    # Cleanup GPU memory
    del data_tensor, out
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return Image.fromarray(edge_map, mode="L").convert("RGB")


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_clip_model_for_encoding(clip_model: str, clip_pretrained: str, device: str = "cpu"):
    try:
        import open_clip
    except Exception as e:
        raise ImportError("open_clip is required for CLIP embedding generation. Install with: pip install open-clip-torch") from e

    model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=clip_pretrained)
    model = model.to(device)
    model.eval()
    return model, preprocess


def encode_image_bytes_to_normalized_numpy(image_bytes: bytes, preprocess, model, device: str = "cpu"):
    img = load_rgb_image_from_bytes(image_bytes)
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        arr = feat.detach().cpu().numpy().astype("float32")[0]
    return arr


def ensure_photos_table(conn: psycopg.Connection, schema: str, table: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}; ").format(sql.Identifier(schema)))
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.{} (
                    id BIGSERIAL PRIMARY KEY,
                    source_app TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'photo',
                    source_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    mime_type TEXT,
                    width INTEGER,
                    height INTEGER,
                    image_sha256 TEXT NOT NULL UNIQUE,
                    clip_model TEXT,
                    clip_embedding BYTEA,
                    clip_embedding_updated_at TIMESTAMPTZ,
                    image_data BYTEA NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            ).format(sql.Identifier(schema), sql.Identifier(table))
        )


def ensure_photos_edge_table(conn: psycopg.Connection, schema: str, table: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}; ").format(sql.Identifier(schema)))
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.{} (
                    id BIGSERIAL PRIMARY KEY,
                    source_app TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'photo_edge',
                    source_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    mime_type TEXT,
                    width INTEGER,
                    height INTEGER,
                    image_sha256 TEXT NOT NULL UNIQUE,
                    photo_sha256 TEXT NOT NULL,
                    image_data BYTEA NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            ).format(sql.Identifier(schema), sql.Identifier(table))
        )


def upsert_photo(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    source_app: str,
    image_path: Path,
) -> str:
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
        VALUES (%s, 'photo', %s, %s, %s, %s, %s, %s, %s)
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
                str(image_path),
                image_path.name,
                mime_type,
                width,
                height,
                img_hash,
                data,
            ),
        )
    return img_hash


def upsert_photo_edge(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    source_app: str,
    image_path: Path,
    photo_sha256: str,
    edge_data: bytes,
) -> None:
    edge_hash = sha256_of_bytes(edge_data)
    with Image.open(io.BytesIO(edge_data)) as img:
        width, height = img.size

    edge_file_name = f"{image_path.stem}_edge.png"
    edge_source_path = str(image_path.with_name(edge_file_name))

    q = sql.SQL(
        """
        INSERT INTO {}.{} (
            source_app, source_type, source_path, file_name, mime_type,
            width, height, image_sha256, photo_sha256, image_data
        )
        VALUES (%s, 'photo_edge', %s, %s, 'image/png', %s, %s, %s, %s, %s)
        ON CONFLICT (image_sha256)
        DO UPDATE SET
            source_app = EXCLUDED.source_app,
            source_type = EXCLUDED.source_type,
            source_path = EXCLUDED.source_path,
            file_name = EXCLUDED.file_name,
            mime_type = EXCLUDED.mime_type,
            width = EXCLUDED.width,
            height = EXCLUDED.height,
            photo_sha256 = EXCLUDED.photo_sha256,
            image_data = EXCLUDED.image_data,
            updated_at = NOW();
        """
    ).format(sql.Identifier(schema), sql.Identifier(table))

    with conn.cursor() as cur:
        cur.execute(
            q,
            (
                source_app,
                edge_source_path,
                edge_file_name,
                width,
                height,
                edge_hash,
                photo_sha256,
                edge_data,
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

    device = select_device(args.device)
    print(f"RBTE device: {device}")
    bdcn_model = load_bdcn_model(device=device)

    clip_model_obj = None
    clip_preprocess = None
    if args.enable_clip:
        print(f"[INFO] CLIP embedding generation enabled: model={args.clip_model} pretrained={args.clip_pretrained}")
        clip_model_obj, clip_preprocess = load_clip_model_for_encoding(args.clip_model, args.clip_pretrained, device=device)

    with psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    ) as conn:
        ensure_photos_table(conn, args.schema, args.photos_table)
        ensure_photos_edge_table(conn, args.schema, args.photos_edge_table)

        for idx, path in enumerate(images, start=1):
            photo_sha = upsert_photo(
                conn=conn,
                schema=args.schema,
                table=args.photos_table,
                source_app=args.source_app,
                image_path=path,
            )

            # If enabled, compute CLIP embedding for the original photo and store into DB
            if args.enable_clip and clip_model_obj is not None and clip_preprocess is not None:
                try:
                    img_bytes = path.read_bytes()
                    emb_np = encode_image_bytes_to_normalized_numpy(img_bytes, clip_preprocess, clip_model_obj, device=device)
                    emb_bytes = emb_np.tobytes()
                    with conn.cursor() as cur:
                        cur.execute(
                            sql.SQL(
                                "UPDATE {}.{} SET clip_model=%s, clip_embedding=%s, clip_embedding_updated_at=NOW() WHERE image_sha256=%s"
                            ).format(sql.Identifier(args.schema), sql.Identifier(args.photos_table)),
                            (f"{args.clip_model}:{args.clip_pretrained}", psycopg.Binary(emb_bytes), photo_sha),
                        )
                except Exception as e:
                    print(f"[WARN] failed to generate/store CLIP embedding for {path}: {e}")

            photo_img = load_rgb_image_from_bytes(path.read_bytes())
            edge_img = detect_edge_from_image(photo_img, bdcn_model, device=device)
            edge_bytes = pil_to_png_bytes(edge_img)

            upsert_photo_edge(
                conn=conn,
                schema=args.schema,
                table=args.photos_edge_table,
                source_app=args.source_app,
                image_path=path,
                photo_sha256=photo_sha,
                edge_data=edge_bytes,
            )
            
            # Cleanup memory after each image
            del photo_img, edge_img, edge_bytes
            if device == "cuda":
                torch.cuda.empty_cache()

            if idx % 10 == 0 or idx == len(images):
                print(f"processed {idx}/{len(images)}")

        conn.commit()

    if device == "cuda":
        bdcn_model.cpu()
        torch.cuda.empty_cache()

    print("done")


if __name__ == "__main__":
    main()

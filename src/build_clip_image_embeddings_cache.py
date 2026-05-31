from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import torch
from PIL import Image
import io
from psycopg import sql


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLIP image embedding cache (.npz) from DB images.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", type=str, default="kakinoki_db")
    parser.add_argument("--user", type=str, required=True)
    parser.add_argument("--password", type=str, default="")
    parser.add_argument("--schema", type=str, default="home_robot")
    parser.add_argument("--table", type=str, default="photos")
    parser.add_argument("--source_type", type=str, default="photo")
    parser.add_argument("--output", type=Path, default=Path("~/workspace/CLIP_DB/cache/clip_image_embeddings.npz"))
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def select_device(arg: str) -> str:
    if arg == "cpu":
        return "cpu"
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was specified, but CUDA is unavailable.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def fetch_db_images(args: argparse.Namespace) -> list[dict[str, Any]]:
    query = sql.SQL(
        """
        SELECT id, file_name, image_data
        FROM {}.{}
        WHERE source_type = %s
        ORDER BY id;
        """
    ).format(sql.Identifier(args.schema), sql.Identifier(args.table))

    with psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (args.source_type,))
            rows = cur.fetchall()

    return [
        {
            "id": int(r[0]),
            "file_name": r[1],
            "image_data": bytes(r[2]),
        }
        for r in rows
    ]


def load_rgb_image_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        img = img.convert("RGBA")
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white_bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    try:
        import open_clip
    except ImportError as e:
        raise ImportError("open_clip is required. Install with: pip install open-clip-torch") from e

    rows = fetch_db_images(args)
    if not rows:
        raise FileNotFoundError(
            f"No rows found in {args.schema}.{args.table} for source_type='{args.source_type}'"
        )

    print(f"[INFO] rows: {len(rows)}")
    print(f"[INFO] device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
    )
    model = model.to(device)
    model.eval()

    ids: list[int] = []
    embeddings: list[np.ndarray] = []

    batch_size = max(1, args.batch_size)
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            imgs = []
            chunk_ids = []
            for row in chunk:
                pil = load_rgb_image_from_bytes(row["image_data"])
                imgs.append(preprocess(pil))
                chunk_ids.append(row["id"])

            batch = torch.stack(imgs, dim=0).to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_np = feats.detach().cpu().numpy().astype(np.float32)

            ids.extend(chunk_ids)
            embeddings.extend([f for f in feats_np])

            if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(rows):
                print(f"[INFO] processed {min(i + batch_size, len(rows))}/{len(rows)}")

    ids_np = np.asarray(ids, dtype=np.int64)
    emb_np = np.asarray(embeddings, dtype=np.float32)

    # Ensure strict L2 normalization before save (critical for dot-product cosine)
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    emb_np = emb_np / norms

    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        ids=ids_np,
        embeddings=emb_np,
    )

    print(f"[INFO] saved: {out}")
    print(f"[INFO] shape: ids={ids_np.shape}, embeddings={emb_np.shape}")


if __name__ == "__main__":
    main()

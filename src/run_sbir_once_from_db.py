from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psycopg
import torch
from PIL import Image
from psycopg import sql


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SBIR for one sketch image against photo images stored in PostgreSQL."
    )
    parser.add_argument("--sketch_path", type=Path, required=True, help="クエリスケッチ画像パス")
    parser.add_argument("--topk", type=int, default=5, help="上位件数")

    parser.add_argument("--host", type=str, default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--dbname", type=str, default=os.getenv("PGDATABASE", "kakinoki_db"))
    parser.add_argument("--user", type=str, default=os.getenv("PGUSER", ""))
    parser.add_argument("--password", type=str, default=os.getenv("PGPASSWORD", ""))

    parser.add_argument("--schema", type=str, default="home_robot")
    parser.add_argument("--table", type=str, default="sketch_images")
    parser.add_argument(
        "--gallery_source_type",
        "--photo_source_type",
        dest="gallery_source_type",
        type=str,
        default="photo",
        help="検索対象の source_type (デフォルト: photo、実行時に RBTE エッジ検出を適用)",
    )
    parser.add_argument(
        "--display_source_type",
        type=str,
        default="photo",
        help="結果表示に使う元画像の source_type",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="推論デバイス",
    )
    parser.add_argument(
        "--sketchscape_root",
        type=Path,
        default=Path("~/workspace/SketchScape"),
        help="SketchScape ルート",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("~/workspace/SketchScape/models/fscoco_normal.pth"),
        help="SketchScape 学習済み重み",
    )
    return parser.parse_args()


def select_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda が指定されましたが CUDA が利用できません。")
        return "cuda"

    if not torch.cuda.is_available():
        return "cpu"

    try:
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        if arch not in set(torch.cuda.get_arch_list()):
            return "cpu"
    except Exception:
        return "cpu"
    return "cuda"


def load_rgb_image(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        img = img.convert("RGBA")
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white_bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return img


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
    """Load BDCN edge detection model."""
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
    """Apply BDCN edge detection to RGB image."""
    # Convert PIL image to numpy array in BGR for BDCN
    img_np = np.array(image_rgb)  # HxWx3 RGB
    img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert to BGR
    
    # Prepare for BDCN (subtract mean)
    data = img_np_bgr.astype(np.float32)
    data -= np.array([[[104.00699, 116.66877, 122.67892]]])  # ImageNet mean
    
    # Convert to tensor (BxCxHxW)
    data_tensor = torch.Tensor(data[np.newaxis, :, :, :].transpose(0, 3, 1, 2)).to(device)
    
    with torch.no_grad():
        out = bdcn_model(data_tensor)
    
    # Get edge map (fuse is the final edge prediction)
    edge_map = torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]  # HxW [0,1]
    edge_map = (edge_map * 255).astype(np.uint8)  # Convert to [0,255]
    
    # Convert to PIL RGB image
    edge_img = Image.fromarray(edge_map, mode="L").convert("RGB")
    return edge_img


def canonical_name_key(file_name: str) -> str:
    stem = Path(file_name).stem.lower()
    stem = re.sub(r"_edge$", "", stem)
    return stem


def fetch_images_from_db(args: argparse.Namespace, source_type: str, with_data: bool) -> list[dict[str, Any]]:
    if not args.user:
        raise ValueError("DB user is empty. Set --user or PGUSER.")

    select_cols = "id, file_name, source_path, image_data" if with_data else "id, file_name, source_path"
    query = sql.SQL(
        f"""
        SELECT {select_cols}
        FROM {{}}.{{}}
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
            cur.execute(query, (source_type,))
            rows = cur.fetchall()

    if with_data:
        return [
            {
                "id": r[0],
                "file_name": r[1],
                "source_path": r[2],
                "image_data": bytes(r[3]),
            }
            for r in rows
        ]

    return [
        {
            "id": r[0],
            "file_name": r[1],
            "source_path": r[2],
        }
        for r in rows
    ]


def main() -> None:
    args = parse_args()

    sketch_path = args.sketch_path.expanduser().resolve()
    if not sketch_path.is_file():
        raise FileNotFoundError(f"sketch_path not found: {sketch_path}")

    gallery_images = fetch_images_from_db(args, args.gallery_source_type, with_data=True)
    if not gallery_images:
        raise FileNotFoundError(
            f"No gallery images found in DB for source_type='{args.gallery_source_type}'. "
            "Register images first."
        )

    display_photos = fetch_images_from_db(args, args.display_source_type, with_data=False)
    if not display_photos:
        raise FileNotFoundError(
            f"No display photos found in DB for source_type='{args.display_source_type}'."
        )

    display_by_key: dict[str, dict[str, Any]] = {}
    for p in display_photos:
        key = canonical_name_key(p["file_name"])
        if key not in display_by_key:
            display_by_key[key] = p

    device = select_device(args.device)

    # Load BDCN edge detection model
    bdcn_model = load_bdcn_model(device=device)

    # Apply BDCN edge detection to gallery images (in-memory)
    gallery_images_edge = []
    for g in gallery_images:
        rgb_img = load_rgb_image_from_bytes(g["image_data"])
        edge_img = detect_edge_from_image(rgb_img, bdcn_model, device=device)
        gallery_images_edge.append({
            **g,
            "image_data_edge": np.array(edge_img),  # Store edge-detected RGB image
        })

    sketchscape_root = args.sketchscape_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model_path not found: {model_path}")

    sys.path.insert(0, str(sketchscape_root))
    from model import SBIRModel

    model = SBIRModel.load_module(str(model_path), strict=False, cuda=(device == "cuda"))
    model = model.to(device)
    model.eval()

    sketch_transform = model.create_sketch_transforms()
    image_transform = model.create_image_transforms()

    # Convert edge-detected images to PIL, then apply transform
    with torch.no_grad():
        gallery_tensors = []
        for g in gallery_images_edge:
            edge_pil = Image.fromarray(g["image_data_edge"])
            edge_tensor = image_transform(edge_pil).unsqueeze(0)
            gallery_tensors.append(edge_tensor)
        
        gallery_batch = torch.cat(gallery_tensors, dim=0).to(device)
        gallery_features = model(gallery_batch)
        gallery_features = gallery_features / gallery_features.norm(dim=-1, keepdim=True)

    sketch_tensor = sketch_transform(load_rgb_image(sketch_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        sketch_feature = model(sketch_tensor)
        sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)

    sims = (sketch_feature @ gallery_features.T).squeeze(0).cpu()
    topk = max(1, args.topk)
    vals, idxs = torch.topk(sims, k=min(topk, len(gallery_images_edge)))

    ranked = []
    for rank, (score, i) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        gallery = gallery_images[i]  # Use original gallery_images for metadata
        key = canonical_name_key(gallery["file_name"])
        photo = display_by_key.get(key)

        ranked.append(
            {
                "rank": rank,
                "gallery_id": int(gallery["id"]),
                "gallery_file": gallery["file_name"],
                "photo_id": int(photo["id"]) if photo is not None else None,
                "photo_file": photo["file_name"] if photo is not None else None,
                "photo_source_path": photo["source_path"] if photo is not None else None,
                "score": float(score),
            }
        )

    result = {
        "sketch_file": sketch_path.name,
        "device": device,
        "topk": ranked,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

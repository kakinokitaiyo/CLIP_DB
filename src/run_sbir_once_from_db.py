from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import hashlib
import gc
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
        default="output",
        help="検索対象の source_type (デフォルト: output、photo の場合のみ RBTE を適用)",
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
    """Apply BDCN edge detection to RGB image (GPU-accelerated)."""
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


def get_db_image_cache_dir() -> Path:
    """Return local cache directory for DB-backed images."""
    cache_dir = Path(os.getenv("DB_IMAGE_CACHE_DIR", "/tmp/clip_db_image_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def sanitize_cache_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def get_db_image_cache_path(cache_dir: Path, source_type: str, image_sha256: str, file_name: str) -> Path:
    suffix = Path(file_name).suffix.lower() or ".bin"
    safe_name = sanitize_cache_name(Path(file_name).stem)
    return cache_dir / f"{source_type}_{safe_name}_{image_sha256[:12]}{suffix}"


def load_cached_image_bytes(cache_path: Path) -> bytes | None:
    if cache_path.exists():
        try:
            return cache_path.read_bytes()
        except Exception as e:
            print(f"[WARN] Failed to load cache {cache_path}: {e}")
    return None


def save_cached_image_bytes(cache_path: Path, data: bytes) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    except Exception as e:
        print(f"[WARN] Failed to save cache {cache_path}: {e}")


def resolve_db_images_with_cache(
    args: argparse.Namespace,
    source_type: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach image_data to metadata rows using a local cache with DB fallback."""
    if not rows:
        return []

    cache_dir = get_db_image_cache_dir()
    resolved: list[dict[str, Any]] = []
    missing_ids: list[int] = []
    cache_paths: dict[int, Path] = {}

    for row in rows:
        cache_path = get_db_image_cache_path(cache_dir, source_type, row["image_sha256"], row["file_name"])
        cache_paths[int(row["id"])] = cache_path
        cached = load_cached_image_bytes(cache_path)
        if cached is None:
            missing_ids.append(int(row["id"]))
            resolved.append({**row})
        else:
            resolved.append({**row, "image_data": cached})

    if missing_ids:
        fetched = fetch_images_from_db(args, source_type, with_data=True, ids=missing_ids)
        fetched_by_id = {int(row["id"]): row for row in fetched}
        for row in resolved:
            row_id = int(row["id"])
            if "image_data" in row:
                continue
            fetched_row = fetched_by_id.get(row_id)
            if fetched_row is None:
                raise FileNotFoundError(f"DB row not found for id={row_id}, source_type={source_type}")
            row["image_data"] = fetched_row["image_data"]
            save_cached_image_bytes(cache_paths[row_id], row["image_data"])

    return resolved


def get_rbte_cache_dir() -> Path:
    """Get RBTE cache directory (with LRU management)."""
    cache_dir = Path(os.getenv("RBTE_CACHE_DIR", "/tmp/rbte_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_rbte_cache_key(image_id: int, image_data: bytes) -> str:
    """Generate cache key as: photo_<id>_<data_hash>.npy"""
    data_hash = hashlib.md5(image_data).hexdigest()[:8]
    return f"photo_{image_id}_{data_hash}.npy"


def load_cached_edge(cache_path: Path) -> np.ndarray | None:
    """Load cached edge-detected image (HxW uint8 grayscale)."""
    if cache_path.exists():
        try:
            edge_map = np.load(cache_path)
            return edge_map
        except Exception as e:
            print(f"[WARN] Failed to load cache {cache_path}: {e}")
    return None


def save_cached_edge(cache_path: Path, edge_map: np.ndarray) -> None:
    """Save edge-detected image to cache (HxW uint8 grayscale)."""
    try:
        np.save(cache_path, edge_map)
    except Exception as e:
        print(f"[WARN] Failed to save cache {cache_path}: {e}")


def cleanup_old_cache(cache_dir: Path, max_size_gb: float = 2.0) -> None:
    """Remove oldest cached files if total size exceeds max_size_gb (LRU policy)."""
    if not cache_dir.exists():
        return
    
    max_bytes = max_size_gb * 1024**3
    
    # Get all cache files with their sizes and modification times
    cache_files = []
    for f in cache_dir.glob("photo_*.npy"):
        stat = f.stat()
        cache_files.append((f, stat.st_size, stat.st_mtime))
    
    total_size = sum(s for _, s, _ in cache_files)
    
    if total_size > max_bytes:
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[2])
        
        # Remove oldest files until under limit
        for cache_file, size, _ in cache_files:
            if total_size <= max_bytes:
                break
            try:
                cache_file.unlink()
                total_size -= size
                print(f"[CACHE] Removed old cache: {cache_file.name}")
            except Exception as e:
                print(f"[WARN] Failed to remove cache {cache_file}: {e}")


def fetch_images_from_db(
    args: argparse.Namespace,
    source_type: str,
    with_data: bool,
    ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    if not args.user:
        raise ValueError("DB user is empty. Set --user or PGUSER.")

    select_cols = "id, file_name, source_path, image_sha256, image_data" if with_data else "id, file_name, source_path, image_sha256"

    if ids:
        where_clause = sql.SQL("WHERE source_type = %s AND id = ANY(%s)")
        params: tuple[Any, ...] = (source_type, ids)
    else:
        where_clause = sql.SQL("WHERE source_type = %s")
        params = (source_type,)

    query = sql.SQL(
        f"""
        SELECT {select_cols}
        FROM {{}}.{{}}
        {{}}
        ORDER BY id;
        """
    ).format(sql.Identifier(args.schema), sql.Identifier(args.table), where_clause)

    with psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    if with_data:
        return [
            {
                "id": r[0],
                "file_name": r[1],
                "source_path": r[2],
                "image_sha256": r[3],
                "image_data": bytes(r[4]),
            }
            for r in rows
        ]

    return [
        {
            "id": r[0],
            "file_name": r[1],
            "source_path": r[2],
            "image_sha256": r[3],
        }
        for r in rows
    ]


def main() -> None:
    args = parse_args()

    sketch_path = args.sketch_path.expanduser().resolve()
    if not sketch_path.is_file():
        raise FileNotFoundError(f"sketch_path not found: {sketch_path}")

    gallery_meta = fetch_images_from_db(args, args.gallery_source_type, with_data=False)
    if not gallery_meta:
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

    sketchscape_root = args.sketchscape_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model_path not found: {model_path}")

    sys.path.insert(0, str(sketchscape_root))
    from model import SBIRModel

    # Extract sketch features in a local scope so model can be garbage collected
    print("[INFO] Loading SBIR model for sketch feature extraction...")
    sbir_model = SBIRModel.load_module(str(model_path), strict=False, cuda=(device == "cuda"))
    sbir_model = sbir_model.to(device)
    sbir_model.eval()

    sketch_transform = sbir_model.create_sketch_transforms()
    sketch_tensor = sketch_transform(load_rgb_image(sketch_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        sketch_feature = sbir_model(sketch_tensor)
        sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)
    
    sketch_feature = sketch_feature.cpu()  # Move to CPU first
    gallery_images = resolve_db_images_with_cache(args, args.gallery_source_type, gallery_meta)

    # Gallery source_type が output の場合は、DB/ローカルキャッシュの画像をそのまま使う
    # photo の場合は RBTE を適用して edge 画像を生成する
    gallery_images_edge: list[dict[str, Any]] = []

    if args.gallery_source_type == "output":
        print("[INFO] Using output images directly from DB/cache; BDCN is skipped.")
        image_transform = sbir_model.create_image_transforms()
        with torch.no_grad():
            gallery_tensors = []
            for g in gallery_images:
                edge_pil = load_rgb_image_from_bytes(g["image_data"])
                gallery_images_edge.append(
                    {
                        **g,
                        "image_data_edge": np.array(edge_pil),
                    }
                )
                edge_tensor = image_transform(edge_pil).unsqueeze(0)
                gallery_tensors.append(edge_tensor)

            gallery_batch = torch.cat(gallery_tensors, dim=0).to(device)
            gallery_features = sbir_model(gallery_batch)
            gallery_features = gallery_features / gallery_features.norm(dim=-1, keepdim=True)

        sketch_feature = sketch_feature.to(device)
        sims = (sketch_feature @ gallery_features.T).squeeze(0).cpu()

        print("[INFO] Cleaning up SBIR model from GPU...")
        sbir_model.cpu()
        del sbir_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    else:
        # photo など RBTE 未処理画像向け
        print("[INFO] Cleaning up SBIR model from GPU...")
        sbir_model.cpu()
        del sbir_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Load BDCN edge detection model
        print("[INFO] Loading BDCN model for edge detection...")
        bdcn_model = load_bdcn_model(device=device)

        # Get cache directory and cleanup old files
        cache_dir = get_rbte_cache_dir()
        cache_max_size_gb = float(os.getenv("RBTE_CACHE_MAX_SIZE_GB", "2.0"))
        cleanup_old_cache(cache_dir, max_size_gb=cache_max_size_gb)

        # Apply BDCN edge detection to gallery images (with caching, GPU-accelerated)
        cache_hits = 0
        cache_misses = 0

        for g in gallery_images:
            cache_key = get_rbte_cache_key(g["id"], g["image_data"])
            cache_path = cache_dir / cache_key

            # Try to load from cache
            edge_map = load_cached_edge(cache_path)

            if edge_map is not None:
                cache_hits += 1
                print(f"[CACHE HIT] {cache_key}")
            else:
                cache_misses += 1
                # Process with BDCN (GPU-accelerated)
                rgb_img = load_rgb_image_from_bytes(g["image_data"])
                edge_img = detect_edge_from_image(rgb_img, bdcn_model, device=device)
                edge_map = np.array(edge_img.convert("L"), dtype=np.uint8)  # HxW grayscale

                # Save to cache
                save_cached_edge(cache_path, edge_map)
                print(f"[CACHE MISS] Processed & cached: {cache_key}")

            # Convert edge_map back to RGB for SBIR processing
            edge_img_rgb = Image.fromarray(edge_map, mode="L").convert("RGB")
            gallery_images_edge.append(
                {
                    **g,
                    "image_data_edge": np.array(edge_img_rgb),  # Store edge-detected RGB image
                }
            )

        total = cache_hits + cache_misses
        if total > 0:
            print(f"[CACHE STATS] Hits: {cache_hits}, Misses: {cache_misses}, HitRate: {cache_hits / total * 100:.1f}%")

        # Unload BDCN model
        print("[INFO] Cleaning up BDCN model from GPU...")
        bdcn_model.cpu()
        del bdcn_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Reload SBIR model for gallery feature extraction and comparison
        print("[INFO] Reloading SBIR model for gallery feature extraction...")
        sbir_model = SBIRModel.load_module(str(model_path), strict=False, cuda=(device == "cuda"))
        sbir_model = sbir_model.to(device)
        sbir_model.eval()

        image_transform = sbir_model.create_image_transforms()

        # Convert edge-detected images to PIL, then apply transform
        with torch.no_grad():
            gallery_tensors = []
            for g in gallery_images_edge:
                edge_pil = Image.fromarray(g["image_data_edge"])
                edge_tensor = image_transform(edge_pil).unsqueeze(0)
                gallery_tensors.append(edge_tensor)

            gallery_batch = torch.cat(gallery_tensors, dim=0).to(device)
            gallery_features = sbir_model(gallery_batch)
            gallery_features = gallery_features / gallery_features.norm(dim=-1, keepdim=True)

        sketch_feature = sketch_feature.to(device)
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

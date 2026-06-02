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
import torchvision.transforms as T
from PIL import Image
from psycopg import sql

"""
run_sbir_once_from_db.py — SBIR runner with optional DINOv2 fusion

New DINOv2 options (added):
- `--enable_dinov2_fusion`: enable re-ranking using DINOv2 image embeddings.
- `--dinov2_weight`: float weight for DINOv2 when fusing with SketchScape score.
- `--dinov2_embeddings_path`: optional .npz of precomputed DINO embeddings.

Behavior:
- The script first performs a SketchScape (shape-based) coarse retrieval using
    `gallery_source_type` (defaults to `photo_edge`). For DINOv2 re-ranking the
    display photo id is used to look up embeddings in this order:
    1) specified `.npz` via `--dinov2_embeddings_path` (ids/embeddings)
    2) `photo_embeddings` table in the DB (bytea) — preferred for production
    3) on-the-fly encoding from the original `photo` image bytes (fallback)

This hybrid keeps the existing edge-based SketchScape scoring while adding
semantic re-ranking using DINOv2 to reduce shape/meaning mismatches.
"""




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SBIR for one sketch image against photo images stored in PostgreSQL."
    )
    parser.add_argument("--sketch_path", type=Path, required=True, help="クエリスケッチ画像パス")
    parser.add_argument("--topk", type=int, default=5, help="上位件数")
    parser.add_argument(
        "--exclude_gallery_ids",
        type=str,
        default="",
        help="一時的に検索対象から除外する gallery id のカンマ区切り（例: 12,34,56）",
    )

    parser.add_argument("--host", type=str, default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--dbname", type=str, default=os.getenv("PGDATABASE", "kakinoki_db"))
    parser.add_argument("--user", type=str, default=os.getenv("PGUSER", ""))
    parser.add_argument("--password", type=str, default=os.getenv("PGPASSWORD", ""))

    parser.add_argument("--schema", type=str, default="home_robot")
    parser.add_argument("--table", type=str, default="sketch_images")
    parser.add_argument("--gallery_table", type=str, default=os.getenv("SBIR_GALLERY_TABLE", "photos_edge"))
    parser.add_argument("--display_table", type=str, default=os.getenv("SBIR_DISPLAY_TABLE", "photos"))
    parser.add_argument(
        "--gallery_source_type",
        "--photo_source_type",
        dest="gallery_source_type",
        type=str,
        default="photo_edge",
        help="検索対象の source_type (デフォルト: photo_edge、photo の場合のみ RBTE を適用)",
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
    parser.add_argument("--coarse_topk", type=int, default=int(os.getenv("SBIR_COARSE_TOPK", "100")), help="粗検索で保持する上位件数")
    parser.add_argument("--enable_clip_fusion", action="store_true", help="CLIPテキスト再ランキングを有効化")
    parser.add_argument("--clip_text_query", type=str, default=os.getenv("SBIR_CLIP_TEXT_QUERY", ""), help="CLIP再ランキング用の短い検索テキスト")
    parser.add_argument("--clip_embeddings_path", type=Path, default=Path(os.getenv("CLIP_IMAGE_EMBEDDINGS_PATH", "~/workspace/CLIP_DB/cache/clip_image_embeddings.npz")), help="事前計算済みCLIP画像埋め込み(.npz)")
    parser.add_argument("--clip_model", type=str, default=os.getenv("CLIP_MODEL_NAME", "ViT-B-32"), help="OpenCLIP model name")
    parser.add_argument("--clip_pretrained", type=str, default=os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k"), help="OpenCLIP pretrained tag")
    parser.add_argument("--scape_weight", type=float, default=float(os.getenv("SBIR_SCAPE_WEIGHT", "0.7")), help="SketchScapeスコア重み")
    parser.add_argument("--clip_weight", type=float, default=float(os.getenv("SBIR_CLIP_WEIGHT", "0.3")), help="CLIPスコア重み")
    parser.add_argument("--enable_dinov2_fusion", action="store_true", help="DINOv2画像埋め込みによる再ランキングを有効化")
    parser.add_argument("--dinov2_weight", type=float, default=float(os.getenv("SBIR_DINO_WEIGHT", "0.3")), help="DINOv2スコア重み")
    parser.add_argument("--dinov2_embeddings_path", type=str, default=os.getenv("DINO_IMAGE_EMBEDDINGS_PATH", ""), help="事前計算済みDINO画像埋め込み(.npz)。未指定なら候補のみオンザフライで計算")
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
    table_name: str,
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
        fetched = fetch_images_from_db(args, table_name, source_type, with_data=True, ids=missing_ids)
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
    table_name: str,
    source_type: str,
    with_data: bool,
    ids: list[int] | None = None,
    exclude_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    if not args.user:
        raise ValueError("DB user is empty. Set --user or PGUSER.")

    select_cols = "id, file_name, source_path, image_sha256, image_data" if with_data else "id, file_name, source_path, image_sha256"

    where_parts = [sql.SQL("source_type = %s")]
    params_list: list[Any] = [source_type]

    if ids:
        where_parts.append(sql.SQL("id = ANY(%s)"))
        params_list.append(ids)

    if exclude_ids:
        where_parts.append(sql.SQL("NOT (id = ANY(%s))"))
        params_list.append(exclude_ids)

    where_clause = sql.SQL("WHERE ") + sql.SQL(" AND ").join(where_parts)
    params: tuple[Any, ...] = tuple(params_list)

    query = sql.SQL(
        f"""
        SELECT {select_cols}
        FROM {{}}.{{}}
        {{}}
        ORDER BY id;
        """
    ).format(sql.Identifier(args.schema), sql.Identifier(table_name), where_clause)

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


def minmax_normalize(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    s_min = torch.min(scores)
    s_max = torch.max(scores)
    if float((s_max - s_min).abs()) < 1e-12:
        return torch.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def load_clip_embeddings_npz(npz_path: Path) -> tuple[dict[int, int], np.ndarray]:
    p = npz_path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"CLIP embeddings cache not found: {p}")

    data = np.load(p, allow_pickle=False)
    if "ids" not in data or "embeddings" not in data:
        raise ValueError(f"Invalid CLIP embeddings cache format: {p} (requires 'ids' and 'embeddings')")

    ids = data["ids"].astype(np.int64)
    embeddings = data["embeddings"].astype(np.float32)
    if embeddings.ndim != 2 or len(ids) != embeddings.shape[0]:
        raise ValueError(f"Invalid CLIP cache shape: ids={ids.shape}, embeddings={embeddings.shape}")

    # Ensure L2 normalized (strictly required for fast dot-product cosine)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    embeddings = embeddings / norms

    id_to_index = {int(pid): i for i, pid in enumerate(ids.tolist())}
    return id_to_index, embeddings


def load_dinov2_embeddings_npz(npz_path: Path) -> tuple[dict[int, int], np.ndarray]:
    if not npz_path or not str(npz_path).strip():
        raise FileNotFoundError("DINO embeddings path is empty or None")
    
    p = npz_path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"DINO embeddings cache not found: {p}")

    data = np.load(p, allow_pickle=False)
    if "ids" not in data or "embeddings" not in data:
        raise ValueError(f"Invalid DINO embeddings cache format: {p} (requires 'ids' and 'embeddings')")

    ids = data["ids"].astype(np.int64)
    embeddings = data["embeddings"].astype(np.float32)
    if embeddings.ndim != 2 or len(ids) != embeddings.shape[0]:
        raise ValueError(f"Invalid DINO cache shape: ids={ids.shape}, embeddings={embeddings.shape}")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    embeddings = embeddings / norms

    id_to_index = {int(pid): i for i, pid in enumerate(ids.tolist())}
    return id_to_index, embeddings


def fetch_dinov2_embeddings_from_db(args: argparse.Namespace, photo_ids: list[int]) -> dict[int, np.ndarray]:
    """Fetch DINO embeddings stored in DB table `photo_embeddings` for given photo_ids.
    Returns mapping photo_id -> numpy float32 vector (L2-normalized).
    """
    if not photo_ids:
        return {}

    # Normalize unique ids
    uniq = sorted({int(x) for x in photo_ids})

    query = sql.SQL(
        "SELECT photo_id, embedding FROM {}.{} WHERE photo_id = ANY(%s)"
    ).format(sql.Identifier(args.schema), sql.Identifier(os.getenv("DINO_EMB_TABLE", "photo_embeddings")))

    result: dict[int, np.ndarray] = {}
    with psycopg.connect(host=args.host, port=args.port, dbname=args.dbname, user=args.user, password=args.password) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (uniq,))
            rows = cur.fetchall()
    for r in rows:
        pid = int(r[0])
        emb_bytes = bytes(r[1])
        try:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            # ensure correct shape (1D) and normalized
            if emb.size > 0:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                result[pid] = emb
        except Exception:
            continue
    return result


def encode_clip_text_feature(
    text_query: str,
    model_name: str,
    pretrained: str,
    device: str,
) -> np.ndarray:
    try:
        import open_clip
    except ImportError as e:
        raise ImportError("open_clip is required for CLIP fusion. Install open-clip-torch.") from e

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    with torch.no_grad():
        tokens = tokenizer([text_query]).to(device)
        txt = model.encode_text(tokens)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt.squeeze(0).detach().cpu().numpy().astype(np.float32)


def load_dinov2_model(device: str):
    try:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    except Exception as e:
        raise RuntimeError("Failed to load DINOv2 model via torch.hub: " + str(e)) from e
    model = model.eval().to(device)
    return model


def encode_dinov2_from_pil(img: Image.Image, model, device: str) -> np.ndarray:
    # Use same transforms as test harness
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img)[:3].unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()

    exclude_gallery_ids: list[int] = []
    if args.exclude_gallery_ids:
        try:
            exclude_gallery_ids = [int(v.strip()) for v in args.exclude_gallery_ids.split(",") if v.strip()]
        except ValueError as e:
            raise ValueError(f"Invalid --exclude_gallery_ids: {args.exclude_gallery_ids}") from e

    sketch_path = args.sketch_path.expanduser().resolve()
    if not sketch_path.is_file():
        raise FileNotFoundError(f"sketch_path not found: {sketch_path}")

    gallery_meta = fetch_images_from_db(
        args,
        args.gallery_table,
        args.gallery_source_type,
        with_data=False,
        exclude_ids=exclude_gallery_ids,
    )
    if not gallery_meta:
        raise FileNotFoundError(
            f"No gallery images found in DB for source_type='{args.gallery_source_type}'. "
            "Register images first."
        )

    display_photos = fetch_images_from_db(args, args.display_table, args.display_source_type, with_data=False)
    if not display_photos:
        raise FileNotFoundError(
            f"No display photos found in DB for source_type='{args.display_source_type}'."
        )

    # Fetch display photos with image bytes available for DINOv2 embedding if needed
    display_photos = fetch_images_from_db(args, args.display_table, args.display_source_type, with_data=True)

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
    gallery_images = resolve_db_images_with_cache(args, args.gallery_table, args.gallery_source_type, gallery_meta)

    # Gallery source_type が output / photo_edge の場合は、DB/ローカルキャッシュの画像をそのまま使う
    # photo の場合は RBTE を適用して edge 画像を生成する
    gallery_images_edge: list[dict[str, Any]] = []

    if args.gallery_source_type in {"output", "photo_edge"}:
        print(f"[INFO] Using precomputed edge images ({args.gallery_source_type}) from DB/cache; BDCN is skipped.")
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
    coarse_topk = max(topk, args.coarse_topk)
    coarse_vals, coarse_idxs = torch.topk(sims, k=min(coarse_topk, len(gallery_images_edge)))

    final_scores = coarse_vals.clone()
    clip_scores = torch.full_like(coarse_vals, fill_value=0.0)
    scape_norm = None
    clip_norm = None
    fusion_weights = None
    fusion_applied = False

    clip_query = (args.clip_text_query or "").strip()
    if args.enable_clip_fusion and clip_query:
        try:
            id_to_index, clip_embeddings = load_clip_embeddings_npz(args.clip_embeddings_path)
            text_feat = encode_clip_text_feature(
                text_query=clip_query,
                model_name=args.clip_model,
                pretrained=args.clip_pretrained,
                device=device,
            )

            valid_count = 0
            for j, cand_i in enumerate(coarse_idxs.tolist()):
                gallery = gallery_images[cand_i]
                key = canonical_name_key(gallery["file_name"])
                photo = display_by_key.get(key)
                if photo is None:
                    continue
                photo_id = int(photo["id"])
                emb_idx = id_to_index.get(photo_id)
                if emb_idx is None:
                    continue
                clip_score = float(np.dot(text_feat, clip_embeddings[emb_idx]))
                clip_scores[j] = clip_score
                valid_count += 1

            if valid_count > 0:
                w_sum = args.scape_weight + args.clip_weight
                if w_sum <= 1e-12:
                    w_scape, w_clip = 1.0, 0.0
                else:
                    w_scape, w_clip = args.scape_weight / w_sum, args.clip_weight / w_sum

                scape_norm = minmax_normalize(coarse_vals)
                clip_norm = minmax_normalize(clip_scores)
                fusion_weights = {"scape": float(w_scape), "clip": float(w_clip)}
                final_scores = w_scape * scape_norm + w_clip * clip_norm
                fusion_applied = True
            else:
                print("[WARN] CLIP fusion requested but no candidate had a matching cached embedding. Falling back to SketchScape-only ranking.")
        except Exception as e:
            print(f"[WARN] CLIP fusion failed: {e}. Falling back to SketchScape-only ranking.")

    # DINOv2 fusion: compute DINO embeddings for sketch and candidate display photos
    dino_scores = torch.zeros(len(coarse_vals), dtype=torch.float32, device="cpu")
    if args.enable_dinov2_fusion:
        try:
            # Optionally load precomputed DINO embeddings
            dino_id_to_index = None
            dino_embeddings = None
            dinov2_path_str = str(args.dinov2_embeddings_path).strip()
            print(f"[DEBUG] dinov2_embeddings_path arg: {args.dinov2_embeddings_path}, normalized: {dinov2_path_str!r}")
            if dinov2_path_str:
                dino_id_to_index, dino_embeddings = load_dinov2_embeddings_npz(args.dinov2_embeddings_path)
                print(f"[INFO] Loaded DINOv2 embeddings from: {args.dinov2_embeddings_path}")
            else:
                print(f"[INFO] No .npz file specified; will fetch from DB or compute on-the-fly")

            # Fetch DINO embeddings from DB if available (priority: .npz > DB > on-the-fly)
            db_dino_map = {}
            candidate_photo_ids = []
            for cand_i in coarse_idxs.tolist():
                gallery = gallery_images[cand_i]
                key = canonical_name_key(gallery["file_name"])
                photo = display_by_key.get(key)
                if photo is not None:
                    candidate_photo_ids.append(int(photo["id"]))
            
            if candidate_photo_ids and dino_embeddings is None:
                # Only fetch from DB if .npz not loaded
                print(f"[INFO] Fetching {len(candidate_photo_ids)} DINOv2 embeddings from DB...")
                db_dino_map = fetch_dinov2_embeddings_from_db(args, candidate_photo_ids)
                if db_dino_map:
                    print(f"[INFO] Retrieved {len(db_dino_map)} embeddings from DB")

            # Load DINO model if on-the-fly encoding is needed
            need_onfly = (dino_embeddings is None) and (len(db_dino_map) < len(candidate_photo_ids))
            dino_model = None
            if need_onfly:
                print(f"[INFO] Loading DINOv2 model for on-the-fly encoding...")
                dino_model = load_dinov2_model(device)

            # Compute sketch DINO embedding
            sketch_dino_emb = None
            if need_onfly:
                sketch_pil = load_rgb_image(sketch_path)
                sketch_dino_emb = encode_dinov2_from_pil(sketch_pil, dino_model, device)
            elif dino_embeddings is not None:
                # If using .npz for candidates, still compute sketch embedding
                sketch_pil = load_rgb_image(sketch_path)
                dino_model = load_dinov2_model(device)
                sketch_dino_emb = encode_dinov2_from_pil(sketch_pil, dino_model, device)

            # Prepare dinov2 score tensor (same shape as coarse_vals)
            valid_count = 0

            for j, cand_i in enumerate(coarse_idxs.tolist()):
                gallery = gallery_images[cand_i]
                key = canonical_name_key(gallery["file_name"])
                photo = display_by_key.get(key)
                if photo is None:
                    continue
                photo_id = int(photo["id"])

                # Prefer precomputed npz embeddings, then DB-stored embeddings, else on-the-fly encode
                emb = None
                if dino_embeddings is not None:
                    emb_idx = dino_id_to_index.get(photo_id)
                    if emb_idx is not None:
                        emb = dino_embeddings[emb_idx]

                if emb is None and photo_id in db_dino_map:
                    emb = db_dino_map[photo_id]

                if emb is None:
                    # on-the-fly encode from photo bytes
                    try:
                        img_pil = load_rgb_image_from_bytes(photo["image_data"])
                    except Exception:
                        continue
                    if dino_model is not None:
                        emb = encode_dinov2_from_pil(img_pil, dino_model, device)

                if emb is None:
                    continue

                # Ensure sketch embedding available (if embeddings cached and sketch not encoded yet)
                if sketch_dino_emb is None:
                    # If dino_embeddings provided but sketch emb not computed, compute on-the-fly
                    sketch_pil = load_rgb_image(sketch_path)
                    if dino_model is None:
                        dino_model = load_dinov2_model(device)
                    sketch_dino_emb = encode_dinov2_from_pil(sketch_pil, dino_model, device)

                # cosine via dot (both normalized)
                dino_score = float(np.dot(sketch_dino_emb, emb))
                dino_scores[j] = dino_score
                valid_count += 1

            if valid_count > 0:
                # fuse with SketchScape score (scape) similar to CLIP fusion
                w_sum = args.scape_weight + args.dinov2_weight
                if w_sum <= 1e-12:
                    w_scape, w_dino = 1.0, 0.0
                else:
                    w_scape, w_dino = args.scape_weight / w_sum, args.dinov2_weight / w_sum

                scape_norm = minmax_normalize(coarse_vals)
                dino_norm = minmax_normalize(dino_scores)
                fusion_weights = {"scape": float(w_scape), "dinov2": float(w_dino)}
                final_scores = w_scape * scape_norm + w_dino * dino_norm
                fusion_applied = True
                print(f"[INFO] DINOv2 fusion applied: {valid_count} candidates fused (weights: scape={w_scape:.3f}, dino={w_dino:.3f})")



            else:
                print("[WARN] DINOv2 fusion requested but no candidate had a matching embedding. Falling back to SketchScape-only ranking.")
        except Exception as e:
            print(f"[WARN] DINOv2 fusion failed: {e}. Falling back to SketchScape-only ranking.")

    sort_order = torch.argsort(final_scores, descending=True)
    selected_order = sort_order[: min(topk, len(sort_order))].tolist()

    ranked = []
    for rank, pos in enumerate(selected_order, start=1):
        i = int(coarse_idxs[pos].item())
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
                "score": float(final_scores[pos].item()),
                "score_sketchscape": float(coarse_vals[pos].item()),
                "score_clip_text": float(clip_scores[pos].item()) if args.enable_clip_fusion else None,
                "score_sketchscape_norm": float(scape_norm[pos].item()) if scape_norm is not None else None,
                "score_clip_text_norm": float(clip_norm[pos].item()) if clip_norm is not None else None,
                "score_dinov2": float(dino_scores[pos].item()) if args.enable_dinov2_fusion else None,
                "score_dinov2_norm": float(dino_norm[pos].item()) if ('dino_norm' in locals() and dino_norm is not None) else None,
            }
        )

    result = {
        "sketch_file": sketch_path.name,
        "device": device,
        "coarse_topk": int(min(coarse_topk, len(gallery_images_edge))),
        "clip_fusion_enabled": bool(args.enable_clip_fusion),
        "clip_fusion_applied": bool(args.enable_clip_fusion and fusion_applied and ('clip' in fusion_weights if fusion_weights else False)),
        "clip_text_query": clip_query if args.enable_clip_fusion else None,
        "dinov2_fusion_enabled": bool(args.enable_dinov2_fusion),
        "dinov2_fusion_applied": bool(args.enable_dinov2_fusion and fusion_applied and ('dinov2' in fusion_weights if fusion_weights else False)),
        "fusion_weights": fusion_weights,
        "topk": ranked,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from PIL import Image
import open_clip


ROOT = Path(__file__).resolve().parents[1]
GALLERY_DIR = ROOT / "output"
SKETCHES_DIR = ROOT / "sketches"
OUTPUT_DIR = ROOT / "outputs" / "clip"

# OpenCLIP model choice
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def load_rgb_image(path: Path) -> Image.Image:
    img = Image.open(path)

    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        img = img.convert("RGBA")
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white_bg, img).convert("RGB")
    else:
        img = img.convert("RGB")

    return img


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return a @ b.T


def resolve_input_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        return (ROOT / path).resolve()

    # Allow Docker-style paths (/zikken/...) when running on host.
    path_str = str(path)
    if path_str == "/zikken" or path_str.startswith("/zikken/"):
        if path.exists():
            return path
        rel = path.relative_to("/zikken")
        return (ROOT / rel).resolve()

    return path


def resolve_output_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        return (ROOT / path).resolve()

    # Allow Docker-style paths (/zikken/...) when running on host.
    path_str = str(path)
    if path_str == "/zikken" or path_str.startswith("/zikken/"):
        rel = path.relative_to("/zikken")
        return (ROOT / rel).resolve()

    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two image folders with OpenCLIP and save top-5 results.")
    parser.add_argument("--gallery_dir", type=Path, default=GALLERY_DIR, help="比較対象の画像ディレクトリ")
    parser.add_argument("--query_dir", type=Path, default=SKETCHES_DIR, help="クエリ画像ディレクトリ")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="結果保存先ディレクトリ")
    parser.add_argument("--topk", type=int, default=5, help="上位何件を保存するか")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gallery_dir = resolve_input_path(args.gallery_dir)
    query_dir = resolve_input_path(args.query_dir)
    output_dir = resolve_output_path(args.output_dir)
    topk_count = max(1, args.topk)

    if not gallery_dir.is_dir():
        raise FileNotFoundError(f"Gallery directory not found: {gallery_dir}")
    if not query_dir.is_dir():
        raise FileNotFoundError(f"Query directory not found: {query_dir}")

    print(f"gallery_dir: {gallery_dir}")
    print(f"query_dir:   {query_dir}")
    print(f"output_dir:  {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    photo_paths = list_images(gallery_dir)
    sketch_paths = list_images(query_dir)

    if not photo_paths:
        raise FileNotFoundError(f"No images found in {gallery_dir}")
    if not sketch_paths:
        raise FileNotFoundError(f"No images found in {query_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # OpenCLIP README style loading
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device)
    model.eval()

    # tokenizer is not used in this image-vs-image baseline,
    # but kept here for future text experiments
    _ = open_clip.get_tokenizer(MODEL_NAME)

    # Encode DB photos
    photo_tensors = []
    for p in photo_paths:
        img = preprocess(load_rgb_image(p)).unsqueeze(0)
        photo_tensors.append(img)
    photo_batch = torch.cat(photo_tensors, dim=0).to(device)

    with torch.no_grad():
        photo_features = model.encode_image(photo_batch)
        photo_features = photo_features / photo_features.norm(dim=-1, keepdim=True)

    summary = []

    # For each sketch, compare against all photos
    for sketch_path in sketch_paths:
        sketch_img = preprocess(load_rgb_image(sketch_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            sketch_feature = model.encode_image(sketch_img)
            sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)

        sims = (sketch_feature @ photo_features.T).squeeze(0).cpu()
        topk_vals, topk_idx = torch.topk(sims, k=min(topk_count, len(photo_paths)))

        top5 = []
        for rank, (score, idx) in enumerate(zip(topk_vals.tolist(), topk_idx.tolist()), start=1):
            top5.append(
                {
                    "rank": rank,
                    "gallery_file": photo_paths[idx].name,
                    "score": float(score),
                }
            )

        result = {
            "sketch_file": sketch_path.name,
            "top5": top5,
        }
        summary.append(result)

        out_json = output_dir / f"{sketch_path.stem}_top{topk_count}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\nSketch: {sketch_path.name}")
        for item in top5:
            print(
                f"  rank {item['rank']}: {item['gallery_file']} "
                f"(score={item['score']:.4f})"
            )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nsaved: {summary_path}")


if __name__ == "__main__":
    main()
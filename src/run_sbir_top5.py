from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
GALLERY_DIR = ROOT / "photos"
SKETCHES_DIR = ROOT / "sketches"
OUTPUT_DIR = ROOT / "outputs" / "sbir"


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


def resolve_input_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        return (ROOT / path).resolve()
    return path


def resolve_output_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        return (ROOT / path).resolve()
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sketch folder and gallery folder with SketchScape and save top-k results."
    )
    parser.add_argument("--gallery_dir", type=Path, default=GALLERY_DIR, help="比較対象の画像ディレクトリ")
    parser.add_argument("--query_dir", type=Path, default=SKETCHES_DIR, help="クエリ画像ディレクトリ")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="結果保存先ディレクトリ")
    parser.add_argument("--topk", type=int, default=5, help="上位何件を保存するか")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="推論デバイス。auto の場合は CUDA が使えない環境で自動的に CPU にフォールバックする",
    )
    parser.add_argument("--sketchscape_root", type=Path, default=Path("~/workspace/SketchScape"), help="SketchScape のルート")
    parser.add_argument("--model_path", type=Path, default=Path("~/workspace/SketchScape/models/fscoco_normal.pth"), help="SketchScape 学習済みモデル")
    return parser.parse_args()


def select_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda が指定されましたが CUDA が利用できません。")
        return "cuda"

    # auto
    if not torch.cuda.is_available():
        return "cpu"

    # PyTorch build may not support the installed GPU architecture.
    try:
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        supported_arch = set(torch.cuda.get_arch_list())
        if arch not in supported_arch:
            print(
                f"[WARN] GPU arch {arch} is not supported by this PyTorch build. "
                "Falling back to CPU."
            )
            return "cpu"
    except Exception:
        return "cpu"

    return "cuda"


def main() -> None:
    args = parse_args()

    gallery_dir = resolve_input_path(args.gallery_dir)
    query_dir = resolve_input_path(args.query_dir)
    output_dir = resolve_output_path(args.output_dir)
    sketchscape_root = args.sketchscape_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    topk_count = max(1, args.topk)

    if not gallery_dir.is_dir():
        raise FileNotFoundError(f"Gallery directory not found: {gallery_dir}")
    if not query_dir.is_dir():
        raise FileNotFoundError(f"Query directory not found: {query_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    photo_paths = list_images(gallery_dir)
    sketch_paths = list_images(query_dir)

    if not photo_paths:
        raise FileNotFoundError(f"No images found in {gallery_dir}")
    if not sketch_paths:
        raise FileNotFoundError(f"No images found in {query_dir}")

    device = select_device(args.device)
    print(f"device: {device}")

    sys.path.insert(0, str(sketchscape_root))

    from model import SBIRModel

    model = SBIRModel.load_module(str(model_path), strict=False, cuda=(device == "cuda"))
    model = model.to(device)
    model.eval()

    sketch_transform = model.create_sketch_transforms()
    image_transform = model.create_image_transforms()

    with torch.no_grad():
        gallery_tensors = [image_transform(load_rgb_image(p)).unsqueeze(0) for p in photo_paths]
        gallery_batch = torch.cat(gallery_tensors, dim=0).to(device)
        gallery_features = model(gallery_batch)
        gallery_features = gallery_features / gallery_features.norm(dim=-1, keepdim=True)

    summary = []

    for sketch_path in sketch_paths:
        sketch_img = load_rgb_image(sketch_path)
        sketch_tensor = sketch_transform(sketch_img).unsqueeze(0).to(device)

        with torch.no_grad():
            sketch_feature = model(sketch_tensor)
            sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)

        sims = (sketch_feature @ gallery_features.T).squeeze(0).cpu()
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
            print(f"  rank {item['rank']}: {item['gallery_file']} (score={item['score']:.4f})")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nsaved: {summary_path}")


if __name__ == "__main__":
    main()
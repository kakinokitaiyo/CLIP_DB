import torch
import torchvision.transforms as T
from PIL import Image

print("Loading DINOv2 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_model = dinov2_model.eval().to(device)

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_rgb_image(path):
    img = Image.open(path)
    # スケッチ画像の透過背景を白として扱う処理
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        img = img.convert("RGBA")
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white_bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return img

def get_dinov2_embedding(image_path):
    img = load_rgb_image(image_path)
    tensor = transform(img)[:3].unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = dinov2_model(tensor)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding

sketch_path = "/home/irsl/workspace/irsl_www/sketch/writing_39.png"
apple_path = "/home/irsl/workspace/CLIP_DB/photos/apple.jpeg"
plate_path = "/home/irsl/workspace/CLIP_DB/photos/IMG_2800.JPG"

print(f"Extracting features for:\n- Sketch: {sketch_path}\n- Apple: {apple_path}\n- Plate: {plate_path}")

try:
    sketch_emb = get_dinov2_embedding(sketch_path)
    photo_apple_emb = get_dinov2_embedding(apple_path)
    photo_plate_emb = get_dinov2_embedding(plate_path)

    sim_apple = (sketch_emb @ photo_apple_emb.T).item()
    sim_plate = (sketch_emb @ photo_plate_emb.T).item()

    print("\n=== Result (Cosine Similarity) ===")
    print(f"Sketch vs Apple photo: {sim_apple:.4f}")
    print(f"Sketch vs Plate photo: {sim_plate:.4f}")
    print("==================================")
except Exception as e:
    import traceback
    traceback.print_exc()

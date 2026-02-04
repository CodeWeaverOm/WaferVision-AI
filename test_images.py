import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os

print("Recursive prediction started...")

# ======================
# CONFIG
# ======================
ROOT_DIR = "test"      # main folder
MODEL_PATH = "model/defect_model.pth"
MAPPING_PATH = "model/class_mapping.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# LOAD CLASS MAPPING
# ======================
with open(MAPPING_PATH, "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
NUM_CLASSES = len(idx_to_class)

# ======================
# TRANSFORMS
# ======================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================
# LOAD MODEL
# ======================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ======================
# WALK THROUGH ALL SUBFOLDERS
# ======================
for root, _, files in os.walk(ROOT_DIR):
    for file in files:
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(root, file)

        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image)
            prob = torch.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)

        label = idx_to_class[pred.item()]
        confidence = conf.item() * 100

        print(f"{img_path:60s} -> {label:25s} ({confidence:.2f}%)")

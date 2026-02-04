import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import Counter
import json
import os
import time

# ======================
# CONFIG
# ======================
start = time.time()
DATA_DIR = "data/train"
MODEL_PATH = "model/defect_model.pth"
MAPPING_PATH = "model/class_mapping.json"

BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# TRANSFORMS (defect-safe)
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
# DATASET
# ======================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

NUM_CLASSES = len(dataset.classes)

print("Classes:", dataset.classes)
print("Class to index:", dataset.class_to_idx)

# ======================
# SAVE CLASS MAPPING (VERY IMPORTANT)
# ======================
os.makedirs("model", exist_ok=True)
with open(MAPPING_PATH, "w") as f:
    json.dump(dataset.class_to_idx, f)

# ======================
# HANDLE CLASS IMBALANCE
# ======================
class_counts = Counter(dataset.targets)
class_weights = [1.0 / class_counts[i] for i in range(NUM_CLASSES)]
class_weights = torch.tensor(class_weights).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# ======================
# MODEL (ResNet18)
# ======================
model = models.resnet18(pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Train only classifier
for param in model.fc.parameters():
    param.requires_grad = True

model.to(DEVICE)

optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

# ======================
# TRAIN LOOP
# ======================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f}")

# ======================
# SAVE MODEL
# ======================
torch.save(model.state_dict(), MODEL_PATH)
print("✅ Model saved successfully!")

stop = time.time()
print("Execution Time:", stop - start)

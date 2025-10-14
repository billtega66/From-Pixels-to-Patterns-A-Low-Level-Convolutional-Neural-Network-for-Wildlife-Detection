# pip install torch torchvision --upgrade

from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import OxfordIIITPet

ROOT = Path("/data")   # <- point to the folder that has images/ and annotations/

# ---- transforms ----
IMG_SIZE = 224
train_tfms = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
    T.ToTensor(),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
val_tfms = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ---- datasets ----
train_ds = OxfordIIITPet(
    root=str(ROOT),
    split="trainval",            # uses annotations/trainval.txt
    target_types="category",     # 37-breed label
    download=True
)
test_ds = OxfordIIITPet(
    root=str(ROOT),
    split="test",                # uses annotations/test.txt
    target_types="category",
)

# attach transforms (torchvision returns PIL Images)
train_ds.transform = train_tfms
test_ds.transform  = val_tfms

# ---- dataloaders ----
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=8, pin_memory=True)
val_loader   = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

# ---- inspect classes ----
idx_to_class = {i:c for i,c in enumerate(train_ds.classes)}  # list of breed names
num_classes = len(train_ds.classes)
print("Classes:", num_classes, list(idx_to_class.items())[:5])
print("Train/Val sizes:", len(train_ds), len(test_ds))

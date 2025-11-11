"""
Simple Keras CNN on Oxford-IIIT Pet, written in the same style as your MNIST example:
- Loads all images into NumPy arrays
- Normalizes to roughly [-0.5, 0.5]
- Uses one-hot targets via to_categorical
- Small Sequential CNN + SGD optimizer

Expected structure:
ROOT/
  images/*.jpg
  annotations/trainval.txt
  annotations/test.txt
"""

import os, re, sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

# -----------------------------
# 0) Paths & basic hyperparams
# -----------------------------
ROOT = r"C:/Users/Admin/machine_learning_CoW/From-Pixels-to-Patterns-A-Low-Level-Convolutional-Neural-Network-for-Wildlife-Detection/data"
IMAGES_DIR = os.path.join(ROOT, "images")
ANN_DIR = os.path.join(ROOT, "annotations")
TRAINVAL_TXT = os.path.join(ANN_DIR, "trainval.txt")
TEST_TXT     = os.path.join(ANN_DIR, "test.txt")

IMG_H, IMG_W = 128, 128     # keep modest for quick testing; you can change to 224 later
CHANNELS = 3                 # RGB
EPOCHS   = 30
BATCH    = 32
SEED     = 42
np.random.seed(SEED)

# ---------------------------------------
# 1) Read split files & build class map
#    - Each line begins with an image basename (no extension)
#    - Label = text before the last "_digits" suffix
# ---------------------------------------
def read_split(path):
    basenames = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                basenames.append(ln.split()[0])  # first token is the id
    return basenames

def basename_to_class(basename):
    # e.g., "American_bulldog_123" -> "American_bulldog"
    m = re.match(r"^(.*)_(\d+)$", basename)
    return m.group(1) if m else basename

train_ids = read_split(TRAINVAL_TXT)
test_ids  = read_split(TEST_TXT)

train_cls_names = [basename_to_class(b) for b in train_ids]
test_cls_names  = [basename_to_class(b) for b in test_ids]

all_classes = sorted(set(train_cls_names + test_cls_names))
class_to_idx = {c:i for i,c in enumerate(all_classes)}
num_classes = len(all_classes)
print(f"[Info] Classes: {num_classes}  (e.g., {all_classes[:5]} ...)")

# ---------------------------------------
# 2) Load images -> NumPy arrays
#    - Resize to (IMG_H, IMG_W)
#    - Normalize to ~[-0.5, 0.5] like your MNIST code
#    - One-hot labels via to_categorical
#    - Skip any corrupt images gracefully
# ---------------------------------------
def load_split_to_numpy(id_list, cls_names):
    X_list, y_list = [], []
    missing, corrupt = 0, 0
    for base, cls_name in zip(id_list, cls_names):
        p = os.path.join(IMAGES_DIR, base + ".jpg")
        if not os.path.exists(p):
            alt = os.path.join(IMAGES_DIR, base + ".png")
            if os.path.exists(alt):
                p = alt
            else:
                missing += 1
                continue
        try:
            img = Image.open(p).convert("RGB")
            img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)     # [0..255]
            arr = (arr / 255.0) - 0.5                   # ~[-0.5, 0.5]
            X_list.append(arr)
            y_list.append(class_to_idx[cls_name])
        except Exception as e:
            corrupt += 1
            continue

    if missing or corrupt:
        print(f"[Warn] Skipped {missing} missing and {corrupt} corrupt images.", file=sys.stderr)

    X = np.stack(X_list, axis=0)                        # (N, H, W, 3)
    y = np.array(y_list, dtype=np.int32)                # (N,)
    Y = to_categorical(y, num_classes=num_classes)      # (N, num_classes) for categorical_crossentropy
    return X, Y

print("[Info] Loading training split into RAM ...")
train_images, train_labels = load_split_to_numpy(train_ids, train_cls_names)
print(f"[Info] Train set: X={train_images.shape}, Y={train_labels.shape}")

print("[Info] Loading test split into RAM ...")
test_images, test_labels = load_split_to_numpy(test_ids, test_cls_names)
print(f"[Info] Test set:  X={test_images.shape}, Y={test_labels.shape}")

# ---------------------------------------
# 3) Define a small CNN (MNIST-like)
#    - Very small network to sanity-check pipeline
#    - You can deepen it later
# ---------------------------------------
model = Sequential([
    # Block 1: 32 filters
    Conv2D(32, 3, padding='same', use_bias=False, input_shape=(IMG_H, IMG_W, 3)),
    BatchNormalization(),
    tf.keras.layers.ReLU(),
    MaxPooling2D(2),

    # Block 2: 64 filters
    Conv2D(64, 3, padding='same', use_bias=False),
    BatchNormalization(),
    tf.keras.layers.ReLU(),
    MaxPooling2D(2),

    # Block 3: 128 filters
    Conv2D(128, 3, padding='same', use_bias=False),
    BatchNormalization(),
    tf.keras.layers.ReLU(),
    MaxPooling2D(2),

    # Replace Flatten -> GAP (big param reduction)
    GlobalAveragePooling2D(),

    # More regularization (you can try 0.3–0.5)
    Dropout(0.4),

    # Small classifier head
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax'),

    # Conv2D(16, 3, padding='same', input_shape=(IMG_H, IMG_W, CHANNELS), use_bias=False, activation='relu'),
    # BatchNormalization(),

    # MaxPooling2D(pool_size=2),

    # Conv2D(32, 3, padding='same', use_bias=False,activation='relu'),
    # BatchNormalization(),
    # MaxPooling2D(pool_size=2),

    # Flatten(),
    # Dropout(0.25),
    # Dense(128, activation='relu'),
    # Dense(num_classes, activation='softmax'),
])

# ---------------------------------------
# 4) Compile with SGD 
# ---------------------------------------
model.compile(
    #optimizer=SGD(learning_rate=0.005),   
    optimizer=Adam(learning_rate=1e-2),
    loss='categorical_crossentropy',      # because we used to_categorical
    metrics=['accuracy']
)

model.summary()

# ---------------------------------------
# 5) Train (use official test split as validation for now)
# ---------------------------------------
history = model.fit(
    train_images,
    train_labels,
    batch_size=BATCH,
    epochs=EPOCHS,
    validation_data=(test_images, test_labels),
)



# Extract training history
acc     = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss    = history.history['loss']
val_loss= history.history['val_loss']
epochs  = range(1, len(acc) + 1)

# ---- Plot Accuracy ----
plt.figure(figsize=(8,5))
plt.plot(epochs, acc, 'bo-', label='Training accuracy')
plt.plot(epochs, val_acc, 'r^-', label='Validation accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ---- Plot Loss ----
plt.figure(figsize=(8,5))
plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'r^-', label='Validation loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# ---------------------------------------
# 6) Save (new Keras format recommended)
# ---------------------------------------
model.save("simple_pets_cnn.keras")

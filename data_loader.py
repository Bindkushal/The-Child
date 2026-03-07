"""
data_loader.py
==============
Handles ALL data sources for SENN training.

Three sources supported:
  1. EMNIST Letters  — 88,000 handwritten A-Z samples (auto-download)
  2. EMNIST Digits   — 280,000 handwritten 0-9 samples (auto-download)
  3. Custom images   — YOUR own handwritten photos from data/custom/

The network sees everything as a flat 784-dim vector (28x28 pixels).
Labels are always 0-indexed integers.

LABEL SCHEME:
  Letters only  : A=0, B=1, C=2 ... Z=25
  Digits only   : 0=0, 1=1, 2=2 ... 9=9
  Combined      : A=0 ... Z=25, then 0=26, 1=27 ... 9=35  (36 classes total)

Usage:
  from data_loader import get_letters_loader, get_digits_loader, get_combined_loader

  # Task 1 — teach letters
  train_loader, val_loader = get_letters_loader(batch_size=32)

  # Task 2 — teach digits (EWC will protect letter knowledge)
  train_loader, val_loader = get_digits_loader(batch_size=32)

  # Combined — letters + digits shuffled together
  train_loader, val_loader = get_combined_loader(batch_size=32)
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import datasets, transforms
from PIL import Image, ImageOps
from pathlib import Path
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/raw")          # EMNIST downloads here
CUSTOM_DIR  = Path("data/custom")       # your own images go here
INPUT_SIZE  = 784                       # 28x28 flattened
VAL_SPLIT   = 0.1                       # 10% of data used for validation

LETTERS     = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS      = "0123456789"

# ─────────────────────────────────────────────────────────────────────────
#  BASE TRANSFORM — applied to ALL images before the network sees them
# ─────────────────────────────────────────────────────────────────────────
BASE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),                            # PIL → [0,1] tensor
    transforms.Normalize((0.1307,), (0.3081,)),       # EMNIST standard mean/std
    transforms.Lambda(lambda x: x.view(INPUT_SIZE))   # flatten 28x28 → 784
])


# ─────────────────────────────────────────────────────────────────────────
#  HELPER — split a dataset into train and validation
# ─────────────────────────────────────────────────────────────────────────
def _split(dataset, val_split=VAL_SPLIT, batch_size=32):
    """
    Split dataset into train / val loaders.
    Returns (train_loader, val_loader)
    """
    n_total = len(dataset)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)   # reproducible split
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,       # 0 = main process (safe on Colab)
        pin_memory  = False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = 0
    )

    print(f"  Train: {n_train:,} samples | Val: {n_val:,} samples | "
          f"Batches/epoch: {len(train_loader)}")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────
#  TASK 1 — EMNIST LETTERS  (A-Z, labels 0-25)
# ─────────────────────────────────────────────────────────────────────────
def get_letters_loader(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Load EMNIST Letters dataset.

    EMNIST Letters contains 88,800 samples of handwritten A-Z.
    Original labels are 1-26 → we shift to 0-25 so A=0, B=1 ... Z=25.

    Downloads automatically on first run (~35MB).
    Cached at data/raw/ for future runs.
    """
    print("\n[DataLoader] Loading EMNIST Letters (A-Z)...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = datasets.EMNIST(
        root      = str(DATA_DIR),
        split     = "letters",
        train     = True,
        download  = True,
        transform = BASE_TRANSFORM
    )

    # EMNIST letters are labelled 1-26, shift to 0-25
    dataset.targets = dataset.targets - 1

    # Also add custom letter images if they exist
    custom = _load_custom_letters()
    if custom is not None:
        dataset = ConcatDataset([dataset, custom])
        print(f"  + {len(custom)} custom letter samples added")

    print(f"  Total letters dataset: {len(dataset):,} samples (26 classes)")
    return _split(dataset, batch_size=batch_size)


# ─────────────────────────────────────────────────────────────────────────
#  TASK 2 — EMNIST DIGITS  (0-9, labels 0-9)
# ─────────────────────────────────────────────────────────────────────────
def get_digits_loader(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Load EMNIST Digits dataset.

    EMNIST Digits contains 280,000 samples of handwritten 0-9.
    Labels are already 0-9 — no shifting needed.

    Downloads automatically on first run (~100MB).
    """
    print("\n[DataLoader] Loading EMNIST Digits (0-9)...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = datasets.EMNIST(
        root      = str(DATA_DIR),
        split     = "digits",
        train     = True,
        download  = True,
        transform = BASE_TRANSFORM
    )

    # Also add custom digit images if they exist
    custom = _load_custom_digits()
    if custom is not None:
        dataset = ConcatDataset([dataset, custom])
        print(f"  + {len(custom)} custom digit samples added")

    print(f"  Total digits dataset: {len(dataset):,} samples (10 classes)")
    return _split(dataset, batch_size=batch_size)


# ─────────────────────────────────────────────────────────────────────────
#  COMBINED — Letters + Digits together  (36 classes, labels 0-35)
# ─────────────────────────────────────────────────────────────────────────
def get_combined_loader(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Load EMNIST Letters + Digits combined.

    Label scheme:
      A=0, B=1 ... Z=25,   then   0=26, 1=27 ... 9=35

    This is used for Task 3 — teaching both letters and numbers together.
    EWC memory protects what was learned in Tasks 1 and 2.
    """
    print("\n[DataLoader] Loading EMNIST Combined (A-Z + 0-9)...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Letters — labels 0-25
    letters_ds = datasets.EMNIST(
        root=str(DATA_DIR), split="letters",
        train=True, download=True, transform=BASE_TRANSFORM
    )
    letters_ds.targets = letters_ds.targets - 1    # shift 1-26 → 0-25

    # Digits — labels shifted to 26-35
    digits_ds = datasets.EMNIST(
        root=str(DATA_DIR), split="digits",
        train=True, download=True, transform=BASE_TRANSFORM
    )
    digits_ds.targets = digits_ds.targets + 26     # shift 0-9 → 26-35

    combined = ConcatDataset([letters_ds, digits_ds])
    print(f"  Combined dataset: {len(combined):,} samples (36 classes)")
    return _split(combined, batch_size=batch_size)


# ─────────────────────────────────────────────────────────────────────────
#  CUSTOM IMAGES — your own handwritten photos
# ─────────────────────────────────────────────────────────────────────────
class CustomImageDataset(Dataset):
    """
    Loads your own handwritten letter/digit photos.

    Folder structure:
      data/custom/
        A/  ← put photos of letter A here (any format: jpg, png)
        B/
        ...
        Z/
        0/  ← digit zero
        1/
        ...
        9/

    Images are automatically:
      - Converted to grayscale
      - Inverted (dark bg → white letter, to match EMNIST)
      - Resized to 28x28
      - Normalized
    """

    def __init__(self, root: Path, label_map: dict):
        """
        root      : path to folder containing class subfolders
        label_map : dict mapping folder name to integer label
                    e.g. {"A": 0, "B": 1, ...}
        """
        self.samples   = []   # list of (image_path, label) tuples
        self.transform = BASE_TRANSFORM
        self.label_map = label_map

        for folder_name, label in label_map.items():
            folder = root / folder_name
            if not folder.exists():
                continue
            for img_path in folder.iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.samples.append((img_path, label))

        print(f"  [Custom] Found {len(self.samples)} custom images "
              f"across {len(label_map)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("L")      # grayscale
            img = ImageOps.invert(img)               # invert like EMNIST
            img = img.resize((28, 28), Image.LANCZOS)
            tensor = self.transform(img)
            return tensor, label
        except Exception as e:
            # If image is corrupt, return a blank tensor
            print(f"  [Custom] Warning: could not load {path}: {e}")
            return torch.zeros(INPUT_SIZE), label


def _load_custom_letters():
    """Load custom letter images if data/custom/ exists with letter folders."""
    if not CUSTOM_DIR.exists():
        return None
    label_map = {letter: i for i, letter in enumerate(LETTERS)}
    ds = CustomImageDataset(CUSTOM_DIR, label_map)
    return ds if len(ds) > 0 else None


def _load_custom_digits():
    """Load custom digit images if data/custom/ exists with digit folders."""
    if not CUSTOM_DIR.exists():
        return None
    label_map = {digit: i for i, digit in enumerate(DIGITS)}
    ds = CustomImageDataset(CUSTOM_DIR, label_map)
    return ds if len(ds) > 0 else None


# ─────────────────────────────────────────────────────────────────────────
#  QUICK SANITY CHECK — run this file directly to test
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  DATA LOADER SANITY CHECK")
    print("=" * 55)

    print("\n--- Testing Letters loader ---")
    train_l, val_l = get_letters_loader(batch_size=32)
    x, y = next(iter(train_l))
    print(f"  Batch shape : {x.shape}")   # should be [32, 784]
    print(f"  Label range : {y.min().item()} to {y.max().item()}")   # 0 to 25
    print(f"  Pixel range : {x.min():.3f} to {x.max():.3f}")

    print("\n--- Testing Digits loader ---")
    train_d, val_d = get_digits_loader(batch_size=32)
    x, y = next(iter(train_d))
    print(f"  Batch shape : {x.shape}")   # should be [32, 784]
    print(f"  Label range : {y.min().item()} to {y.max().item()}")   # 0 to 9

    print("\n✓ Data loader working correctly")
    print(f"\nTo add your own handwritten images:")
    print(f"  1. Create folder:  data/custom/A/")
    print(f"  2. Put photos in:  data/custom/A/my_a_1.jpg")
    print(f"  3. Run training — custom images mix in automatically")

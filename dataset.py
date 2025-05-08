import os
print("=== dataset.py loaded ===", flush=True)
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class PairedSeasonDataset(Dataset):
    """
    PyTorch Dataset for paired seasonal images from three folders: fall, winter, summer.
    Assumes each folder contains images named as integers (e.g., '0.png', '1.png', ...).
    Returns normalized tensors of shape (3, 256, 256).
    """
    def __init__(self, root_dir, src_season='fall', tgt_season='winter', transform=None):
        print(f"Initializing dataset: {src_season} -> {tgt_season}", flush=True)
        self.src_dir = os.path.join(root_dir, src_season)
        self.tgt_dir = os.path.join(root_dir, tgt_season)
        self.src_images = sorted(os.listdir(self.src_dir), key=lambda x: int(os.path.splitext(x)[0]))
        self.tgt_images = sorted(os.listdir(self.tgt_dir), key=lambda x: int(os.path.splitext(x)[0]))
        assert len(self.src_images) == len(self.tgt_images), (
            f"Mismatch: '{src_season}' has {len(self.src_images)} files, "
            f"'{tgt_season}' has {len(self.tgt_images)} files"
        )
        self.transform = transform
        print(f"Found {len(self.src_images)} pairs.", flush=True)

    def __len__(self):
        return len(self.src_images)

    def __getitem__(self, idx):
        src_path = os.path.join(self.src_dir, self.src_images[idx])
        tgt_path = os.path.join(self.tgt_dir, self.tgt_images[idx])
        src_img = np.array(Image.open(src_path).convert('RGB'))
        tgt_img = np.array(Image.open(tgt_path).convert('RGB'))
        if self.transform:
            augmented = self.transform(image=src_img, image0=tgt_img)
            src_img = augmented['image']
            tgt_img = augmented['image0']
        return src_img, tgt_img

# Automatic shape test on import
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

print("Running dataset shape tests...", flush=True)
transform = Compose([
    Resize(256, 256),
    Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2(),
], additional_targets={'image0': 'image'})

pairs = [('fall','winter'), ('winter','summer'), ('summer','fall')]
for src, tgt in pairs:
    ds = PairedSeasonDataset(
        root_dir='nordland-dataset',
        src_season=src,
        tgt_season=tgt,
        transform=transform
    )
    s_img, t_img = ds[0]
    print(f"{src}->{tgt}: src shape {s_img.shape}, tgt shape {t_img.shape}", flush=True)

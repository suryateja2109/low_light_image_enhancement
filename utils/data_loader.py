import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def histogram_equalization(img_np):
    """
    Applies histogram equalization to RGB image in [0, 1].
    Converts to YCrCb space, equalizes Y channel, and converts back to RGB.
    """
    img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    rgb_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return rgb_eq.astype(np.float32) / 255.0

class LowLightDataset(Dataset):
    """
    Unsupervised dataset loader for SPNet (IEEE TMM 2025).
    Loads raw low-light image S1 and computes:
    - SHE: Histogram Equalization of S1
    - S2: Pseudo-reference via quadratic curve S2 = S1 + alpha * S1 * (1 - S1)
    """
    def __init__(self, img_dir, patch_size=128, is_train=True, alpha=0.5):
        self.img_dir = img_dir
        self.patch_size = patch_size
        self.is_train = is_train
        self.alpha = alpha
        
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        self.img_paths = [
            os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))
            if f.lower().endswith(valid_exts)
        ]
        if len(self.img_paths) == 0:
            raise ValueError(f"No valid images found in {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert('RGB')
        img_np = np.array(img, dtype=np.float32) / 255.0 # (H, W, 3)

        H, W, _ = img_np.shape

        # Patch cropping for training
        if self.is_train:
            if H > self.patch_size and W > self.patch_size:
                top = random.randint(0, H - self.patch_size)
                left = random.randint(0, W - self.patch_size)
                img_np = img_np[top:top+self.patch_size, left:left+self.patch_size, :]
            else:
                img_np = cv2.resize(img_np, (self.patch_size, self.patch_size))

            # Random Horizontal & Vertical Flips
            if random.random() > 0.5:
                img_np = np.fliplr(img_np)
            if random.random() > 0.5:
                img_np = np.flipud(img_np)

        # 1. Raw image S1
        S1 = img_np.copy()

        # 2. Histogram Equalized image SHE
        SHE = histogram_equalization(S1)

        # 3. Pseudo-reference image S2 via Quadratic Curve Transformation (Eq. 3)
        # S2 = S1 + alpha * S1 * (1 - S1)
        alpha_val = self.alpha if not self.is_train else random.uniform(0.4, 0.7)
        S2 = np.clip(S1 + alpha_val * S1 * (1.0 - S1), 0.0, 1.0)

        # Convert to PyTorch tensors (C, H, W)
        tensor_S1 = torch.from_numpy(S1.transpose(2, 0, 1)).float()
        tensor_SHE = torch.from_numpy(SHE.transpose(2, 0, 1)).float()
        tensor_S2 = torch.from_numpy(S2.transpose(2, 0, 1)).float()

        return tensor_S1, tensor_SHE, tensor_S2, alpha_val

def get_dataloader(img_dir, batch_size=16, patch_size=128, is_train=True, num_workers=0):
    dataset = LowLightDataset(img_dir, patch_size=patch_size, is_train=is_train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )
    return loader

if __name__ == "__main__":
    loader = get_dataloader(r"data\train\spnet_324", batch_size=4)
    s1, she, s2, alphas = next(iter(loader))
    print("S1 batch shape:", s1.shape)
    print("SHE batch shape:", she.shape)
    print("S2 batch shape:", s2.shape)
    print("Alphas:", alphas)


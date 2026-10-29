import os
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
from models.denoiser import DenoiseModule

class DoubleNoiseDataset(Dataset):
    """
    Implements the base paper's pre-training data setting (Section III-D, Eq. 8-10):
    X: low-light image
    Y = X + N(0, sigma^2)
    Z = Y + N(0, sigma^2)
    Model maps Z -> Y using MSE Loss.
    """
    def __init__(self, img_dir, patch_size=128, sigma=15.0/255.0):
        self.patch_size = patch_size
        self.sigma = sigma
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        self.paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
        print(f"Loaded {len(self.paths)} images for denoiser pre-training from {img_dir}")

    def __len__(self):
        # Repeat to create enough mini-batches per epoch
        return len(self.paths) * 20

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        img = Image.open(path).convert('RGB')
        img_np = np.array(img, dtype=np.float32) / 255.0

        H, W, _ = img_np.shape
        if H > self.patch_size and W > self.patch_size:
            top = random.randint(0, H - self.patch_size)
            left = random.randint(0, W - self.patch_size)
            X = img_np[top:top+self.patch_size, left:left+self.patch_size, :]
        else:
            X = cv2.resize(img_np, (self.patch_size, self.patch_size))

        # Add first noise layer: Y = X + N(0, sigma^2)
        noise1 = np.random.normal(0, self.sigma, X.shape).astype(np.float32)
        Y = np.clip(X + noise1, 0.0, 1.0)

        # Add second noise layer: Z = Y + N(0, sigma^2)
        noise2 = np.random.normal(0, self.sigma, Y.shape).astype(np.float32)
        Z = np.clip(Y + noise2, 0.0, 1.0)

        tensor_Z = torch.from_numpy(Z.transpose(2, 0, 1)).float()
        tensor_Y = torch.from_numpy(Y.transpose(2, 0, 1)).float()
        return tensor_Z, tensor_Y

def pretrain_denoiser(img_dir=r"data\train\denoise_10", save_path=r"weights\denoiser_pretrained.pth", epochs=50):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for denoiser pre-training: {device}")

    dataset = DoubleNoiseDataset(img_dir=img_dir, patch_size=128)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    model = DenoiseModule().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print(f"--- Starting Stage 1: Denoise Module Pre-training ({epochs} epochs) ---")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for Z, Y in loader:
            Z, Y = Z.to(device), Y.to(device)
            optimizer.zero_grad()
            out = model(Z)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] - LDNM Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Pre-trained Denoise Module saved successfully to: {save_path}")

if __name__ == "__main__":
    pretrain_denoiser()


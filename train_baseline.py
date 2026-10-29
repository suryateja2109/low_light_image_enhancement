import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from models.spnet_base import SPNet
from utils.data_loader import get_dataloader
from losses.retinex_loss import RetinexLoss
from losses.spl_loss import SPLLoss

def train_spnet(
    train_dir=r"data\train\spnet_324",
    denoiser_weights=r"weights\denoiser_pretrained.pth",
    save_path=r"weights\spnet_baseline.pth",
    epochs=150,
    batch_size=16,
    patch_size=128,
    lr=1e-4
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"Starting SPNet Training (IEEE TMM 2025 Base Paper Replication)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Training Images: 324 (Unsupervised)")
    print(f"Total Epochs: {epochs}, Batch Size: {batch_size}, Patch Size: {patch_size}x{patch_size}")
    print("=" * 60)

    # 1. Initialize Dataset & DataLoader
    train_loader = get_dataloader(train_dir, batch_size=batch_size, patch_size=patch_size, is_train=True)

    # 2. Initialize Model
    model = SPNet(denoiser_weights=denoiser_weights).to(device)

    # 3. Initialize Losses
    retinex_criterion = RetinexLoss(omega0=1.0, omega1=0.2, omega2=0.02, omega3=0.2, omega4=1.0).to(device)
    spl_criterion = SPLLoss(device=device)

    # 4. Optimizer: Optimize only trainable parameters (FEM + LIDM, denoiser is frozen)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.9, 0.999)
    )

    # Initialize guidance weight lambda = 1.0 (Algorithm 1)
    current_lambda = 1.0
    start_time = time.time()

    model.train()
    # Keep denoiser in eval mode
    model.denoiser.eval()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_lhep = 0.0
        batch_count = 0

        for S1, SHE, S2, alpha in train_loader:
            S1, SHE, S2 = S1.to(device), SHE.to(device), S2.to(device)
            alpha = alpha.to(device)

            optimizer.zero_grad()

            # Forward pass through FEM + SPL Fusion + LIDM
            R1, I1, R2, I2, _ = model(S1, SHE, S2, lam=current_lambda)

            # Compute Lhep using pre-trained VGG-19 features (Eq. 11)
            Lhep, new_lambda = spl_criterion(R1, SHE)

            # Compute overall unsupervised loss (Eq. 7): Lall = omega4 * LR + LGSI
            L_all, loss_dict = retinex_criterion(S1, R1, I1, S2, R2, I2, Lhep, alpha)

            # Backpropagation & Update (Algorithm 1, Step 4)
            L_all.backward()
            optimizer.step()

            # Dynamically update lambda for the next iteration (Algorithm 1)
            current_lambda = new_lambda

            epoch_loss += L_all.item()
            epoch_lhep += Lhep.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        avg_lhep = epoch_lhep / batch_count

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:03d}/{epochs}] | Loss: {avg_loss:.4f} | Lhep: {avg_lhep:.4f} | SPL lambda: {current_lambda:.3f} | Elapsed: {elapsed:.1f}s")

    # Save trained model weights
    torch.save(model.state_dict(), save_path)
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"SPNet Baseline Training Completed in {total_time:.2f} seconds ({total_time/60.0:.2f} mins)!")
    print(f"Checkpoint saved to: {save_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_spnet()


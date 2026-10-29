import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from models.aspnet_plus import ASPNetPlus
from utils.data_loader import get_dataloader
from losses.retinex_loss import RetinexLoss
from losses.spl_loss import SPLLoss
from losses.advanced_loss import LightnessOrderLoss, ColorConstancyLoss, ExposureControlLoss

def train_aspnet_plus(
    train_dir=r"data\train\spnet_324",
    save_path=r"weights\aspnet_plus.pth",
    epochs=150,
    batch_size=16,
    patch_size=128,
    lr=1e-4
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("Training Upgraded Model: A-SPNet+ (Adaptive SPNet Plus)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("Integrated Innovations:")
    print(" 1. Spatial-SPL (Pixel-level difficulty gating)")
    print(" 2. SALEM (Spatial Exposure Attention to prevent overexposure)")
    print(" 3. Wavelet-Domain Denoising (Lightweight high-frequency suppression)")
    print(" 4. Multi-Constraint Objective (Retinex + Lord + Color + Lexp)")
    print("=" * 65)

    train_loader = get_dataloader(train_dir, batch_size=batch_size, patch_size=patch_size, is_train=True)
    model = ASPNetPlus().to(device)

    # Core base losses
    retinex_criterion = RetinexLoss(omega0=1.0, omega1=0.2, omega2=0.02, omega3=0.2, omega4=1.0).to(device)
    spl_criterion = SPLLoss(device=device)

    # Advanced losses solving the identified limitations
    lord_criterion = LightnessOrderLoss().to(device)
    color_criterion = ColorConstancyLoss().to(device)
    exp_criterion = ExposureControlLoss(patch_size=16, mean_val=0.55).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    current_lambda = 1.0
    start_time = time.time()

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batch_count = 0

        for S1, SHE, S2, alpha in train_loader:
            S1, SHE, S2 = S1.to(device), SHE.to(device), S2.to(device)
            alpha = alpha.to(device)

            optimizer.zero_grad()

            # Forward pass through A-SPNet+
            clean_R1, I1_adapt, clean_R2, I2_adapt, final_enh = model(S1, SHE, S2, global_lam=current_lambda)

            # Compute Lhep for difficulty assessment
            Lhep, new_lambda = spl_criterion(clean_R1, SHE)

            # Base Retinex & Illumination gradient similarity loss
            L_base, _ = retinex_criterion(S1, clean_R1, I1_adapt, S2, clean_R2, I2_adapt, Lhep, alpha)

            # Novel Regularization Losses
            L_exp = exp_criterion(final_enh)
            L_lord = lord_criterion(S1, final_enh)
            L_color = color_criterion(final_enh)

            # Combined Upgraded Objective
            L_total = L_base + 1.0 * L_exp + 0.3 * L_lord + 0.1 * L_color

            L_total.backward()
            optimizer.step()

            current_lambda = new_lambda
            epoch_loss += L_total.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:03d}/{epochs}] | Loss: {avg_loss:.4f} | SPL lambda: {current_lambda:.3f} | Elapsed: {elapsed:.1f}s")

    torch.save(model.state_dict(), save_path)
    total_time = time.time() - start_time
    print("=" * 65)
    print(f"A-SPNet+ Upgraded Training Completed in {total_time:.2f}s ({total_time/60.0:.2f} mins)!")
    print(f"Weights saved to: {save_path}")
    print("=" * 65)

if __name__ == "__main__":
    train_aspnet_plus()

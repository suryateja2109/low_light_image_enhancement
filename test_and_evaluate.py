import os
import cv2
import numpy as np
import torch
from PIL import Image

from models.spnet_base import SPNet
from utils.data_loader import histogram_equalization
from utils.metrics import calculate_psnr, calculate_ssim

def evaluate_model(
    model_weights=r"weights\spnet_baseline.pth",
    test_low_dir=r"data\test\LOL\low",
    test_high_dir=r"data\test\LOL\high",
    output_dir=r"results\baseline",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print(f"Evaluating SPNet Baseline on LOL Test Benchmark (15 Paired Images)")
    print(f"Device: {device}")
    print("=" * 60)

    # Load model
    model = SPNet().to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()

    test_imgs = sorted([f for f in os.listdir(test_low_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
    
    psnr_scores = []
    ssim_scores = []

    print(f"{'Image Name':<15} | {'PSNR (dB)':<12} | {'SSIM':<10}")
    print("-" * 45)

    with torch.no_grad():
        for fname in test_imgs:
            low_path = os.path.join(test_low_dir, fname)
            high_path = os.path.join(test_high_dir, fname)

            # Load low-light input
            low_pil = Image.open(low_path).convert('RGB')
            low_np = np.array(low_pil, dtype=np.float32) / 255.0
            
            # Compute SHE and S2
            she_np = histogram_equalization(low_np)
            s2_np = np.clip(low_np + 0.5 * low_np * (1.0 - low_np), 0.0, 1.0)

            # To Tensor
            tensor_S1 = torch.from_numpy(low_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            tensor_SHE = torch.from_numpy(she_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            tensor_S2 = torch.from_numpy(s2_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

            # During inference, SPL lambda is 0.0 (paper: fully adapted network relies on learned features)
            R1, I1, R2, I2, enhanced = model(tensor_S1, tensor_SHE, tensor_S2, lam=0.0)

            enhanced_np = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
            enhanced_np = np.clip(enhanced_np, 0.0, 1.0)

            # Save result image
            save_img = (enhanced_np * 255.0).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, fname), cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))

            # Compute metrics against GT
            if os.path.exists(high_path):
                high_pil = Image.open(high_path).convert('RGB')
                high_np = np.array(high_pil, dtype=np.float32) / 255.0

                psnr = calculate_psnr(enhanced_np, high_np)
                ssim = calculate_ssim(enhanced_np, high_np)

                psnr_scores.append(psnr)
                ssim_scores.append(ssim)

                print(f"{fname:<15} | {psnr:<12.2f} | {ssim:<10.4f}")

    mean_psnr = np.mean(psnr_scores)
    mean_ssim = np.mean(ssim_scores)

    print("=" * 45)
    print(f"Mean PSNR: {mean_psnr:.2f} dB  (Paper Table I reports ~20.12 dB)")
    print(f"Mean SSIM: {mean_ssim:.4f}     (Paper Table I reports ~0.7900)")
    print(f"Enhanced images saved in: {output_dir}")
    print("=" * 45)
    return mean_psnr, mean_ssim

if __name__ == "__main__":
    evaluate_model()


import os
import time
import cv2
import numpy as np
import torch
from PIL import Image

from models.spnet_base import SPNet
from models.aspnet_plus import ASPNetPlus
from utils.data_loader import histogram_equalization
from utils.metrics import calculate_psnr, calculate_ssim

def calculate_loe(input_img, output_img, down_size=50):
    """
    Computes Lightness Order Error (LOE) between input and enhanced output.
    Lower score indicates better lightness order preservation.
    """
    if isinstance(input_img, torch.Tensor):
        input_img = input_img.detach().cpu().numpy()
    if isinstance(output_img, torch.Tensor):
        output_img = output_img.detach().cpu().numpy()

    lum_in = 0.299 * input_img[:, :, 0] + 0.587 * input_img[:, :, 1] + 0.114 * input_img[:, :, 2]
    lum_out = 0.299 * output_img[:, :, 0] + 0.587 * output_img[:, :, 1] + 0.114 * output_img[:, :, 2]

    h, w = lum_in.shape
    r_h = min(down_size, h)
    r_w = min(down_size, w)
    small_in = cv2.resize(lum_in, (r_w, r_h), interpolation=cv2.INTER_NEAREST)
    small_out = cv2.resize(lum_out, (r_w, r_h), interpolation=cv2.INTER_NEAREST)

    flat_in = small_in.flatten()
    flat_out = small_out.flatten()
    N = len(flat_in)

    if N > 1000:
        idx = np.random.choice(N, 1000, replace=False)
        flat_in = flat_in[idx]
        flat_out = flat_out[idx]
        N = 1000

    diff_in = flat_in[:, None] - flat_in[None, :]
    diff_out = flat_out[:, None] - flat_out[None, :]

    order_violations = ((diff_in > 0) & (diff_out < 0)) | ((diff_in < 0) & (diff_out > 0))
    loe = np.sum(order_violations) / float(N)
    return float(loe)

def evaluate_both(
    baseline_weights=r"weights\spnet_baseline.pth",
    upgraded_weights=r"weights\aspnet_plus.pth",
    test_low_dir=r"data\test\LOL\low",
    test_high_dir=r"data\test\LOL\high",
    results_dir=r"results\comparisons"
):
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print("HEAD-TO-HEAD BENCHMARK EVALUATION (LOL TEST BENCHMARK - 15 IMAGES)")
    print("Base Paper SPNet vs. Proposed A-SPNet+ (Adaptive SPNet Plus)")
    print(f"Device: {device}")
    print("=" * 88)

    # Load Baseline Model
    baseline_model = SPNet().to(device)
    baseline_model.load_state_dict(torch.load(baseline_weights, map_location=device))
    baseline_model.eval()

    # Load Upgraded Model
    upgraded_model = ASPNetPlus().to(device)
    upgraded_model.load_state_dict(torch.load(upgraded_weights, map_location=device))
    upgraded_model.eval()

    test_imgs = sorted([f for f in os.listdir(test_low_dir) if f.lower().endswith(('.png', '.jpg'))])

    base_psnr, base_ssim, base_loe = [], [], []
    upg_psnr, upg_ssim, upg_loe = [], [], []
    base_times, upg_times = [], []

    print(f"{'Image':<10} | {'Base PSNR':<11} | {'Upg PSNR':<11} | {'Base SSIM':<11} | {'Upg SSIM':<11} | {'Base LOE':<10} | {'Upg LOE':<10}")
    print("-" * 88)

    with torch.no_grad():
        for fname in test_imgs:
            low_path = os.path.join(test_low_dir, fname)
            high_path = os.path.join(test_high_dir, fname)

            low_pil = Image.open(low_path).convert('RGB')
            low_np = np.array(low_pil, dtype=np.float32) / 255.0
            high_np = np.array(Image.open(high_path).convert('RGB'), dtype=np.float32) / 255.0

            she_np = histogram_equalization(low_np)
            s2_np = np.clip(low_np + 0.5 * low_np * (1.0 - low_np), 0.0, 1.0)

            t_s1 = torch.from_numpy(low_np.transpose(2,0,1)).unsqueeze(0).float().to(device)
            t_she = torch.from_numpy(she_np.transpose(2,0,1)).unsqueeze(0).float().to(device)
            t_s2 = torch.from_numpy(s2_np.transpose(2,0,1)).unsqueeze(0).float().to(device)

            # 1. Baseline Inference
            t0 = time.time()
            _, _, _, _, enh_base = baseline_model(t_s1, t_she, t_s2, lam=0.85)
            base_times.append(time.time() - t0)
            np_base = np.clip(enh_base.squeeze(0).permute(1,2,0).cpu().numpy(), 0.0, 1.0)

            # 2. Upgraded Inference
            t0 = time.time()
            _, _, _, _, enh_upg = upgraded_model(t_s1, t_she, t_s2, global_lam=0.5)
            upg_times.append(time.time() - t0)
            np_upg = np.clip(enh_upg.squeeze(0).permute(1,2,0).cpu().numpy(), 0.0, 1.0)

            # Metrics
            bpsnr = calculate_psnr(np_base, high_np)
            upsnr = calculate_psnr(np_upg, high_np)
            bssim = calculate_ssim(np_base, high_np)
            ussim = calculate_ssim(np_upg, high_np)
            bloe = calculate_loe(low_np, np_base)
            uloe = calculate_loe(low_np, np_upg)

            base_psnr.append(bpsnr)
            upg_psnr.append(upsnr)
            base_ssim.append(bssim)
            upg_ssim.append(ussim)
            base_loe.append(bloe)
            upg_loe.append(uloe)

            print(f"{fname:<10} | {bpsnr:<11.2f} | {upsnr:<11.2f} | {bssim:<11.4f} | {ussim:<11.4f} | {bloe:<10.1f} | {uloe:<10.1f}")

            # Save comparison strip
            vis_strip = np.hstack([
                (low_np * 255).astype(np.uint8),
                (np_base * 255).astype(np.uint8),
                (np_upg * 255).astype(np.uint8),
                (high_np * 255).astype(np.uint8)
            ])
            cv2.imwrite(os.path.join(results_dir, f"compare_{fname}"), cv2.cvtColor(vis_strip, cv2.COLOR_RGB2BGR))

    p_base = sum(p.numel() for p in baseline_model.parameters()) / 1e6
    p_upg = sum(p.numel() for p in upgraded_model.parameters()) / 1e6

    print("=" * 88)
    print(f"{'METRIC SUMMARY':<25} | {'SPNet (Baseline)':<25} | {'A-SPNet+ (Proposed)':<25}")
    print("-" * 88)
    print(f"{'Mean PSNR (dB)':<25} | {np.mean(base_psnr):<25.2f} | {np.mean(upg_psnr):<25.2f}")
    print(f"{'Mean SSIM':<25} | {np.mean(base_ssim):<25.4f} | {np.mean(upg_ssim):<25.4f}")
    print(f"{'Mean LOE (Error)':<25} | {np.mean(base_loe):<25.1f} | {np.mean(upg_loe):<25.1f}")
    print(f"{'Model Parameters (M)':<25} | {p_base:<25.4f}M | {p_upg:<25.4f}M")
    print(f"{'Avg Inference Time':<25} | {np.mean(base_times)*1000:<25.1f}ms | {np.mean(upg_times)*1000:<25.1f}ms")
    print("=" * 88)
    print(f"Side-by-side visual comparisons saved to: {results_dir}")

if __name__ == "__main__":
    evaluate_both()

import numpy as np
import torch
import torch.nn.functional as F

def calculate_psnr(img1, img2):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between two images in [0, 1].
    Can accept numpy arrays or torch tensors of shape (H, W, 3) or (3, H, W).
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)

def gaussian_window(window_size=11, sigma=1.5):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size=11, channel=3):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def calculate_ssim(img1, img2, window_size=11):
    """
    Computes Structural Similarity Index (SSIM) between two images in [0, 1].
    Expects torch tensors of shape (1, 3, H, W) or (B, 3, H, W).
    """
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float()

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

if __name__ == "__main__":
    t1 = torch.rand(1, 3, 128, 128)
    t2 = t1 + torch.randn(1, 3, 128, 128) * 0.05
    t2 = torch.clamp(t2, 0, 1)
    print("PSNR:", calculate_psnr(t1, t2))
    print("SSIM:", calculate_ssim(t1, t2))


import torch
import torch.nn as nn
import torch.nn.functional as F

class LightnessOrderLoss(nn.Module):
    """
    Lightness Order Preservation Loss (Lord).
    Explicitly penalizes inversions in relative brightness between neighboring pixels,
    directly driving down the Lightness Order Error (LOE) metric.
    """
    def __init__(self, pool_size=8):
        super(LightnessOrderLoss, self).__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=pool_size)

    def forward(self, input_low, enhanced):
        low_lum = 0.299 * input_low[:, 0:1, :, :] + 0.587 * input_low[:, 1:2, :, :] + 0.114 * input_low[:, 2:3, :, :]
        enh_lum = 0.299 * enhanced[:, 0:1, :, :] + 0.587 * enhanced[:, 1:2, :, :] + 0.114 * enhanced[:, 2:3, :, :]

        p_low = self.pool(low_lum)
        p_enh = self.pool(enh_lum)

        dx_low = p_low[:, :, :, 1:] - p_low[:, :, :, :-1]
        dy_low = p_low[:, :, 1:, :] - p_low[:, :, :-1, :]

        dx_enh = p_enh[:, :, :, 1:] - p_enh[:, :, :, :-1]
        dy_enh = p_enh[:, :, 1:, :] - p_enh[:, :, :-1, :]

        order_penalty_x = F.relu(-torch.sign(dx_low) * dx_enh)
        order_penalty_y = F.relu(-torch.sign(dy_low) * dy_enh)

        return torch.mean(order_penalty_x) + torch.mean(order_penalty_y)

class ColorConstancyLoss(nn.Module):
    """
    Gray-World Color Constancy Loss.
    Ensures that chromatic channels (R, G, B) maintain balanced natural proportions,
    preventing color shifts and yellow/green artifacts.
    """
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, x):
        mean_r = torch.mean(x[:, 0, :, :], dim=[1, 2])
        mean_g = torch.mean(x[:, 1, :, :], dim=[1, 2])
        mean_b = torch.mean(x[:, 2, :, :], dim=[1, 2])

        d_rg = (mean_r - mean_g) ** 2
        d_rb = (mean_r - mean_b) ** 2
        d_gb = (mean_g - mean_b) ** 2

        return torch.mean(torch.sqrt(d_rg + d_rb + d_gb + 1e-6))

class ExposureControlLoss(nn.Module):
    """
    Exposure Control Loss (Lexp).
    Measures the distance between the local average intensity and a well-exposed reference value (E=0.55).
    Prevents both under-exposure and local over-exposure.
    """
    def __init__(self, patch_size=16, mean_val=0.55):
        super(ExposureControlLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size, stride=patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        # Grayscale intensity
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        mean_intensity = self.pool(gray)
        return torch.mean((mean_intensity - self.mean_val) ** 2)

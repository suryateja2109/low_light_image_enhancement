import torch
import torch.nn as nn
import torch.nn.functional as F

def gradient_x(img):
    return torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])

def gradient_y(img):
    return torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])

def total_gradient(img):
    gx = gradient_x(img)
    gy = gradient_y(img)
    # Average gradient magnitude
    return torch.mean(gx) + torch.mean(gy)

class RetinexLoss(nn.Module):
    """
    Retinex Decomposition Loss from SPNet (IEEE TMM 2025, Eq. 2 & Eq. 6):
    LR = omega0 * ||S1 - R1 * I1||1 
       + omega1 * Lhep 
       + omega2 * ||grad(R1)||1 
       + omega3 * ||grad(S1) - grad(I1)||1
    LGSI = ||(1 + alpha) * grad(I1) - grad(I2)||1
    Lall = omega4 * LR + LGSI
    """
    def __init__(self, omega0=1.0, omega1=0.2, omega2=0.02, omega3=0.2, omega4=1.0):
        super(RetinexLoss, self).__init__()
        self.w0 = omega0
        self.w1 = omega1
        self.w2 = omega2
        self.w3 = omega3
        self.w4 = omega4
        self.l1 = nn.L1Loss()

    def forward(self, S1, R1, I1, S2, R2, I2, Lhep, alpha):
        # 1. Reconstruction Loss: ||S1 - R1 * I1||1
        # Broadcast 1-channel I1 across 3 channels of R1
        reconstructed = R1 * I1
        L_recon = self.l1(reconstructed, S1)

        # 2. Smoothness Loss: ||grad(R1)||1
        L_smooth = total_gradient(R1)

        # 3. Illumination Gradient Alignment: ||grad(S1) - grad(I1)||1
        # Convert S1 to grayscale for gradient matching with I1
        S1_gray = 0.299 * S1[:, 0:1, :, :] + 0.587 * S1[:, 1:2, :, :] + 0.114 * S1[:, 2:3, :, :]
        gx_S1 = gradient_x(S1_gray)
        gx_I1 = gradient_x(I1)
        gy_S1 = gradient_y(S1_gray)
        gy_I1 = gradient_y(I1)
        L_grad_align = torch.mean(torch.abs(gx_S1 - gx_I1)) + torch.mean(torch.abs(gy_S1 - gy_I1))

        # Combine Retinex Loss LR
        LR = self.w0 * L_recon + self.w1 * Lhep + self.w2 * L_smooth + self.w3 * L_grad_align

        # 4. Illumination Gradient Similarity Loss (LGSI, Eq. 6):
        # LGSI = ||(1 + alpha) * grad(I1) - grad(I2)||1
        gx_I2 = gradient_x(I2)
        gy_I2 = gradient_y(I2)
        
        # Handle alpha batch shape
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.view(-1, 1, 1, 1)
        target_gx = (1.0 + alpha) * gx_I1
        target_gy = (1.0 + alpha) * gy_I1
        LGSI = torch.mean(torch.abs(target_gx - gx_I2)) + torch.mean(torch.abs(target_gy - gy_I2))

        # Overall Loss (Eq. 7): Lall = omega4 * LR + LGSI
        L_all = self.w4 * LR + LGSI

        loss_dict = {
            "L_all": L_all.item(),
            "L_recon": L_recon.item(),
            "L_smooth": L_smooth.item(),
            "L_grad_align": L_grad_align.item(),
            "L_hep": Lhep.item(),
            "LGSI": LGSI.item()
        }

        return L_all, loss_dict


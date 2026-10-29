import torch
import torch.nn as nn

try:
    from .fem import FeatureExtractionModule
    from .lidm import LowLightImageDecompositionModule
    from .denoiser import DenoiseModule
except (ImportError, ValueError):
    from models.fem import FeatureExtractionModule
    from models.lidm import LowLightImageDecompositionModule
    from models.denoiser import DenoiseModule

class SPNet(nn.Module):
    """
    Complete SPNet model from IEEE TMM 2025 (Algorithm 1).
    Combines:
    - Feature Extraction Module (FEM)
    - Low-Light Image Decomposition Module (LIDM)
    - SPL Dynamic Guidance Fusion
    - Pre-trained & Frozen Denoise Module
    """
    def __init__(self, denoiser_weights=None):
        super(SPNet, self).__init__()
        self.fem = FeatureExtractionModule()
        self.lidm = LowLightImageDecompositionModule()
        self.denoiser = DenoiseModule()
        
        if denoiser_weights:
            self.denoiser.load_state_dict(torch.load(denoiser_weights, map_location='cpu'))
        
        # Freeze denoiser during LIE training as specified in the base paper
        for param in self.denoiser.parameters():
            param.requires_grad = False
        self.denoiser.eval()

    def forward(self, S1, SHE, S2, lam=1.0):
        """
        Args:
            S1: raw low-light image (B, 3, H, W)
            SHE: histogram-equalized image (B, 3, H, W)
            S2: quadratic pseudo-reference image (B, 3, H, W)
            lam: current self-paced learning guidance weight (scalar in [0, 1])
        Returns:
            R1: Reflectance of S1 (preliminary enhanced image)
            I1: Illumination of S1
            R2: Reflectance of S2
            I2: Illumination of S2
            final_enhanced: Output after passing R1 through denoiser
        """
        # 1. Feature Extraction (shared parameters)
        F1 = self.fem(S1)
        FHE = self.fem(SHE)
        F2 = self.fem(S2)

        # 2. SPL Guided Fusion: F_fusion = (1 - lam) * F + lam * FHE
        F_fusion1 = (1.0 - lam) * F1 + lam * FHE
        F_fusion2 = (1.0 - lam) * F2 + lam * FHE

        # 3. Retinex Decomposition
        R1, I1 = self.lidm(F_fusion1)
        R2, I2 = self.lidm(F_fusion2)

        # 4. Denoise Module applied to preliminary enhanced reflectance R1
        with torch.no_grad():
            final_enhanced = self.denoiser(R1)

        return R1, I1, R2, I2, final_enhanced

if __name__ == "__main__":
    net = SPNet()
    s1 = torch.randn(2, 3, 128, 128)
    she = torch.randn(2, 3, 128, 128)
    s2 = torch.randn(2, 3, 128, 128)
    r1, i1, r2, i2, out = net(s1, she, s2, lam=0.5)
    print("R1 shape:", r1.shape)
    print("I1 shape:", i1.shape)
    print("Final output shape:", out.shape)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total Params: {total_params / 1e6:.4f}M")

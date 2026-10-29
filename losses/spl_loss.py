import torch
import torch.nn as nn
import torchvision.models as models

class SPLLoss(nn.Module):
    """
    Self-Paced Learning Guidance and Histogram Equalization Prior Loss (Eq. 11 & Eq. 13).
    Extracts deep features using pre-trained VGG-19 network and computes MSE loss:
    Lhep = ||phi(R1) - phi(SHE)||^2_2
    Dynamically computes lambda guidance weight:
    - lambda = 1.0 if Lhep >= 1.0
    - lambda = (Lhep - 0.3) / (1.0 - 0.3) if 0.3 < Lhep < 1.0
    - lambda = 0.0 if Lhep <= 0.3
    """
    def __init__(self, device='cpu'):
        super(SPLLoss, self).__init__()
        # Load VGG-19 up to relu4_1 / relu5_1
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        # Features up to conv4_2 / relu4_2
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:18]).to(device).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.mse = nn.MSELoss()
        
        # ImageNet normalization parameters
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def compute_lhep(self, R1, SHE):
        feat_R1 = self.feature_extractor(self.normalize(R1))
        feat_SHE = self.feature_extractor(self.normalize(SHE))
        lhep = self.mse(feat_R1, feat_SHE)
        return lhep

    def compute_lambda(self, lhep_val):
        """
        Dynamically adjusts guidance weight lambda (Eq. 13):
        """
        if lhep_val >= 1.0:
            lam = 1.0
        elif lhep_val > 0.3:
            lam = (lhep_val - 0.3) / (1.0 - 0.3)
        else:
            lam = 0.0
        return float(lam)

    def forward(self, R1, SHE):
        lhep = self.compute_lhep(R1, SHE)
        lam = self.compute_lambda(lhep.item())
        return lhep, lam

if __name__ == "__main__":
    spl = SPLLoss()
    r1 = torch.rand(2, 3, 128, 128)
    she = torch.rand(2, 3, 128, 128)
    lhep, lam = spl(r1, she)
    print("Lhep:", lhep.item())
    print("Guidance weight lambda:", lam)


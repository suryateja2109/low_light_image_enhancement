import torch
import torch.nn as nn

class FeatureExtractionModule(nn.Module):
    """
    Feature Extraction Module (FEM) from SPNet (IEEE TMM 2025).
    Consists of two convolutional layers:
    - Conv1: 9x9 kernel (captures global contextual information)
    - Conv2: 3x3 kernel + ReLU (extracts fine-grained local details)
    Parameters are shared across S1 (raw), SHE (equalized), and S2 (pseudo-reference).
    """
    def __init__(self, in_channels=3, out_channels=64):
        super(FeatureExtractionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.relu(self.conv2(feat1))
        return feat2

if __name__ == "__main__":
    fem = FeatureExtractionModule()
    x = torch.randn(2, 3, 128, 128)
    out = fem(x)
    print("FEM output shape:", out.shape)

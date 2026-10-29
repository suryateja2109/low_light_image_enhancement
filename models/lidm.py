import torch
import torch.nn as nn

class LowLightImageDecompositionModule(nn.Module):
    """
    Low-Light Image Decomposition Module (LIDM) from SPNet (IEEE TMM 2025).
    Consists of 5 convolutional layers and 1 deconvolution layer.
    Outputs:
    - R: Reflectance map (3-channel RGB in [0, 1])
    - I: Illumination map (1-channel or 3-channel broadcast in [0, 1])
    """
    def __init__(self, in_channels=64):
        super(LowLightImageDecompositionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.deconv = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu_deconv = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Conv5 outputs 4 channels: 3 for Reflectance R, 1 for Illumination I
        self.conv5 = nn.Conv2d(64, 4, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.relu1(self.conv1(x))
        feat = self.relu2(self.conv2(feat))
        feat = self.relu_deconv(self.deconv(feat))
        feat = self.relu3(self.conv3(feat))
        feat = self.relu4(self.conv4(feat))
        out = self.sigmoid(self.conv5(feat))
        
        # Slice into Reflectance (first 3 channels) and Illumination (4th channel)
        R = out[:, 0:3, :, :]
        I = out[:, 3:4, :, :]
        return R, I

if __name__ == "__main__":
    lidm = LowLightImageDecompositionModule()
    x = torch.randn(2, 64, 128, 128)
    R, I = lidm(x)
    print("Reflectance R shape:", R.shape)
    print("Illumination I shape:", I.shape)

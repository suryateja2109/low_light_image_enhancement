import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual Block with Conv -> BN -> LeakyReLU -> Conv -> BN and skip connection.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = self.act(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        return self.act(x + res)

class LocalSelfAttention(nn.Module):
    """
    Local Self-Attention module following Vaswani et al. [55] and SPNet (Section III-D).
    Calculates attention weights across local receptive windows to capture salient areas
    without quadratic (H*W)^2 memory explosion.
    """
    def __init__(self, channels):
        super(LocalSelfAttention, self).__init__()
        self.local_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // 4, channels // 4, kernel_size=3, padding=1, groups=channels // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        attn_map = self.local_attn(x)
        out = self.gamma * (x * attn_map) + x
        return out

class DenoiseModule(nn.Module):
    """
    Denoise Module from SPNet (IEEE TMM 2025, Section III-D).
    Consists of input conv, cascaded Residual Blocks, Local Self-Attention, and output conv.
    Trained on double-noise formulation (Z -> Y) and frozen during LIE enhancement.
    """
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_resblocks=4):
        super(DenoiseModule, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        res_blocks = [ResidualBlock(num_features) for _ in range(num_resblocks)]
        self.body = nn.Sequential(*res_blocks)
        
        self.attention = LocalSelfAttention(num_features)
        
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        feat = self.attention(feat)
        out = self.tail(feat)
        return out

if __name__ == "__main__":
    dnm = DenoiseModule()
    x = torch.randn(2, 3, 128, 128)
    out = dnm(x)
    print("Denoise Module output shape:", out.shape)
    params = sum(p.numel() for p in dnm.parameters())
    print(f"Denoiser parameters: {params / 1e6:.4f}M")

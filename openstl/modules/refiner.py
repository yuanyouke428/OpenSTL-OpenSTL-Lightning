import torch
import torch.nn as nn


class ResidualRefiner(nn.Module):
    """
    轻量级 UNet，用于从模糊预测中恢复细节
    Input: [B, T, C, H, W] (Base Prediction)
    Output: [B, T, C, H, W] (Residual Details)
    """

    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()

        # 输入通道 = 1 (Prediction) + 1 (Input Last Frame) = 2
        # 我们将 Prediction 和 Last Frame 拼接作为输入
        self.encoder1 = nn.Sequential(nn.Conv2d(2, base_channels, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.encoder2 = nn.Sequential(nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1),
                                      nn.LeakyReLU(0.2, inplace=True))  # /2
        self.encoder3 = nn.Sequential(nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1),
                                      nn.LeakyReLU(0.2, inplace=True))  # /4

        # Bottleneck (中间层加宽一点感受野)
        self.middle = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Output layer (输出残差)
        self.final = nn.Conv2d(base_channels, 1, 3, 1, 1)

    def forward(self, coarse_pred, last_input_frame):
        # coarse_pred: [B, T, 1, H, W]
        # last_input_frame: [B, 1, 1, H, W]

        B, T, C, H, W = coarse_pred.shape

        # 为了高效处理，我们将 T 维度合并到 Batch 维度 (即把每一帧当作独立图片处理)
        x_base = coarse_pred.view(B * T, C, H, W)

        # 将最后一帧复制 T 次，作为纹理参考
        x_ref = last_input_frame.repeat(1, T, 1, 1, 1).view(B * T, C, H, W)

        # 拼接输入: [B*T, 2, H, W]
        x = torch.cat([x_base, x_ref], dim=1)

        # U-Net Forward
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        m = self.middle(e3)

        # Skip Connections (残差连接对恢复细节至关重要)
        d1 = self.decoder1(m + e3)
        d2 = self.decoder2(d1 + e2)

        residual = self.final(d2 + e1)

        # 恢复形状 [B, T, C, H, W]
        residual = residual.view(B, T, C, H, W)

        # 最终输出 = 粗糙预测 + 预测的残差
        return coarse_pred + residual
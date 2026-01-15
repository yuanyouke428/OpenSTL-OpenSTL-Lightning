import torch
from torch import nn
# 尝试从 simvp_model 导入 Encoder 和 Decoder，复用现有组件
try:
    from openstl.models.simvp_model import Encoder, Decoder
except ImportError:
    # 如果导入失败，手动定义简化的 Encoder/Decoder (防报错兜底)
    print("Warning: Importing Encoder/Decoder from simvp_model failed. Using fallback.")


    class Encoder(nn.Module):
        def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(C_in, C_hid, spatio_kernel, 1, spatio_kernel // 2),
                nn.ReLU(inplace=act_inplace),
                nn.Conv2d(C_hid, C_hid, 3, 1, 1),
                nn.ReLU(inplace=act_inplace)
            )

        def forward(self, x): return self.enc(x), x


    class Decoder(nn.Module):
        def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
            super().__init__()
            self.dec = nn.Sequential(
                nn.Conv2d(C_hid, C_hid, 3, 1, 1),
                nn.ReLU(inplace=act_inplace),
                nn.Conv2d(C_hid, C_out, spatio_kernel, 1, spatio_kernel // 2)
            )

        def forward(self, x, e): return self.dec(x)

from openstl.modules.mamba_modules import MambaBlock


class MambaCast(nn.Module):
    def __init__(self, in_shape, hid_S=64, hid_T=256, N_S=4, N_T=8, model_type='gSTA',
                 mlp_ratio=4., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super().__init__()
        T, C, H, W = in_shape  # Input shape, e.g., (10, 1, 64, 64)

        # 1. Encoder: 提取特征并下采样
        # 注意: SimVP 通常将 T 维度合并到 Channel 维度进行处理: (B, T*C, H, W)
        # 这里我们遵循这个经典设计
        H_down, W_down = H // (2 ** (N_S // 2)), W // (2 ** (N_S // 2))

        self.enc = Encoder(C * T, hid_S, N_S, spatio_kernel_enc, act_inplace)

        # 2. Translator: 用 Mamba 替代原来的卷积/Attention
        # 输入: (B, hid_S, H', W')
        modules = []
        for _ in range(N_T):
            modules.append(MambaBlock(in_channels=hid_S, d_state=16))
        self.hid = nn.Sequential(*modules)

        # 3. Decoder: 上采样并恢复预测
        self.dec = Decoder(hid_S, C * T, N_S, spatio_kernel_dec, act_inplace)

    def forward(self, x_raw, **kwargs):
        # x_raw: [B, T, C, H, W]
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B, T * C, H, W)

        # Encoder
        embed, skip = self.enc(x)

        # Mamba Translator (处理空间全局依赖)
        # embed shape: [B, hid_S, H_down, W_down]
        z = self.hid(embed)

        # Decoder
        Y = self.dec(z, skip)

        # Reshape back: [B, T*C, H, W] -> [B, T, C, H, W]
        Y = Y.reshape(B, T, C, H, W)
        return Y
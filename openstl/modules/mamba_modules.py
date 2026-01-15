# import torch
# import torch.nn as nn
# from mamba_ssm import Mamba
#
#
# class MambaLayer(nn.Module):
#     """
#     MambaLayer: 将 2D 图像特征展平为 1D 序列，输入 Mamba 后再恢复。
#     """
#
#     def __init__(self, dim, d_state=16, d_conv=4, expand=2):
#         super().__init__()
#         self.dim = dim
#         # 注意：LayerNorm 需要作用在 Channel 维度上
#         self.norm = nn.LayerNorm(dim)
#         self.mamba = Mamba(
#             d_model=dim,  # Model dimension
#             d_state=d_state,  # SSM state expansion factor
#             d_conv=d_conv,  # Local convolution width
#             expand=expand,  # Block expansion factor
#         )
#
#     # 引入“2D 选择性扫描” (2D Selective Scan / SS2D)
#     def forward(self, x):
#         # x: [B, C, H, W]
#         B, C, H, W = x.shape
#
#         # === 修复点 1: 必须先进行 LayerNorm ===
#         # LayerNorm 期望输入形状为 [..., C]，所以我们需要先 permute
#         x_norm = x.permute(0, 2, 3, 1)  # [B, H, W, C]
#         x_norm = self.norm(x_norm)  # 执行归一化，防止输入过大
#
#         # 变回 [B, C, H, W] 以便进行下面的切片操作，或者直接在下面处理
#         x_clean = x_norm.permute(0, 3, 1, 2)  # [B, C, H, W]
#
#         # 1. 生成 4 个方向的序列
#         # 使用归一化后的 x_clean 进行处理
#         x_hw = x_clean.view(B, C, -1).transpose(1, 2)  # [B, L, C] (左上->右下)
#         x_wh = x_clean.transpose(2, 3).contiguous().view(B, C, -1).transpose(1, 2)  # (转置扫描)
#         x_hw_rev = x_hw.flip(1)  # (右下->左上)
#         x_wh_rev = x_wh.flip(1)  # (转置反向)
#
#         # 2. Mamba Forward
#         # 官方库内部通常抗溢出能力尚可，只要输入是 Norm 过的
#         out1 = self.mamba(x_hw)
#         out2 = self.mamba(x_wh)
#         out3 = self.mamba(x_hw_rev)
#         out4 = self.mamba(x_wh_rev)
#
#         # 3. 还原并融合 (Merge)
#         y1 = out1.transpose(1, 2).view(B, C, H, W)
#         y2 = out2.transpose(1, 2).view(B, C, W, H).transpose(2, 3)
#         y3 = out3.flip(1).transpose(1, 2).view(B, C, H, W)
#         y4 = out4.flip(1).transpose(1, 2).view(B, C, W, H).transpose(2, 3)
#
#         # === 修复点 2: 防止数值爆炸 ===
#         # 直接相加会导致信号方差扩大 4 倍。建议取平均，或者除以 2
#         return (y1 + y2 + y3 + y4) / 4.0
#
# ###展平的方式进行扫苗
#     # def forward(self, x):
#     #     # x shape: [B, C, H, W]
#     #     B, C, H, W = x.shape
#     #
#     #     # 1. (B, C, H, W) -> (B, H*W, C) [Batch, Length, Dim]
#     #     x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
#     #
#     #     # 2. Norm
#     #     x_norm = self.norm(x_flat)
#     #
#     #     # 3. Mamba Forward (Standard)
#     #     out = self.mamba(x_norm)
#     #
#     #     # 4. (B, H*W, C) -> (B, C, H, W)
#     #     out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
#     #
#     #     return out
#
#
# class MambaBlock(nn.Module):
#     """
#     MambaBlock: 包含 MambaLayer 和 残差连接
#     """
#
#     def __init__(self, in_channels, d_state=16, **kwargs):
#         super().__init__()
#         self.layer = MambaLayer(in_channels, d_state=d_state)
#         # 可以在这里添加一个 FeedForward Network (FFN) 来增强非线性，类似于 Transformer Block
#         # 这里为了保持简洁和显存效率，暂时只用 MambaLayer
#
#     def forward(self, x):
#         # 残差连接
#         return x + self.layer(x)

######上面是1D和2D扫描，可以用于消融实验


import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MambaLayer(nn.Module):
    """
    MambaLayer: 结合了“风向感知门控”的 2D 扫描 Mamba 层。
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim

        # 1. LayerNorm (关键修复：防止数值爆炸)
        self.norm = nn.LayerNorm(dim)

        # 2. Mamba 核心模块
        self.mamba = Mamba(
            d_model=dim,  # Model dimension
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

        # 3. [新增] 风向感知门控网络 (Gating Network)
        # 输入维度 dim，压缩后输出 4 个权重 (对应 4 个方向)
        # 这是一个极轻量的网络，几乎不增加计算量
        self.scan_gating = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),  # Tanh 激活增加非线性
            nn.Linear(dim // 4, 4)  # 输出 [B, H, W, 4]
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # === 步骤 1: 归一化与权重生成 ===
        # Permute 变成 [B, H, W, C] 以便通过 Linear 层和 LayerNorm
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x_perm)  # [B, H, W, C]

        # 计算 4 个方向的动态权重
        # 输出形状: [B, H, W, 4] -> Softmax 保证权重之和为 1，数值稳定
        gate_weights = F.softmax(self.scan_gating(x_norm), dim=-1)

        # 准备输入给 Mamba 的干净数据
        x_clean = x_norm.permute(0, 3, 1, 2)  # [B, C, H, W]

        # === 步骤 2: 四方向扫描 (Cross Scan) ===
        # 1. 生成序列
        x_hw = x_clean.view(B, C, -1).transpose(1, 2)  # [B, L, C] (左上->右下 / SE)
        x_wh = x_clean.transpose(2, 3).contiguous().view(B, C, -1).transpose(1, 2)  # (转置扫描 / SW)
        x_hw_rev = x_hw.flip(1)  # (右下->左上 / NW)
        x_wh_rev = x_wh.flip(1)  # (转置反向 / NE)

        # 2. Mamba Forward (共享权重)
        out1 = self.mamba(x_hw)
        out2 = self.mamba(x_wh)
        out3 = self.mamba(x_hw_rev)
        out4 = self.mamba(x_wh_rev)

        # 3. 还原形状
        y1 = out1.transpose(1, 2).view(B, C, H, W)
        y2 = out2.transpose(1, 2).view(B, C, W, H).transpose(2, 3)
        y3 = out3.flip(1).transpose(1, 2).view(B, C, H, W)
        y4 = out4.flip(1).transpose(1, 2).view(B, C, W, H).transpose(2, 3)

        # === 步骤 3: 风向感知融合 (Anisotropic Fusion) ===

        # 将 4 个结果堆叠: [B, C, H, W] -> [B, C, H, W, 4]
        y_stack = torch.stack([y1, y2, y3, y4], dim=-1)

        # 调整 y_stack 形状以匹配 gate_weights: [B, H, W, C, 4]
        y_stack = y_stack.permute(0, 2, 3, 1, 4)

        # 调整 weights 形状: [B, H, W, 4] -> [B, H, W, 1, 4] (在 Channel 维度广播)
        weights = gate_weights.unsqueeze(-2)

        # 加权求和: sum( y_i * w_i )
        y_fused = (y_stack * weights).sum(dim=-1)  # [B, H, W, C]

        # 变回输出形状 [B, C, H, W]
        return y_fused.permute(0, 3, 1, 2)


class MambaBlock(nn.Module):
    """
    MambaBlock: 包含 MambaLayer 和 残差连接
    """

    def __init__(self, in_channels, d_state=16, **kwargs):
        super().__init__()
        self.layer = MambaLayer(in_channels, d_state=d_state)

    def forward(self, x):
        # 残差连接
        return x + self.layer(x)
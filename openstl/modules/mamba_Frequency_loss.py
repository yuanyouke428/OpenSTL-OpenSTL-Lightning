import torch
import torch.nn as nn
import torch.fft


class FrequencyEnhancedLoss(nn.Module):
    """
    Unified Frequency-Enhanced Hybrid Loss (HFFS-Loss)

    集成了三种核心机制，可通过参数灵活开关：
    1. Mask Weighting (物理先验): 对强降水区域进行静态加权。
    2. Focal Mechanism (动态聚焦): 对难样本(大误差)进行动态加权。
    3. Frequency Loss (频域保真): 恢复高频纹理细节。
    """

    def __init__(self, w_pixel=1.0, w_freq=0.01, threshold=0.5, gamma=0.5, mask_ratio=3.0):
        """
        Args:
            w_pixel (float): 空间域 Loss 的基础权重，通常为 1.0。
            w_freq (float):  频域 Loss 的权重。设为 0.0 可关闭频域损失 (用于消融实验)。
            threshold (float): 强降水掩码的阈值 (归一化后的值)。
            gamma (float):   Focal Loss 的聚焦参数。
                             - gamma = 0.5: Ours (V5) 默认设置。
                             - gamma = 0.0: 关闭 Focal 机制 (回退到普通 L1)。
            mask_ratio (float): 强降水区域的总权重倍率。
                             - mask_ratio = 3.0: Ours (V5) 默认 (1 + 2)。
                             - mask_ratio = 1.0: 关闭 Mask 机制 (所有像素平等)。
        """
        super().__init__()
        self.w_pixel = w_pixel
        self.w_freq = w_freq
        self.threshold = threshold
        self.gamma = gamma
        self.mask_ratio = mask_ratio

        self.l1_mean = nn.L1Loss()

    def forward(self, pred, target):
        # pred, target shape: [B, T, C, H, W]

        # ==========================================
        # Part A: 空间域损失 (Spatial Loss)
        # 包含: Base L1 + Focal Dynamic + Mask Static
        # ==========================================

        # 1. 计算基础绝对误差
        diff = torch.abs(pred - target)

        # 2. Focal Weight (动态权重)
        # 如果 gamma=0，focal_weight 恒为 1 (等效于普通 L1)
        # 如果 gamma=0.5，误差越大权重越大
        focal_weight = (diff + 1e-8) ** self.gamma

        # 3. Mask Weight (静态权重)
        # mask_ratio 是总倍率。例如 mask_ratio=3.0，
        # 也就是 1.0 (基础) + (target>thresh) * 2.0 (额外)
        # 如果 mask_ratio=1.0，则 rain_mask 恒为 1 (关闭 Mask)
        rain_mask = 1.0 + (target > self.threshold).float() * (self.mask_ratio - 1.0)

        # 4. 组合计算空间 Loss
        loss_pixel = (diff * focal_weight * rain_mask).mean()

        # ==========================================
        # Part B: 频域损失 (Frequency Loss)
        # ==========================================
        loss_freq = torch.tensor(0.0, device=pred.device)

        # 只有当权重 > 0 时才计算 FFT，节省算力
        if self.w_freq > 0:
            # 1. 转换到频域 (只对 H, W 维度做 FFT)
            fft_pred = torch.fft.rfft2(pred, norm='ortho')
            fft_target = torch.fft.rfft2(target, norm='ortho')

            # 2. 提取幅度谱 (Amplitude)
            amp_pred = torch.abs(fft_pred) + 1e-8
            amp_target = torch.abs(fft_target) + 1e-8

            # 3. 计算 Log 幅度差异
            loss_freq = self.l1_mean(torch.log(amp_pred), torch.log(amp_target))

        # ==========================================
        # Total Loss
        # ==========================================
        return self.w_pixel * loss_pixel + self.w_freq * loss_freq

import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 配置 =================
pred_path = '/home/ps/data2/zp/OpenSTL-OpenSTL-Lightning/work_dirs/mamba_Anisotropic_scan_FrequencyLoss_v5/saved_batches/batch_0_pred.npy'  # V5 的预测结果 [B, T, C, H, W]
gt_path = '/home/ps/data2/zp/OpenSTL-OpenSTL-Lightning/work_dirs/mamba_Anisotropic_scan_FrequencyLoss_v5/saved_batches/batch_0_true.npy'  # 真实值 [B, T, C, H, W]

save_dir = './vis_residual_check_enhanced'
os.makedirs(save_dir, exist_ok=True)


# ========================================

def visualize_enhanced():
    print("Loading data...")
    pred = np.load(pred_path)  # [B, T, C, H, W]
    gt = np.load(gt_path)

    # 确保维度对齐
    if len(pred.shape) == 4:
        pred = pred[:, :, None, :, :]
        gt = gt[:, :, None, :, :]

    # 计算绝对残差
    residual = np.abs(gt - pred)

    # 选择 Sample 8 (对应你之前的图)
    idx = 0  # 如果你保存的 batch 只有16个样本，Sample 8 就是 idx=8。
    # 之前的代码用的 idx 是循环里的，这里你可以手动指定 8

    # 假设我们要看 batch 里的第 8 个样本
    target_idx = 8
    if target_idx >= pred.shape[0]:
        target_idx = 0  # 防止越界

    time_steps = [0, 5, 11]

    fig, axes = plt.subplots(3, len(time_steps), figsize=(12, 10))

    for i, t in enumerate(time_steps):
        img_gt = gt[target_idx, t, 0]
        img_pred = pred[target_idx, t, 0]
        img_res = residual[target_idx, t, 0]

        # Row 1: GT
        ax = axes[0, i]
        ax.imshow(img_gt, cmap='jet')  # 用 jet 看强弱更明显
        ax.set_title(f'GT (T={t})')
        ax.axis('off')

        # Row 2: V5 Pred
        ax = axes[1, i]
        ax.imshow(img_pred, cmap='jet')
        ax.set_title(f'V5 Pred (T={t})')
        ax.axis('off')

        # Row 3: Residual (增强版!)
        ax = axes[2, i]
        # 【关键】这里不固定 vmax=255，而是用残差自己的最大值来归一化
        # 这样微弱的纹理也会变得非常亮
        res_max = np.max(img_res) + 1e-5
        im = ax.imshow(img_res, cmap='magma', vmin=0, vmax=res_max)
        ax.set_title(f'Residual (Max={res_max:.1f})')
        ax.axis('off')

        # 加个 colorbar 方便看数值范围
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f'Residual Check (Sample {target_idx}) - Texture Confirmation', fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'residual_enhanced_sample_{target_idx}.png')
    plt.savefig(save_path, dpi=150)
    print(f"✅ Saved: {save_path}")


if __name__ == '__main__':
    visualize_enhanced()
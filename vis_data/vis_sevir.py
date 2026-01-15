import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ==========================================
# 1. 核心配置
# ==========================================
# 数据存放的文件夹
DATA_DIR = '/home/ps/data2/zp/OpenSTL-OpenSTL-Lightning/work_dirs/mamba_Anisotropic_scan_FrequencyLoss_v5/saved_batches'

# 您想看哪个 Batch 文件?
BATCH_ID = 0

# 您想看该 Batch 里的第几个样本? (0 到 15)
SAMPLE_IDX = 8

# 结果保存路径
OUTPUT_DIR = 'vis_sevir_mamba_Anisotropic_scan_FrequencyLoss_v5_results'

# 显示设置
INTERVAL = 1  # 设为 1 表示画出每一帧 (12帧全部画出)
SAVE_FRAMES = True  # 是否将每一张小图单独保存到文件夹中

# SEVIR 数据反归一化参数
NORM_MEAN = 33.44
NORM_STD = 47.54


# ==========================================
# 2. 配色方案 (SEVIR VIL)
# ==========================================
def get_sevir_cmap():
    """
    参考 '可视化构建好的sevir数据集.py' 中的配色逻辑。
    定义 RGB 颜色列表和对应的 VIL 值边界。
    """
    # 1. 定义 RGB 颜色列表 (0-1 float)
    colors = [
        [0.0, 0.0, 0.0],  # 0-16 (Black)
        [0.30196078, 0.30196078, 0.30196078],  # 16-31 (Gray)
        [0.15686275, 0.74509804, 0.15686275],  # 31-59 (Green)
        [0.09803922, 0.58823529, 0.09803922],  # 59-74 (Dark Green)
        [0.03921569, 0.41176471, 0.03921569],  # 74-100
        [0.0, 0.35294118, 0.0],  # 100-133
        [0.99215686, 0.97254902, 0.00784314],  # 133-160 (Yellow)
        [0.89803922, 0.7372549, 0.0],  # 160-181
        [0.99215686, 0.58431373, 0.0],  # 181-219 (Orange)
        [0.99215686, 0.0, 0.0],  # 219-255 (Red)
        [0.83137255, 0.0, 0.0],  # > 255 (Dark Red)
        [0.97254902, 0.0, 0.99215686]  # (Magenta)
    ]

    # 2. 定义数值边界
    bounds = [0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 255, 256]

    # 3. 创建 Colormap 和 Norm
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm, bounds


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 构造文件路径
    true_path = os.path.join(DATA_DIR, f'batch_{BATCH_ID}_true.npy')
    pred_path = os.path.join(DATA_DIR, f'batch_{BATCH_ID}_pred.npy')

    print(f"Loading Batch {BATCH_ID} ...")
    if not os.path.exists(true_path):
        print(f"错误：找不到文件 {true_path}，请检查 BATCH_ID 是否存在。")
        return

    # 2. 加载数据
    trues_batch = np.load(true_path)
    preds_batch = np.load(pred_path)

    # 3. 提取指定样本并反归一化
    print(f"提取第 {SAMPLE_IDX} 号样本 (Batch {BATCH_ID})...")
    true_sample = trues_batch[SAMPLE_IDX] * NORM_STD + NORM_MEAN
    pred_sample = preds_batch[SAMPLE_IDX] * NORM_STD + NORM_MEAN

    if true_sample.ndim == 4:
        true_sample = true_sample.squeeze(1)
        pred_sample = pred_sample.squeeze(1)

    # 限制范围 [0, 255]
    true_sample = np.clip(true_sample, 0, 255)
    pred_sample = np.clip(pred_sample, 0, 255)

    T = true_sample.shape[0]
    print(f"序列总长度: {T} 帧")

    # 【修复1】这里必须接收 3 个返回值
    cmap, norm, bounds = get_sevir_cmap()

    # ==========================================
    # 功能 A: 绘制对比大图
    # ==========================================
    frames = list(range(0, T, INTERVAL))
    num_cols = len(frames)

    print(f"正在生成对比大图 (共 {num_cols} 列)...")

    fig, axes = plt.subplots(2, num_cols, figsize=(num_cols * 2.2, 4.5))

    if num_cols == 1:
        axes = axes[:, None]

    cols_title = [f"t={t + 1}" for t in frames]
    rows_title = ['Ground Truth', 'Prediction']

    for ax, col in zip(axes[0], cols_title):
        ax.set_title(col, fontsize=12)
    for ax, row in zip(axes[:, 0], rows_title):
        ax.set_ylabel(row, rotation=90, fontsize=14)

    # 循环绘图
    # 注意：im 对象必须在这里保存，供下面 Colorbar 使用
    im = None
    for idx, frame_t in enumerate(frames):
        # GT
        im = axes[0, idx].imshow(true_sample[frame_t], cmap=cmap, norm=norm)
        axes[0, idx].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Pred
        axes[1, idx].imshow(pred_sample[frame_t], cmap=cmap, norm=norm)
        axes[1, idx].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # 【修复2】添加 Colorbar (关键修改)
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])

    # 使用 bounds 作为刻度，spacing='uniform' 让不均匀的区间等宽显示
    cbar = fig.colorbar(im, cax=cbar_ax,
                        ticks=bounds[:-1],
                        spacing='uniform',
                        orientation='vertical')

    cbar.set_label('VIL Levels (0-255)', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=8)

    # 保存大图
    save_name_full = f"Vis_Batch{BATCH_ID}_Sample{SAMPLE_IDX}_Full.png"
    save_path_full = os.path.join(OUTPUT_DIR, save_name_full)
    plt.savefig(save_path_full, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"--> 对比大图已保存: {save_path_full}")

    # ==========================================
    # 功能 B: 单独保存每一帧
    # ==========================================
    if SAVE_FRAMES:
        frames_dir = os.path.join(OUTPUT_DIR, f"frames_batch{BATCH_ID}_sample{SAMPLE_IDX}")
        os.makedirs(frames_dir, exist_ok=True)
        print(f"正在保存单帧图片至: {frames_dir} ...")

        for t in range(T):
            gt_filename = os.path.join(frames_dir, f"gt_{t + 1:02d}.png")
            plt.imsave(gt_filename, true_sample[t], cmap=cmap, vmin=0, vmax=255)

            pred_filename = os.path.join(frames_dir, f"pred_{t + 1:02d}.png")
            plt.imsave(pred_filename, pred_sample[t], cmap=cmap, vmin=0, vmax=255)

        print(f"--> 所有 {T * 2} 张单帧图片已保存完毕。")


if __name__ == '__main__':
    main()
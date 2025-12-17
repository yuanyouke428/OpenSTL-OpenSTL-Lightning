import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# 1. 配置路径
work_dir = './work_dirs/cikm_b16_e100_run/saved'
save_dir = './vis_results_cikm_style'
os.makedirs(save_dir, exist_ok=True)

print(f"Loading data from {work_dir} ...")

# 2. 加载数据
try:
    inputs = np.load(os.path.join(work_dir, 'inputs.npy'))
    preds = np.load(os.path.join(work_dir, 'preds.npy'))
    trues = np.load(os.path.join(work_dir, 'trues.npy'))
except FileNotFoundError:
    print("Error: 找不到 .npy 文件")
    exit()

# 3. 关键：定义气象雷达的自定义配色 (0-75 dBZ)
# 参考标准雷达图：透明/白 -> 蓝 -> 青 -> 绿 -> 黄 -> 红 -> 紫
# CIKM 数据归一化是 0-1，对应 0-76 dBZ
colors = [
    (1, 1, 1, 0),  # 0.0: 透明/白色 (无雨)
    (0, 0.6, 1),  # 0.2: 蓝色 (小雨)
    (0, 0.8, 0),  # 0.4: 绿色 (中雨)
    (1, 1, 0),  # 0.6: 黄色 (大雨)
    (1, 0, 0),  # 0.8: 红色 (暴雨)
    (0.6, 0, 0.6)  # 1.0: 紫色 (特大暴雨)
]
# 创建自定义的线性分段 Colormap
cmap_radar = mcolors.LinearSegmentedColormap.from_list('radar', colors)


# 4. 可视化函数
def save_styled_plot(idx):
    in_len = inputs.shape[1]
    out_len = preds.shape[1]

    seq_in = inputs[idx].squeeze(1)
    seq_true = trues[idx].squeeze(1)
    seq_pred = preds[idx].squeeze(1)

    # 创建画布：宽一点，给右边的 colorbar 留位置
    total_cols = max(in_len, out_len)
    fig, axes = plt.subplots(3, total_cols, figsize=(total_cols * 2 + 2, 6), constrained_layout=True)

    # 设置行标题
    row_labels = ['Input\n(Past)', 'Ground Truth\n(Future)', 'Prediction\n(Future)']

    # 统一绘图参数
    plot_args = dict(cmap=cmap_radar, vmin=0.05, vmax=1.0)  # vmin=0.05 过滤掉底噪，让背景变白

    # --- 第一行：Inputs ---
    for t in range(total_cols):
        ax = axes[0, t]
        if t < in_len:
            im = ax.imshow(seq_in[t], **plot_args)
            ax.set_title(f't - {in_len - t}', fontsize=10)
        ax.axis('off')

    # --- 第二行：Ground Truth ---
    for t in range(total_cols):
        ax = axes[1, t]
        if t < out_len:
            im = ax.imshow(seq_true[t], **plot_args)
            ax.set_title(f't + {t + 1}', fontsize=10)
        ax.axis('off')

    # --- 第三行：Prediction ---
    for t in range(total_cols):
        ax = axes[2, t]
        if t < out_len:
            im = ax.imshow(seq_pred[t], **plot_args)
        ax.axis('off')

    # 添加左侧文字标签
    for ax, row_label in zip(axes[:, 0], row_labels):
        ax.text(-0.2, 0.5, row_label, transform=ax.transAxes,
                rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')

    # --- 添加 Colorbar ---
    # 在图的最右侧添加一个色条
    cbar = fig.colorbar(im, ax=axes[:, :], location='right', shrink=0.6, pad=0.02)
    # 设置刻度标签 (0-1 对应 0-76 dBZ)
    cbar.set_label('Reflectivity (Normalized)', rotation=270, labelpad=15)

    # 保存
    save_path = os.path.join(save_dir, f'sample_{idx}_styled.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"美化后的结果已保存: {save_path}")


# 5. 运行 (自动寻找一个降雨比较明显的样本)
# 计算每个样本的像素总和，找“雨最大”的那个，这样显示效果最明显
max_rain_idx = np.argmax(trues.sum(axis=(1, 2, 3, 4)))
print(f"正在可视化降雨量最大的样本 (Index: {max_rain_idx}) ...")
save_styled_plot(max_rain_idx)

# 如果想看第0个，也可以取消下面这行的注释
# save_styled_plot(0)
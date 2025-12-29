# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import matplotlib.patches as patches
# import os
#
# # 1. 配置路径
# work_dir = './work_dirs/CIKM_PhyDNet/saved'
# save_dir = './vis_results_cikm_PhyDNet'
# os.makedirs(save_dir, exist_ok=True)
#
# print(f"Loading data from {work_dir} ...")
#
# # 2. 加载数据
# try:
#     inputs = np.load(os.path.join(work_dir, 'inputs.npy'))
#     preds = np.load(os.path.join(work_dir, 'preds.npy'))
#     trues = np.load(os.path.join(work_dir, 'trues.npy'))
# except FileNotFoundError:
#     print("Error: 找不到 .npy 文件，请检查路径")
#     exit()
#
# # 3. 定义 Paper-Replica 配色方案 (复刻 inspect_cikm.py)
# # 对应 dBZ 范围: 0-76
# # 分级: 0-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-76
# color_list = [
#     (1, 1, 1, 0),  # 0-10:   完全透明/白色 (无雨)
#     '#0000FF',  # 10-20:  纯蓝 (Blue)
#     '#00FFFF',  # 20-30:  青色 (Cyan)
#     '#00FF00',  # 30-40:  纯绿 (Green)
#     '#FFFF00',  # 40-50:  纯黄 (Yellow)
#     '#FF0000',  # 50-60:  纯红 (Red)
#     '#FF00FF',  # 60-70:  洋红 (Magenta)
#     '#8B008B',  # 70-85:  深紫 (Purple) - 极端区域
# ]
#
# # 定义分级阈值 (dBZ)
# levels = [-10, 10, 20, 30, 40, 50, 60, 70, 85]
#
# # 创建 Colormap 和 Normalization
# cmap_radar = mcolors.ListedColormap(color_list)
# norm_radar = mcolors.BoundaryNorm(boundaries=levels, ncolors=len(color_list))
#
# # 4. 可视化函数
# def save_styled_plot(idx):
#     in_len = inputs.shape[1]
#     out_len = preds.shape[1]
#
#     # 获取数据并 squeeze 掉通道维度 (T, H, W)
#     # 【关键】乘以 76.0，将 0-1 归一化数据还原为 0-76 dBZ 物理量
#     scale_factor = 95.0
#
#     seq_in = inputs[idx].squeeze(1) * scale_factor-10.0
#     seq_true = trues[idx].squeeze(1) * scale_factor-10.0
#     seq_pred = preds[idx].squeeze(1) * scale_factor-10.0
#
#     # 创建画布
#     total_cols = max(in_len, out_len)
#     fig, axes = plt.subplots(3, total_cols, figsize=(total_cols * 2 + 2, 6), constrained_layout=True)
#
#     row_labels = ['Input\n(Past)', 'Ground Truth\n(Future)', 'Prediction\n(Future)']
#
#     # 统一绘图参数
#     plot_args = dict(cmap=cmap_radar, norm=norm_radar)
#
#     # --- 第一行：Inputs ---
#     for t in range(total_cols):
#         ax = axes[0, t]
#         if t < in_len:
#             im = ax.imshow(seq_in[t], **plot_args)
#             ax.set_title(f'T - {in_len - t}', fontsize=10, color='blue', fontweight='bold')
#             # 添加蓝色边框表示输入
#             rect = patches.Rectangle((0, 0), 127, 127, linewidth=1, edgecolor='blue', facecolor='none')
#             ax.add_patch(rect)
#         ax.axis('off')
#
#     # --- 第二行：Ground Truth ---
#     for t in range(total_cols):
#         ax = axes[1, t]
#         if t < out_len:
#             im = ax.imshow(seq_true[t], **plot_args)
#             ax.set_title(f'T + {t + 1}', fontsize=10)
#         ax.axis('off')
#
#     # --- 第三行：Prediction ---
#     for t in range(total_cols):
#         ax = axes[2, t]
#         if t < out_len:
#             im = ax.imshow(seq_pred[t], **plot_args)
#             ax.set_title(f'T + {t + 1}', fontsize=10, color='#D60000', fontweight='bold')
#             # 添加红色边框表示预测
#             rect = patches.Rectangle((0, 0), 127, 127, linewidth=1, edgecolor='#D60000', facecolor='none')
#             ax.add_patch(rect)
#         ax.axis('off')
#
#     # 添加左侧文字标签
#     for ax, row_label in zip(axes[:, 0], row_labels):
#         ax.text(-0.2, 0.5, row_label, transform=ax.transAxes,
#                 rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')
#
#     # --- 添加 Colorbar ---
#     # 使用离散刻度
#     cbar = fig.colorbar(im, ax=axes[:, :], location='right', shrink=0.6, pad=0.02,
#                         ticks=levels, spacing='uniform')
#     cbar.set_label('Reflectivity (dBZ)', rotation=270, labelpad=15)
#
#     # 保存
#     save_path = os.path.join(save_dir, f'sample_{idx}_styled.png')
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"结果已保存: {save_path}")
#
#
# # 5. 运行逻辑
# # 自动寻找一个降雨比较明显的样本 (根据 Ground Truth 的总像素值)
# # 注意：trues 也是 0-1 范围，求和即可
# sum_rain = trues.sum(axis=(1, 2, 3, 4))
# max_rain_idx = np.argmax(sum_rain)
#
# print(f"正在可视化降雨量最大的样本 (Index: {max_rain_idx}) ...")
# save_styled_plot(3)
#
# # 你也可以手动指定 index，例如可视化前 5 个
# # for i in range(5):
# #     save_styled_plot(i)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.cm as cm  # 引入 cm 模块用于颜色映射处理
import os

# 1. 配置路径
work_dir = './work_dirs/CIKM_PhyDNet/saved'
save_dir = './vis_results_cikm_PhyDNet'
os.makedirs(save_dir, exist_ok=True)

print(f"Loading data from {work_dir} ...")

# 2. 加载数据
try:
    inputs = np.load(os.path.join(work_dir, 'inputs.npy'))
    preds = np.load(os.path.join(work_dir, 'preds.npy'))
    trues = np.load(os.path.join(work_dir, 'trues.npy'))
except FileNotFoundError:
    print("Error: 找不到 .npy 文件，请检查路径")
    exit()

# 3. 定义 Paper-Replica 配色方案
# 对应 dBZ 范围: 0-76
color_list = [
    (1, 1, 1, 0),  # 0-10:   完全透明 (无雨)
    '#0000FF',  # 10-20:  纯蓝
    '#00FFFF',  # 20-30:  青色
    '#00FF00',  # 30-40:  纯绿
    '#FFFF00',  # 40-50:  纯黄
    '#FF0000',  # 50-60:  纯红
    '#FF00FF',  # 60-70:  洋红
    '#8B008B',  # 70-85:  深紫
]

# 定义分级阈值 (dBZ)
levels = [-10, 10, 20, 30, 40, 50, 60, 70, 85]

# 创建 Colormap 和 Normalization
cmap_radar = mcolors.ListedColormap(color_list)
norm_radar = mcolors.BoundaryNorm(boundaries=levels, ncolors=len(color_list))


# 4. 可视化函数 (已修改)
def save_styled_plot(idx):
    in_len = inputs.shape[1]
    out_len = preds.shape[1]

    # 获取数据并转换 (T, H, W) -> dBZ
    scale_factor = 95.0

    # 这里的 -10.0 偏移量是根据你提供的代码保留的，请确认是否符合你的归一化逻辑
    seq_in = inputs[idx].squeeze(1) * scale_factor - 10.0
    seq_true = trues[idx].squeeze(1) * scale_factor - 10.0
    seq_pred = preds[idx].squeeze(1) * scale_factor - 10.0

    # ==========================
    # Part A: 保存完整的拼接大图 (原有逻辑)
    # ==========================
    total_cols = max(in_len, out_len)
    fig, axes = plt.subplots(3, total_cols, figsize=(total_cols * 2 + 2, 6), constrained_layout=True)
    row_labels = ['Input\n(Past)', 'Ground Truth\n(Future)', 'Prediction\n(Future)']
    plot_args = dict(cmap=cmap_radar, norm=norm_radar)

    # --- 第一行：Inputs ---
    for t in range(total_cols):
        ax = axes[0, t]
        if t < in_len:
            im = ax.imshow(seq_in[t], **plot_args)
            ax.set_title(f'T - {in_len - t}', fontsize=10, color='blue', fontweight='bold')
            rect = patches.Rectangle((0, 0), 127, 127, linewidth=1, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')

    # --- 第二行：Ground Truth ---
    for t in range(total_cols):
        ax = axes[1, t]
        if t < out_len:
            im = ax.imshow(seq_true[t], **plot_args)
            ax.set_title(f'T + {t + 1}', fontsize=10)
        ax.axis('off')

    # --- 第三行：Prediction ---
    for t in range(total_cols):
        ax = axes[2, t]
        if t < out_len:
            im = ax.imshow(seq_pred[t], **plot_args)
            ax.set_title(f'T + {t + 1}', fontsize=10, color='#D60000', fontweight='bold')
            rect = patches.Rectangle((0, 0), 127, 127, linewidth=1, edgecolor='#D60000', facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')

    # 添加文字和 Colorbar
    for ax, row_label in zip(axes[:, 0], row_labels):
        ax.text(-0.2, 0.5, row_label, transform=ax.transAxes,
                rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')

    cbar = fig.colorbar(im, ax=axes[:, :], location='right', shrink=0.6, pad=0.02,
                        ticks=levels, spacing='uniform')
    cbar.set_label('Reflectivity (dBZ)', rotation=270, labelpad=15)

    # 保存大图
    save_path_full = os.path.join(save_dir, f'sample_{idx}_styled.png')
    plt.savefig(save_path_full, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"完整对比图已保存: {save_path_full}")

    # ==========================
    # Part B: 单独保存预测帧 (新增逻辑)
    # ==========================

    # 1. 创建存放单独帧的子文件夹
    frames_dir = os.path.join(save_dir, f'sample_{idx}_frames')
    os.makedirs(frames_dir, exist_ok=True)

    # --- 修改开始：创建背景为纯白的专用 Colormap ---
    # 复制一份原始的全局颜色列表
    color_list_solid = list(color_list)
    # 将列表中第一个颜色（对应最低数值区间，即背景）修改为不透明的纯白色
    color_list_solid[0] = 'white'  # 强制背景变为不透明白色

    # 使用修改后的颜色列表创建新的 ListedColormap
    cmap_radar_solid = mcolors.ListedColormap(color_list_solid)

    # 2. 初始化颜色映射器
    # 【关键修改】：这里必须传入 cmap_radar_solid
    sm = cm.ScalarMappable(cmap=cmap_radar_solid, norm=norm_radar)

    # 3. 循环保存每一帧
    for t in range(out_len):
        # 将 dBZ 数据转换为 RGBA 图像数据
        frame_rgba = sm.to_rgba(seq_pred[t])

        # 保存路径
        frame_path = os.path.join(frames_dir, f'pred_t{t + 1}.png')

        # 使用 plt.imsave 直接保存图像数组
        plt.imsave(frame_path, frame_rgba)

    print(f"  └─ 单独预测帧(白底)已保存至: {frames_dir}")


# 5. 运行逻辑
sum_rain = trues.sum(axis=(1, 2, 3, 4))
max_rain_idx = np.argmax(sum_rain)

print(f"正在可视化降雨量最大的样本 (Index: {max_rain_idx}) ...")
save_styled_plot(3)  # 你可以把 3 改为 max_rain_idx
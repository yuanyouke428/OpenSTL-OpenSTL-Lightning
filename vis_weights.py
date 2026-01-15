import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import os
import sys
import warnings
import argparse

from openstl.utils import create_parser, load_config, update_config
from openstl.utils import get_dataset

warnings.filterwarnings('ignore')

# ================= 🔧 配置区域 =================
config_file = 'configs/sevir/Mamba.py'
ckpt_path = 'work_dirs/mamba_Anisotropic_scan/checkpoints/best.ckpt'
data_root_path = './data/sevir'

# 反归一化参数
NORM_MEAN = 33.44
NORM_STD = 47.54


def import_model_class():
    try:
        from openstl.models import MambaCast
        return MambaCast
    except ImportError:
        pass
    try:
        sys.path.append('./openstl/models')
        from mamba_model import MambaCast
        return MambaCast
    except ImportError:
        pass
    raise ImportError("❌ Cannot find MambaCast model definition!")


def get_sevir_cmap():
    """SEVIR 配色方案"""
    colors = [
        [0.0, 0.0, 0.0], [0.30, 0.30, 0.30], [0.15, 0.74, 0.15], [0.09, 0.58, 0.09],
        [0.03, 0.41, 0.03], [0.0, 0.35, 0.0], [0.99, 0.97, 0.00], [0.89, 0.73, 0.0],
        [0.99, 0.58, 0.0], [0.99, 0.0, 0.0], [0.83, 0.0, 0.0], [0.97, 0.0, 0.99]
    ]
    bounds = [0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 255, 300]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend='max')
    return cmap, norm, bounds


def add_direction_arrow(ax, direction_idx):
    """在图中心添加半透明的方向箭头水印"""
    rotations = [-45, -135, 135, 45]
    ax.text(0.5, 0.5, '➤', transform=ax.transAxes,
            ha='center', va='center', fontsize=60, color='black',
            alpha=0.15, rotation=rotations[direction_idx], fontname='DejaVu Sans')


def visualize_sequence():
    print(f"🚀 Starting Sequence Visualization (Fix Layout)...")

    # --- 1. 配置与模型加载 ---
    args_namespace = create_parser().parse_args([])
    args_dict = vars(args_namespace)
    args_dict.update({
        'config_file': config_file, 'ckpt_path': ckpt_path,
        'dataname': 'sevir', 'data_root': data_root_path,
        'batch_size': 16, 'val_batch_size': 16
    })

    config_dict = load_config(config_file)
    update_config(args_dict, config_dict)
    if 'in_shape' not in args_dict and 'input_shape' in args_dict:
        args_dict['in_shape'] = args_dict['input_shape']
    args = argparse.Namespace(**args_dict)

    MambaCast = import_model_class()
    try:
        model = MambaCast(**args_dict).cuda()
    except TypeError:
        model = MambaCast(in_shape=args_dict['in_shape'], hid_S=args_dict['hid_S'],
                          hid_T=args_dict['hid_T'], N_S=args_dict['N_S'],
                          N_T=args_dict['N_T'], model_type=args_dict.get('model_type', 'MambaCast')).cuda()

    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint not found at {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {k[6:] if k.startswith('model.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # --- 2. 数据加载 ---
    try:
        _, _, test_loader = get_dataset(args_dict['dataname'], args_dict)
    except Exception as e:
        print(f"❌ Data load failed: {e}")
        return

    print("📦 Extracting Data Batch...")
    batch_x, _ = next(iter(test_loader))
    batch_x = batch_x.cuda().float()

    # --- 3. Hook 获取权重 ---
    weights_storage = []

    def hook_fn(module, input, output):
        weights_storage.append(output.detach().cpu())

    target_layer = None
    for name, module in model.named_modules():
        if 'scan_gating' in name:
            target_layer = module
            break

    if target_layer is None:
        print("❌ Cannot find 'scan_gating' layer")
        return

    handle = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(batch_x)
    handle.remove()

    gate_weights = F.softmax(weights_storage[0], dim=-1).numpy()

    # --- 4. 绘图逻辑 (修复版) ---
    indices_to_plot = [6, 10]
    indices_to_plot = [i for i in indices_to_plot if i < batch_x.shape[0]]
    cmap, norm, bounds = get_sevir_cmap()
    time_steps = [0, 3, 6, 9, 12]

    for idx in indices_to_plot:
        print(f"🎨 Plotting sequence for Sample {idx}...")

        sample_seq = batch_x[idx].cpu().numpy()
        sample_frames = sample_seq[time_steps, 0, :, :] * NORM_STD + NORM_MEAN
        sample_frames = np.clip(sample_frames, 0, 255)
        weights = gate_weights[idx]
        max_vals_per_dir = [np.max(weights[:, :, k]) for k in range(4)]
        dominant_dir_idx = np.argmax(max_vals_per_dir)

        # === 布局设计 ===
        # 【关键修改】：去掉了 constrained_layout=True，改用手动调整
        fig = plt.figure(figsize=(20, 9))

        # 左右留白给 Colorbar
        plt.subplots_adjust(left=0.05, right=0.92, top=0.92, bottom=0.05, hspace=0.3, wspace=0.1)

        gs = fig.add_gridspec(2, 20)

        # --- Row 1: VIL 演化序列 ---
        for i, t in enumerate(time_steps):
            ax = fig.add_subplot(gs[0, i * 4:(i + 1) * 4])
            im = ax.imshow(sample_frames[i], cmap=cmap, norm=norm, interpolation='nearest')
            ax.set_title(f"T={t}", fontsize=14, fontweight='bold')
            ax.axis('off')

            if i == 0:
                ax.text(0.05, 0.95, "Start", transform=ax.transAxes, color='white',
                        fontweight='bold', fontsize=12, va='top')
            if i == 4:
                ax.text(0.95, 0.95, "End", transform=ax.transAxes, color='white',
                        fontweight='bold', fontsize=12, va='top', ha='right')

        # 【关键修改】：手动指定 Top Colorbar 位置 [left, bottom, width, height]
        # 位置大致在第一排右侧
        cax_top = fig.add_axes([0.93, 0.55, 0.015, 0.35])
        cbar = fig.colorbar(im, cax=cax_top, ticks=bounds[:-1])
        cbar.set_label('VIL (Pixel Value)', fontsize=12)

        # --- Row 2: Attention Weights ---
        directions_text = ['SE (South-East)', 'SW (South-West)', 'NW (North-West)', 'NE (North-East)']
        w_mean, w_std = np.mean(weights), np.std(weights)
        vmin, vmax = max(0.0, w_mean - 2.5 * w_std), min(1.0, w_mean + 2.5 * w_std)

        im_w = None  # 占位符

        for i in range(4):
            ax = fig.add_subplot(gs[1, 2 + i * 4: 2 + (i + 1) * 4])

            im_w = ax.imshow(weights[:, :, i], cmap='coolwarm', vmin=vmin, vmax=vmax)

            is_dominant = (i == dominant_dir_idx)
            title_color = 'red' if is_dominant else 'darkblue'
            title_weight = 'bold' if is_dominant else 'normal'
            title_text = directions_text[i]
            if is_dominant: title_text += " [Dominant]"

            ax.set_title(title_text, fontsize=13, color=title_color, fontweight=title_weight)
            ax.axis('off')
            add_direction_arrow(ax, i)

            max_val = max_vals_per_dir[i]
            ax.text(0.05, 0.05, f"Max: {max_val:.2f}", transform=ax.transAxes,
                    fontsize=12, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

            if is_dominant:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
                    spine.set_visible(True)
                rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                     linewidth=4, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

        # 【关键修改】：手动指定 Bottom Colorbar 位置
        # 位置大致在第二排右侧
        cax_bot = fig.add_axes([0.93, 0.10, 0.015, 0.35])
        cbar_w = fig.colorbar(im_w, cax=cax_bot)
        cbar_w.set_label('Gating Weight', fontsize=12)

        fig.suptitle(f"Spatiotemporal Evolution & Wind-Aware Gating (Sample {idx})",
                     fontsize=18, y=0.98, fontweight='bold')

        save_name = f'vis_sequence_sample_{idx}_enhanced.png'
        # 使用 bbox_inches='tight' 会自动处理边缘，但不会像 constrained_layout 那样崩溃
        plt.savefig(save_name, dpi=200, bbox_inches='tight')
        print(f"✅ Saved: {save_name}")
        plt.close()


if __name__ == '__main__':
    visualize_sequence()
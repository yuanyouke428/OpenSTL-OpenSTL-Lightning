import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys

# ================= 项目路径设置 =================
sys.path.append(os.getcwd())

# 1. 尝试导入 metrics
try:
    from openstl.core.metrics import metric as calc_all_metrics

    print("✅ Successfully loaded 'openstl.core.metrics.py'")
except ImportError:
    try:
        import sys

        sys.path.append(os.path.join(os.getcwd(), 'openstl', 'core'))
        from metrics import metric as calc_all_metrics

        print("✅ Successfully loaded 'metrics.py' from openstl/core/")
    except ImportError:
        raise ImportError("❌ Cannot find 'metrics.py'. Please check path.")

from openstl.utils import load_config, update_config, get_dataset
from openstl.modules.refiner import ResidualRefiner


def import_mamba():
    """鲁棒的模型导入"""
    try:
        from openstl.models.mamba_model import MambaCast
        return MambaCast
    except ImportError:
        pass
    try:
        from openstl.models import MambaCast
        return MambaCast
    except ImportError:
        pass
    try:
        import sys
        model_path = os.path.join(os.getcwd(), 'openstl', 'models')
        if model_path not in sys.path: sys.path.append(model_path)
        from mamba_model import MambaCast
        return MambaCast
    except ImportError as e:
        pass
    raise ImportError("Fatal: Cannot find 'MambaCast'. Check path.")


# ================= 2. SEVIR 配色方案 =================
def get_sevir_cmap():
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
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 255, 300]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


# ================= 3. 完美可视化函数 (大图) =================
def save_vis_complete(base, refined, gt, batch_idx, sample_idx, save_dir):
    """
    可视化 12 帧序列，Row1=GT, Row2=Mamba, Row3=Refined
    """
    cmap, norm = get_sevir_cmap()
    T = min(12, gt.shape[0])

    # 画布大小
    fig, axes = plt.subplots(3, T, figsize=(20, 5), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    # 顺序：真值 -> Mamba -> Refiner
    row_titles = ["Ground Truth", "Mamba", "Refined"]
    data_list = [gt, base, refined]

    for row in range(3):
        for t in range(T):
            ax = axes[row, t]
            img_data = data_list[row][t, 0]  # [H, W]

            # 使用 imshow (它支持 norm)
            im = ax.imshow(img_data, cmap=cmap, norm=norm)

            # 时间标签 (仅第一行)
            if row == 0:
                ax.set_title(f"t={t + 1}", fontsize=12)

            # 行标签 (仅第一列)
            if t == 0:
                ax.set_ylabel(row_titles[row], fontsize=14, fontweight='bold', labelpad=10)

            ax.set_xticks([])
            ax.set_yticks([])

    # Colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[16, 74, 133, 160, 181, 219, 255])
    cbar.ax.set_yticklabels(['16', '74', '133', '160', '181', '219', '255'], fontsize=10)
    cbar.set_label('VIL Levels', fontsize=12)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'batch{batch_idx}_sample{sample_idx}_overview.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Saved overview: {save_path}")


# ================= 4. 单张纯净保存函数 (修复报错版) =================
def save_clean_frames(base, refined, gt, batch_idx, sample_idx, save_dir):
    """
    保存单独的帧，手动应用 Colormap 避免 imsave 不支持 norm 的问题
    """
    cmap, norm = get_sevir_cmap()
    T = min(12, gt.shape[0])

    frames_dir = os.path.join(save_dir, f'batch{batch_idx}_sample{sample_idx}_frames')
    os.makedirs(frames_dir, exist_ok=True)

    for t in range(T):
        # 1. Refiner
        ref_rgba = cmap(norm(refined[t, 0]))
        plt.imsave(os.path.join(frames_dir, f'refiner_t{t + 1}.png'), ref_rgba)

        # 2. GT
        gt_rgba = cmap(norm(gt[t, 0]))
        plt.imsave(os.path.join(frames_dir, f'gt_t{t + 1}.png'), gt_rgba)

        # 3. Mamba
        base_rgba = cmap(norm(base[t, 0]))
        plt.imsave(os.path.join(frames_dir, f'mamba_t{t + 1}.png'), base_rgba)

    print(f"📂 Saved clean frames to: {frames_dir}")


# ================= 5. 主测试逻辑 =================
def test():
    parser = argparse.ArgumentParser()
    # 路径配置
    parser.add_argument('--config_file', type=str, default='configs/sevir/Mamba.py')
    parser.add_argument('--mamba_ckpt', type=str, required=True)
    parser.add_argument('--refiner_ckpt', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='/home/ps/data2/zp/OpenSTL-OpenSTL-Lightning/data/sevir')
    parser.add_argument('--dataname', type=str, default='sevir')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='vis_results/paper_ready')

    # 可视化控制
    parser.add_argument('--vis_mode', type=str, default='manual', choices=['auto', 'manual'])
    parser.add_argument('--vis_target_batch', type=int, default=0)
    parser.add_argument('--vis_target_sample', type=int, default=8)

    args = parser.parse_args()

    config = load_config(args.config_file)
    if 'val_batch_size' not in config: config['val_batch_size'] = args.batch_size
    update_config(vars(args), config)

    if not hasattr(args, 'in_shape'):
        args.in_shape = (13, 1, 128, 128)
    elif args.in_shape[0] is None:
        t_s = list(args.in_shape); t_s[0] = 13; args.in_shape = tuple(t_s)

    print("🚀 Loading Models...")
    MambaCast = import_mamba()
    model_mamba = MambaCast(**vars(args)).cuda().eval()

    ckpt = torch.load(args.mamba_ckpt, map_location='cpu')
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model_mamba.load_state_dict({k.replace('model.', ''): v for k, v in state.items()}, strict=False)

    model_refiner = ResidualRefiner().cuda().eval()
    model_refiner.load_state_dict(torch.load(args.refiner_ckpt))

    print(f"📦 Loading Test Set...")
    _, _, test_loader = get_dataset(args.dataname, vars(args))

    all_metrics = []

    with torch.no_grad():
        for idx, (batch_x, batch_y) in enumerate(tqdm(test_loader)):
            batch_x = batch_x.cuda().float()
            batch_y = batch_y.cuda().float()

            # 1. Mamba 推理
            base_pred = model_mamba(batch_x)
            if base_pred.shape[1] != batch_y.shape[1]:
                min_t = min(base_pred.shape[1], batch_y.shape[1])
                base_pred = base_pred[:, :min_t].contiguous()
                batch_y = batch_y[:, :min_t].contiguous()

            # 2. Refiner 推理
            last_frame = batch_x[:, -1:, :, :, :]
            final_pred = model_refiner(base_pred, last_frame)

            # 3. 指标计算 (传入 0-1 的数据)
            batch_metric_res, _ = calc_all_metrics(
                pred=final_pred.cpu().numpy(),
                true=batch_y.cpu().numpy(),
                dataset_name='sevir',
                metrics=['mae', 'mse', 'rmse', 'ssim', 'csi', 'lpips'],
                return_log=False
            )
            all_metrics.append(batch_metric_res)

            # 4. 可视化判断
            need_vis = False
            sample_to_vis = -1

            if args.vis_mode == 'manual':
                if idx == args.vis_target_batch:
                    need_vis = True
                    sample_to_vis = min(args.vis_target_sample, batch_x.shape[0] - 1)

            elif args.vis_mode == 'auto':
                if idx % 10 == 0:
                    tmp_y = batch_y * 47.54 + 33.44
                    max_vals = tmp_y.cpu().numpy().max(axis=(1, 2, 3, 4))
                    best_idx = np.argmax(max_vals)
                    if max_vals[best_idx] > 74:
                        need_vis = True
                        sample_to_vis = best_idx

            if need_vis:
                # 反归一化并截断到 0-255
                mean, std = 33.44, 47.54

                vis_base = np.clip((base_pred[sample_to_vis] * std + mean).cpu().numpy(), 0, 255)
                vis_refined = np.clip((final_pred[sample_to_vis] * std + mean).cpu().numpy(), 0, 255)
                vis_gt = np.clip((batch_y[sample_to_vis] * std + mean).cpu().numpy(), 0, 255)

                # 保存总览图 (带标签和色条)
                save_vis_complete(vis_base, vis_refined, vis_gt, idx, sample_to_vis, args.save_dir)

                # 保存纯净单帧 (无标签，修复了报错)
                save_clean_frames(vis_base, vis_refined, vis_gt, idx, sample_to_vis, args.save_dir)

    # 5. 指标汇总
    print("\n" + "=" * 50)
    print(f"📊 Final Evaluation Report (Averaged over {len(all_metrics)} batches)")
    print("=" * 50)

    avg_results = {}
    if len(all_metrics) > 0:
        keys = all_metrics[0].keys()
        for k in keys:
            values = [m[k] for m in all_metrics if k in m]
            if len(values) > 0:
                avg_results[k] = np.mean(values)

    print(f"{'Metric':<15} | {'Value':<10}")
    print("-" * 30)

    # 核心指标 (加入 avg_csi, avg_hss)
    for k in ['mae', 'mse', 'rmse', 'ssim', 'lpips', 'pod','avg_csi', 'avg_hss']:
        if k in avg_results:
            print(f"{k.upper():<15} | {avg_results[k]:.4f}")

    print("-" * 30)
    # CSI 各个阈值
    for k in sorted(avg_results.keys()):
        if 'csi' in k and 'avg' not in k:
            print(f"{k.upper():<15} | {avg_results[k]:.4f}")

    print("-" * 30)
    # HSS 各个阈值 (新增)
    for k in sorted(avg_results.keys()):
        if 'hss' in k and 'avg' not in k:
            print(f"{k.upper():<15} | {avg_results[k]:.4f}")

    print("=" * 50)
    print(f"📝 Images saved to: {args.save_dir}")


if __name__ == '__main__':
    test()
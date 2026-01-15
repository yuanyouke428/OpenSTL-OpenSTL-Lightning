import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import sys

# 把当前目录加入路径
sys.path.append(os.getcwd())

from openstl.utils import create_parser, load_config, update_config, get_dataset


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
    raise ImportError("Cannot find MambaCast model definition")


def generate_data():
    # --- 1. 配置参数 ---
    parser = create_parser()
    args = parser.parse_args([])

    # 【请确认路径】
    config_file = 'configs/sevir/Mamba.py'
    args.batch_size = 32
    ckpt_path = '/home/ps/data2/zp/OpenSTL-OpenSTL-Lightning/work_dirs/mamba_Anisotropic_scan_FrequencyLoss_v5/checkpoints/best.ckpt'
    save_root = './data/refiner_data'

    args.config_file = config_file
    args.dataname = 'sevir'
    args.data_root = '/home/ps/data2/zp/OpenSTL-OpenSTL-Lightning/data/sevir'
    args.batch_size = 32

    config = load_config(config_file)
    update_config(vars(args), config)

    # ================= 🔧 强力修复: 确保 T 不是 None =================
    # 1. 尝试获取 in_shape
    if not hasattr(args, 'in_shape'):
        if hasattr(args, 'input_shape'):
            args.in_shape = args.input_shape
        else:
            args.in_shape = None  # 先置空，后面处理

    # 2. 如果获取到了，但第一维是 None (比如 (None, 1, 128, 128))，则强制修正
    if args.in_shape is not None and args.in_shape[0] is None:
        print(f"⚠️ Detect None in time dimension: {args.in_shape}")
        # 强制转换为 list 修改，再转回 tuple
        temp_shape = list(args.in_shape)
        temp_shape[0] = 13  # SEVIR 固定输入 13 帧
        args.in_shape = tuple(temp_shape)

    # 3. 如果完全没有 in_shape，手动构造标准形状
    if args.in_shape is None:
        # SEVIR 标准: (13帧输入, 1通道, 128高, 128宽)
        args.in_shape = (13, 1, 128, 128)

    # 4. 双重保险：再次检查 args.pre_seq_length
    if not hasattr(args, 'pre_seq_length') or args.pre_seq_length is None:
        args.pre_seq_length = 13

    print(f"✅ Final Model Input Shape: {args.in_shape}")
    # ================= 🔧 修复结束 =================

    os.makedirs(save_root, exist_ok=True)

    # --- 2. 加载模型 ---
    print("🚀 Loading Stage 1 Model (Mamba)...")
    MambaCast = import_model_class()

    # 现在 args.in_shape[0] 一定是 13 (int)，不会报错了
    model = MambaCast(**vars(args)).cuda()

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("✅ Model loaded from best.ckpt")

    # --- 3. 获取数据加载器 ---
    train_loader, val_loader, test_loader = get_dataset(args.dataname, vars(args))

    # 我们只处理 train 和 val
    modes = ['train', 'val']
    loaders = [train_loader, val_loader]

    # --- 4. 分块生成与保存 ---
    CHUNK_SIZE = 5000

    for mode, loader in zip(modes, loaders):
        print(f"\n📦 Processing {mode} set...")

        chunk_preds, chunk_gts, chunk_last = [], [], []
        chunk_idx = 0
        total_samples = 0

        mode_save_dir = os.path.join(save_root, mode)
        os.makedirs(mode_save_dir, exist_ok=True)

        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(tqdm(loader)):
                batch_x = batch_x.cuda().float()

                # Mamba 推理
                pred_y = model(batch_x)

                # 提取数据 (转 float16 节省一半硬盘空间，精度足够)
                last_frame = batch_x[:, -1:, :, :, :].cpu().numpy().astype(np.float16)
                pred_numpy = pred_y.cpu().numpy().astype(np.float16)
                gt_numpy = batch_y.cpu().numpy().astype(np.float16)

                chunk_preds.append(pred_numpy)
                chunk_gts.append(gt_numpy)
                chunk_last.append(last_frame)

                current_len = sum([x.shape[0] for x in chunk_preds])

                # 如果积攒够了 CHUNK_SIZE，就存一次
                if current_len >= CHUNK_SIZE:
                    save_chunk(mode_save_dir, mode, chunk_idx, chunk_preds, chunk_gts, chunk_last)
                    chunk_idx += 1
                    total_samples += current_len
                    # 清空缓存
                    chunk_preds, chunk_gts, chunk_last = [], [], []

        # 循环结束后，保存剩下的数据
        if chunk_preds:
            save_chunk(mode_save_dir, mode, chunk_idx, chunk_preds, chunk_gts, chunk_last)
            total_samples += sum([x.shape[0] for x in chunk_preds])

        print(f"✅ {mode} set finished! Total samples: {total_samples}")
        print(f"📂 Saved in: {mode_save_dir}")


def save_chunk(root, mode, idx, preds, gts, lasts):
    """辅助函数：保存分块数据"""
    p = np.concatenate(preds, axis=0)
    g = np.concatenate(gts, axis=0)
    l = np.concatenate(lasts, axis=0)

    np.save(os.path.join(root, f'{mode}_preds_{idx:03d}.npy'), p)
    np.save(os.path.join(root, f'{mode}_gts_{idx:03d}.npy'), g)
    np.save(os.path.join(root, f'{mode}_last_{idx:03d}.npy'), l)


if __name__ == '__main__':
    generate_data()
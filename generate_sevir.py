import h5py
import pandas as pd
import numpy as np
import cv2
import os
import os.path as osp
import argparse
from tqdm import tqdm
import datetime

# --- 参数配置 ---
PARSER = argparse.ArgumentParser(description="Convert SEVIR to SimVP H5 (Clone Logic)")
PARSER.add_argument('--raw_root', type=str, default='./data/sevir', help='CATALOG.csv 所在的根目录')
PARSER.add_argument('--output_path', type=str, default='./data/sevir/sevir.h5', help='输出文件路径')
PARSER.add_argument('--img_size', type=int, default=128, help='目标分辨率')
args = PARSER.parse_args()

# --- 核心配置 ---
SEQ_LEN = 25
STRIDE = 12


# 已移除 MIN_VIL_THRESH 和 MIN_COVERAGE

def find_file_path_clone(root_dir, filename):
    """
    完全复刻 generate_sevir.py 的 _open_files 逻辑
    """
    # 1. 确定 data_home
    # generate_sevir.py 默认 data_home 是 'data/sevir/data'
    # 如果 root_dir 下面有个 data 文件夹，我们优先用它
    potential_data_home = osp.join(root_dir, 'data')
    if osp.exists(potential_data_home):
        data_home = potential_data_home
    else:
        data_home = root_dir

    # 2. 复刻 if-elif 查找逻辑
    # 逻辑来源：generate_sevir.py -> SEVIRDataset._open_files

    # 尝试 A: 直接拼接
    possible_path = osp.join(data_home, filename)
    if osp.exists(possible_path):
        return possible_path

    # 尝试 B: 特殊规则匹配
    basename = osp.basename(filename)

    # 规则 1: 如果路径里没 vil 且 data_home 也没 vil，拼一个 vil 进去
    if 'vil' not in filename and 'vil' not in data_home:
        path = osp.join(data_home, 'vil', filename)
        if osp.exists(path): return path

    # 规则 2: 按年份暴力匹配 (2017, 2018, 2019)
    if '2017' in filename:
        path = osp.join(data_home, 'vil', '2017', basename)
        if osp.exists(path): return path
    elif '2018' in filename:
        path = osp.join(data_home, 'vil', '2018', basename)
        if osp.exists(path): return path
    elif '2019' in filename:
        path = osp.join(data_home, 'vil', '2019', basename)
        if osp.exists(path): return path

    return None


def get_sliding_windows(event_data):
    num_frames = event_data.shape[0]
    samples = []
    for start_idx in range(0, num_frames - SEQ_LEN + 1, STRIDE):
        end_idx = start_idx + SEQ_LEN
        sample = event_data[start_idx:end_idx]
        samples.append(sample)
    return samples


def resize_seq(data, target_size=128):
    resized = []
    for img in data:
        img_r = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        resized.append(img_r)
    return np.array(resized)


def process_and_write(split_name, events, raw_root, hf_out):
    print(f"\n🚀 Processing {split_name} set... Raw Events: {len(events)}")

    if split_name in hf_out: del hf_out[split_name]
    grp = hf_out.create_group(split_name)

    dset = grp.create_dataset('data',
                              shape=(0, SEQ_LEN, args.img_size, args.img_size),
                              maxshape=(None, SEQ_LEN, args.img_size, args.img_size),
                              dtype='uint8',
                              chunks=(1, SEQ_LEN, args.img_size, args.img_size),
                              compression="lzf")

    grouped = events.groupby('file_name')
    buffer = []
    buffer_limit = 500
    total_samples = 0
    missing_files = set()

    pbar = tqdm(total=len(events))

    for file_name, group in grouped:
        # --- 使用复刻的查找逻辑 ---
        file_path = find_file_path_clone(raw_root, file_name)

        if file_path is None:
            if file_name not in missing_files:
                missing_files.add(file_name)
            pbar.update(len(group))
            continue

        try:
            with h5py.File(file_path, 'r') as hf_in:
                # 兼容 key
                raw_dataset = hf_in['vil'] if 'vil' in hf_in else hf_in[list(hf_in.keys())[0]]

                for _, row in group.iterrows():
                    idx = int(row['file_index'])
                    raw_event = raw_dataset[idx]

                    if raw_event.ndim == 3 and raw_event.shape[-1] == 49:
                        raw_event = raw_event.transpose(2, 0, 1)

                    slices = get_sliding_windows(raw_event)

                    for s in slices:
                        # 【修改】无过滤，直接 resize 并添加
                        s_resized = resize_seq(s, args.img_size)
                        buffer.append(s_resized)

                    pbar.update(1)

                    if len(buffer) >= buffer_limit:
                        current_len = dset.shape[0]
                        add_len = len(buffer)
                        dset.resize(current_len + add_len, axis=0)
                        dset[current_len:] = np.array(buffer, dtype='uint8')
                        total_samples += add_len
                        buffer = []

        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            pbar.update(len(group))
            continue

    if len(buffer) > 0:
        current_len = dset.shape[0]
        dset.resize(current_len + len(buffer), axis=0)
        dset[current_len:] = np.array(buffer, dtype='uint8')
        total_samples += len(buffer)

    pbar.close()
    if len(missing_files) > 0:
        print(f"⚠️ Warning: {len(missing_files)} files missing.")
        # print(list(missing_files)[:3]) # 打印前3个看看
    print(f"✅ {split_name} Done. Valid Samples: {total_samples}")


def main():
    # 1. 读取 Catalog
    # 优先找 raw_root 下的 CATALOG，找不到就找 raw_root 上一层的（防止指向了 data/sevir/data）
    catalog_path = os.path.join(args.raw_root, 'CATALOG.csv')
    if not os.path.exists(catalog_path):
        catalog_path = os.path.join(os.path.dirname(args.raw_root), 'CATALOG.csv')

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"CATALOG.csv not found in or above {args.raw_root}")

    print(f"📖 Loading Catalog from {catalog_path}...")
    catalog = pd.read_csv(catalog_path, parse_dates=['time_utc'], low_memory=False)

    # 基础校验：只取 VIL 和 完整图
    catalog = catalog[catalog['img_type'] == 'vil']
    catalog = catalog[catalog['pct_missing'] == 0]
    print(f"Filtered Catalog: {len(catalog)} events.")

    # 2. 划分
    val_start = datetime.datetime(2019, 6, 1)
    test_start = datetime.datetime(2019, 10, 1)

    train_df = catalog[catalog['time_utc'] < val_start]
    val_df = catalog[(catalog['time_utc'] >= val_start) & (catalog['time_utc'] < test_start)]
    test_df = catalog[catalog['time_utc'] >= test_start]

    print(f"Split Sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 3. 执行
    if os.path.exists(args.output_path):
        os.remove(args.output_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with h5py.File(args.output_path, 'w') as hf_out:
        process_and_write('train', train_df, args.raw_root, hf_out)
        process_and_write('val', val_df, args.raw_root, hf_out)
        process_and_write('test', test_df, args.raw_root, hf_out)

    print(f"\n🎉 All Done! Saved to {args.output_path}")


if __name__ == '__main__':
    main()
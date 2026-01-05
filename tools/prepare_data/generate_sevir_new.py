"""
Input generator for sevir (Resize 128x128 + Sliding Window + Area Coverage Filtering)
"""

import argparse
import os
import os.path as osp
import logging
import numpy as np
import pandas as pd
import h5py
import torch
import datetime
import cv2

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

from torch.utils.data import Dataset

TYPES = ['vil']

import pathlib

_thisdir = str(pathlib.Path(__file__).parent.absolute())

DEFAULT_CATALOG = 'data/sevir/CATALOG.csv'
DEFAULT_DATA_HOME = 'data/sevir/data'


class SEVIRDataset(Dataset):
    def __init__(self, catalog, data_types=None, sevir_data_home=None,
                 batch_size=8, seq_len=25, raw_seq_len=49,
                 start_date=None, end_date=None,
                 datetime_filter=None, catalog_filter=None,
                 unwrap_time=False, shuffle=False, shuffle_seed=1,
                 verbose=True):
        super(SEVIRDataset, self).__init__()

        if data_types is None or (isinstance(data_types, list) and None in data_types):
            self.data_types = TYPES
        else:
            self.data_types = data_types

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.raw_seq_len = raw_seq_len
        self.start_date = start_date
        self.end_date = end_date
        self.unwrap_time = unwrap_time
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.verbose = verbose

        self.sevir_data_home = sevir_data_home if sevir_data_home else DEFAULT_DATA_HOME

        if isinstance(catalog, str):
            if verbose: print(f'Reading catalog from {catalog}')
            self.catalog = pd.read_csv(catalog, parse_dates=['time_utc'], low_memory=False)
        else:
            self.catalog = catalog

        if start_date:
            self.catalog = self.catalog[self.catalog.time_utc > start_date]
        if end_date:
            self.catalog = self.catalog[self.catalog.time_utc <= end_date]

        if verbose:
            print(f'Catalog size after filtering: {len(self.catalog)}')

        def process_group(df):
            d = {}
            d['id'] = df['id'].iloc[0]
            d['time_utc'] = df['time_utc'].iloc[0]
            for t in self.data_types:
                row = df[df['img_type'] == t]
                if not row.empty:
                    d[f'{t}_filename'] = row['file_name'].iloc[0]
                    d[f'{t}_index'] = row['file_index'].iloc[0]
                else:
                    d[f'{t}_filename'] = None
                    d[f'{t}_index'] = None
            return pd.Series(d)

        if verbose: print("Grouping and processing catalog...")
        try:
            self._samples = self.catalog.groupby('id').apply(process_group)
        except TypeError:
            self._samples = self.catalog.groupby('id').apply(process_group)

        if isinstance(self._samples, pd.DataFrame):
            self._samples = self._samples.reset_index(drop=True)
        else:
            self._samples = pd.DataFrame(self._samples.tolist())

        col_name = f'{self.data_types[0]}_filename'
        if col_name in self._samples.columns:
            self._samples = self._samples.dropna(subset=[col_name])

        if verbose:
            print(f"DEBUG: Valid samples found: {len(self._samples)}")

        if shuffle:
            self._samples = self._samples.sample(frac=1, random_state=shuffle_seed)

        self._open_files(verbose=self.verbose)

    def _open_files(self, verbose=True):
        if verbose: print("Opening HDF5 files...")
        self.hdf_files = {}
        for t in self.data_types:
            col_name = f'{t}_filename'
            if col_name not in self._samples.columns: continue

            hdf_filenames = list(np.unique(self._samples[col_name].values))
            for f in hdf_filenames:
                possible_path = osp.join(self.sevir_data_home, f)
                if not osp.exists(possible_path):
                    if 'vil' not in f and 'vil' not in self.sevir_data_home:
                        possible_path = osp.join(self.sevir_data_home, 'vil', f)
                    elif '2017' in f:
                        possible_path = osp.join(self.sevir_data_home, 'vil', '2017', osp.basename(f))
                    elif '2018' in f:
                        possible_path = osp.join(self.sevir_data_home, 'vil', '2018', osp.basename(f))
                    elif '2019' in f:
                        possible_path = osp.join(self.sevir_data_home, 'vil', '2019', osp.basename(f))

                if osp.exists(possible_path):
                    self.hdf_files[f] = h5py.File(possible_path, 'r')

    def load_batches(self, n_batches=10, offset=0, progress_bar=False,
                     target_size=128, stride=12, min_vil_thresh=16, min_coverage=0.015):
        """
        min_vil_thresh: 像素值的门槛 (16 代表轻微降雨)
        min_coverage:  面积占比门槛 (0.015 代表 1.5% 的区域必须有雨)
        """
        batch_data = []
        end_idx = min(offset + n_batches, len(self._samples))

        for idx in range(offset, end_idx):
            s = self._samples.iloc[idx]
            fname = s.get('vil_filename')
            fidx = s.get('vil_index')

            if fname is None or fname not in self.hdf_files: continue

            try:
                with h5py.File(self.hdf_files[fname].filename, 'r') as hf:
                    data = hf['vil'][int(fidx)]

                    # 维度修正 (384,384,49) -> (49,384,384)
                if data.shape[-1] == 49 and data.ndim == 3:
                    data = data.transpose(2, 0, 1)

                    # 滑动窗口
                num_frames = data.shape[0]
                for start_t in range(0, num_frames - self.seq_len + 1, stride):
                    seq = data[start_t: start_t + self.seq_len]  # [25, 384, 384]

                    # --- 核心修改：双重过滤 ---

                    # 1. 快速检查：最大值是否达标
                    if seq.max() < min_vil_thresh:
                        continue

                    # 2. 面积检查：计算大于阈值的像素比例
                    # seq > min_vil_thresh 生成一个 Boolean 矩阵
                    # np.mean 计算 True 的比例
                    coverage = np.mean(seq > min_vil_thresh)

                    # 如果降雨区域小于总面积的 1.5%，则丢弃
                    if coverage < min_coverage:
                        continue

                    # --- 通过过滤，开始处理 ---

                    # Resize to 128x128
                    resized_seq = []
                    for t in range(seq.shape[0]):
                        img = cv2.resize(seq[t], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                        resized_seq.append(img)

                    seq = np.array(resized_seq)  # [25, 128, 128]
                    batch_data.append(seq)

            except Exception as e:
                print(f"Error reading {fname}: {e}")
                continue

        return np.array(batch_data) if len(batch_data) > 0 else np.array([])


class NowcastGenerator(SEVIRDataset):
    def __init__(self, catalog=None, sevir_data_home=None, **kwargs):
        if catalog is None: catalog = DEFAULT_CATALOG
        super(NowcastGenerator, self).__init__(catalog=catalog, sevir_data_home=sevir_data_home, **kwargs)


def get_nowcast_generator(sevir_data, batch_size=8, data_name='sevir', start_date=None, end_date=None):
    return NowcastGenerator(
        catalog=DEFAULT_CATALOG,
        sevir_data_home=sevir_data,
        batch_size=batch_size,
        start_date=start_date,
        end_date=end_date,
        data_types=['vil']
    )


def read_write_chunks(filename, generator, split, chunks):
    print(f"Writing {split} data to {filename}...")
    chunk_size = 50

    # 预读取确定 Shape (循环直到找到有效数据)
    print(f"[{split}] Checking data shape...")
    sample_batch = []
    offset = 0
    # 尝试查找有效数据，最多查 500 个样本，防止无限循环
    while len(sample_batch) == 0 and offset < min(500, len(generator._samples)):
        sample_batch = generator.load_batches(n_batches=10, offset=offset)
        offset += 10

    if len(sample_batch) == 0:
        print(f"Error: No valid data found for {split} (Filter might be too strict?).")
        return

    _, t_total, h, w = sample_batch.shape
    print(f"[{split}] Detected Shape: {t_total} frames x {h}x{w}")

    with h5py.File(filename, 'a') as hf:
        if split in hf: del hf[split]
        grp = hf.create_group(split)
        grp.create_dataset('data', (0, t_total, h, w), maxshape=(None, t_total, h, w), dtype='uint8', chunks=True)

    total_events = len(generator._samples)
    print(f"Scanning {total_events} raw events...")

    samples_written = 0
    for i in range(0, total_events, chunk_size):
        if chunks is not None and i >= chunks * chunk_size: break

        if i % 500 == 0:
            print(f"Scanning {i}/{total_events} | Written: {samples_written}")

        batch_data = generator.load_batches(n_batches=chunk_size, offset=i)

        if len(batch_data) == 0: continue

        with h5py.File(filename, 'a') as hf:
            ds = hf[split]['data']
            curr_len = ds.shape[0]
            new_len = curr_len + batch_data.shape[0]
            ds.resize((new_len, t_total, h, w))
            ds[curr_len:] = batch_data

        samples_written += len(batch_data)

    print(f"Finished {split}. Total samples written: {samples_written}")


def main(args):
    logging.basicConfig(level=logging.INFO)

    split_date_val_start = datetime.datetime(2019, 1, 1)
    split_date_test_start = datetime.datetime(2019, 6, 1)

    final_file = osp.join(args.output_dir, f'{args.data_name}.h5')

    if osp.exists(final_file):
        print(f"Removing old file: {final_file}")
        os.remove(final_file)

    print("\n--- Processing Training Set ---")
    trn_gen = get_nowcast_generator(args.sevir_data, end_date=split_date_val_start)
    read_write_chunks(final_file, trn_gen, split='train', chunks=args.chunks)

    print("\n--- Processing Validation Set ---")
    val_gen = get_nowcast_generator(args.sevir_data, start_date=split_date_val_start, end_date=split_date_test_start)
    read_write_chunks(final_file, val_gen, split='val', chunks=args.chunks)

    print("\n--- Processing Testing Set ---")
    test_gen = get_nowcast_generator(args.sevir_data, start_date=split_date_test_start)
    read_write_chunks(final_file, test_gen, split='test', chunks=args.chunks)

    print(f"\nAll Done! Data saved to {final_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sevir_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_name', type=str, default='sevir')
    parser.add_argument('--chunks', type=int, default=None)
    args = parser.parse_args()

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from openstl.datasets.utils import create_loader


class CIKMDataset(Dataset):
    def __init__(self, data_path, mode='train', input_len=10, predict_len=5, target_size=64):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.input_len = input_len
        self.predict_len = predict_len
        self.target_size = target_size

        self.mean = 0
        self.std = 1
        self.data_name = 'cikm'

        # 打开 H5 文件读取 keys
        with h5py.File(data_path, 'r') as f:
            if mode not in f.keys():
                raise ValueError(f"H5文件中找不到 {mode} 分组，请检查文件结构")
            self.keys = list(f[mode].keys())

        # 排序确保顺序一致
        # 假设 key 格式为 sample_1, sample_10...
        try:
            self.keys.sort(key=lambda x: int(x.split('_')[-1]))
        except:
            self.keys.sort()  # 如果 key 不是标准格式，就用默认字符排序

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with h5py.File(self.data_path, 'r') as f:
            key = self.keys[index]
            data = f[self.mode][key][()]

        # 归一化
        data = data.astype(np.float32) / 255.0

        # Resize
        frames = []
        for i in range(data.shape[0]):
            img = data[i]
            if self.target_size != 101:
                img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            frames.append(img)
        data = np.stack(frames, axis=0)

        # 增加通道维度 (T, H, W) -> (T, 1, H, W)
        data = data[:, np.newaxis, :, :]
        data_tensor = torch.from_numpy(data)

        # 切分输入输出
        input_seq = data_tensor[:self.input_len]
        target_seq = data_tensor[self.input_len: self.input_len + self.predict_len]

        return input_seq, target_seq


# --- 这是刚才缺失的关键函数 ---
def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=10, aft_seq_length=10, in_shape=[10, 1, 64, 64],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):
    target_size = in_shape[-1] if in_shape is not None else 64

    # 实例化 Dataset
    train_set = CIKMDataset(data_path=data_root, mode='train',
                            input_len=pre_seq_length, predict_len=aft_seq_length,
                            target_size=target_size)
    val_set = CIKMDataset(data_path=data_root, mode='valid',
                          input_len=pre_seq_length, predict_len=aft_seq_length,
                          target_size=target_size)
    test_set = CIKMDataset(data_path=data_root, mode='test',
                           input_len=pre_seq_length, predict_len=aft_seq_length,
                           target_size=target_size)

    # 封装成 DataLoader
    dataloader_train = create_loader(train_set, batch_size=batch_size, shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True, num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_val = create_loader(val_set, batch_size=val_batch_size, shuffle=False, is_training=False,
                                   pin_memory=True, drop_last=drop_last, num_workers=num_workers,
                                   distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set, batch_size=val_batch_size, shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last, num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_val, dataloader_test
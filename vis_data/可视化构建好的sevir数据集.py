import os
import os.path as osp
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from openstl.datasets.utils import create_loader

# 忽略 HDF5 文件锁警告
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'


class SEVIRDataset(Dataset):
    """
    适配新生成的 sevir.h5 (Group模式: train/val/test) 的数据加载器
    保留了官方的 Z-Score 归一化逻辑 (mean=33.44, std=47.54)
    """

    def __init__(self,
                 data_root,
                 data_name='sevir',
                 mode='train',
                 seq_len=25,
                 pre_seq_length=13,
                 aft_seq_length=12,
                 use_augment=False,
                 use_sevir_stats=True):

        super(SEVIRDataset, self).__init__()

        self.data_root = data_root
        self.data_name = data_name
        self.mode = mode
        self.seq_len = seq_len
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.use_augment = use_augment
        self.use_sevir_stats = use_sevir_stats

        # --- 官方统计参数 (SEVIR VIL) ---
        # 为什么重要？因为 SEVIR 论文中使用的是 Z-Score 标准化，
        # 这能让数据分布符合 N(0,1)，有利于 MSE Loss 的收敛。
        self.mean = 33.44
        self.std = 47.54

        # 1. 锁定文件名
        self.file_name = f'{data_name}.h5'
        self.file_path = osp.join(data_root, self.file_name)

        if not osp.exists(self.file_path):
            raise IOError(f"Dataset file not found: {self.file_path}")

        # 2. 初始化时读取元数据 (长度/形状)
        # 注意：不要在 __init__ 里持久化 h5py.File 对象，防止多线程死锁
        with h5py.File(self.file_path, 'r') as hf:
            if self.mode not in hf:
                raise KeyError(f"Group '{self.mode}' not found in {self.file_name}. Available: {list(hf.keys())}")

            # 获取数据引用
            ds = hf[self.mode]['data']
            self.length = ds.shape[0]
            self.shape = ds.shape  # (N, 25, 128, 128)

            print(f"[{self.mode.upper()}] Dataset loaded. Length: {self.length}, Shape: {self.shape}")

    def __len__(self):
        return self.length

    def _augment_seq(self, x, y):
        # 简单的随机翻转增强
        if np.random.rand() > 0.5:
            x = torch.flip(x, dims=[-1])  # 水平翻转
            y = torch.flip(y, dims=[-1])
        return x, y

    def __getitem__(self, index):
        # 3. 每次读取时单独打开文件 (多进程安全)
        with h5py.File(self.file_path, 'r') as hf:
            # 读取数据: [25, 128, 128] (uint8)
            # 这是一个包含整个序列的张量
            data = hf[self.mode]['data'][index]

            # 4. 维度调整: (T, H, W) -> (T, C, H, W)
        if data.ndim == 3:
            data = data[..., np.newaxis]  # (25, 128, 128, 1)

        # 5. 转 Tensor 并转 float
        # permute: (T, H, W, C) -> (T, C, H, W)
        data = torch.tensor(data).permute(0, 3, 1, 2).float()

        # 6. 切分输入和预测目标
        # inputs: 前13帧, targets: 后12帧
        X = data[:self.pre_seq_length]
        Y = data[self.pre_seq_length: self.pre_seq_length + self.aft_seq_length]

        # 7. 数据增强
        if self.use_augment:
            X, Y = self._augment_seq(X, Y)

        # 8. 归一化 (核心部分)
        if self.use_sevir_stats:
            # 使用官方均值方差归一化: (x - 33.44) / 47.54
            X = (X - self.mean) / self.std
            Y = (Y - self.mean) / self.std
        else:
            # 普通归一化: x / 255.0
            X = X / 255.0
            Y = Y / 255.0

        return X, Y

    @staticmethod
    def denormalize(x):
        """
        反归一化工具：用于可视化，将数据还原回原始值
        """
        mean = 33.44
        std = 47.54
        return x * std + mean


def load_data(batch_size,
              val_batch_size,
              data_root='./data/sevir/',
              num_workers=4,
              data_name='sevir',
              pre_seq_length=13,
              aft_seq_length=12,
              distributed=False,
              use_augment=False,
              use_prefetcher=False,
              drop_last=False,
              **kwargs):
    # 实例化三个 Dataset，分别对应 h5 文件中的 train/val/test 组
    train_set = SEVIRDataset(data_root=data_root,
                             data_name=data_name,
                             mode='train',
                             pre_seq_length=pre_seq_length,
                             aft_seq_length=aft_seq_length,
                             use_augment=use_augment,
                             use_sevir_stats=True)  # 默认开启官方统计归一化

    vali_set = SEVIRDataset(data_root=data_root,
                            data_name=data_name,
                            mode='val',
                            pre_seq_length=pre_seq_length,
                            aft_seq_length=aft_seq_length,
                            use_augment=False,
                            use_sevir_stats=True)

    test_set = SEVIRDataset(data_root=data_root,
                            data_name=data_name,
                            mode='test',
                            pre_seq_length=pre_seq_length,
                            aft_seq_length=aft_seq_length,
                            use_augment=False,
                            use_sevir_stats=True)

    # 创建 DataLoader
    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     is_training=True,
                                     pin_memory=True,
                                     drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed,
                                     use_prefetcher=use_prefetcher)

    dataloader_vali = create_loader(vali_set,
                                    batch_size=val_batch_size,
                                    shuffle=False,
                                    is_training=False,
                                    pin_memory=True,
                                    drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed,
                                    use_prefetcher=use_prefetcher)

    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False,
                                    is_training=False,
                                    pin_memory=True,
                                    drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed,
                                    use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    # 测试代码
    #path = 'data/sevir'  # 确保这里指向包含 sevir.h5 的文件夹
    path = '/home/ps/data2/zp/data'
    print(f"Testing DataLoader from: {path}")

    try:
        train_loader, val_loader, test_loader = load_data(
            batch_size=4,
            val_batch_size=4,
            data_root=path,
            data_name='sevir'
        )

        print("DataLoader created successfully.")

        # 测试读取一个 batch
        for i, (x, y) in enumerate(train_loader):
            print(f"Batch {i}:")
            print(f"  Input Shape: {x.shape} (Expect: [4, 13, 1, 128, 128])")
            print(f"  Target Shape: {y.shape} (Expect: [4, 12, 1, 128, 128])")
            print(f"  Mean value: {x.mean().item():.2f}")
            print(f"  Max value: {x.max().item():.2f}")
            break

    except Exception as e:
        print(f"Error: {e}")
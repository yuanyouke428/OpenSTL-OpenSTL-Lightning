import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from openstl.datasets.utils import create_loader


class MeteoNetDataset(Dataset):
    """MeteoNet Dataset (Adapted from DiffCast logic)
    The h5 file is expected to have groups 'train' and 'test',
    where samples are stored as keys '0', '1', ...
    """

    def __init__(self, data_root, mode='train', pre_seq_length=5, aft_seq_length=20, use_augment=False):
        super().__init__()
        self.data_root = data_root
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.use_augment = use_augment  # OpenSTL 标准接口保留

        # DiffCast 参数
        self.pixel_scale = 70.0  # 原始数据归一化因子
        self.target_size = (128, 128)

        # !!! 新增部分开始 !!!
        # OpenSTL BaseDataModule 必须要求的属性
        # 因为我们只是做了简单的除法归一化 (x / 70)，没有做减均值除方差
        # 所以这里 mean 设为 0，std 设为 1，表示不需要额外反归一化
        self.mean = 0.0
        self.std = 1.0
        self.data_name = 'meteonet'
        # !!! 新增部分结束 !!!

        # 确定读取模式
        if mode == 'train':
            self.type = 'train'
        else:
            self.type = 'test'  # Val 和 Test 都用 DiffCast 的 test 集

        # 获取数据集长度
        # 注意：为了避免多进程读取冲突，这里只读取长度，不保持文件句柄
        with h5py.File(self.data_root, 'r') as f:
            # DiffCast 的 h5 文件里存储了长度信息，key 为 'train_len' 或 'test_len'
            len_key = f'{self.type}_len'
            if len_key in f.keys():
                self.all_len = int(f[len_key][()])
            else:
                # 如果没有存长度，则回退到计算 keys 数量 (排除 _len 后缀)
                keys = [k for k in f[self.type].keys()]
                self.all_len = len(keys)

    def __len__(self):
        return self.all_len

    def __getitem__(self, index):
        # 在 getitem 中打开文件以支持多线程 (num_workers > 0)
        with h5py.File(self.data_root, 'r') as f:
            # 读取原始数据 (25, 565, 784)
            # DiffCast 存储逻辑: f['train']['123']
            imgs = f[self.type][str(index)][()]

            # 1. 转为 Tensor 并归一化
        # 原始数据通常是 uint8 或 float，DiffCast 除以 90.0
        frames = torch.from_numpy(imgs).float()
        frames = frames / self.pixel_scale

        # 2. 调整维度 (T, H, W) -> (T, C, H, W) 或 (T, 1, H, W)
        if frames.ndim == 3:
            frames = frames.unsqueeze(1)

        # 3. 缩放 (Resize) 到 128x128
        # interpolate 需要输入 (N, C, H, W)，这里 T 维度充当 N
        frames = F.interpolate(frames, size=self.target_size, mode='bilinear', align_corners=False)

        # 4. 切分输入和输出
        data = frames[:self.pre_seq_length]
        labels = frames[self.pre_seq_length: self.pre_seq_length + self.aft_seq_length]

        return data, labels


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=5, aft_seq_length=20, distributed=False, **kwargs):
    # 训练集
    train_set = MeteoNetDataset(data_root, mode='train',
                                pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
    # 验证集 (复用测试集)
    val_set = MeteoNetDataset(data_root, mode='val',
                              pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
    # 测试集
    test_set = MeteoNetDataset(data_root, mode='test',
                               pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)

    dataloader_train = create_loader(train_set, batch_size=batch_size, shuffle=True,
                                     is_training=True, pin_memory=True, num_workers=num_workers,
                                     distributed=distributed)

    dataloader_val = create_loader(val_set, batch_size=val_batch_size, shuffle=False,
                                   is_training=False, pin_memory=True, num_workers=num_workers,
                                   distributed=distributed)

    dataloader_test = create_loader(test_set, batch_size=val_batch_size, shuffle=False,
                                    is_training=False, pin_memory=True, num_workers=num_workers,
                                    distributed=distributed)

    return dataloader_train, dataloader_val, dataloader_test
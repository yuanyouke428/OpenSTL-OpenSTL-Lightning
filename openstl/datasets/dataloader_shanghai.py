# import h5py
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# from openstl.datasets.utils import create_loader
#
#
# class ShanghaiDataset(Dataset):
#     """MeteoNet Dataset (Adapted from DiffCast logic)
#     The h5 file is expected to have groups 'train' and 'test',
#     where samples are stored as keys '0', '1', ...
#     """
#
#     def __init__(self, data_root, mode='train', pre_seq_length=5, aft_seq_length=20, use_augment=False):
#         super().__init__()
#         self.data_root = data_root
#         self.pre_seq_length = pre_seq_length
#         self.aft_seq_length = aft_seq_length
#         self.use_augment = use_augment  # OpenSTL 标准接口保留
#
#         # DiffCast 参数
#         self.pixel_scale = 255.0  # 原始数据归一化因子
#         self.target_size = (128, 128)
#
#         # !!! 新增部分开始 !!!
#         # OpenSTL BaseDataModule 必须要求的属性
#         # 因为我们只是做了简单的除法归一化 (x / 255)，没有做减均值除方差
#         # 所以这里 mean 设为 0，std 设为 1，表示不需要额外反归一化
#         self.mean = 0.0
#         self.std = 1.0
#         self.data_name = 'shanghai'
#         # !!! 新增部分结束 !!!
#
#         # 确定读取模式
#         if mode == 'train':
#             self.type = 'train'
#         else:
#             self.type = 'test'  # Val 和 Test 都用  的 test 集
#
#         # 获取数据集长度
#         # 注意：为了避免多进程读取冲突，这里只读取长度，不保持文件句柄
#         with h5py.File(self.data_root, 'r') as f:
#             # DiffCast 的 h5 文件里存储了长度信息，key 为 'train_len' 或 'test_len'
#             len_key = f'{self.type}_len'
#             if len_key in f.keys():
#                 self.all_len = int(f[len_key][()])
#             else:
#                 # 如果没有存长度，则回退到计算 keys 数量 (排除 _len 后缀)
#                 keys = [k for k in f[self.type].keys()]
#                 self.all_len = len(keys)
#
#     def __len__(self):
#         return self.all_len
#
#     def __getitem__(self, index):
#         # 在 getitem 中打开文件以支持多线程 (num_workers > 0)
#         with h5py.File(self.data_root, 'r') as f:
#             # 读取原始数据 (25, 565, 784)
#             # DiffCast 存储逻辑: f['train']['123']
#             imgs = f[self.type][str(index)][()]
#
#             # 1. 转为 Tensor 并归一化
#         # 原始数据通常是 uint8 或 float，DiffCast 除以 90.0
#         frames = torch.from_numpy(imgs).float()
#         frames = frames / self.pixel_scale
#
#         # 2. 调整维度 (T, H, W) -> (T, C, H, W) 或 (T, 1, H, W)
#         if frames.ndim == 3:
#             frames = frames.unsqueeze(1)
#
#         # 3. 缩放 (Resize) 到 128x128
#         # interpolate 需要输入 (N, C, H, W)，这里 T 维度充当 N
#         frames = F.interpolate(frames, size=self.target_size, mode='bilinear', align_corners=False)
#
#         # 4. 切分输入和输出
#         data = frames[:self.pre_seq_length]
#         labels = frames[self.pre_seq_length: self.pre_seq_length + self.aft_seq_length]
#
#         return data, labels
#
#
# def load_data(batch_size, val_batch_size, data_root, num_workers=4,
#               pre_seq_length=5, aft_seq_length=20, distributed=False, **kwargs):
#     # 训练集
#     train_set = ShanghaiDataset(data_root, mode='train',
#                                 pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
#     # 验证集 (复用测试集)
#     val_set = ShanghaiDataset(data_root, mode='val',
#                               pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
#     # 测试集
#     test_set = ShanghaiDataset(data_root, mode='test',
#                                pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
#
#     dataloader_train = create_loader(train_set, batch_size=batch_size, shuffle=True,
#                                      is_training=True, pin_memory=True, num_workers=num_workers,
#                                      distributed=distributed)
#
#     dataloader_val = create_loader(val_set, batch_size=val_batch_size, shuffle=False,
#                                    is_training=False, pin_memory=True, num_workers=num_workers,
#                                    distributed=distributed)
#
#     dataloader_test = create_loader(test_set, batch_size=val_batch_size, shuffle=False,
#                                     is_training=False, pin_memory=True, num_workers=num_workers,
#                                     distributed=distributed)
#
#     return dataloader_train, dataloader_val, dataloader_test


import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from openstl.datasets.utils import create_loader


class ShanghaiDataset(Dataset):
    """
    Shanghai Radar Dataset
    解决 Key 不连续和非 "0","1"..."N" 命名的问题
    """

    def __init__(self, data_root, mode='train', pre_seq_length=5, aft_seq_length=20,
                 use_augment=False, target_size=128):
        super().__init__()
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.use_augment = use_augment
        self.target_size = (target_size, target_size) if isinstance(target_size, int) else target_size

        # OpenSTL 要求的基础属性
        self.mean = 0.0
        self.std = 1.0
        self.data_name = 'shanghai'

        # 1. 自动判断路径 (文件还是文件夹)
        if os.path.isdir(data_root):
            self.file_path = os.path.join(data_root, 'shanghai.h5')
        else:
            self.file_path = data_root

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Shanghai dataset file not found at: {self.file_path}")

        # 2. 确定 Group 名称
        self.group_name = 'train' if mode == 'train' else 'test'

        # 3. 预读取所有合法的 Keys (关键修复步骤)
        with h5py.File(self.file_path, 'r') as f:
            if self.group_name not in f.keys():
                raise ValueError(f"Group '{self.group_name}' not found in {self.file_path}")

            # 获取该组下所有的 Key
            all_keys = list(f[self.group_name].keys())

            # 过滤掉非样本 Key (比如有些 h5 会存 'length', 'config' 等)
            # 假设样本 Key 都是数字组成的字符串
            self.keys = [k for k in all_keys if k.isdigit()]

            # 排序，确保训练顺序一致
            self.keys.sort(key=lambda x: int(x))

            if len(self.keys) == 0:
                raise ValueError(f"No valid sample keys found in group {self.group_name}")

            print(f"[{mode}] Loaded {len(self.keys)} samples. First key: {self.keys[0]}, Last key: {self.keys[-1]}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # !!! 关键修复: 使用列表映射，而不是直接 str(index) !!!
        # 这样即使 index=0, 对应的 key 是 "533" 也没问题
        key = self.keys[index]

        with h5py.File(self.file_path, 'r') as f:
            # 读取数据
            data = f[self.group_name][key][()]

        # 增加通道维度 (T, H, W) -> (T, C, H, W)
        data = data[:, np.newaxis, :, :]

        # 归一化 (0-169 -> 0-1)
        # 建议用 255.0 以防未来有更大值，或者根据统计数据调整
        data = torch.from_numpy(data).float() / 255.0

        # Resize (501 -> 128)
        # F.interpolate 需要 (N, C, H, W)，把 T 当作 N
        if self.target_size is not None:
            H, W = data.shape[-2], data.shape[-1]
            if (H, W) != self.target_size:
                data = F.interpolate(data, size=self.target_size, mode='bilinear', align_corners=False)

        # 检查长度
        total_req = self.pre_seq_length + self.aft_seq_length
        if data.shape[0] < total_req:
            # 如果样本长度不够，这通常是个数据问题，这里做个简单截断或报错
            # 为了防止报错，可以做 padding，但最好是报错
            raise ValueError(f"Sample {key} has {data.shape[0]} frames, need {total_req}")

        input_data = data[:self.pre_seq_length]
        output_data = data[self.pre_seq_length: total_req]

        return input_data, output_data


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=5, aft_seq_length=20, in_shape=None,
              distributed=False, **kwargs):
    target_size = in_shape[-1] if (in_shape and len(in_shape) == 4) else 128

    train_set = ShanghaiDataset(data_root, mode='train',
                                pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
                                target_size=target_size)
    val_set = ShanghaiDataset(data_root, mode='val',
                              pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
                              target_size=target_size)
    test_set = ShanghaiDataset(data_root, mode='test',
                               pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
                               target_size=target_size)

    dataloader_train = create_loader(train_set, batch_size=batch_size, shuffle=True,
                                     is_training=True, pin_memory=True, num_workers=num_workers,
                                     distributed=distributed)
    dataloader_val = create_loader(val_set, batch_size=val_batch_size, shuffle=False,
                                   is_training=False, pin_memory=True, num_workers=num_workers, distributed=distributed)
    dataloader_test = create_loader(test_set, batch_size=val_batch_size, shuffle=False,
                                    is_training=False, pin_memory=True, num_workers=num_workers,
                                    distributed=distributed)

    return dataloader_train, dataloader_val, dataloader_test
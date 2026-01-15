import numpy as np
import os

# 这里换成你实际生成的任意一个 preds 文件路径
file_path = '/home/ps/data2/zp/OpenSTL-OpenSTL-Lightning/tools/data/refiner_data/train/train_preds_000.npy'

if os.path.exists(file_path):
    data = np.load(file_path, mmap_mode='r')
    print(f"📂 文件路径: {file_path}")
    print(f"📏 数据形状 (Shape): {data.shape}")
    print(f"⏱️ 时间维度 (Time Dim): {data.shape[1]}")

    gt_path = file_path.replace('preds', 'gts')
    if os.path.exists(gt_path):
        gt_data = np.load(gt_path, mmap_mode='r')
        print(f"📏 真值形状 (GT Shape): {gt_data.shape}")
else:
    print("❌ 找不到文件，请检查路径")
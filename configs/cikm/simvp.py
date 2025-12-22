method = 'SimVP'
dataname = 'cikm'

# 数据集参数
dataset_params = dict(
    name='cikm',
    data_root='/home/ps/data2/zp/data/cikm.h5',
    input_len=5,
    predict_len=10,
    # 【修改 1】将尺寸改为 128 (原 64)，配合 DataLoader 的 Padding
    in_shape=[5, 1, 128, 128],
)

# 模型参数
model_params = dict(
    # 【修改 2】模型输入尺寸同步改为 128
    in_shape=(5, 1, 128, 128),
    hid_S=64,
    hid_T=256,
    N_S=4,
    N_T=8,
)

# 训练参数
training_params = dict(
    batch_size=32, # 如果显存允许，建议调大一点，例如 16
    val_batch_size=16,
    epochs=100,     # 建议跑多一点 epoch
    lr=1e-3,
)

# 【修改 3】新增测试参数，启用 cikm 评估指标
test_params = dict(
    # 'cikm' 会触发我们刚才在 metrics.py 里写的逻辑
    metrics=['mse', 'mae', 'ssim', 'radar_metrics'],
)


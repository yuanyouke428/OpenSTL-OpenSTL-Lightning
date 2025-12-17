method = 'SimVP'
dataname = 'cikm'

# 数据集参数
dataset_params = dict(
    name='cikm',
    # 关键修改：把 'root' 改为 'data_root'
    data_root='/home/ps/data2/zp/data/cikm.h5',
    input_len=5,
    predict_len=10,
    in_shape=[5, 1, 64, 64],
)

# 模型参数
model_params = dict(
    in_shape=(5, 1, 64, 64),
    hid_S=64,
    hid_T=256,
    N_S=4,
    N_T=8,
)

# 训练参数
training_params = dict(
    batch_size=2,
    val_batch_size=2,
    epochs=1,
    lr=1e-3,
)


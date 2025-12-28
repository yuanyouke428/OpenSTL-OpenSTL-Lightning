method = 'ConvLSTM'
dataname = 'cikm'

# ==========================================================
# 1. 数据集参数 (对应 parser.py 中的标准参数名)
# ==========================================================
data_root = '/home/ps/data2/zp/data/cikm.h5'
# 注意：OpenSTL 标准参数名是 pre_seq_length 和 aft_seq_length
# 原 input_len -> pre_seq_length
pre_seq_length = 5
# 原 predict_len -> aft_seq_length
aft_seq_length = 10
# CIKM 的总长度通常是 输入+预测
total_length = 15

# 数据维度 [T, C, H, W]
# 注意：ConvLSTM 并不直接使用 in_shape 参数来构建网络结构，
# 它主要依赖 patch_size 和 num_hidden 来决定中间维度。
# 但为了确保 dataloader 正确裁剪，保留它有助于记录
in_shape = [5, 1, 128, 128]

# ==========================================================
# 2. 模型参数 (必须平铺在最外层，不能放在 model_params 里)
# ==========================================================
# ConvLSTM 特有参数
num_hidden = '64,64,64,64'  # 4层，每层64通道。如果爆显存，尝试 '32,32,32,32'
filter_size = 5
stride = 1
patch_size = 4             # 重要：将 128x128 下采样为 32x32 处理，否则显存必爆
layer_norm = 1             # 推荐开启

# Scheduled Sampling 参数 (ConvLSTM 训练关键)
reverse_scheduled_sampling = 0
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002

# ==========================================================
# 3. 训练参数 (平铺)
# ==========================================================
batch_size = 8             # ConvLSTM 显存占用大，建议从 8 或 16 开始
val_batch_size = 8
epoch = 100                # 注意 parser 里参数名是 epoch 不是 epochs
lr = 1e-3
sched = 'cosine'
warmup_epoch = 5

# ==========================================================
# 4. 评估指标
# ==========================================================
metrics = ['mse', 'mae', 'ssim', 'pod', 'radar_metrics']
method = 'PredRNN'
dataname = 'cikm'

# ==========================================================
# 1. 数据集参数 (与 dataloader_cikm.py 配合)
# ==========================================================
# 请修改为你实际的 H5 文件路径
data_root = '/home/ps/data2/zp/data/cikm.h5'

# 序列长度设置
pre_seq_length = 5   # 输入长度
aft_seq_length = 10  # 预测长度
total_length = 15    # 总长度

# 输入形状: [输入帧数, 通道数, 高, 宽]
# CIKM 是单通道雷达图，分辨率通常 resize/pad 到 128
in_shape = [5, 1, 128, 128]

# ==========================================================
# 2. 模型参数 (PredRNN 特有参数)
# ==========================================================
# 隐藏层通道数
# 注意：CIKM (128x128) 比 MNIST (64x64) 大4倍。
# 为了防止显存溢出(OOM)，建议设为 '64,64,64,64'。若显存充足可尝试 '128,128,128,128'
num_hidden = '128,128,128,128'

filter_size = 5
stride = 1

# Patch Size 是显存占用的关键。
# 128/4 = 32x32 的特征图。如果爆显存，可以尝试改为 8 (即 16x16 特征图)
patch_size = 2

layer_norm = 0             # PredRNN 默认设置

# Scheduled Sampling (计划采样) - PredRNN 训练的核心策略
reverse_scheduled_sampling = 0
r_sampling_step_1 = 25000
r_sampling_step_2 = 50000
r_exp_alpha = 5000

scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002

# ==========================================================
# 3. 训练参数
# ==========================================================
# Batch Size 根据显存调整。
# 单卡 24G 显存建议从 4 或 8 开始
batch_size = 8
val_batch_size = 8
epoch = 100
lr = 5e-4                  # PredRNN 常用学习率
sched = 'onecycle'         # 学习率调度器
warmup_epoch = 5

# ==========================================================
# 4. 评估指标
# ==========================================================
# radar_metrics 是针对雷达回波的特定指标 (CSI/POD 等)
metrics = ['mse', 'mae', 'ssim', 'psnr', 'pod', 'radar_metrics']

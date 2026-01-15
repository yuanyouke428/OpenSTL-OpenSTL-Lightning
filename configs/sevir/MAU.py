method = 'MAU'
# ==========================================================
# 1. 数据集参数
# ==========================================================
dataname = 'sevir'
pre_seq_length = 13
aft_seq_length = 12
total_length = 25
in_shape = [13, 1, 128, 128]

# ==========================================================
# 2. 模型参数
# ==========================================================
num_hidden = '64,64,64,64'
filter_size = 5
stride = 1
patch_size = 1
layer_norm = 0

# MAU 特有参数
sr_size = 4          # 下采样倍率，MAU 核心参数，通常设为 4 或 2
tau = 5              # 记忆回溯步数，通常设为 5
cell_mode = 'normal'
model_mode = 'normal'

# Scheduled Sampling
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002

# ==========================================================
# 3. 训练参数
# ==========================================================
batch_size = 16
val_batch_size = 16
lr = 1e-3            # MAU 结构稳定，通常可以使用 1e-3 的学习率
sched = 'onecycle'
warmup_epoch = 0
epoch = 100
patience = 10
# ==========================================================
# 4. 评估指标
# ==========================================================
metrics = ['mse', 'mae', 'ssim', 'lpips', 'radar_metrics']
metric_threshold = [16, 74, 133, 160, 181, 219]
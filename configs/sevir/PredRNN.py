method = 'PredRNN'
# ==========================================================
# 1. 数据集参数
# ==========================================================
dataname = 'sevir'
# 输入 13 帧 -> 预测 12 帧
pre_seq_length = 13
aft_seq_length = 12
total_length = 25
# 确保这里匹配你的 128x128 数据
in_shape = [13, 1, 128, 128]

# ==========================================================
# 2. 模型参数
# ==========================================================
# 64通道对于128x128是可以跑起来的，显存应该在10G-16G左右
num_hidden = '64,64,64,64'
filter_size = 5
stride = 1
patch_size = 4
layer_norm = 0

# 【关键修改】关闭反向计划采样，避免 IndexError
reverse_scheduled_sampling = 0
r_sampling_step_1 = 25000
r_sampling_step_2 = 50000
r_exp_alpha = 5000

# 保留标准的计划采样 (Teacher Forcing decay)
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002

# ==========================================================
# 3. 训练参数
# ==========================================================
batch_size = 16
val_batch_size = 16
lr = 3e-4   # 建议用 3e-4，PredRNN 比 ConvLSTM 更容易梯度爆炸
sched = 'onecycle'
warmup_epoch = 0
epoch = 100
patience = 10 # 既然设置了patience，建议配合EarlyStopping callback使用

# ==========================================================
# 4. 评估指标
# ==========================================================
metrics = ['mse', 'mae', 'ssim', 'lpips', 'radar_metrics']
metric_threshold = [16, 74, 133, 160, 181, 219]



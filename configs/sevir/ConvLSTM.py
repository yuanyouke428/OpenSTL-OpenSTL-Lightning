method = 'ConvLSTM'
# ==========================================================
# 1. 数据集参数
# ==========================================================
dataname = 'sevir'
# 序列长度配置：输入13帧 -> 预测12帧
pre_seq_length = 13
aft_seq_length = 12
total_length = 25

# 数据维度 [T, C, H, W]
# 修正：匹配你的下采样尺寸 128x128
in_shape = [13, 1, 128, 128]

# ==========================================================
# 2. 模型参数
# ==========================================================
# 4层 LSTM，每层 128 通道。对于 128x128 输入，这个容量足够大，能捕获很好的特征。
num_hidden = '128,128,128,128'
filter_size = 5
stride = 1
# 128 / 4 = 32，特征图大小为 32x32，兼顾了精度和显存
patch_size = 4
layer_norm = 0

# Scheduled Sampling (ConvLSTM 标准训练策略)
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
# 128x128 下 batch_size=16 显存占用预估在 10GB-16GB 左右（取决于具体实现开销）

batch_size = 32
val_batch_size = 32 # 验证集 batch size 可以和训练集一致，或者更大
lr = 5e-4
sched = 'onecycle'
#预热 总步数的0.05-0.1之间
warmup_epoch = 5
epoch = 100

# ==========================================================
# 4. 评估指标
# ==========================================================
# 包含常用的图像指标 (MSE, MAE, SSIM, LPIPS) 和气象指标 (CSI, POD 等)
metrics = ['mse', 'mae', 'ssim', 'lpips', 'radar_metrics']

# 如果验证集 Loss 在 10 个 Epoch 内没有下降，则停止训练
patience = 10


# SEVIR VIL 像素阈值，用于 radar_metrics 计算
metric_threshold = [16, 74, 133, 160, 181, 219]
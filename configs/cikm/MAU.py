method = 'MAU'
dataname = 'cikm'

# ==========================================================
# 1. 数据集参数 (与 CIKM 通用配置保持一致)
# ==========================================================
# 请确认你的 .h5 文件路径是否正确
data_root = '/home/ps/data2/zp/data/cikm.h5'

pre_seq_length = 5
aft_seq_length = 10
total_length = 15

# CIKM 输入形状: [输入帧数, 通道数, 高, 宽]
in_shape = [5, 1, 128, 128]

# ==========================================================
# 2. 模型参数 (MAU 特有参数)
# ==========================================================
# 隐藏层通道数
num_hidden = '64,64,64,64'

filter_size = 5
stride = 1

# 重要：CIKM 图片较大(128x128)，必须使用 Patch 降低维度
# 128 / 4 = 32，模型将在 32x32 的特征图上进行计算
patch_size = 1

layer_norm = 0             # MAU 默认通常不加 LayerNorm

# MAU 特有超参数
sr_size = 4                # 建议与 patch_size 保持一致，用于解码恢复
tau = 5                    # 时序滑动窗口参数
cell_mode = 'normal'
model_mode = 'normal'

# Scheduled Sampling (计划采样)
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002

# ==========================================================
# 3. 训练参数
# ==========================================================
# MAU 显存占用适中，128x128下建议从 8 开始尝试
batch_size = 16
val_batch_size = 8
epoch = 100
lr = 1e-3                  # MAU 常用学习率
sched = 'onecycle'         # MAU 常用调度器
warmup_epoch = 5

# ==========================================================
# 4. 评估指标
# ==========================================================
metrics = ['mse', 'mae', 'ssim', 'radar_metrics']
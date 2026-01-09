method = 'SimVP'
# 关键修改 1: 显式指定数据集名称，这将触发 metrics.py 中 METRIC_CONFIGS['sevir'] 的逻辑
dataname = 'sevir'

# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'IncepU'
hid_S = 64
hid_T = 256
N_T = 4
N_S = 2

# training
lr = 1e-3           # 128分辨率下建议稍微调小学习率
batch_size = 16     # 128分辨率下显存占用小，建议开大 Batch Size (如 16 或 32)
drop_path = 0.1
sched = 'onecycle'
warmup_epoch = 0
epoch = 100
# --- 关键修改 2: 适配 128x128 分辨率和 13->12 预测 ---
# 输入形状 [T_in, C, H, W]
in_shape = [13, 1, 128, 128]
# 序列长度
pre_seq_length = 13
aft_seq_length = 12


patience = 10

# --- 关键修改 3: 启用气象评估指标 ---
# 只有加上 'radar_metrics'，才会去计算 CSI, POD, HSS 等指标
# 只有加上 'sevir' 的 dataname，才会使用 metrics.py 中修正后的阈值 [16, 74...]
metrics = ['mse', 'mae','ssim', 'lpips','radar_metrics']

metric_threshold = [16, 74, 133, 160, 181, 219]
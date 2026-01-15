method = 'mamba' # 使用 SimVP 的训练流程 (MSE Loss, 这里的 method 指的是 Training Recipe)
dataname = 'cikm'
data_root = '/home/ps/data2/zp/data/cikm.h5'
# 模型配置
model_type = 'MambaCast'  # 对应 __init__.py 里的名字
hid_S = 64      # 隐藏层通道数
hid_T = 256     # (在MambaCast里这个参数可能没用到，为了兼容性保留)
N_S = 4         # Encoder/Decoder 下采样层数
N_T = 4         # Mamba 层数 (可以调大，比如 8 或 12)
input_shape = (5, 1, 128, 128) # SEVIR 的标准输入: 13帧 (12输入+1预测 或 其它切分)
# 序列长度
pre_seq_length = 5
aft_seq_length = 10
total_length = 15

clip_grad = 1.0  # 原来是 None
clip_mode = 'norm' # 保持默认即可


patience = 10
# 训练配置
batch_size = 32  # Mamba 显存占用低，如果显存够大可以改到 8 或 16
val_batch_size = 32

#drop_path = 0.1  # 建议设置为 0.1 或 0.2

lr = 0.0005
epoch = 100

sched = 'onecycle'
warmup_epoch = 5
fp16 = False     # 开启混合精度，加速训练
metrics = ['mse', 'mae','ssim', 'lpips','radar_metrics']

metric_threshold = [16, 74, 133, 160, 181, 219]

w_freq = 0.01
gamma = 0.5
mask_ratio = 3.0

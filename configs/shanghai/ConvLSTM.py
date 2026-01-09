method = 'ConvLSTM'
# 1. 数据集定义
dataname = 'shanghai'
data_root = '/home/ps/data2/zp/data/shanghai.h5'

# 2. 序列长度与形状
# Input: 5 frames -> Output: 20 frames
pre_seq_length = 5
aft_seq_length = 20
total_length = 25
# Shape: (T, C, H, W) -> C=1 (Radar Reflectivity)
in_shape = [5, 1, 128, 128]

# 3. 采样策略 (Curriculum Learning)
# 确保 dataset_len / batch_size * epoch > 50000，否则采样率不会衰减
reverse_scheduled_sampling = 0
r_sampling_step_1 = 25000
r_sampling_step_2 = 50000
r_exp_alpha = 5000

scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002

# 4. 模型结构
num_hidden = '128,128,128,128'
filter_size = 5
stride = 1
# patch_size=4 对应 128x128 输入产生 32x32 特征图，保留较好气象细节
patch_size = 4
layer_norm = 0

# 5. 训练参数
lr = 5e-4
# 注意：128x128 分辨率下，Batch=16 对显存要求较高。若 OOM 请改为 8 或 4
batch_size = 32
sched = 'onecycle'
epoch = 100

patience = 10
# 6. 评价指标
# 包含基础指标 + 感知指标(LPIPS) + 气象指标(CSI/POD/FAR)

metrics = ['mse', 'mae', 'ssim', 'lpips', 'radar_metrics']
metric_threshold=[20, 30, 35, 40]

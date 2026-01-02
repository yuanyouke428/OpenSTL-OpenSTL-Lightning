import numpy as np
import torch
import torch.nn as nn

try:
    import lpips
    from skimage.metrics import structural_similarity as cal_ssim
except ImportError:
    lpips = None
    cal_ssim = None


# =========================================================================
# 1. 基础工具函数
# =========================================================================

def _threshold(x, y, t):
    """ 计算二值化的预测图和真实图 """
    t_map = np.greater_equal(x, t).astype(np.float32)
    p_map = np.greater_equal(y, t).astype(np.float32)
    return t_map, p_map


def calculate_confusion(pred, true, threshold):
    """
    核心混淆矩阵计算 (TP, FP, FN, TN)
    """
    t_map, p_map = _threshold(pred, true, threshold)

    hits = np.sum(t_map * p_map)  # TP
    fas = np.sum(t_map * (1 - p_map))  # FP (误报)
    misses = np.sum((1 - t_map) * p_map)  # FN (漏报)

    return hits, fas, misses


def POD(hits, misses, eps=1e-6): return hits / (hits + misses + eps)


def SUCR(hits, fas, eps=1e-6): return hits / (hits + fas + eps)


def CSI(hits, fas, misses, eps=1e-6): return hits / (hits + misses + fas + eps)


def HSS(hits, fas, misses, cn, eps=1e-6):
    num = 2 * (hits * cn - misses * fas)
    den = (hits + misses) * (misses + cn) + (hits + fas) * (fas + cn)
    return num / (den + eps)


def MAE(pred, true, spatial_norm=False): return np.mean(np.abs(pred - true))


def MSE(pred, true, spatial_norm=False): return np.mean((pred - true) ** 2)


def RMSE(pred, true, spatial_norm=False): return np.sqrt(MSE(pred, true))


# =========================================================================
# 2. LPIPS 计算类
# =========================================================================
class LPIPS(nn.Module):
    def __init__(self, net='alex', use_gpu=True):
        super().__init__()
        self.use_gpu = use_gpu and torch.cuda.is_available()
        try:
            self.loss_fn = lpips.LPIPS(net=net)
            if self.use_gpu: self.loss_fn.cuda()
        except:
            self.loss_fn = None

    def forward(self, pred, true):
        if self.loss_fn is None: return 0.0

        # 1. 维度转换 (B, T, C, H, W) -> (N, C, H, W)
        B, T, C, H, W = pred.shape
        pred = pred.reshape(-1, C, H, W)
        true = true.reshape(-1, C, H, W)

        # 2. 转换为 Tensor
        pred_t = torch.from_numpy(pred).float()
        true_t = torch.from_numpy(true).float()

        # 3. 单通道转伪彩色
        if C == 1:
            pred_t = pred_t.repeat(1, 3, 1, 1)
            true_t = true_t.repeat(1, 3, 1, 1)

        # 4. 归一化调整 [-1, 1]
        pred_t = pred_t * 2 - 1
        true_t = true_t * 2 - 1

        # 5. 【关键修改】分 Batch 计算，防止 OOM
        # 每次只算 32 张图 (你可以根据显存大小调整这个 mini_batch)
        mini_batch = 32
        scores = []
        N = pred_t.shape[0]

        with torch.no_grad():
            for i in range(0, N, mini_batch):
                p_batch = pred_t[i:i + mini_batch]
                t_batch = true_t[i:i + mini_batch]

                if self.use_gpu:
                    p_batch = p_batch.cuda()
                    t_batch = t_batch.cuda()

                # 计算当前 mini-batch 的分数并转回 CPU
                batch_score = self.loss_fn(p_batch, t_batch).squeeze().cpu().numpy()

                # 兼容处理：如果只有一个数，转化为列表
                if batch_score.ndim == 0:
                    scores.append(batch_score.item())
                else:
                    scores.extend(batch_score.tolist())

        return np.mean(scores)

# =========================================================================
# 3. 主评估函数 (metric)
# =========================================================================

def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse', 'radar_metrics'],
           clip_range=[0, 1], spatial_norm=False, return_log=True,
           dataset_name='cikm', channel_names=None, **kwargs):
    # 1. 反归一化
    if mean is not None and std is not None:
        pred = pred * std + mean
        true = true * std + mean

    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])
    true = np.maximum(true, clip_range[0])
    true = np.minimum(true, clip_range[1])

    eval_res = {}
    eval_log = ""

    # 配置中心
    METRIC_CONFIGS = {
        'cikm': {
            'scale': 95.0,
            'offset': -10.0,
            # 【修复1】加回 10，并保留常用的阈值
            'thresholds': [10, 20, 30, 35, 40],
            'use_pixel_threshold': False
        },
        'sevir': {
            'scale': 255.0,
            'offset': 0.0,
            'thresholds': [16, 74, 133, 160, 181, 219],
            'use_pixel_threshold': True
        },
        'default': {
            'scale': 1.0,
            'offset': 0.0,
            'thresholds': [0.1, 0.25, 0.5],
            'use_pixel_threshold': False
        }
    }

    cfg = METRIC_CONFIGS.get(dataset_name, METRIC_CONFIGS['default'])

    # 计算物理值 (dBZ)
    pred_eval = pred * cfg['scale'] + cfg['offset']
    true_eval = true * cfg['scale'] + cfg['offset']

    # 【修复2】决定使用哪种数据计算 MSE/MAE
    # 如果你想看“大数值”的物理误差，使用 pred_eval
    # 如果你想看“标准”的归一化误差，使用 pred
    # 这里我改为 pred_eval 以恢复你之前的数值大小
    target_pred = pred_eval if 'radar_metrics' in metrics else pred
    target_true = true_eval if 'radar_metrics' in metrics else true

    # 基础指标
    if 'mse' in metrics: eval_res['mse'] = MSE(target_pred, target_true)
    if 'mae' in metrics: eval_res['mae'] = MAE(target_pred, target_true)
    if 'rmse' in metrics: eval_res['rmse'] = RMSE(target_pred, target_true)

    # 气象指标
    if 'radar_metrics' in metrics or 'csi' in metrics:
        total_pixels = np.prod(pred.shape)
        avg_csi = []
        avg_hss = []
        avg_pod = []  # 初始化

        for t in cfg['thresholds']:
            current_t = t
            if cfg['use_pixel_threshold']:
                current_t = t / 255.0

            hits, fas, misses = calculate_confusion(pred_eval, true_eval, t)  # 始终用物理值对比
            cn = total_pixels - hits - misses - fas

            t_csi = CSI(hits, fas, misses)
            t_hss = HSS(hits, fas, misses, cn)
            t_pod = POD(hits, misses)  # 【修复3】计算 POD

            key_suffix = str(t).replace('.', 'p')
            eval_res[f'csi_{key_suffix}'] = t_csi
            eval_res[f'hss_{key_suffix}'] = t_hss
            # eval_res[f'pod_{key_suffix}'] = t_pod # 如果你想看每个阈值的POD就取消注释

            avg_csi.append(t_csi)
            avg_hss.append(t_hss)
            avg_pod.append(t_pod)

        eval_res['avg_csi'] = np.mean(avg_csi)  # 即 cikm_csi
        eval_res['avg_hss'] = np.mean(avg_hss)  # 即 cikm_hss
        eval_res['pod'] = np.mean(avg_pod)  # 把平均 POD 加回去

    # 图像质量
    if 'ssim' in metrics and cal_ssim is not None:
        ssim_sum = 0
        B, T = pred.shape[:2]
        # SSIM 建议在归一化尺度 [0,1] 计算，或者与之前保持一致
        # 这里使用归一化数据以保持标准性
        for b in range(B):
            for t in range(T):
                ssim_sum += cal_ssim(pred[b, t].squeeze(), true[b, t].squeeze(), data_range=1.0)
        eval_res['ssim'] = ssim_sum / (B * T)

    if 'lpips' in metrics and lpips is not None:
        try:
            scorer = LPIPS(net='alex', use_gpu=True)
            eval_res['lpips'] = scorer(pred, true)
        except Exception as e:
            print(f"\n[Warning] LPIPS calculation failed: {e}")  # 打印具体的错误信息

    if return_log:
        for k, v in eval_res.items():
            val = f"{v:.4f}" if isinstance(v, (float, np.float32)) else str(v)
            eval_log += f"{k}:{val}, "

    return eval_res, eval_log
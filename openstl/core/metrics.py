import numpy as np
import torch
import torch.nn as nn

try:
    import lpips
    from skimage.metrics import structural_similarity as cal_ssim
except ImportError:
    lpips = None
    cal_ssim = None

# 定义一个全局变量，初始为 None
_GLOBAL_LPIPS_SCORER = None
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


def MAE(pred, true): return np.mean(np.abs(pred - true))


def MSE(pred, true): return np.mean((pred - true) ** 2)


def RMSE(pred, true): return np.sqrt(MSE(pred, true))


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

        # 5. 分 Batch 计算，防止 OOM
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

                # 兼容处理
                if batch_score.ndim == 0:
                    scores.append(batch_score.item())
                else:
                    scores.extend(batch_score.tolist())

        return np.mean(scores)


# =========================================================================
# 3. 主评估函数 (metric)
# =========================================================================

def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse', 'radar_metrics'],
           clip_range=[0,1], spatial_norm=False, return_log=True,
           dataset_name='default', channel_names=None, **kwargs):


    print(f"DEBUG: Running metric() for dataset: {dataset_name}")


    # -------------------------------------------------------------------------
    # 阶段 A: 初步反归一化 (如果 base_method 传了 mean/std)
    # -------------------------------------------------------------------------
    # 如果您的 base_method 没有传 mean/std (即传了 0/1 或 None)，这步相当于没做
    if mean is not None and std is not None:
        pred = pred * std + mean
        true = true * std + mean

    # -------------------------------------------------------------------------
    # 阶段 B: 数据集配置中心 (核心修改)
    # -------------------------------------------------------------------------
    METRIC_CONFIGS = {
        'cikm': {
            'scale': 95.0,  # 0-1 映射到 0-95
            'offset': -10.0,  # 偏移到 -10 到 85 dBZ
            'thresholds': [10, 20, 30, 35, 40],
            'use_pixel_threshold': False
        },
        'meteonet': {
            'scale': 70.0,  #
            'offset': 0,  #
            'thresholds': [12, 18, 24, 32],
            'use_pixel_threshold': False
        },
        'shanghai': {
            'scale': 90.0,  #根据Diffcast
            'offset': 0,  #
            'thresholds': [20, 30, 35, 40],
            'use_pixel_threshold': False
        },
        'sevir': {
            # 【核心修改点】
            # 在这里定义 SEVIR 的反归一化逻辑 (Mean=33.44, Std=47.54)
            # 逻辑：Pixel_Value = Normalized_Value * scale + offset
            'scale': 47.54,
            'offset': 33.44,
            'thresholds': [16, 74, 133, 160, 181, 219],
            # 设为 False，因为经过上面的 scale/offset 计算后，数据已经是 0-255 的像素值了
            # 我们直接用 16, 74 跟 0-255 比较即可
            'use_pixel_threshold': False
        },

        'default': {
            'scale': 1.0,
            'offset': 0.0,
            'thresholds': [0.1, 0.25, 0.5],
            'use_pixel_threshold': False
        }
    }

    # 1. 获取配置
    # 使用包含匹配，这样 'sevir_13to12' 也能匹配到 'sevir'
    cfg = METRIC_CONFIGS['default']
    for k in METRIC_CONFIGS.keys():
        if k in dataset_name.lower():
            cfg = METRIC_CONFIGS[k]
            break

    # 2. 计算物理值 (Physical Values)
    # 这是最关键的一步：将归一化的数据转换回真实的物理量/像素值 (如 0-255)
    pred_eval = pred * cfg['scale'] + cfg['offset']
    true_eval = true * cfg['scale'] + cfg['offset']

    # 3. 截断保护 (Clip)
    # 确保数值在合理范围内，SEVIR 是 0-255
    #max_val = 255.0 if 'sevir' in dataset_name.lower() or 'cikm' in dataset_name.lower() else 1.0
    # 修改后：根据配置动态决定，或者把 meteo 加入白名单
    if 'meteonet' in dataset_name.lower():
        max_val = 70.0
    # === 新增 shanghai 分支 ===
    elif 'shanghai' in dataset_name.lower():
        max_val = 90.0  # 对应 METRIC_CONFIGS 中的 scale
    # ==========================
    elif 'sevir' in dataset_name.lower() or 'cikm' in dataset_name.lower():
        max_val = 255.0
    else:
        max_val = 1.0


    pred_eval = np.maximum(pred_eval, 0)
    pred_eval = np.minimum(pred_eval, max_val)
    true_eval = np.maximum(true_eval, 0)
    true_eval = np.minimum(true_eval, max_val)

    # -------------------------------------------------------------------------
    # 阶段 C: 指标计算
    # -------------------------------------------------------------------------
    eval_res = {}
    eval_log = ""

    # 【重要】决定计算 MSE/MAE 使用哪种数据
    # 只要包含 radar_metrics，我们就强制使用物理值 (pred_eval) 来计算所有指标
    # 这样您的 MSE 就会是 1800+ 而不是 0.2
    target_pred = pred_eval if ('radar_metrics' in metrics or 'sevir' in dataset_name.lower()) else pred
    target_true = true_eval if ('radar_metrics' in metrics or 'sevir' in dataset_name.lower()) else true

    # 1. 基础指标 (MSE, MAE, RMSE)
    if 'mse' in metrics: eval_res['mse'] = MSE(target_pred, target_true)
    if 'mae' in metrics: eval_res['mae'] = MAE(target_pred, target_true)
    if 'rmse' in metrics: eval_res['rmse'] = RMSE(target_pred, target_true)

    # 2. 气象分类指标 (CSI, HSS, POD)
    if 'radar_metrics' in metrics or 'csi' in metrics:
        total_pixels = np.prod(pred.shape)
        avg_csi, avg_hss, avg_pod = [], [], []

        for t in cfg['thresholds']:
            # 处理阈值
            current_t = t
            if cfg['use_pixel_threshold']:
                current_t = t / 255.0

            # 【Bug 修复】这里之前写成了 t，必须用 current_t
            hits, fas, misses = calculate_confusion(target_pred, target_true, current_t)
            cn = total_pixels - hits - misses - fas

            t_csi = CSI(hits, fas, misses)
            t_hss = HSS(hits, fas, misses, cn)
            t_pod = POD(hits, misses)

            key_suffix = str(int(t)) if t > 1 else str(t).replace('.', 'p')
            eval_res[f'csi_{key_suffix}'] = t_csi
            eval_res[f'hss_{key_suffix}'] = t_hss
            # eval_res[f'pod_{key_suffix}'] = t_pod

            avg_csi.append(t_csi)
            avg_hss.append(t_hss)
            avg_pod.append(t_pod)

        eval_res['avg_csi'] = np.mean(avg_csi)
        eval_res['avg_hss'] = np.mean(avg_hss)
        eval_res['pod'] = np.mean(avg_pod)

    # 3. 感知质量指标 (SSIM)
    if 'ssim' in metrics and cal_ssim is not None:
        ssim_sum = 0
        B, T = pred.shape[:2]
        # SSIM 建议使用 data_range 参数来适配数据范围
        data_range = max_val

        for b in range(B):
            for t in range(T):
                ssim_sum += cal_ssim(target_pred[b, t].squeeze(), target_true[b, t].squeeze(),
                                     data_range=data_range)
        eval_res['ssim'] = ssim_sum / (B * T)

        # 引用全局变量
        global _GLOBAL_LPIPS_SCORER

        # 4. 感知误差 (LPIPS)
        if 'lpips' in metrics and lpips is not None:
            try:
                # 【修复点】: 只有当全局变量为空时才实例化，否则直接复用！
                if _GLOBAL_LPIPS_SCORER is None:
                    # 实例化一次，权重加载进显存
                    _GLOBAL_LPIPS_SCORER = LPIPS(net='alex', use_gpu=True)
                    # 可选：确保模型处于评估模式
                    if hasattr(_GLOBAL_LPIPS_SCORER, 'loss_fn') and _GLOBAL_LPIPS_SCORER.loss_fn:
                        _GLOBAL_LPIPS_SCORER.loss_fn.eval()

                # 使用全局实例进行计算
                eval_res['lpips'] = _GLOBAL_LPIPS_SCORER(pred, true)

            except Exception as e:
                print(f"\n[Warning] LPIPS calculation failed: {e}")
                # 如果出错（比如显存不足），可以将全局变量重置，防止错误持续
                _GLOBAL_LPIPS_SCORER = None

        if return_log:
            for k, v in eval_res.items():
                val = f"{v:.4f}" if isinstance(v, (float, np.float32)) else str(v)
                eval_log += f"{k}:{val}, "

        return eval_res, eval_log


    # # 4. 感知误差 (LPIPS)
    # if 'lpips' in metrics and lpips is not None:
    #     try:
    #         scorer = LPIPS(net='alex', use_gpu=True)
    #         # LPIPS 内部会自动处理归一化，但为了稳健，我们传归一化后的数据
    #         # 这里的 pred 还是原始输入 (normalized)，可以直接用
    #         eval_res['lpips'] = scorer(pred, true)
    #     except Exception as e:
    #         print(f"\n[Warning] LPIPS calculation failed: {e}")
    #
    # if return_log:
    #     for k, v in eval_res.items():
    #         val = f"{v:.4f}" if isinstance(v, (float, np.float32)) else str(v)
    #         eval_log += f"{k}:{val}, "
    #
    # return eval_res, eval_log
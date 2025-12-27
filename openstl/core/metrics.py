import cv2
import numpy as np
import torch

try:
    import lpips
    from skimage.metrics import structural_similarity as cal_ssim
except:
    lpips = None
    cal_ssim = None

def HSS(hits, fas, misses, cn, eps=1e-6):
    """Heisenberg-Kutner Skill Score"""
    num = 2 * (hits * cn - misses * fas)
    den = (hits + misses) * (misses + cn) + (hits + fas) * (fas + cn)
    return np.mean(num / (den + eps))

def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1

def _threshold(x, y, t):
    t = np.greater_equal(x, t).astype(np.float32)
    p = np.greater_equal(y, t).astype(np.float32)
    is_nan = np.logical_or(np.isnan(x), np.isnan(y))
    t = np.where(is_nan, np.zeros_like(t, dtype=np.float32), t)
    p = np.where(is_nan, np.zeros_like(p, dtype=np.float32), p)
    return t, p

def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred-true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred-true) / norm, axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred-true)**2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred-true)**2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred-true)**2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred-true)**2 / norm, axis=(0, 1)).sum())


def PSNR(pred, true, min_max_norm=True):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    else:
        if min_max_norm:  # [0, 1] normalized by min and max
            return 20. * np.log10(1. / np.sqrt(mse))  # i.e., -10. * np.log10(mse)
        else:
            return 20. * np.log10(255. / np.sqrt(mse))  # [-1, 1] normalized by mean and std


def SNR(pred, true):
    """Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    """
    signal = ((true)**2).mean()
    noise = ((true - pred)**2).mean()
    return 10. * np.log10(signal / noise)


def SSIM(pred, true, **kwargs):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = pred.astype(np.float64)
    img2 = true.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def POD(hits, misses, eps=1e-6):
    """
    probability_of_detection
    Inputs:
    Outputs:
        pod = hits / (hits + misses) averaged over the T channels
        
    """
    pod = (hits + eps) / (hits + misses + eps)
    return np.mean(pod)

def SUCR(hits, fas, eps=1e-6):
    """
    success_rate
    Inputs:
    Outputs:
        sucr = hits / (hits + false_alarms) averaged over the D channels
    """
    sucr = (hits + eps) / (hits + fas + eps)
    return np.mean(sucr)

def CSI(hits, fas, misses, eps=1e-6):
    """
    critical_success_index 
    Inputs:
    Outputs:
        csi = hits / (hits + false_alarms + misses) averaged over the D channels
    """
    csi = (hits + eps) / (hits + misses + fas + eps)
    return np.mean(csi)

def sevir_metrics(pred, true, threshold):
    """
    calcaulate t, p, hits, fas, misses
    Inputs:
    pred: [N, T, C, L, L]
    true: [N, T, C, L, L]
    threshold: float
    """
    pred = pred.transpose(1, 0, 2, 3, 4)
    true = true.transpose(1, 0, 2, 3, 4)
    hits, fas, misses = [], [], []
    for i in range(pred.shape[0]):
        t, p = _threshold(pred[i], true[i], threshold)
        hits.append(np.sum(t * p))
        fas.append(np.sum((1 - t) * p))
        misses.append(np.sum(t * (1 - p)))
    return np.array(hits), np.array(fas), np.array(misses)


class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity, LPIPS.

    Modified from
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
    """

    def __init__(self, net='alex', use_gpu=True):
        super().__init__()
        assert net in ['alex', 'squeeze', 'vgg']
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.loss_fn = lpips.LPIPS(net=net)
        if use_gpu:
            self.loss_fn.cuda()
#原有的
    # def forward(self, img1, img2):
    #     # Load images, which are min-max norm to [0, 1]
    #     img1 = lpips.im2tensor(img1 * 255)  # RGB image from [-1,1]
    #     img2 = lpips.im2tensor(img2 * 255)
    #     if self.use_gpu:
    #         img1, img2 = img1.cuda(), img2.cuda()
    #     return self.loss_fn.forward(img1, img2).squeeze().detach().cpu().numpy()
#改进的
    def forward(self, img1, img2):
        """
        修正后的 LPIPS 计算逻辑：
        1. 显式处理单通道 (Grayscale) 转 3通道 (RGB)
        2. 显式处理归一化 ([0, 1] -> [-1, 1])
        3. 跳过 lpips.im2tensor，防止维度识别错误
        """
        # 1. 转换为 Tensor 并增加 Batch 维度: (C, H, W) -> (1, C, H, W)
        # 假设输入是 numpy array，形状为 (1, 128, 128)
        img1 = torch.from_numpy(img1).unsqueeze(0).float()
        img2 = torch.from_numpy(img2).unsqueeze(0).float()

        # 2. 如果是单通道 (Grayscale)，重复 3 次变成伪 RGB: (1, 3, H, W)
        # LPIPS 网络 (AlexNet/VGG) 第一层需要 3 个通道
        if img1.shape[1] == 1:
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)

        # 3. 数据归一化
        # 你的原始数据在 [0, 1] 之间，但 LPIPS 预训练模型期望输入在 [-1, 1] 之间
        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1

        # 4. 放入 GPU (如果启用了)
        if self.use_gpu:
            img1 = img1.cuda()
            img2 = img2.cuda()

        # 5. 计算并返回
        return self.loss_fn.forward(img1, img2).squeeze().detach().cpu().numpy()


# def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse'],
#            clip_range=[0, 1], channel_names=None,
#            spatial_norm=False, return_log=True, threshold=74.0):
#
#     """The evaluation function to output metrics.
#
#     Args:
#         pred (tensor): The prediction values of output prediction.
#         true (tensor): The prediction values of output prediction.
#         mean (tensor): The mean of the preprocessed video data.
#         std (tensor): The std of the preprocessed video data.
#         metric (str | list[str]): Metrics to be evaluated.
#         clip_range (list): Range of prediction to prevent overflow.
#         channel_names (list | None): The name of different channels.
#         spatial_norm (bool): Weather to normalize the metric by HxW.
#         return_log (bool): Whether to return the log string.
#
#     Returns:
#         dict: evaluation results
#     """
#     if mean is not None and std is not None:
#         pred = pred * std + mean
#         true = true * std + mean
#     eval_res = {}
#     eval_log = ""
#     allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'psnr', 'snr', 'lpips', 'pod', 'sucr', 'csi','radar_metrics']
#     invalid_metrics = set(metrics) - set(allowed_metrics)
#     if len(invalid_metrics) != 0:
#         raise ValueError(f'metric {invalid_metrics} is not supported.')
#     if isinstance(channel_names, list):
#         assert pred.shape[2] % len(channel_names) == 0 and len(channel_names) > 1
#         c_group = len(channel_names)
#         c_width = pred.shape[2] // c_group
#     else:
#         channel_names, c_group, c_width = None, None, None
#
#     if 'mse' in metrics:
#         if channel_names is None:
#             eval_res['mse'] = MSE(pred, true, spatial_norm)
#         else:
#             mse_sum = 0.
#             for i, c_name in enumerate(channel_names):
#                 eval_res[f'mse_{str(c_name)}'] = MSE(pred[:, :, i*c_width: (i+1)*c_width, ...],
#                                                      true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
#                 mse_sum += eval_res[f'mse_{str(c_name)}']
#             eval_res['mse'] = mse_sum / c_group
#
#     if 'mae' in metrics:
#         if channel_names is None:
#             eval_res['mae'] = MAE(pred, true, spatial_norm)
#         else:
#             mae_sum = 0.
#             for i, c_name in enumerate(channel_names):
#                 eval_res[f'mae_{str(c_name)}'] = MAE(pred[:, :, i*c_width: (i+1)*c_width, ...],
#                                                      true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
#                 mae_sum += eval_res[f'mae_{str(c_name)}']
#             eval_res['mae'] = mae_sum / c_group
#
#     if 'rmse' in metrics:
#         if channel_names is None:
#             eval_res['rmse'] = RMSE(pred, true, spatial_norm)
#         else:
#             rmse_sum = 0.
#             for i, c_name in enumerate(channel_names):
#                 eval_res[f'rmse_{str(c_name)}'] = RMSE(pred[:, :, i*c_width: (i+1)*c_width, ...],
#                                                        true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
#                 rmse_sum += eval_res[f'rmse_{str(c_name)}']
#             eval_res['rmse'] = rmse_sum / c_group
#
#     if 'radar_metrics' in metrics:
#         # CIKM 物理参数 (参考 Diffcast)
#         scale = 90.0
#         thresholds = [20, 30, 35, 40]
#
#         # 还原数值到 [0, 90]
#         pred_rescaled = pred * scale
#         true_rescaled = true * scale
#
#         # 重新计算物理尺度上的基础指标 (覆盖之前的归一化指标)
#         if 'mse' in metrics:
#             eval_res['mse'] = MSE(pred_rescaled, true_rescaled, spatial_norm)
#         if 'mae' in metrics:
#             eval_res['mae'] = MAE(pred_rescaled, true_rescaled, spatial_norm)
#         if 'rmse' in metrics:
#             eval_res['rmse'] = RMSE(pred_rescaled, true_rescaled, spatial_norm)
#
#         # 计算多阈值指标
#         avg_csi, avg_pod, avg_far, avg_hss = [], [], [], []
#         total_pixels = np.prod(pred.shape[-2:])
#
#         for t in thresholds:
#             # 计算 hits, false alarms, misses
#             hits, fas, misses = sevir_metrics(pred_rescaled, true_rescaled, t)
#             hits, fas, misses = np.sum(hits), np.sum(fas), np.sum(misses)
#
#             # 计算 Correct Negatives
#             cn = total_pixels * pred.shape[0] * pred.shape[1] - hits - misses - fas
#
#             t_csi = CSI(hits, fas, misses)
#             t_hss = HSS(hits, fas, misses, cn)
#
#             # 记录每个阈值的详细分数
#             eval_res[f'csi_{t}'] = t_csi
#             eval_res[f'hss_{t}'] = t_hss
#
#             avg_csi.append(t_csi)
#             avg_hss.append(t_hss)
#
#         # 记录平均分
#         eval_res['cikm_avg_csi'] = np.mean(avg_csi)
#         eval_res['cikm_avg_hss'] = np.mean(avg_hss)
#
#
#     if 'pod' in metrics:
#         current_threshold = threshold
#
#         # 判断：如果数据在 [0, 1] 范围内，但阈值是针对 0-255 的（比如 74），则归一化阈值
#         if pred.max() <= 1.05 and threshold > 1.0:
#             current_threshold = threshold / 255.0
#         hits, fas, misses = sevir_metrics(pred, true, current_threshold)
#         eval_res['pod'] = POD(hits, misses)
#         eval_res['sucr'] = SUCR(hits, fas)
#         eval_res['csi'] = CSI(hits, fas, misses)
#
#     pred = np.maximum(pred, clip_range[0])
#     pred = np.minimum(pred, clip_range[1])
#     # if 'ssim' in metrics:
#     #     ssim = 0
#     #     for b in range(pred.shape[0]):
#     #         for f in range(pred.shape[1]):
#     #             ssim += cal_ssim(pred[b, f].swapaxes(0, 2),
#     #                              true[b, f].swapaxes(0, 2), multichannel=True)
#     #     eval_res['ssim'] = ssim / (pred.shape[0] * pred.shape[1])
#
#     if 'ssim' in metrics:
#         # 严格对齐：在物理尺度(0-90)上计算 SSIM
#         ssim_val = 0
#         for b in range(pred.shape[0]):
#             for f in range(pred.shape[1]):
#                 # 注意：这里传入 data_range=scale (即90.0)
#                 ssim_val += cal_ssim(pred_rescaled[b, f].squeeze(),
#                                      true_rescaled[b, f].squeeze(),
#                                      data_range=scale)
#         eval_res['ssim'] = ssim_val / (pred.shape[0] * pred.shape[1])
#
#     if 'psnr' in metrics:
#         psnr = 0
#         for b in range(pred.shape[0]):
#             for f in range(pred.shape[1]):
#                 psnr += PSNR(pred[b, f], true[b, f])
#         eval_res['psnr'] = psnr / (pred.shape[0] * pred.shape[1])
#
#     if 'snr' in metrics:
#         snr = 0
#         for b in range(pred.shape[0]):
#             for f in range(pred.shape[1]):
#                 snr += SNR(pred[b, f], true[b, f])
#         eval_res['snr'] = snr / (pred.shape[0] * pred.shape[1])
#
#     if 'lpips' in metrics:
#         lpips = 0
#         cal_lpips = LPIPS(net='alex', use_gpu=False)
#         pred = pred.transpose(0, 1, 3, 4, 2)
#         true = true.transpose(0, 1, 3, 4, 2)
#         for b in range(pred.shape[0]):
#             for f in range(pred.shape[1]):
#                 lpips += cal_lpips(pred[b, f], true[b, f])
#         eval_res['lpips'] = lpips / (pred.shape[0] * pred.shape[1])
#
#     if return_log:
#         for k, v in eval_res.items():
#             eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
#             eval_log += eval_str
#
#     return eval_res, eval_log

# ... (保留前面的 imports 和辅助函数: HSS, rescale, _threshold, MAE, MSE, RMSE, PSNR, SNR, SSIM, POD, SUCR, CSI, sevir_metrics, LPIPS 类)

def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse'],
           clip_range=[0, 1], channel_names=None,
           spatial_norm=False, return_log=True, threshold=74.0):
    # 1. 反归一化处理 (如果输入经过了标准化，先还原到 [0, 1])
    if mean is not None and std is not None:
        pred = pred * std + mean
        true = true * std + mean

    # 确保输入严格在 [0, 1] 范围内，防止转换物理量时出现异常值
    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])
    true = np.maximum(true, clip_range[0])
    true = np.minimum(true, clip_range[1])

    eval_res = {}
    eval_log = ""

    # =========================================================
    #  核心修改：物理量转换逻辑 (dBZ)
    # =========================================================
    use_radar_metrics = 'radar_metrics' in metrics

    if use_radar_metrics:
        # CIKM / ShenZhen Met 物理参数
        # 0-255 对应 -10 到 85 dBZ，跨度为 95
        scale = 95.0
        offset = -10.0

        # 还原数值到物理空间 [-10, 85] dBZ
        # 公式: dBZ = Pixel_Norm * 95 - 10
        pred_eval = pred * scale + offset
        true_eval = true * scale + offset
    else:
        # 默认情况，保持 [0, 1]
        pred_eval = pred
        true_eval = true

    # 2. 计算基础指标 (MSE, MAE, RMSE) - 使用物理值或归一化值
    if 'mse' in metrics:
        eval_res['mse'] = MSE(pred_eval, true_eval, spatial_norm)
    if 'mae' in metrics:
        eval_res['mae'] = MAE(pred_eval, true_eval, spatial_norm)
    if 'rmse' in metrics:
        eval_res['rmse'] = RMSE(pred_eval, true_eval, spatial_norm)

    # 3. 计算气象分类指标 (CSI, HSS)
    if use_radar_metrics:
        # 常用阈值 (dBZ)
        # 注意：现在 pred_eval 已经是真实的 dBZ 值了，直接比较即可
        thresholds = [10, 20, 35, 40]
        # 注：CIKM比赛中常用阈值可能包括更高，如 [30, 40, 50]，可按需调整

        avg_csi, avg_hss = [], []
        total_pixels = np.prod(pred.shape[-2:])

        for t in thresholds:
            # 计算 hits, fas, misses
            # 这里传入的是物理值 (dBZ) 和 物理阈值 (t)
            hits_list, fas_list, misses_list = sevir_metrics(pred_eval, true_eval, t)

            # 累计所有样本的统计量 (避免单样本分母为0)
            hits_sum = np.sum(hits_list)
            fas_sum = np.sum(fas_list)
            misses_sum = np.sum(misses_list)
            cn_sum = total_pixels * pred.shape[0] * pred.shape[1] - hits_sum - misses_sum - fas_sum

            t_csi = CSI(hits_sum, fas_sum, misses_sum)
            t_hss = HSS(hits_sum, fas_sum, misses_sum, cn_sum)

            eval_res[f'csi_{t}'] = t_csi
            eval_res[f'hss_{t}'] = t_hss
            avg_csi.append(t_csi)
            avg_hss.append(t_hss)

        # 记录平均分
        eval_res['cikm_csi'] = np.mean(avg_csi)
        eval_res['cikm_hss'] = np.mean(avg_hss)

    # 兼容旧逻辑的 POD/CSI (针对单一阈值)
    if 'pod' in metrics or ('csi' in metrics and not use_radar_metrics):
        # 这里的 threshold 参数通常是 0-255 或 0-1
        current_threshold = threshold
        # 简单的归一化判断
        if threshold > 1.0:
            current_threshold = threshold / 255.0

        hits, fas, misses = sevir_metrics(pred, true, current_threshold)
        if 'pod' in metrics: eval_res['pod'] = POD(hits, misses)
        if 'sucr' in metrics: eval_res['sucr'] = SUCR(hits, fas)
        if 'csi' in metrics: eval_res['csi'] = CSI(hits, fas, misses)

    # 4. 图像质量指标 (SSIM, PSNR, LPIPS)
    if 'ssim' in metrics:
        # SSIM 建议在物理尺度下计算，或者保持一致即可
        # 物理尺度的 range 是 95 (-10到85)
        dr = 95.0 if use_radar_metrics else (clip_range[1] - clip_range[0])
        ssim_val = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim_val += cal_ssim(pred_eval[b, f].squeeze(),
                                     true_eval[b, f].squeeze(),
                                     data_range=dr)
        eval_res['ssim'] = ssim_val / (pred.shape[0] * pred.shape[1])

    if 'psnr' in metrics:
        # PSNR 建议始终使用 [0, 1] 归一化数据计算，便于与其他论文对比
        # 因为 PSNR 依赖于 Peak Value (1.0 或 255)，dBZ 的 Peak 定义比较模糊
        psnr_val = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                psnr_val += PSNR(pred[b, f], true[b, f], min_max_norm=True)
        eval_res['psnr'] = psnr_val / (pred.shape[0] * pred.shape[1])

    if 'lpips' in metrics:
        # LPIPS 模型是在 ImageNet (RGB) 上训练的，输入必须是归一化的 [0, 1] 或 [-1, 1]
        # 不要传 dBZ 进去，否则数值太大网络会输出无效值
        lpips_score = 0
        try:
            # 实例化放在外面更好，避免重复加载模型，但这里为了兼容保持不动
            cal_lpips = LPIPS(net='alex', use_gpu=True)
            for b in range(pred.shape[0]):
                for f in range(pred.shape[1]):
                    lpips_score += cal_lpips(pred[b, f], true[b, f])
            eval_res['lpips'] = lpips_score / (pred.shape[0] * pred.shape[1])
        except Exception as e:
            print(f"Warning: LPIPS calculation failed: {e}")

    if return_log:
        for k, v in eval_res.items():
            # 格式化输出，保留4位小数
            v_str = f"{v:.4f}" if isinstance(v, (float, np.float32, np.float64)) else v
            eval_str = f"{k}:{v_str}" if len(eval_log) == 0 else f", {k}:{v_str}"
            eval_log += eval_str

    return eval_res, eval_log
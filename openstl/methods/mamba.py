import torch
import torch.nn as nn
from openstl.models import MambaCast
from .base_method import Base_method
# 确保引用路径正确
from openstl.modules.mamba_Frequency_loss import FrequencyEnhancedLoss


class MambaMethod(Base_method):
    r"""MambaCast 的专用训练方法"""

    def __init__(self, **args):
        super().__init__(**args)

        # ================== 核心修改点 ==================
        # ❌ 错误写法: self.args.get(...) -> self.args 不存在
        # ✅ 正确写法: args.get(...)      -> 直接从传入的字典里取

        self.criterion = FrequencyEnhancedLoss(
            w_pixel=args.get('w_pixel', 1.0),

            # 读取 config 里的 w_freq
            w_freq=args.get('w_freq', 0.01),

            # 读取 config 里的 threshold
            threshold=args.get('threshold', 0.5),

            # 读取 config 里的 gamma
            gamma=args.get('gamma', 0.5),

            # 读取 config 里的 mask_ratio
            mask_ratio=args.get('mask_ratio', 3.0)
        )
        # ==============================================

    def _build_model(self, **args):
        return MambaCast(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        if aft_seq_length == pre_seq_length:
            pred_y = self.model(batch_x)
        elif aft_seq_length < pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :aft_seq_length]
        else:
            pred_y = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length

            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])

            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)

        loss = self.criterion(pred_y, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
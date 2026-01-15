import os
import numpy as np
import torch.nn as nn
import os.path as osp
import lightning as l
from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler, timm_schedulers
from openstl.core import metric


class Base_method(l.LightningModule):

    def __init__(self, **args):
        super().__init__()

        #if 'weather' in args['dataname']:
        if 'weather' in args['dataname'] or 'cikm' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        self.save_hyperparameters()
        self.model = self._build_model(**args)
        self.criterion = nn.MSELoss()
        self.test_outputs = []

    def _build_model(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        }
    
    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch):
        NotImplementedError
    
    def training_step(self, batch, batch_idx):
        NotImplementedError


    # ... (前面的 __init__, configure_optimizers 等代码保持不变) ...

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)

        # 获取数据集名称，统一转小写处理
        d_name = self.hparams.dataname.lower() if hasattr(self.hparams, 'dataname') else 'default'
        # 定义哪些数据集需要开启“内存节省模式”
        large_datasets = ['sevir']  #sevir
        is_large_dataset = any(name in d_name for name in large_datasets)

        outputs = {
            'inputs': batch_x.cpu().numpy(),
            'preds': pred_y.cpu().numpy(),
            'trues': batch_y.cpu().numpy()
        }

        # -----------------------------------------------------------
        # 模式 A: 大规模数据集 (SEVIR) - 边测边算，防止 OOM
        # -----------------------------------------------------------
        if is_large_dataset:
            # 1. 仅保存第 0 个 Batch 用于可视化
            if batch_idx == 0:
                save_dir = osp.join(self.hparams.save_dir, 'saved_batches')
                os.makedirs(save_dir, exist_ok=True)
                np.save(osp.join(save_dir, f'batch_{batch_idx}_true.npy'), outputs['trues'])
                np.save(osp.join(save_dir, f'batch_{batch_idx}_pred.npy'), outputs['preds'])
                print(f"\n--> [Visualization] Batch {batch_idx} saved to {save_dir} for plotting.")

            # 2. 处理反归一化参数
            if 'sevir' in d_name:
                current_mean, current_std = 0.0, 1.0
            else:
                current_mean = self.hparams.test_mean
                current_std = self.hparams.test_std

            # 3. 立即计算指标
            batch_eval_res, _ = metric(
                outputs['preds'], outputs['trues'],
                current_mean, current_std,
                metrics=self.metric_list,
                channel_names=self.channel_names,
                spatial_norm=self.spatial_norm,
                threshold=self.hparams.get('metric_threshold', None),
                dataset_name=d_name
            )

            # 返回计算好的指标，不再返回图片数据
            self.test_outputs.append(batch_eval_res)
            return {'loss': 0}

        # -----------------------------------------------------------
        # 模式 B: 小型数据集 (CIKM, shanghai,Meteo) - 收集所有数据，最后统一算
        # -----------------------------------------------------------
        else:
            # 直接将整个 batch 的数据存入列表，留给 on_test_epoch_end 处理
            self.test_outputs.append(outputs)
            return outputs

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return

        d_name = self.hparams.dataname.lower() if hasattr(self.hparams, 'dataname') else 'default'
        large_datasets = ['sevir']
        is_large_dataset = any(name in d_name for name in large_datasets)

        # -----------------------------------------------------------
        # 逻辑分支 A: 大规模数据集 (SEVIR) - 汇总已计算的指标
        # -----------------------------------------------------------
        if is_large_dataset:
            metrics_avg = {}
            first_res = self.test_outputs[0]
            # 此时 self.test_outputs 里全是指标字典，不是 numpy 数组
            for key in first_res.keys():
                values = [x[key] for x in self.test_outputs]
                metrics_avg[key] = np.mean(values)

            print("\n" + "=" * 40)
            print(f" Final Test Results ({self.hparams.dataname}) - Memory Efficient Mode")
            print("=" * 40)
            for k, v in metrics_avg.items():
                print(f"{k}: {v:.6f}")
            print("=" * 40 + "\n")

            self.log_dict(metrics_avg)

        # -----------------------------------------------------------
        # 逻辑分支 B: 小型数据集 (CIKM) - 拼接全量数据，计算指标并保存
        # -----------------------------------------------------------
        else:
            results_all = {}
            # 此时 self.test_outputs 里全是 inputs/preds/trues
            for k in self.test_outputs[0].keys():
                results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)

            # 统一计算指标
            eval_res, eval_log = metric(
                results_all['preds'], results_all['trues'],
                self.hparams.test_mean, self.hparams.test_std,
                metrics=self.metric_list,
                channel_names=self.channel_names,
                spatial_norm=self.spatial_norm,
                threshold=self.hparams.get('metric_threshold', None),
                dataset_name=self.hparams.dataname
            )

            # 记录关键指标
            results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

            if self.trainer.is_global_zero:
                print_log(eval_log)
                folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

                # 保存全量 .npy 文件
                print(f"Saving full test results to {folder_path}...")
                for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                    np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])

        # 清空列表释放内存
        self.test_outputs = []

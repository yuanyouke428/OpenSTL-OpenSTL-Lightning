import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
import sys

# 引入模块
sys.path.append(os.getcwd())
from openstl.modules.refiner import ResidualRefiner
from openstl.modules.discriminator import NLayerDiscriminator

# 尝试引入 LPIPS
try:
    import lpips

    use_lpips = True
    print("✅ LPIPS library found.")
except ImportError:
    use_lpips = False
    print("⚠️ LPIPS not found. Falling back to L1.")


# ================= 1. Dataloader (保持优化版不变) =================
class RefinerDataset(Dataset):
    def __init__(self, data_root, mode='train'):
        super().__init__()
        self.preds_files = sorted(glob.glob(os.path.join(data_root, mode, f'{mode}_preds_*.npy')))
        self.gts_files = sorted(glob.glob(os.path.join(data_root, mode, f'{mode}_gts_*.npy')))
        self.lasts_files = sorted(glob.glob(os.path.join(data_root, mode, f'{mode}_last_*.npy')))

        if len(self.preds_files) == 0:
            print(f"⚠️ Warning: No files found for {mode}. Trying train set (DEBUG).")
            self.preds_files = sorted(glob.glob(os.path.join(data_root, 'train', f'train_preds_*.npy')))
            self.gts_files = sorted(glob.glob(os.path.join(data_root, 'train', f'train_gts_*.npy')))
            self.lasts_files = sorted(glob.glob(os.path.join(data_root, 'train', f'train_last_*.npy')))

        self.chunk_sizes = []
        for f in self.preds_files:
            shape = np.load(f, mmap_mode='r').shape
            self.chunk_sizes.append(shape[0])

        self.total_len = sum(self.chunk_sizes)
        self.cumulative_sizes = np.cumsum(self.chunk_sizes)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        if file_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[file_idx - 1]

        p_path = self.preds_files[file_idx]
        g_path = self.gts_files[file_idx]
        l_path = self.lasts_files[file_idx]

        try:
            pred_mmap = np.load(p_path, mmap_mode='r')
            gt_mmap = np.load(g_path, mmap_mode='r')
            last_mmap = np.load(l_path, mmap_mode='r')

            pred_data = np.array(pred_mmap[sample_idx])
            gt_data = np.array(gt_mmap[sample_idx])
            last_data = np.array(last_mmap[sample_idx])
        except Exception as e:
            print(f"Error reading {p_path} at idx {sample_idx}: {e}")
            return torch.zeros(13, 1, 128, 128), torch.zeros(12, 1, 128, 128), torch.zeros(1, 1, 128, 128)

        pred = torch.from_numpy(pred_data).float()
        gt = torch.from_numpy(gt_data).float()
        last = torch.from_numpy(last_data).float()

        return pred, gt, last


# ================= 2. 验证函数 (已修复维度问题) =================
def validate(loader, refiner, l1_loss, epoch):
    refiner.eval()
    total_val_loss = 0
    count = 0
    with torch.no_grad():
        for preds, gts, lasts in tqdm(loader, desc=f"Validating Epoch {epoch}", leave=False):
            preds, gts, lasts = preds.cuda(), gts.cuda(), lasts.cuda()

            # --- 维度自动对齐 ---
            if preds.shape[1] != gts.shape[1]:
                min_t = min(preds.shape[1], gts.shape[1])
                preds = preds[:, :min_t].contiguous()  # 这里也加
                gts = gts[:, :min_t].contiguous()  # 这里也加
            # -------------------

            refined_imgs = refiner(preds, lasts)
            loss = l1_loss(refined_imgs, gts)
            total_val_loss += loss.item()
            count += 1
    return total_val_loss / (count + 1e-6)


# ================= 3. 训练主循环 (已修复维度问题) =================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/refiner_data')
    parser.add_argument('--batch_size', type=int, default=16)  # 之前是 32，如果显存紧张可以保持 16
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='work_dirs/refiner_v1')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading datasets...")
    train_ds = RefinerDataset(args.data_root, 'train')
    val_ds = RefinerDataset(args.data_root, 'val')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"🚀 Train: {len(train_ds)}, Val: {len(val_ds)}")

    refiner = ResidualRefiner().cuda()
    discriminator = NLayerDiscriminator().cuda()

    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    lpips_loss = None
    if use_lpips:
        lpips_loss = lpips.LPIPS(net='alex').cuda()

    opt_G = optim.Adam(refiner.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        refiner.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")

        for batch_idx, (preds, gts, lasts) in enumerate(pbar):
            preds, gts, lasts = preds.cuda(), gts.cuda(), lasts.cuda()

            # === ⚠️ 关键修复: 维度对齐 (Auto-Crop) ===
            # 如果 preds 是 13 帧，gts 是 12 帧，就砍掉 preds 的最后一帧
            if preds.shape[1] != gts.shape[1]:
                min_t = min(preds.shape[1], gts.shape[1])
                # preds = preds[:, :min_t]  # [B, 13] -> [B, 12]
                # gts = gts[:, :min_t]  # 如果反过来也兼容
                preds = preds[:, :min_t].contiguous()
                gts = gts[:, :min_t].contiguous()
            # ==========================================

            # --- Train Discriminator ---
            real_score = discriminator(gts)
            loss_d_real = mse_loss(real_score, torch.ones_like(real_score))

            refined_imgs = refiner(preds, lasts)
            fake_score = discriminator(refined_imgs.detach())
            loss_d_fake = mse_loss(fake_score, torch.zeros_like(fake_score))

            loss_d = (loss_d_real + loss_d_fake) * 0.5

            opt_D.zero_grad();
            loss_d.backward();
            opt_D.step()

            # --- Train Generator ---
            fake_score_g = discriminator(refined_imgs)
            loss_g_gan = mse_loss(fake_score_g, torch.ones_like(fake_score_g))
            loss_g_l1 = l1_loss(refined_imgs, gts)

            loss_g_lpips = 0
            if use_lpips:
                t_mid = refined_imgs.shape[1] // 2

                def norm(x): return (x - x.mean()) / (x.std() + 1e-5)

                flat_refined = norm(refined_imgs[:, t_mid].repeat(1, 3, 1, 1))
                flat_gt = norm(gts[:, t_mid].repeat(1, 3, 1, 1))
                loss_g_lpips = lpips_loss(flat_refined, flat_gt).mean()

            # 权重组合
            loss_g = 1.0 * loss_g_l1 + 0.05 * loss_g_gan + 0.5 * loss_g_lpips

            opt_G.zero_grad();
            loss_g.backward();
            opt_G.step()

            if batch_idx % 10 == 0:
                pbar.set_postfix({'G': loss_g.item(), 'D': loss_d.item()})

        # --- Validation ---
        val_loss = validate(val_loader, refiner, l1_loss, epoch)
        print(f"📊 Epoch {epoch + 1} Val L1 Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(refiner.state_dict(), os.path.join(args.save_dir, 'best_refiner.pth'))
            print(f"🏆 Best Model Saved!")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement: {epochs_no_improve}/{args.patience}")

        torch.save(refiner.state_dict(), os.path.join(args.save_dir, 'latest_refiner.pth'))

        if epochs_no_improve >= args.patience:
            print(f"🛑 Early stopping triggered.")
            break


if __name__ == '__main__':
    train()